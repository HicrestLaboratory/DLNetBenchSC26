// AllReduce microbenchmark (NCCL/CUDA or RCCL/HIP)
//
// Compile Alps (uenv=prgenv-gnu/26.3:v1, view=default)
// nvcc -O3 -std=c++17 -I/user-environment/env/default/include -arch=sm_90 -o ar ar.cu  -L/user-environment/env/default/lib -L/user-environment/env/default/lib64 -lmpi -lcudart -lnccl -lpthread -ldl
// nvcc -O3 -std=c++17 -DPER_ITER_MPI_BARRIER=1 -I/user-environment/env/default/include -arch=sm_90 -o ar_iter_barr ar.cu  -L/user-environment/env/default/lib -L/user-environment/env/default/lib64 -lmpi -lcudart -lnccl -lpthread -ldl
//
// Optional: -DPER_ITER_MPI_BARRIER=1 to add an MPI_Barrier before every
// timed iteration (off by default -- see the timed loop below).
//
// Optional: -DVARIANCE_WARN_REL_THRESHOLD=0.10 to change the relative
// (max-min)/mean threshold, per iteration, above which a straggler warning
// is printed to stderr (default 10%).

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------------------------------------------------------------
// GPU / collective-comm backend abstraction: CUDA+NCCL or HIP+RCCL.
// Select HIP+RCCL by compiling with -DUSE_HIP.
// ---------------------------------------------------------------------
#if defined(USE_HIP)
    #include <hip/hip_runtime.h>
    #include <hip/hip_bf16.h>
    #include <rccl/rccl.h>

    typedef hipError_t     gpuError_t;
    typedef hipEvent_t     gpuEvent_t;
    typedef __hip_bfloat16 bf16_t;

    #define gpuSuccess           hipSuccess
    #define gpuGetErrorString    hipGetErrorString
    #define gpuMalloc            hipMalloc
    #define gpuMemset            hipMemset
    #define gpuFree              hipFree
    #define gpuSetDevice         hipSetDevice
    #define gpuGetDeviceCount    hipGetDeviceCount
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuEventCreate       hipEventCreate
    #define gpuEventRecord       hipEventRecord
    #define gpuEventSynchronize  hipEventSynchronize
    #define gpuEventElapsedTime  hipEventElapsedTime
    #define gpuEventDestroy      hipEventDestroy
    #define ncclBf16             ncclBfloat16   // RCCL mirrors the NCCL enum names
#else
    #include <cuda_runtime.h>
    #include <cuda_bf16.h>
    #include <nccl.h>

    typedef cudaError_t   gpuError_t;
    typedef cudaEvent_t   gpuEvent_t;
    typedef __nv_bfloat16 bf16_t;

    #define gpuSuccess           cudaSuccess
    #define gpuGetErrorString    cudaGetErrorString
    #define gpuMalloc            cudaMalloc
    #define gpuMemset            cudaMemset
    #define gpuFree              cudaFree
    #define gpuSetDevice         cudaSetDevice
    #define gpuGetDeviceCount    cudaGetDeviceCount
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuEventCreate       cudaEventCreate
    #define gpuEventRecord       cudaEventRecord
    #define gpuEventSynchronize  cudaEventSynchronize
    #define gpuEventElapsedTime  cudaEventElapsedTime
    #define gpuEventDestroy      cudaEventDestroy
    #define ncclBf16             ncclBfloat16
#endif

// ---- Hardcoded benchmark parameters ----
#define BUFFER_BYTES   25300000UL          // 25.3 MB, decimal
#define WARMUP_ITERS   0
#define TIMED_ITERS    2000

// Default: no barrier inside the timed loop. Compile with
// -DPER_ITER_MPI_BARRIER=1 to re-enable a barrier before every iteration.
#ifndef PER_ITER_MPI_BARRIER
#define PER_ITER_MPI_BARRIER 0
#endif

// Relative spread ((max-min)/mean) across ranks, per iteration, above which
// a straggler/variance warning is printed to stderr.
#ifndef VARIANCE_WARN_REL_THRESHOLD
#define VARIANCE_WARN_REL_THRESHOLD 0.10
#endif

#define GPU_CHECK(cmd) do {                                              \
    gpuError_t e = cmd;                                                  \
    if (e != gpuSuccess) {                                               \
        fprintf(stderr, "GPU error %s:%d '%s'\n", __FILE__, __LINE__,    \
                gpuGetErrorString(e));                                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                    \
    }                                                                    \
} while (0)

#define NCCL_CHECK(cmd) do {                                             \
    ncclResult_t r = cmd;                                                \
    if (r != ncclSuccess) {                                              \
        fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
                ncclGetErrorString(r));                                  \
        MPI_Abort(MPI_COMM_WORLD, 1);                                    \
    }                                                                    \
} while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Determine local rank (per-node) to pick the right GPU
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                         MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);

    int ndevices;
    GPU_CHECK(gpuGetDeviceCount(&ndevices));
    GPU_CHECK(gpuSetDevice(local_rank % ndevices));

    const size_t N = BUFFER_BYTES / sizeof(bf16_t);  // element count

    // ---- NCCL/RCCL init ----
    ncclUniqueId id;
    ncclComm_t comm;
    if (world_rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

    // ---- Allocate + init buffers ----
    bf16_t *d_send, *d_recv;
    GPU_CHECK(gpuMalloc(&d_send, N * sizeof(bf16_t)));
    GPU_CHECK(gpuMalloc(&d_recv, N * sizeof(bf16_t)));
    GPU_CHECK(gpuMemset(d_send, 0, N * sizeof(bf16_t)));
    GPU_CHECK(gpuMemset(d_recv, 0, N * sizeof(bf16_t)));

    gpuEvent_t start, stop;
    GPU_CHECK(gpuEventCreate(&start));
    GPU_CHECK(gpuEventCreate(&stop));

    // ---- Warmup ----
    for (int i = 0; i < WARMUP_ITERS; i++) {
        NCCL_CHECK(ncclAllReduce(d_send, d_recv, N, ncclBf16, ncclSum,
                                  comm, 0));
    }
    GPU_CHECK(gpuDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Bytes actually moved per rank in a ring AllReduce (standard formula),
    // used as the numerator for "goodput" below.
    double moved_bytes = (double)BUFFER_BYTES * 2.0 * (world_size - 1) / world_size;

    // Per-iteration local (this-rank) elapsed time, in milliseconds.
    double *local_ms = (double*)malloc(sizeof(double) * TIMED_ITERS);
    
    GPU_CHECK(gpuDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Timed loop: no cross-rank communication other than the GPU
    // collective itself (plus an optional barrier). Times are only
    // gathered/analyzed once, after the loop, not on every iteration.
    for (int i = 0; i < TIMED_ITERS; i++) {
#if PER_ITER_MPI_BARRIER
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        GPU_CHECK(gpuEventRecord(start, 0));
        NCCL_CHECK(ncclAllReduce(d_send, d_recv, N, ncclBf16, ncclSum,
                                  comm, 0));
        GPU_CHECK(gpuEventRecord(stop, 0));
        GPU_CHECK(gpuEventSynchronize(stop));

        float ms;
        GPU_CHECK(gpuEventElapsedTime(&ms, start, stop));
        local_ms[i] = (double)ms;
    }

    // ---- Gather every rank's full timing series to rank 0 (no reduction
    // yet). all_ms is laid out as [rank][iter], i.e. rank r's data for
    // iteration i is at all_ms[r * TIMED_ITERS + i].
    double *all_ms = (world_rank == 0)
        ? (double*)malloc(sizeof(double) * (size_t)world_size * TIMED_ITERS)
        : NULL;
    MPI_Gather(local_ms, TIMED_ITERS, MPI_DOUBLE,
               all_ms,   TIMED_ITERS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- Analysis + CSV output + summary (rank 0 only) ----
    // Lines starting with '#' are comments/summary; data rows are plain CSV
    // so this stdout can be redirected straight into a .csv and parsed with
    // e.g. pandas.read_csv(path, comment='#').
    if (world_rank == 0) {
        double *max_ms  = (double*)malloc(sizeof(double) * TIMED_ITERS);
        double *goodput = (double*)malloc(sizeof(double) * TIMED_ITERS);

        printf("iter,time_ms,goodput_GBs\n");
        for (int i = 0; i < TIMED_ITERS; i++) {
            // Per-iteration min/max/mean across ranks, and which ranks hit
            // the extremes -- used both for the reported metric (worst-case
            // rank) and for straggler detection.
            double lo = all_ms[i], hi = all_ms[i];
            int lo_rank = 0, hi_rank = 0;
            double sum = 0.0;
            for (int r = 0; r < world_size; r++) {
                double v = all_ms[(size_t)r * TIMED_ITERS + i];
                sum += v;
                if (v < lo) { lo = v; lo_rank = r; }
                if (v > hi) { hi = v; hi_rank = r; }
            }
            double mean = sum / world_size;

            max_ms[i]  = hi;
            goodput[i] = moved_bytes / (hi / 1000.0) / 1e9;
            printf("%d,%.6f,%.3f\n", i, max_ms[i], goodput[i]);

            // Flag iterations where ranks disagree significantly on timing
            // (e.g. a straggler GPU/NIC) -- printed to stderr so it doesn't
            // pollute the CSV on stdout.
            if (world_size > 1 && mean > 0.0) {
                double rel_spread = (hi - lo) / mean;
                if (rel_spread > VARIANCE_WARN_REL_THRESHOLD) {
                    fprintf(stderr,
                        "WARNING: iter %d high rank variance: min=%.6f ms (rank %d), "
                        "max=%.6f ms (rank %d), spread=%.1f%% (threshold %.1f%%)\n",
                        i, lo, lo_rank, hi, hi_rank,
                        rel_spread * 100.0, VARIANCE_WARN_REL_THRESHOLD * 100.0);
                }
            }
        }

        double sum_ms = 0.0, sum_goodput = 0.0;
        for (int i = 0; i < TIMED_ITERS; i++) {
            sum_ms      += max_ms[i];
            sum_goodput += goodput[i];
        }
        double mean_ms      = sum_ms / TIMED_ITERS;
        double mean_goodput = sum_goodput / TIMED_ITERS;

        double var_ms = 0.0, var_goodput = 0.0;
        for (int i = 0; i < TIMED_ITERS; i++) {
            double dt = max_ms[i] - mean_ms;
            double dg = goodput[i] - mean_goodput;
            var_ms      += dt * dt;
            var_goodput += dg * dg;
        }
        double std_ms      = sqrt(var_ms / TIMED_ITERS);
        double std_goodput = sqrt(var_goodput / TIMED_ITERS);

        printf("#\n");
        printf("# === AllReduce benchmark summary (worst-case rank per iter) ===\n");
        printf("# Backend:          %s\n",
#if defined(USE_HIP)
               "HIP + RCCL"
#else
               "CUDA + NCCL"
#endif
        );
        printf("# Ranks (GPUs):     %d\n", world_size);
        printf("# Buffer size:      %lu bytes (%.2f MB)\n",
               (unsigned long)BUFFER_BYTES, BUFFER_BYTES / 1e6);
        printf("# Dtype:            bfloat16\n");
        printf("# Warmup iters:     %d\n", WARMUP_ITERS);
        printf("# Timed iters:      %d\n", TIMED_ITERS);
        printf("# Per-iter barrier: %s\n", PER_ITER_MPI_BARRIER ? "on" : "off");
        printf("# Variance thresh:  %.1f%% (relative, per-iter, warnings on stderr)\n",
               VARIANCE_WARN_REL_THRESHOLD * 100.0);
        printf("# Mean time (max):  %.6f ms  (std: %.6f ms)\n", mean_ms, std_ms);
        printf("# Mean goodput:     %.3f GB/s  (std: %.3f GB/s)\n", mean_goodput, std_goodput);

        free(goodput);
        free(max_ms);
        free(all_ms);
    }

    GPU_CHECK(gpuEventDestroy(start));
    GPU_CHECK(gpuEventDestroy(stop));
    free(local_ms);
    GPU_CHECK(gpuFree(d_send));
    GPU_CHECK(gpuFree(d_recv));
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
