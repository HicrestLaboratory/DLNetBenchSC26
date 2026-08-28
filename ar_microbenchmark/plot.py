#!/usr/bin/env python3

import csv
import io
import math
import re
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sbatchman as sbm

mpl.rcParams.update({
    #"text.usetex": True,
    #"text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 20,
    "legend.fontsize": 9,
    "legend.title_fontsize": 12,
    "figure.titlesize": 20,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

# Regex for the per-iteration straggler/variance warnings the benchmark
# prints to stderr, e.g.:
#   WARNING: iter 173 high rank variance: min=1.234567 ms (rank 0),
#   max=2.345678 ms (rank 5), spread=42.1% (threshold 10.0%)
WARNING_RE = re.compile(
    r"WARNING: iter (?P<iter>\d+) high rank variance: "
    r"min=(?P<min_ms>[\d.]+) ms \(rank (?P<min_rank>\d+)\), "
    r"max=(?P<max_ms>[\d.]+) ms \(rank (?P<max_rank>\d+)\), "
    r"spread=(?P<spread>[\d.]+)% \(threshold (?P<threshold>[\d.]+)%\)"
)

# Metric -> (column index in the stdout CSV, axis label, title fragment).
METRICS = {
    "time":    {"col": 1, "label": "Time (ms)",        "title": "Time"},
    "goodput": {"col": 2, "label": "Goodput (GB/s)",   "title": "Goodput"},
}

# Larger fonts.
# plt.rcParams.update({
#     "font.size": 14,
#     "axes.titlesize": 18,
#     "axes.labelsize": 16,
#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,
# })


def parse_stdout(text):
    """
    Parse the benchmark's stdout CSV text (header 'iter,time_ms,goodput_GBs',
    data rows, '#'-prefixed comment/summary lines) into a list of
    (iter, time_ms, goodput_GBs) tuples.
    """
    rows = []

    reader = csv.reader(io.StringIO(text or ""))
    next(reader, None)  # header

    for row in reader:
        if not row:
            continue

        if not row[0].strip() or row[0].strip().startswith("#"):
            continue

        if len(row) < 3:
            continue

        try:
            it = int(row[0])
            time_ms = float(row[1])
            goodput = float(row[2])
        except ValueError:
            continue

        rows.append((it, time_ms, goodput))

    return rows


def parse_stderr_warnings(text):
    """Parse per-iteration rank-variance warnings out of stderr text."""
    warnings = []

    for line in (text or "").splitlines():
        m = WARNING_RE.search(line)

        if not m:
            continue

        warnings.append({
            "iter": int(m.group("iter")),
            "min_ms": float(m.group("min_ms")),
            "min_rank": int(m.group("min_rank")),
            "max_ms": float(m.group("max_ms")),
            "max_rank": int(m.group("max_rank")),
            "spread_pct": float(m.group("spread")),
            "threshold_pct": float(m.group("threshold")),
        })

    return warnings


def choose_bin_count(values):
    """Choose a reasonable number of histogram bins."""
    if len(values) < 2:
        return 1

    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        return 1

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if iqr > 0:
        bin_width = 2 * iqr / (len(values) ** (1 / 3))

        if bin_width > 0:
            bins = math.ceil((vmax - vmin) / bin_width)
            return max(1, min(bins, 100))

    return max(1, min(math.ceil(math.sqrt(len(values))), 100))


def find_y_break(counts):
    """
    Find a useful y-axis break.

    Returns (low_max, high_min, high_max), or None if no
    meaningful break exists.
    """
    positive = np.sort(np.unique(counts[counts > 0]))

    if len(positive) < 3:
        return None

    ratios = positive[1:] / positive[:-1]
    index = np.argmax(ratios)

    low = positive[index]
    high = positive[index + 1]

    # Only introduce a break for a substantial gap.
    if high / low < 4:
        return None

    if low > 0.5 * high:
        return None

    # Padding around the two visible ranges.
    low_max = max(1, math.ceil(low * 1.25))
    high_min = max(low_max + 1, math.floor(high * 0.97))
    high_max = math.ceil(high * 1.03)

    if high_min <= low_max:
        return None

    return low_max, high_min, high_max


def rotate_y_ticks(ax):
    pass


def draw_break_marks(ax_low, ax_high):
    """Draw diagonal marks indicating the broken y-axis."""
    d = 0.012

    kwargs = dict(
        color="k",
        clip_on=False,
        linewidth=1.5,
    )

    ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)

    ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, **kwargs)


def draw_histogram(ax_low, ax_high, values, bins, y_break):
    """Draw the histogram on the lower and upper broken axes."""
    counts, bin_edges = np.histogram(values, bins=bins)

    ax_low.hist(values, bins=bin_edges)
    ax_high.hist(values, bins=bin_edges)

    if y_break is None:
        return

    low_max, high_min, high_max = y_break

    ax_low.set_ylim(0, low_max)
    ax_high.set_ylim(high_min, high_max)

    ax_low.spines["top"].set_visible(False)
    ax_high.spines["bottom"].set_visible(False)

    ax_high.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    draw_break_marks(ax_low, ax_high)

SYSTEM_NAMES_MAP = {
    'alps_clariden': 'Alps (Clariden)',
    'lumi': 'LUMI',
}
def group_label(group):
    """Human-readable subplot title for a (system, nodes, tasks_per_node) group."""
    system, nodes, tasks_per_node = group
    return f"{SYSTEM_NAMES_MAP.get(system, system)} - {nodes} nodes - {tasks_per_node} ranks/node"


def sorted_groups(data):
    return sorted(data.items(), key=lambda item: (str(item[0][0]), str(item[0][1]), str(item[0][2])))


def remove_high_runtime_outliers(rows, system):
    """
    Remove runs/iterations with unusually high runtimes using the
    standard upper IQR outlier rule.

    Returns rows with:
        time_ms <= Q3 + 100 * IQR
    """
    if len(rows) < 4:
        return rows

    times = np.array([r[1] for r in rows])

    q1, q3 = np.percentile(times, [25, 75])
    iqr = q3 - q1

    # If all runtimes are nearly identical, don't remove anything.
    if iqr <= 0:
        return rows

    upper_limit = q3 + 100 * iqr
    abs_upper_limit = 50 if system == 'lumi' else 150

    filtered_rows = [
        row for row in rows
        # if row[1] <= upper_limit
        if row[1] <= abs_upper_limit
    ]

    removed = len(rows) - len(filtered_rows)

    if removed:
        print(
            f"Removed {removed}/{len(rows)} high-runtime outliers "
            f"(runtime > {upper_limit:.3f} ms)"
        )

    return filtered_rows

def collect_data():
    """
    Pull all jobs from sbatchman and organize them as:

      metric_data[program][group] = {"time": [...], "goodput": [...]}
          merged across every job in that (program, group) -- used for
          the histogram figures.

      seq_data[program][group] = {"time": [...], "goodput": [...]}
          the ordered per-iteration series from a single representative
          job per group -- used for the sequence figures. sbatchman jobs
          don't carry an explicit run index, so we just keep the first
          job encountered per group (stable but arbitrary ordering from
          sbm.jobs_list()).

      variance_data[program][group] = [warning_dict, ...]
          all rank-variance warnings parsed from stderr, merged across
          every job in that (program, group).

    where group = (system, nodes, tasks_per_node).
    """
    jobs = sbm.jobs_list()

    metric_data = defaultdict(lambda: defaultdict(lambda: {"time": [], "goodput": []}))
    seq_data = defaultdict(dict)
    variance_data = defaultdict(lambda: defaultdict(list))

    for j in jobs:
        system = j.cluster_name
        variables = j.variables or {}

        program = variables.get("program", "unknown")
        nodes = variables.get("nodes", "?")
        tasks_per_node = variables.get("tasks_per_node", "?")
        group = (system, nodes, tasks_per_node)

        if nodes <= 1:
            continue

        rows = parse_stdout(j.get_stdout())

        # Remove unusually slow runtime outliers.
        rows = remove_high_runtime_outliers(rows, system)

        times = [r[1] for r in rows]
        goodputs = [r[2] for r in rows]

        metric_data[program][group]["time"].extend(times)
        metric_data[program][group]["goodput"].extend(goodputs)

        if group not in seq_data[program]:
            seq_data[program][group] = {
                "time": times,
                "goodput": goodputs,
            }

        variance_data[program][group].extend(
            parse_stderr_warnings(j.get_stderr())
        )

    return metric_data, seq_data, variance_data


def create_histogram_figure(data, metric, program):
    """Create and save the merged histogram figure for one metric/program."""
    cfg = METRICS[metric]
    items = sorted_groups(data)

    if not items:
        return None

    nplots = len(items)
    ncols = min(3, nplots)
    nrows = math.ceil(nplots / ncols)

    fig = plt.figure(figsize=(7 * ncols, 5.5 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.3)

    for index, (group, values_by_metric) in enumerate(items):
        values = values_by_metric[metric]
        row, col = divmod(index, ncols)
        title = group_label(group)

        if not values:
            ax = fig.add_subplot(outer[row, col])
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                     transform=ax.transAxes, fontsize=16)
            ax.set_title(title)
            continue

        bins = choose_bin_count(values)
        counts, bin_edges = np.histogram(values, bins=bins)
        y_break = find_y_break(counts)

        if y_break is None:
            ax = fig.add_subplot(outer[row, col])
            ax.hist(values, bins=bin_edges)
            ax.set_title(title)
            ax.set_xlabel(cfg["label"])
            ax.set_ylabel("Occurrences")
            ax.grid(axis="y", alpha=0.3)
            rotate_y_ticks(ax)
        else:
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.05)
            ax_high = fig.add_subplot(inner[0])
            ax_low = fig.add_subplot(inner[1], sharex=ax_high)

            draw_histogram(ax_low, ax_high, values, bin_edges, y_break)

            ax_high.set_title(title, fontsize=18)
            ax_low.set_xlabel(cfg["label"])
            ax_low.set_ylabel("Occurrences")
            ax_high.grid(axis="y", alpha=0.3)
            ax_low.grid(axis="y", alpha=0.3)
            rotate_y_ticks(ax_high)
            rotate_y_ticks(ax_low)

    for index in range(nplots, nrows * ncols):
        row, col = divmod(index, ncols)
        ax = fig.add_subplot(outer[row, col])
        ax.set_visible(False)

    # fig.suptitle(f"AllReduce {cfg['title']} Distributions — {program}", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"ar_{metric}_distribution_{program}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fname


def create_sequence_figure(data, metric, program):
    """Create and save the ordered-sequence figure for one metric/program."""
    cfg = METRICS[metric]
    items = sorted_groups(data)

    if not items:
        return None

    nplots = len(items)
    ncols = min(2, nplots)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax_i, (ax, (group, values_by_metric)) in enumerate(zip(axes, items)):
        values = values_by_metric[metric]

        if not values:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                     transform=ax.transAxes, fontsize=16)
        else:
            sample_indices = np.arange(len(values))
            ax.plot(sample_indices, values, linewidth=1)

        row_i = ax_i // ncols

        ax.set_title(group_label(group), fontsize=20)
        if row_i == nrows - 1:
            ax.set_xlabel("AllReduce Run")
        ax.set_ylabel(cfg["label"])
        ax.grid(axis="y", alpha=0.3)
        rotate_y_ticks(ax)

    for ax in axes[nplots:]:
        ax.set_visible(False)

    # fig.suptitle(f"AllReduce {cfg['title']} Sequences — {program}", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"ar_{metric}_sequence_{program}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fname


def create_variance_figure(data, program):
    """
    Create and save a figure visualizing rank-variance ("straggler")
    warnings parsed from stderr: per-iteration spread (%) against the
    configured threshold, one subplot per (system, nodes, tasks_per_node)
    group.
    """
    items = sorted_groups(data)

    if not items:
        return None

    nplots = len(items)
    ncols = min(3, nplots)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (group, warnings) in zip(axes, items):
        title = group_label(group)

        if not warnings:
            ax.text(0.5, 0.5, "No high-variance\niterations detected",
                     ha="center", va="center", transform=ax.transAxes,
                     fontsize=14, color="gray")
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Spread (%)")
            ax.grid(axis="y", alpha=0.3)
            continue

        iters = [w["iter"] for w in warnings]
        spreads = [w["spread_pct"] for w in warnings]
        threshold = warnings[0]["threshold_pct"]

        ax.scatter(iters, spreads, s=18, alpha=0.7, color="tab:red")
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1,
                   label=f"threshold ({threshold:.1f}%)")

        ax.set_title(f"{title}\n({len(warnings)} flagged iterations)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Spread (%)")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[nplots:]:
        ax.set_visible(False)

    fig.suptitle(f"AllReduce Rank-Variance Warnings — {program}",
                 fontsize=22, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"ar_variance_{program}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    metric_data, seq_data, variance_data = collect_data()

    if not metric_data:
        print("No jobs found.")
        return

    generated = []

    for program in sorted(metric_data):
        for metric in METRICS:
            fname = create_histogram_figure(metric_data[program], metric, program)
            if fname:
                generated.append(fname)

            fname = create_sequence_figure(seq_data.get(program, {}), metric, program)
            if fname:
                generated.append(fname)

        fname = create_variance_figure(variance_data.get(program, {}), program)
        if fname:
            generated.append(fname)

    print("Generated:")
    for fname in generated:
        print(f"  {fname}")


if __name__ == "__main__":
    main()