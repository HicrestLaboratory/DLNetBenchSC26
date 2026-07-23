#!/bin/bash

cd common

git clone https://github.com/HicrestLaboratory/JobPlacer.git
git clone https://github.com/ThomasPasquali/ccutils.git

cd JobPlacer
git checkout SC26
cargo build --release

cd ../ccutils
git checkout SC26
if [[ -z "$CCUTILS_INCLUDE" ]]; then
    wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/ccutils_json/install.sh | env bash
fi

cd ../../DLNetBench/
git clone https://github.com/HicrestLaboratory/DLNetBench.git
cd DLNetBench
git checkout SC26