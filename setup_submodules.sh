#!/bin/bash

git submodule init
git submodule update

cd common/JobPlacer
git checkout SC26
cargo build --release

cd ../ccutils
git checkout SC26
if [[ -z "$CCUTILS_INCLUDE" ]]; then
    wget -qO- https://raw.githubusercontent.com/ThomasPasquali/ccutils/ccutils_json/install.sh | env bash
fi

cd ../../DLNetBench/DLNetBench
git checkout SC26
