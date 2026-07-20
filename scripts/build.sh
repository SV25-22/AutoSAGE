#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
TORCH_PREFIX="$($PYTHON_BIN -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --preset release -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset release --parallel "${BUILD_JOBS:-$(nproc)}"

echo "Built build/libautosage_cuda.so"
