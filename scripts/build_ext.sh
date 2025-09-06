#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "is_available:", torch.cuda.is_available())
PY
# Build (editable)
pip install -e .
# Optional: build via CMake into ./build
mkdir -p cmake-build && cd cmake-build

TORCH_CMAKE_PREFIX=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))
PY
)

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
      -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      ..

cmake --build . -j


cd ..
echo "Build done."
