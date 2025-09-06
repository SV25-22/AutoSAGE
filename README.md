# AutoSAGE (P0)

CUDA-only, input-aware auto-scheduler scaffold for sparse GNN aggregators.

## What works in P0
- Buildable PyTorch CUDA extension via setuptools **and** CMake
- Custom `autosage::spmm_csr` op registered with Torch dispatcher
- Safe runtime fallback (CPU reference path) when CUDA/op not available
- Smoke tests (CPU + CUDA) and simple PyG-style wrapper (`AutoSAGEConv` stub)

## Quickstart
```bash
python -m pip install -U pip
pip install -e .
bash scripts/build_ext.sh
python scripts/env_info.py
pytest -q
python scripts/run_smoke.sh
