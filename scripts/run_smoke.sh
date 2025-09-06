#!/usr/bin/env bash
set -euo pipefail
pytest -q
python - <<'PY'
import torch
from autosage.utils.csr import edge_index_to_csr
from autosage.ops import spmm_csr
N=4; F=4
edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]])
x = torch.randn(N,F)
crow, col = edge_index_to_csr(edge_index, N)
y = spmm_csr(crow, col, None, x)
print("OK shape:", tuple(y.shape))
PY
