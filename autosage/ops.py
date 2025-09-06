import torch
from typing import Optional

# Try to load the compiled library once (works both for editable installs and CMake builds)
try:
    torch.ops.load_library("autosage_cuda")  # e.g., build/lib*/autosage_cuda.so
except OSError:
    try:
        # If setuptools built it, torch usually auto-loads; keep a loose try
        pass
    except Exception:
        pass

def _cpu_reference_spmm(crow: torch.Tensor,
                        col: torch.Tensor,
                        val: Optional[torch.Tensor],
                        x: torch.Tensor) -> torch.Tensor:
    # Slow but correct baseline using scatter-add (for smoke/fallback)
    n = x.shape[0]
    y = x.new_zeros((n, x.shape[1]))
    # Build per-row loop from CSR
    crow_l = crow.tolist()
    col_l = col.tolist()
    w = val if val is not None else None
    for r in range(n):
        start, end = crow_l[r], crow_l[r+1]
        if start == end:
            continue
        cols = col_l[start:end]
        if w is None:
            y[r] = x[cols].sum(dim=0)
        else:
            weights = w[start:end].unsqueeze(1).to(x.dtype)
            y[r] = (x[cols] * weights).sum(dim=0)
    return y

def spmm_csr(crow: torch.Tensor,
             col: torch.Tensor,
             val: Optional[torch.Tensor],
             x: torch.Tensor) -> torch.Tensor:
    """
    crow: [N+1] int64 CSR rowptr
    col:  [nnz]  int64 CSR colidx
    val:  [nnz]  optional float edge weights
    x:    [N, F] dense features
    """
    if x.is_cuda and torch.cuda.is_available():
        try:
            return torch.ops.autosage.spmm_csr(crow, col, val, x)
        except (RuntimeError, AttributeError):
            # Extension not loaded—fall through
            pass
    # CPU or fallback path
    return _cpu_reference_spmm(crow.cpu(), col.cpu(), None if val is None else val.cpu(), x.cpu()).to(x.device)
