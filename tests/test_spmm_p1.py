import torch
import pytest
from autosage.ops import spmm_csr

def make_random_csr(n, p=0.05, device="cpu"):
    # Simple ER graph (ensure all tensors are on `device`)
    idx = (torch.rand((n, n), device=device) < p).nonzero(as_tuple=False).t()
    if idx.numel() == 0:
        idx = torch.tensor([[0],[0]], dtype=torch.long, device=device)
    row = idx[0]
    col = idx[1]
    perm = row.argsort(stable=True)
    row = row[perm]; col = col[perm]
    crow = torch.zeros(n+1, dtype=torch.long, device=device)
    crow.index_add_(0, row + 1, torch.ones_like(row, dtype=torch.long))
    crow = crow.cumsum(0)
    return crow, col

def cpu_ref(crow, col, val, x):
    # mirrors ops._cpu_reference_spmm
    n, f = x.shape
    y = x.new_zeros((n, f))
    crow_l = crow.tolist(); col_l = col.tolist()
    for r in range(n):
        s,e = crow_l[r], crow_l[r+1]
        if s==e: continue
        cols = col_l[s:e]
        if val is None:
            y[r] = x[cols].sum(dim=0)
        else:
            w = val[s:e].unsqueeze(1).to(x.dtype)
            y[r] = (x[cols] * w).sum(dim=0)
    return y

@pytest.mark.parametrize("n,f", [(64, 16), (64, 64), (64, 128), (128, 37)])
def test_spmm_cpu_matches_reference(n, f):
    crow, col = make_random_csr(n, p=0.1)
    x = torch.randn(n, f)
    y = spmm_csr(crow, col, None, x)
    y_ref = cpu_ref(crow, col, None, x)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=0)

@pytest.mark.cuda
@pytest.mark.parametrize("n,f", [(256, 32), (256, 64), (256, 128), (256, 113)])
def test_spmm_cuda_matches_reference(n, f):
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    crow, col = make_random_csr(n, p=0.05, device="cuda")
    x = torch.randn(n, f, device="cuda", dtype=torch.float32)
    y = spmm_csr(crow, col, None, x)
    y_ref = cpu_ref(crow.cpu(), col.cpu(), None, x.cpu()).to(x.device)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=0)

@pytest.mark.cuda
def test_spmm_cuda_hub_row():
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    n, f = 256, 64
    crow = torch.zeros(n+1, dtype=torch.long, device="cuda")
    # Make node 0 connected to all others (hub)
    col0 = torch.arange(1, n, dtype=torch.long, device="cuda")
    # The rest small
    col_rest = torch.tensor([0,2,3,4,5,6,7,8,9,10], dtype=torch.long, device="cuda")
    col = torch.cat([col0, col_rest])
    # crow: row 0 has len(col0), row 1 gets len(col_rest), others 0
    crow[1] = col0.numel()
    crow[2] = col0.numel() + col_rest.numel()
    crow = crow.cumsum(0)  # Already cumulative
    x = torch.randn(n, f, device="cuda", dtype=torch.float32)
    y = spmm_csr(crow, col, None, x)
    y_ref = x[col0].sum(0).unsqueeze(0)
    assert torch.allclose(y[0], y_ref[0], atol=1e-5)
