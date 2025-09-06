import os, torch, pytest
torch.ops.load_library(os.path.abspath("cmake-build/libautosage_cuda.so"))
from autosage.ops import spmm_csr

@pytest.mark.cuda
@pytest.mark.parametrize("F", [32, 64, 128, 256])
def test_vec4_matches_cpu(F):
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    N = 512
    device = "cuda"
    idx = (torch.rand((N,N), device=device) < 0.003).nonzero(as_tuple=False).t()
    if idx.numel()==0:
        idx = torch.tensor([[0,1],[1,0]], device=device)
    row, col = idx[0], idx[1]
    perm = row.argsort(stable=True); row=row[perm]; col=col[perm]
    crow = torch.zeros(N+1, dtype=torch.long, device=device)
    crow.index_add_(0, row+1, torch.ones_like(row, dtype=torch.long)); crow = crow.cumsum(0)
    x = torch.randn(N, F, device=device, dtype=torch.float32).contiguous()
    y = spmm_csr(crow, col, None, x)
    # CPU ref
    y_ref = torch.zeros_like(x.cpu())
    cr, co = crow.cpu().tolist(), col.cpu().tolist()
    for r in range(N):
        s,e = cr[r], cr[r+1]
        if s<e: y_ref[r] = x.cpu()[co[s:e]].sum(0)
    assert torch.allclose(y, y_ref.to(device), atol=1e-5, rtol=0)
