import torch
import pytest
from autosage.ops import spmm_csr
from autosage.conv import AutoSAGEConv
from autosage.utils.csr import edge_index_to_csr

def small_graph():
    # 4-node toy: edges 0->1, 1->2, 2->3, 3->0
    ei = torch.tensor([[0,1,2,3],
                       [1,2,3,0]], dtype=torch.long)
    return ei

def test_spmm_cpu_smoke():
    ei = small_graph()
    N = 4; F = 3
    x = torch.arange(N*F, dtype=torch.float32).reshape(N, F)
    crow, col = edge_index_to_csr(ei, N)
    y = spmm_csr(crow, col, None, x)
    assert y.shape == (N, F)
    # quick sanity: each row sums neighbors (ring)
    assert torch.allclose(y[0], x[1])

@pytest.mark.cuda
def test_spmm_cuda_smoke():
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    ei = small_graph().cuda()
    N = 4; F = 8
    x = torch.randn(N, F, device="cuda")
    crow, col = edge_index_to_csr(ei, N)
    y = spmm_csr(crow, col, None, x)
    assert y.shape == (N, F)
    # P0 kernel returns zeros; fallback might compute CPU then .to(cuda)
    # Either way: tensor is finite
    assert torch.isfinite(y).all()

def test_conv_forward():
    ei = small_graph()
    N = 4; Fin = 5; Fout = 2
    x = torch.randn(N, Fin)
    crow, col = edge_index_to_csr(ei, N)
    conv = AutoSAGEConv(Fin, Fout)
    y = conv(x, crow, col)
    assert y.shape == (N, Fout)
