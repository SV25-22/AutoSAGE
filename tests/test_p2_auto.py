import os
import torch
import pytest

# Load compiled lib once
torch.ops.load_library(os.path.abspath("cmake-build/libautosage_cuda.so"))

from autosage.ops import spmm_csr_auto

def _tiny_graph(device="cuda"):
    crow = torch.tensor([0,1,3,3,3], dtype=torch.long, device=device)
    col  = torch.tensor([1,2,3],     dtype=torch.long, device=device)
    return crow, col

@pytest.mark.cuda
def test_model_topk_smoke():
    device = "cuda"
    crow, col = _tiny_graph(device)
    top = torch.ops.autosage.model_topk(crow, col, 128, 3)
    assert top.shape == (3, 3)
    ftile = top[0,0].item()
    assert ftile in (32, 64, 128)

@pytest.mark.cuda
def test_spmm_csr_auto_matches_kernel_and_reports_info():
    device = "cuda"
    crow, col = _tiny_graph(device)
    F = 128
    x = torch.randn(crow.numel()-1, F, device=device, dtype=torch.float32)

    # run auto
    y_auto, info = spmm_csr_auto(crow, col, None, x, verbose=False)
    # reference (your P1 kernel)
    y_ref = torch.ops.autosage.spmm_csr(crow, col, None, x)

    assert y_auto.shape == y_ref.shape
    assert torch.allclose(y_auto, y_ref, rtol=1e-3, atol=1e-3)
    # info dict sanity
    assert set(["choice","use_autosage","tb_ms","ta_ms","speedup","guardrail","top","probe_rows","probe_nnz"]).issubset(info.keys())

@pytest.mark.cuda
def test_guardrail_can_force_baseline_branch():
    device = "cuda"
    crow, col = _tiny_graph(device)
    F = 64
    x = torch.randn(crow.numel()-1, F, device=device)
    # Set an extremely strict guardrail so baseline is chosen even if autosage is faster
    y_base, info = spmm_csr_auto(crow, col, None, x, guardrail=0.01, verbose=False)
    assert info["choice"] == "baseline"
