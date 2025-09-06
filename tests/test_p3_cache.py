import os, torch, pytest
torch.ops.load_library(os.path.abspath("cmake-build/libautosage_cuda.so"))
from autosage._auto import spmm_csr_auto

@pytest.mark.cuda
def test_cache_roundtrip(tmp_path, monkeypatch):
    if not torch.cuda.is_available(): pytest.skip("no cuda")
    device="cuda"
    crow = torch.tensor([0,1,3,3,3], dtype=torch.long, device=device)
    col  = torch.tensor([1,2,3],     dtype=torch.long, device=device)
    x    = torch.randn(4, 32, device=device)
    monkeypatch.setenv("AUTOSAGE_CACHE_DIR", str(tmp_path))
    y1, info1 = spmm_csr_auto(crow, col, None, x, verbose=False)   # populate cache
    monkeypatch.setenv("AUTOSAGE_REPLAY_ONLY", "1")
    y2, info2 = spmm_csr_auto(crow, col, None, x, verbose=False)   # replay-only
    assert y1.shape == y2.shape
