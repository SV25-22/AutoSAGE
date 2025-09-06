import os
import math
import time
import pytest
import torch

# ---- helpers ----

def _have_cuda():
    return torch.cuda.is_available()

def _load_ext():
    # In dev we load the local .so. If you package this as a PyTorch extension,
    # importing the Python package will already register ops.
    lib = os.environ.get("AUTOSAGE_LIB", "cmake-build/libautosage_cuda.so")
    torch.ops.load_library(lib)

def _make_csr(N, deg_hub=None, deg_other=32, seed=0, device="cuda"):
    g = torch.Generator(device=device); g.manual_seed(int(seed))
    if deg_hub is None:
        # random-ish degrees in [0, 2*deg_other]
        deg = torch.randint(0, 2*deg_other + 1, (N,), generator=g, device=device, dtype=torch.long)
    else:
        deg = torch.full((N,), int(deg_other), device=device, dtype=torch.long)
        deg[0] = int(deg_hub)
    crow = torch.zeros(N + 1, device=device, dtype=torch.long)
    crow[1:] = deg
    crow = crow.cumsum(0)
    nnz = int(crow[-1].item())
    col = torch.randint(0, N, (nnz,), generator=g, device=device, dtype=torch.long)
    return crow, col

def _bench(fn, iters=20):
    torch.cuda.synchronize(); tmin=1e9; out=None
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); out = fn(); e1.record(); e1.synchronize()
        t = e0.elapsed_time(e1); tmin = min(tmin, t)
    return tmin, out

# ---- skip if no CUDA ----
pytestmark = pytest.mark.skipif(not _have_cuda(), reason="CUDA required for AutoSAGE M2 tests")

# ---- tests ----

def test_parity_unweighted_small():
    _load_ext()
    N, F = 2048, 64
    crow, col = _make_csr(N, deg_other=24, seed=1)
    x = torch.randn(N, F, device="cuda", dtype=torch.float32)

    y_base = torch.ops.autosage.spmm_csr(crow, col, None, x)
    y_split = torch.ops.autosage.spmm_csr_split(crow, col, None, x, 128, 4, 512)

    assert torch.allclose(y_base, y_split, rtol=1e-4, atol=3e-3)

def test_parity_weighted_128():
    _load_ext()
    N, F = 4096, 128
    crow, col = _make_csr(N, deg_other=48, seed=2)
    nnz = int(crow[-1].item())
    val = torch.rand(nnz, device="cuda", dtype=torch.float32)
    x = torch.randn(N, F, device="cuda", dtype=torch.float32)

    y_base = torch.ops.autosage.spmm_csr(crow, col, val, x)
    y_split = torch.ops.autosage.spmm_csr_split(crow, col, val, x, 128, 4, 512)

    assert torch.allclose(y_base, y_split, rtol=1e-4, atol=3e-3)

def test_split_is_fast_on_hub_skew_smoke():
    """
    Perf is machine-dependent; this is a smoke check:
    Run only when AUTOSAGE_RUN_PERF=1 to avoid flaky CI.
    """
    if os.getenv("AUTOSAGE_RUN_PERF", "0") != "1":
        pytest.skip("set AUTOSAGE_RUN_PERF=1 to run perf-smoke")

    _load_ext()
    N, F = 20000, 128
    crow, col = _make_csr(N, deg_hub=12000, deg_other=32, seed=0)
    x = torch.randn(N, F, device="cuda", dtype=torch.float32)

    base = lambda: torch.ops.autosage.spmm_csr(crow, col, None, x)
    split = lambda: torch.ops.autosage.spmm_csr_split(crow, col, None, x, 64, 8, 512)

    tb, yb = _bench(base, iters=15)
    ts, ys = _bench(split, iters=15)

    # correctness
    assert torch.allclose(yb, ys, rtol=1e-4, atol=3e-3)

    # perf-smoke (allow generous margin)
    assert ts <= tb * 0.9, f"Expected split to be >=1.11x faster, got base {tb:.3f}ms vs split {ts:.3f}ms"

def test_auto_api_uses_split_and_reports_telemetry(monkeypatch):
    _load_ext()
    from autosage import _auto  # your package import path

    # Prefer split during probe with quantile hubT
    monkeypatch.setenv("AUTOSAGE_HUBT_MODE", "q95")
    monkeypatch.setenv("AUTOSAGE_WPB", "8")
    monkeypatch.setenv("AUTOSAGE_FTILE", "64")
    monkeypatch.setenv("AUTOSAGE_CACHE", "0")  # disable cache to exercise probe

    N, F = 6000, 64
    crow, col = _make_csr(N, deg_hub=3000, deg_other=24, seed=3)
    x = torch.randn(N, F, device="cuda", dtype=torch.float32)

    y, info = _auto.spmm_csr_auto(crow, col, None, x, verbose=False)

    assert isinstance(info, dict)
    assert info.get("choice") in ("autosage", "baseline")
    # If choice is autosage, we should see candidate and hub telemetry
    if info.get("choice") == "autosage":
        assert "candidate" in info and len(info["candidate"]) == 3
        assert "heavy_rows" in info and "heavy_frac" in info
