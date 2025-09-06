# AutoSAGE micro-probe + cache
import os
from typing import Dict, Any, Tuple, Optional
import torch
from ._cache import ScheduleCache, _device_sig, _graph_sig, log_probe

# compiled ops
_spmm = torch.ops.autosage.spmm_csr
_model_topk = torch.ops.autosage.model_topk

def _has_cusparse_op() -> bool:
    return hasattr(torch.ops.autosage, "cusparse_spmm_csr")

def _has_torch_sparse() -> bool:
    try:
        import torch_sparse  # noqa
        return True
    except Exception:
        return False

def _baseline_spmm(crow: torch.Tensor,
                   col:  torch.Tensor,
                   val:  Optional[torch.Tensor],
                   x:    torch.Tensor) -> torch.Tensor:
    if _has_cusparse_op():
        return torch.ops.autosage.cusparse_spmm_csr(crow, col, val if val is not None else None, x)
    if _has_torch_sparse():
        import torch_sparse
        nrows = int(crow.numel() - 1)
        row_counts = (crow[1:] - crow[:-1]).to(torch.long)
        row_idx = torch.repeat_interleave(
            torch.arange(nrows, device=col.device, dtype=torch.long),
            row_counts
        )
        coo = torch.stack([row_idx, col], dim=0)
        v = val if val is not None else torch.ones(int(col.numel()), device=x.device, dtype=x.dtype)
        return torch_sparse.spmm(coo, v, nrows, x.size(0), x)
    return _spmm(crow, col, val, x)

def _time_ms(fn, warmup: int = 1, iters: int = 5) -> Tuple[float, torch.Tensor]:
    # warmup
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    torch.cuda.synchronize()
    tmin = float("inf")
    last_out = out
    for _ in range(max(1, iters)):
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); last_out = fn(); t1.record(); t1.synchronize()
        t = t0.elapsed_time(t1)
        if t < tmin:
            tmin = t
    return tmin, last_out

def _induced_sample_csr(crow: torch.Tensor,
                        col:  torch.Tensor,
                        val:  Optional[torch.Tensor],
                        *, frac: float, min_rows: int, seed: int = 0):
    """Sample node set S and build induced square CSR on S. Returns (crow_s, col_s, val_s, rows_sel)."""
    assert crow.dtype == torch.long and col.dtype == torch.long
    N = int(crow.numel() - 1)
    m = min(max(int(N * float(frac)), int(min_rows), 1), N)
    dev = crow.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    rows_sel = torch.randperm(N, generator=gen, device=dev)[:m].sort().values

    mapping = torch.full((N,), -1, dtype=torch.long, device=dev)
    mapping[rows_sel] = torch.arange(m, device=dev, dtype=torch.long)

    crow_s = torch.zeros(m + 1, dtype=torch.long, device=dev)
    cols_buf = []
    vals_buf = [] if val is not None else None

    crow_cpu = crow.cpu()
    for i in range(m):
        r = int(rows_sel[i])
        s_ptr = int(crow_cpu[r]); e_ptr = int(crow_cpu[r+1])
        if e_ptr <= s_ptr:
            crow_s[i+1] = crow_s[i]; continue
        cols = col[s_ptr:e_ptr]
        mapped = mapping[cols]
        keep = mapped.ge(0)
        k = int(keep.sum().item())
        crow_s[i+1] = crow_s[i] + k
        if k > 0:
            cols_buf.append(mapped[keep])
            if val is not None:
                vals_buf.append(val[s_ptr:e_ptr][keep])

    col_s = (torch.cat(cols_buf, dim=0) if cols_buf else torch.zeros(0, dtype=torch.long, device=dev)).to(dev)
    val_s = None
    if val is not None:
        val_s = (torch.cat(vals_buf, dim=0) if vals_buf else torch.zeros(0, dtype=val.dtype, device=dev)).to(dev)
    return crow_s, col_s, val_s, rows_sel

def spmm_csr_auto(crow: torch.Tensor,
                  col:  torch.Tensor,
                  val:  Optional[torch.Tensor],
                  x:    torch.Tensor,
                  *, guardrail: float = 0.95, k: int = 3, seed: int = 0, verbose: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Decide baseline vs AutoSAGE via micro-probe and run on the full graph."""
    assert crow.dtype == torch.long and col.dtype == torch.long
    F = int(x.size(1))
    top = None

    cache = ScheduleCache()
    dev_sig = _device_sig()
    g_sig = _graph_sig(crow, col, F)
    cache_on = os.getenv("AUTOSAGE_CACHE", "1") == "1"
    replay_only = os.getenv("AUTOSAGE_REPLAY_ONLY", "0") == "1"

    if cache_on:
        rec = cache.lookup(dev_sig, g_sig)
        if rec is not None:
            choice_cached = rec.get("result", {}).get("choice")
            if choice_cached in ("autosage", "baseline"):
                if verbose or os.getenv("AUTOSAGE_VERBOSE") == "1":
                    print(f"[auto-cache] hit -> {choice_cached}")
                Y = _spmm(crow, col, val, x) if choice_cached == "autosage" else _baseline_spmm(crow, col, val, x)
                info = rec["result"] | {"choice": choice_cached, "from_cache": True, "top": rec["result"].get("top", [])}
                return Y, info

    if replay_only:
        Y = _baseline_spmm(crow, col, val, x)
        info = {
            "probe_rows": 0,
            "probe_nnz": 0,
            "use_autosage": False,
            "choice": "baseline",
            "tb_ms": None,
            "ta_ms": None,
            "speedup": None,
            "guardrail": guardrail,
            "top": [],
            "from_cache": False,
            "replay_only": True,
        }
        return Y, info

    # --- induced-subgraph micro-probe ---
    frac = float(os.getenv("AUTOSAGE_PROBE_FRAC", "0.02"))
    min_rows = int(os.getenv("AUTOSAGE_PROBE_MIN_ROWS", "256"))
    iters = int(os.getenv("AUTOSAGE_PROBE_ITERS", "5"))
    crow_s, col_s, val_s, rows_sel = _induced_sample_csr(crow, col, val, frac=frac, min_rows=min_rows, seed=seed)
    x_s = x.index_select(0, rows_sel).contiguous()
    probe_rows = int(crow_s.numel() - 1)
    probe_nnz  = int(col_s.numel())

    tb, _ = _time_ms(lambda: _baseline_spmm(crow_s, col_s, val_s, x_s), iters=iters)
    ta, _ = _time_ms(lambda: _spmm(crow_s, col_s, val_s, x_s), iters=iters)
    speedup = (tb / ta) if ta > 0 else float("inf")
    use_auto = ta <= guardrail * tb
    if (probe_rows < 64) or (probe_nnz < 512):
        use_auto = False
    choice = "autosage" if use_auto else "baseline"

    # Compute top-k candidates once (for logging & cache)
    if top is None:
        try:
            top = _model_topk(crow, col, F, int(k)).cpu().tolist()
        except Exception:
            top = []
    if verbose or os.getenv("AUTOSAGE_VERBOSE") == "1":
        print(f"[auto] rows={probe_rows} nnz={probe_nnz}  baseline={tb:.3f}ms  autosage={ta:.3f}ms  speedup={speedup:.3f}x  -> {choice}")
        print(f"[auto] top-{k} candidates: {top}")

    Y = _spmm(crow, col, val, x) if use_auto else _baseline_spmm(crow, col, val, x)

    info = {
        "probe_rows": probe_rows,
        "probe_nnz": probe_nnz,
        "use_autosage": use_auto,
        "choice": choice,
        "tb_ms": tb,
        "ta_ms": ta,
        "speedup": speedup,
        "guardrail": guardrail,
        "top": top,
        "from_cache": False,
    }
    if cache_on:
        cache.store(dev_sig, g_sig, info)
    log_probe({"ts": __import__("time").time(), "device": dev_sig, "graph": g_sig, "result": info})
    return Y, info

def calibrate_full(crow: torch.Tensor, col: torch.Tensor, val: Optional[torch.Tensor], x: torch.Tensor, guardrail: float = 0.95):
    """Time baseline vs autosage on the FULL graph once, cache the winner, and return (choice, info)."""
    from ._cache import ScheduleCache, _device_sig, _graph_sig, log_probe
    tb, _ = _time_ms(lambda: _baseline_spmm(crow, col, val, x), iters=3)
    ta, _ = _time_ms(lambda: _spmm(crow, col, val, x), iters=3)
    use_auto = (ta <= guardrail * tb)
    choice = "autosage" if use_auto else "baseline"
    # Also compute a quick top-k (best-effort; non-fatal)
    try:
        F = int(x.size(1))
        top = _model_topk(crow, col, F, int(3)).cpu().tolist()
    except Exception:
        top = []
    info = {
        "probe_rows": int(crow.numel()-1), "probe_nnz": int(col.numel()),
        "use_autosage": use_auto, "choice": choice, "tb_ms": tb, "ta_ms": ta,
        "speedup": (tb/ta) if ta > 0 else float("inf"), "guardrail": guardrail, "top": top
    }
    dev_sig = _device_sig(); g_sig = _graph_sig(crow, col, int(x.size(1)))
    ScheduleCache().store(dev_sig, g_sig, info)
    log_probe({"ts": __import__("time").time(), "device": dev_sig, "graph": g_sig, "result": info})
    return choice, info
