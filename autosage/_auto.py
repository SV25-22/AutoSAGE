# AutoSAGE micro-probe + cache + M2 CTA-per-hub plumbing
import os
from typing import Dict, Any, Tuple, Optional, List
import torch
from ._cache import ScheduleCache, _device_sig, _graph_sig, log_probe

# ----- compiled ops -----
_spmm = torch.ops.autosage.spmm_csr
_spmm_split = getattr(torch.ops.autosage, "spmm_csr_split", None)   # M2 (may be missing if not built)
_model_topk = getattr(torch.ops.autosage, "model_topk", None)       # may be missing

# ----- env toggles / overrides -----
AUTOSAGE_HUB_CTA = os.getenv("AUTOSAGE_HUB_CTA", "1") == "1"  # enable CTA-per-hub split path
_AUTOSAGE_FTILE  = os.getenv("AUTOSAGE_FTILE")                # optional override (e.g., 64/128)
_AUTOSAGE_WPB    = os.getenv("AUTOSAGE_WPB")                  # optional override (e.g., 2/4/8)
_AUTOSAGE_HUB_T  = os.getenv("AUTOSAGE_HUB_T")                # optional override (degree threshold)


# ===== internals ===============================================================

def _has_cusparse_op() -> bool:
    return hasattr(torch.ops.autosage, "cusparse_spmm_csr")


def _has_torch_sparse() -> bool:
    try:
        import torch_sparse  # noqa: F401
        return True
    except Exception:
        return False


def _baseline_spmm(crow: torch.Tensor,
                   col:  torch.Tensor,
                   val:  Optional[torch.Tensor],
                   x:    torch.Tensor) -> torch.Tensor:
    """Prefer vendor baselines when present (cuSPARSE > torch_sparse > ours)."""
    if _has_cusparse_op():
        return torch.ops.autosage.cusparse_spmm_csr(
            crow, col, val if val is not None else None, x
        )
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
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); last_out = fn(); e1.record(); e1.synchronize()
        t = e0.elapsed_time(e1)
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
    cols_buf: List[torch.Tensor] = []
    vals_buf: Optional[List[torch.Tensor]] = [] if val is not None else None

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
            if val is not None and vals_buf is not None:
                vals_buf.append(val[s_ptr:e_ptr][keep])

    col_s = (torch.cat(cols_buf, dim=0) if cols_buf else torch.zeros(0, dtype=torch.long, device=dev)).to(dev)
    val_s = None
    if val is not None:
        val_s = (torch.cat(vals_buf, dim=0) if vals_buf else torch.zeros(0, dtype=val.dtype, device=dev)).to(dev)
    return crow_s, col_s, val_s, rows_sel


def _pick_candidate(top: Optional[List], default=(128, 4, 1024)) -> Tuple[int, int, int]:
    """Return (ftile, wpb, hubT) from top-1; allow env overrides and dict/list shapes."""
    ft, wpb, hubT = default
    if top and len(top) > 0:
        cand = top[0]
        if isinstance(cand, dict):
            ft   = int(cand.get("ftile", ft))
            wpb  = int(cand.get("wpb", wpb))
            hubT = int(cand.get("hubT", hubT))
        else:
            try:
                ft   = int(cand[0]); wpb = int(cand[1]); hubT = int(cand[2])
            except Exception:
                pass
    if _AUTOSAGE_FTILE:
        ft = int(_AUTOSAGE_FTILE)
    if _AUTOSAGE_WPB:
        wpb = int(_AUTOSAGE_WPB)
    if _AUTOSAGE_HUB_T:
        hubT = int(_AUTOSAGE_HUB_T)
    return ft, wpb, hubT


# ===== public API ==============================================================

def spmm_csr_auto(crow: torch.Tensor,
                  col:  torch.Tensor,
                  val:  Optional[torch.Tensor],
                  x:    torch.Tensor,
                  *, guardrail: float = 0.95, k: int = 3, seed: int = 0, verbose: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Decide baseline vs AutoSAGE (M2 split when available) via micro-probe and run on the full graph."""
    assert crow.dtype == torch.long and col.dtype == torch.long
    F = int(x.size(1))
    top: Optional[List] = None

    cache = ScheduleCache()
    dev_sig = _device_sig()
    g_sig = _graph_sig(crow, col, F)
    cache_on = os.getenv("AUTOSAGE_CACHE", "1") == "1"
    replay_only = os.getenv("AUTOSAGE_REPLAY_ONLY", "0") == "1"

    # -- cache fast-path --
    if cache_on:
        rec = cache.lookup(dev_sig, g_sig)
        if rec is not None:
            choice_cached = rec.get("result", {}).get("choice")
            if choice_cached in ("autosage", "baseline"):
                if verbose or os.getenv("AUTOSAGE_VERBOSE") == "1":
                    print(f"[auto-cache] hit -> {choice_cached}")
                Y = _spmm(crow, col, val, x) if choice_cached == "autosage" else _baseline_spmm(crow, col, val, x)
                info = rec["result"] | {
                    "choice": choice_cached,
                    "from_cache": True,
                    "top": rec["result"].get("top", []),
                }
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

    # baseline time on sample
    tb, _ = _time_ms(lambda: _baseline_spmm(crow_s, col_s, val_s, x_s), iters=iters)

    # autosage time on sample:
    # Prefer M2 split (CTA-per-hub) if compiled & enabled; else core _spmm.
    if AUTOSAGE_HUB_CTA and (_spmm_split is not None):
        if top is None:
            try:
                if _model_topk is not None:
                    top = _model_topk(crow, col, F, int(k)).cpu().tolist()
                else:
                    top = []
            except Exception:
                top = []
        ftile, wpb, hubT = _pick_candidate(top, default=(128, 4, 1024))
        ta, _ = _time_ms(lambda: _spmm_split(crow_s, col_s, val_s, x_s, int(ftile), int(wpb), int(hubT)),
                         iters=iters)
    else:
        ta, _ = _time_ms(lambda: _spmm(crow_s, col_s, val_s, x_s), iters=iters)

    speedup = (tb / ta) if ta > 0 else float("inf")
    use_auto = ta <= guardrail * tb
    if (probe_rows < 64) or (probe_nnz < 512):
        use_auto = False
    choice = "autosage" if use_auto else "baseline"

    # Compute top-k candidates once (for logging & full-run)
    if top is None:
        try:
            if _model_topk is not None:
                top = _model_topk(crow, col, F, int(k)).cpu().tolist()
            else:
                top = []
        except Exception:
            top = []
    if verbose or os.getenv("AUTOSAGE_VERBOSE") == "1":
        print(f"[auto] rows={probe_rows} nnz={probe_nnz}  baseline={tb:.3f}ms  autosage={ta:.3f}ms  speedup={speedup:.3f}x  -> {choice}")
        print(f"[auto] top-{k} candidates: {top}")

    # --- full run selection ---
    if use_auto:
        ftile, wpb, hubT = _pick_candidate(top, default=(128, 4, 1024))
        if AUTOSAGE_HUB_CTA and (_spmm_split is not None):
            Y = _spmm_split(crow, col, val, x, int(ftile), int(wpb), int(hubT))
            hub_cta_enabled = True
        else:
            Y = _spmm(crow, col, val, x)
            hub_cta_enabled = False
    else:
        Y = _baseline_spmm(crow, col, val, x)
        hub_cta_enabled = False

    info: Dict[str, Any] = {
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
    if use_auto:
        ftile, wpb, hubT = _pick_candidate(top, default=(128, 4, 1024))
        info["candidate"] = [int(ftile), int(wpb), int(hubT)]
        info["hub_cta_enabled"] = bool(hub_cta_enabled)

    if cache_on:
        cache.store(dev_sig, g_sig, info)
    log_probe({"ts": __import__("time").time(), "device": dev_sig, "graph": g_sig, "result": info})
    return Y, info


def calibrate_full(crow: torch.Tensor,
                   col: torch.Tensor,
                   val: Optional[torch.Tensor],
                   x: torch.Tensor,
                   guardrail: float = 0.95):
    """Time baseline vs autosage on the FULL graph once, cache the winner, and return (choice, info)."""
    from ._cache import ScheduleCache, _device_sig, _graph_sig, log_probe

    # Baseline timing
    tb, _ = _time_ms(lambda: _baseline_spmm(crow, col, val, x), iters=3)

    # Autosage timing: prefer split if built+enabled; else core _spmm
    F = int(x.size(1))
    if AUTOSAGE_HUB_CTA and (_spmm_split is not None):
        try:
            top = _model_topk(crow, col, F, int(3)).cpu().tolist() if _model_topk is not None else []
        except Exception:
            top = []
        ftile, wpb, hubT = _pick_candidate(top, default=(128, 4, 1024))
        ta, _ = _time_ms(lambda: _spmm_split(crow, col, val, x, int(ftile), int(wpb), int(hubT)), iters=3)
    else:
        top = []
        ta, _ = _time_ms(lambda: _spmm(crow, col, val, x), iters=3)

    use_auto = (ta <= guardrail * tb)
    choice = "autosage" if use_auto else "baseline"

    info = {
        "probe_rows": int(crow.numel()-1),
        "probe_nnz": int(col.numel()),
        "use_autosage": use_auto,
        "choice": choice,
        "tb_ms": tb,
        "ta_ms": ta,
        "speedup": (tb/ta) if ta > 0 else float("inf"),
        "guardrail": guardrail,
        "top": top
    }
    dev_sig = _device_sig(); g_sig = _graph_sig(crow, col, F)
    ScheduleCache().store(dev_sig, g_sig, info)
    log_probe({"ts": __import__("time").time(), "device": dev_sig, "graph": g_sig, "result": info})
    return choice, info
