import os, torch
from typing import Dict, Any, Tuple, Optional

# compiled ops
_spmm = torch.ops.autosage.spmm_csr
_model_topk = torch.ops.autosage.model_topk

def _has_cusparse_op() -> bool:
    return hasattr(torch.ops.autosage, 'cusparse_spmm_csr')

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
        nnz = col.numel()
        nrows = crow.numel() - 1
        row_counts = (crow[1:] - crow[:-1]).to(col.dtype)
        row_idx = torch.repeat_interleave(torch.arange(nrows, device=col.device, dtype=col.dtype), row_counts)
        coo = torch.stack([row_idx, col])
        v = val if val is not None else torch.ones(nnz, device=x.device, dtype=x.dtype)
        return torch_sparse.spmm(coo, v, nrows, x.size(0), x)
    return _spmm(crow, col, val, x)

def _slice_rows(crow: torch.Tensor, col: torch.Tensor, rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = rows.to(crow.device)
    m = rows.numel()
    crow_s = torch.empty(m + 1, dtype=torch.long, device=crow.device)
    segs = []
    nnz = 0
    crow_cpu = crow.tolist() if not crow.is_cuda else crow.cpu().tolist()
    for i, r in enumerate(rows.tolist()):
        start = crow_cpu[r]; end = crow_cpu[r + 1]
        seg = col[start:end]
        segs.append(seg); nnz += seg.numel()
        crow_s[i + 1] = nnz
    col_s = torch.empty(nnz, dtype=col.dtype, device=col.device)
    off = 0
    for seg in segs:
        k = seg.numel()
        col_s[off:off + k] = seg
        off += k
    crow_s[0] = 0
    return crow_s, col_s

def _stratified_sample_rows(crow: torch.Tensor, frac: float = 0.02, min_rows: int = 256, seed: int = 0) -> torch.Tensor:
    deg = (crow[1:] - crow[:-1]).cpu()
    N = deg.numel()
    if N == 0: return torch.zeros(0, dtype=torch.long)
    q50 = torch.quantile(deg.float(), 0.50).item()
    q90 = torch.quantile(deg.float(), 0.90).item()
    b0 = (deg <=  q50).nonzero(as_tuple=False).flatten()
    b1 = ((deg > q50) & (deg <= q90)).nonzero(as_tuple=False).flatten()
    b2 = (deg >  q90).nonzero(as_tuple=False).flatten()
    m = max(min_rows, int(N * frac))
    g = torch.Generator().manual_seed(seed)
    def pick(b, t):
        if b.numel() == 0: return b
        idx = torch.randperm(b.numel(), generator=g)[:min(t, b.numel())]
        return b[idx]
    t0, t1, t2 = int(0.5*m), int(0.3*m), int(0.2*m)
    rows = torch.unique(torch.cat([pick(b0,t0), pick(b1,t1), pick(b2,t2)]), sorted=True)
    if rows.numel() == 0:
        rows = torch.arange(min(N, max(1, m)), dtype=torch.long)
    return rows

def _time_ms(fn, warmup: int = 1, iters: int = 5) -> Tuple[float, torch.Tensor]:
    for _ in range(warmup): out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record(); out = fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return (min(times) if times else float('inf')), out

def spmm_csr_auto(crow: torch.Tensor,
                  col:  torch.Tensor,
                  val:  Optional[torch.Tensor],
                  x:    torch.Tensor,
                  *,
                  guardrail: float = 0.95,
                  k: int = 3,
                  seed: int = 0,
                  verbose: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Decide baseline vs AutoSAGE via micro-probe and run on the full graph.

    Returns: (Y, info) where info includes times, speedup, choice, and candidates.
    """
    assert crow.dtype == torch.long and col.dtype == torch.long
    F = x.size(1)
    top = _model_topk(crow, col, int(F), int(k)).cpu().tolist()

    frac = float(os.getenv('AUTOSAGE_PROBE_FRAC', '0.02'))
    min_rows = int(os.getenv('AUTOSAGE_PROBE_MIN_ROWS', '256'))
    iters = int(os.getenv('AUTOSAGE_PROBE_ITERS', '5'))
    rows = _stratified_sample_rows(crow, frac=frac, min_rows=min_rows, seed=seed)
    crow_s, col_s = _slice_rows(crow, col, rows)

    tb, yb = _time_ms(lambda: _baseline_spmm(crow_s, col_s, val, x), iters=iters)
    ta, ya = _time_ms(lambda: _spmm(crow_s, col_s, val, x), iters=iters)

    use_auto = (ta <= guardrail * tb)
    choice = "autosage" if use_auto else "baseline"
    speedup = (tb / ta) if ta > 0 else float('inf')

    # tiny-slice guard: if probe too small, trust baseline
    if rows.numel() < 64 or col_s.numel() < 512:
        use_auto = False
        choice = 'baseline'

    if verbose or os.environ.get("AUTOSAGE_VERBOSE") == "1":
        print(f"[auto] rows={rows.numel()} nnz={col_s.numel()}  baseline={tb:.3f}ms  autosage={ta:.3f}ms  "
              f"speedup={speedup:.3f}x  -> {choice}")
        print(f"[auto] top-{k} candidates: {top}")

    if use_auto:
        Y = _spmm(crow, col, val, x)
    else:
        Y = _baseline_spmm(crow, col, val, x)

    info = {"use_autosage": use_auto, "choice": choice, "tb_ms": tb, "ta_ms": ta,
            "speedup": speedup, "guardrail": guardrail, "top": top,
            "probe_rows": int(rows.numel()), "probe_nnz": int(col_s.numel())}
    return Y, info
