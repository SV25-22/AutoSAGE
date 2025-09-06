import os, torch, math

# Load your compiled extension
torch.ops.load_library(os.path.abspath("cmake-build/libautosage_cuda.so"))

def _has_torch_sparse():
    try:
        import torch_sparse  # noqa: F401
        return True
    except Exception:
        return False

def _spmm_baseline(crow, col, x):
    if _has_torch_sparse():
        import torch_sparse
        nnz = col.numel()
        nrows = crow.numel() - 1
        # build COO for the sampled CSR
        row_counts = (crow[1:] - crow[:-1]).to(col.dtype)
        row_idx = torch.repeat_interleave(torch.arange(nrows, device=col.device, dtype=col.dtype), row_counts)
        coo = torch.stack([row_idx, col])
        val = torch.ones(nnz, device=x.device, dtype=x.dtype)
        return torch_sparse.spmm(coo, val, nrows, x.size(0), x)
    # Fallback baseline = our kernel (conservative but keeps pipeline working)
    return torch.ops.autosage.spmm_csr(crow, col, None, x)

def _slice_rows(crow, col, rows):
    """Build a new CSR with reindexed rows [0..m-1] but original column IDs."""
    rows = rows.to(crow.device)
    m = rows.numel()
    crow_s = torch.empty(m + 1, dtype=torch.long, device=crow.device)
    segs = []
    nnz = 0
    crow_s[0] = 0
    crow_cpu = crow.tolist() if not crow.is_cuda else crow.cpu().tolist()
    col_dev = col
    for i, r in enumerate(rows.tolist()):
        start = crow_cpu[r]
        end   = crow_cpu[r + 1]
        seg = col_dev[start:end]
        segs.append(seg)
        nnz += seg.numel()
        crow_s[i + 1] = nnz
    col_s = torch.empty(nnz, dtype=col.dtype, device=col.device)
    off = 0
    for seg in segs:
        k = seg.numel()
        col_s[off:off + k] = seg
        off += k
    return crow_s, col_s

def _stratified_sample_rows(crow, frac=0.02, min_rows=256, seed=0):
    """Sample rows from <=q50, (q50,q90], >q90 degree buckets."""
    deg = (crow[1:] - crow[:-1]).cpu()
    N = deg.numel()
    if N == 0:
        return torch.zeros(0, dtype=torch.long)
    q50 = torch.quantile(deg.float(), 0.50).item()
    q90 = torch.quantile(deg.float(), 0.90).item()
    b0 = (deg <=  q50).nonzero(as_tuple=False).flatten()
    b1 = ((deg > q50) & (deg <= q90)).nonzero(as_tuple=False).flatten()
    b2 = (deg >  q90).nonzero(as_tuple=False).flatten()
    m = max(min_rows, int(N * frac))
    # bucket targets
    t0, t1, t2 = int(0.5*m), int(0.3*m), int(0.2*m)
    g = torch.Generator().manual_seed(seed)
    def pick(b, t):
        if b.numel() == 0: return b
        idx = torch.randperm(b.numel(), generator=g)[:min(t, b.numel())]
        return b[idx]
    rows = torch.unique(torch.cat([pick(b0, t0), pick(b1, t1), pick(b2, t2)]), sorted=True)
    if rows.numel() == 0:
        rows = torch.arange(min(N, max(1, m)), dtype=torch.long)
    return rows

def _time_ms(fn, warmup=1, iters=5):
    # CUDA event timing
    for _ in range(warmup):
        y = fn();  # warm
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return min(times), y  # ms

def run_probe(crow, col, x, F, k=3, guardrail=0.95, seed=0):
    # shortlist candidates (we're not using ftile/wpB until P1.5, but we still compute them)
    top = torch.ops.autosage.model_topk(crow, col, int(F), int(k)).cpu().tolist()

    # stratified slice (~2% rows)
    rows = _stratified_sample_rows(crow, frac=0.02, min_rows=256, seed=seed)
    crow_s, col_s = _slice_rows(crow, col, rows)
    # build a sliced X that still uses original N columns (so just reuse x)
    # our spmm returns [m,F] where m = rows.numel()

    # baseline time
    tb_ms, yb = _time_ms(lambda: _spmm_baseline(crow_s, col_s, x))
    # our kernel time (single variant for now; P1 ignores knobs)
    ta_ms, ya = _time_ms(lambda: torch.ops.autosage.spmm_csr(crow_s, col_s, None, x))

    print(f"[micro] rows={rows.numel()} nnz={col_s.numel()}  baseline={tb_ms:.3f} ms  autosage={ta_ms:.3f} ms")
    # optional correctness check on the slice (coarse)
    if torch.allclose(ya, yb, rtol=1e-3, atol=1e-3):
        print("[micro] outputs match (rtol=1e-3, atol=1e-3)")
    else:
        print("[micro] WARNING: outputs differ on the probe slice")

    use_auto = (ta_ms <= guardrail * tb_ms)
    speedup = tb_ms / ta_ms if ta_ms > 0 else float('inf')
    decision = "autosage" if use_auto else "baseline"
    print(f"[decision] guardrail={guardrail:.2f}  speedup={speedup:.3f}×  -> choose {decision}")
    print(f"[candidates] top-{k} (ftile, wpb, hubT): {top}")
    return {"use_autosage": use_auto, "speedup": speedup, "tb_ms": tb_ms, "ta_ms": ta_ms, "top": top}

if __name__ == "__main__":
    device = "cuda"
    # --- Small sanity graph (your earlier example) ---
    crow = torch.tensor([0,1,3,3,3], dtype=torch.long, device=device)
    col  = torch.tensor([1,2,3],     dtype=torch.long, device=device)
    F = 128
    x = torch.randn(crow.numel()-1, F, device=device, dtype=torch.float32)
    print("=== Sanity run ===")
    run_probe(crow, col, x, F=F, k=3, guardrail=0.95, seed=0)

    # --- Optional: medium synthetic if you want a more realistic micro-slice ---
    # N, p, F = 20000, 5e-4, 128
    # idx = (torch.rand(N, N, device=device) < p).nonzero(as_tuple=False).t()
    # row, col_full = idx[0], idx[1]
    # perm = row.argsort(stable=True); row=row[perm]; col_full=col_full[perm]
    # crow_full = torch.zeros(N+1, dtype=torch.long, device=device)
    # crow_full.index_add_(0, row+1, torch.ones_like(row, dtype=torch.long))
    # crow_full = crow_full.cumsum(0)
    # x_full = torch.randn(N, F, device=device)
    # print("=== Synthetic run ===")
    # run_probe(crow_full, col_full, x_full, F=F, k=3, guardrail=0.95, seed=0)
