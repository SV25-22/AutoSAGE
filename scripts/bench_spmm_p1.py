import os, time, torch
from autosage.ops import spmm_csr

# Dev convenience: point to your built .so (works everywhere)
os.environ.setdefault("AUTOSAGE_NATIVE_PATH",  # comment out if you prefer auto-search
                      os.path.abspath("cmake-build/libautosage_cuda.so"))
os.environ.setdefault("AUTOSAGE_DEBUG", "1")

def bench(n=20000, f=128, p=5e-4, repeats=10):
    # Build CSR ER graph on CPU
    idx = (torch.rand(n, n) < p).nonzero(as_tuple=False).t()
    row, col = idx[0], idx[1]
    perm = row.argsort(stable=True); row=row[perm]; col=col[perm]
    crow = torch.zeros(n+1, dtype=torch.long)
    crow.index_add_(0, row+1, torch.ones_like(row, dtype=torch.long))
    crow = crow.cumsum(0)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    crow = crow.to(device)
    col  = col.to(device)
    x    = torch.randn(n, f, device=device, dtype=torch.float32)

    # warmup
    for _ in range(5):
        spmm_csr(crow, col, None, x)
        if use_cuda: torch.cuda.synchronize()

    ts=[]
    for _ in range(repeats):
        if use_cuda: torch.cuda.synchronize()
        t0=time.time()
        y = spmm_csr(crow, col, None, x)
        if use_cuda: torch.cuda.synchronize()
        ts.append((time.time()-t0)*1e3)
    print(f"device={device} N={n} F={f} p={p}  mean={sum(ts)/len(ts):.3f} ms")

if __name__ == "__main__":
    bench()
