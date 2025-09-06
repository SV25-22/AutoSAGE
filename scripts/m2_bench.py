#!/usr/bin/env python3
import argparse, os, sys, torch

def load_ext(lib):
    try:
        torch.ops.load_library(lib)
    except Exception as e:
        print(f"[error] failed to load '{lib}': {e}", file=sys.stderr)
        sys.exit(1)

def bench(fn, iters=20):
    torch.cuda.synchronize()
    tmin = 1e9
    y = None
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); y = fn(); e1.record(); e1.synchronize()
        t = e0.elapsed_time(e1)
        if t < tmin: tmin = t
    return tmin, y

def make_hub_csr(N, deg_hub, deg_other, device):
    crow = torch.zeros(N+1, device=device, dtype=torch.long)
    crow[1] = int(deg_hub)
    if N > 1:
        crow[2:] = int(deg_other)
    crow = crow.cumsum(0)
    nnz = int(crow[-1].item())
    col = torch.randint(0, N, (nnz,), device=device, dtype=torch.long)
    return crow, col

def main():
    ap = argparse.ArgumentParser(description="AutoSAGE M2 microbench (CTA-per-hub)")
    ap.add_argument("--lib", default="cmake-build/libautosage_cuda.so", help="path to compiled extension .so")
    ap.add_argument("--N", type=int, default=20000, help="nodes")
    ap.add_argument("--F", type=int, default=128, help="feature dim")
    ap.add_argument("--deg-hub", type=int, default=5000, help="degree of the hub row (row 0)")
    ap.add_argument("--deg-other", type=int, default=64, help="degree of all other rows")
    ap.add_argument("--iters", type=int, default=20, help="timing iterations (min of)")
    ap.add_argument("--seed", type=int, default=0, help="torch seed")
    ap.add_argument("--ftile", type=int, default=128, help="feature tile for hub CTA")
    ap.add_argument("--wpb", type=int, default=4, help="warps per block for hub CTA")
    ap.add_argument("--hubT", type=int, default=512, help="degree threshold for hub CTA")
    ap.add_argument("--val", action="store_true", help="use random edge weights")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[error] CUDA required", file=sys.stderr); sys.exit(1)

    load_ext(args.lib)
    g = torch.Generator(device="cuda"); g.manual_seed(int(args.seed))

    N, F = int(args.N), int(args.F)
    crow, col = make_hub_csr(N, args.deg_hub, args.deg_other, device="cuda")
    nnz = int(crow[-1].item())

    x = torch.randn(N, F, device="cuda", dtype=torch.float32, generator=g)
    val = torch.rand(nnz, device="cuda", dtype=torch.float32, generator=g) if args.val else None

    base = lambda: torch.ops.autosage.spmm_csr(crow, col, val, x)
    split = lambda: torch.ops.autosage.spmm_csr_split(crow, col, val, x,
                                                      int(args.ftile), int(args.wpb), int(args.hubT))

    tb, yb = bench(base, iters=args.iters)
    ts, ys = bench(split, iters=args.iters)

    max_delta = float((yb - ys).abs().max().item())
    print(f"N={N} F={F} nnz={nnz} hub={args.deg_hub} other={args.deg_other} "
          f"ftile={args.ftile} wpb={args.wpb} hubT={args.hubT} val={bool(args.val)}")
    print(f"baseline {tb:.3f} ms | split {ts:.3f} ms | speedup {tb/ts:.3f}x | maxΔ {max_delta:.4g}")

    # parity check (tolerates FP32 reordering on heavy rows)
    ok = torch.allclose(yb, ys, rtol=1e-4, atol=3e-3)
    print("allclose:", ok)

if __name__ == "__main__":
    main()
