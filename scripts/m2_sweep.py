#!/usr/bin/env python3
import argparse, csv, os, sys, itertools, time
import torch

def load_ext(lib):
    try:
        torch.ops.load_library(lib)
    except Exception as e:
        print(f"[error] failed to load '{lib}': {e}", file=sys.stderr); sys.exit(1)

def bench(fn, iters=15):
    torch.cuda.synchronize(); tmin=1e9; y=None
    for _ in range(iters):
        e0,e1=torch.cuda.Event(True),torch.cuda.Event(True)
        e0.record(); y=fn(); e1.record(); e1.synchronize()
        t=e0.elapsed_time(e1); tmin=min(tmin,t)
    return tmin,y

def make_hub_csr(N,deg_hub,deg_other,device):
    crow=torch.zeros(N+1,device=device,dtype=torch.long)
    crow[1]=int(deg_hub)
    if N>1: crow[2:]=int(deg_other)
    crow=crow.cumsum(0); nnz=int(crow[-1].item())
    col=torch.randint(0,N,(nnz,),device=device,dtype=torch.long)
    return crow,col,nnz

def main():
    ap=argparse.ArgumentParser(description="AutoSAGE M2 ablation sweep")
    ap.add_argument("--lib", default="cmake-build/libautosage_cuda.so")
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--F", type=int, default=128)
    ap.add_argument("--deg-hub", type=int, default=5000)
    ap.add_argument("--deg-other", type=int, default=64)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--ftiles", default="64,128")
    ap.add_argument("--wpbs", default="2,4,8")
    ap.add_argument("--hubTs", default="256,512,1024,2048")
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--out", default="results/m2/sweep.csv")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not torch.cuda.is_available():
        print("[error] CUDA required", file=sys.stderr); sys.exit(1)

    load_ext(args.lib)
    torch.manual_seed(0)

    N,F=args.N,args.F
    crow,col,nnz=make_hub_csr(N,args.deg_hub,args.deg_other,"cuda")
    x=torch.randn(N,F,device="cuda",dtype=torch.float32)
    val=torch.rand(nnz,device="cuda",dtype=torch.float32) if args.val else None

    base=lambda: torch.ops.autosage.spmm_csr(crow,col,val,x)
    tb,_=bench(base,iters=args.iters)

    combos=list(itertools.product(
        [int(s) for s in args.ftiles.split(",") if s],
        [int(s) for s in args.wpbs.split(",") if s],
        [int(s) for s in args.hubTs.split(",") if s],
    ))

    rows=[["N","F","nnz","deg_hub","deg_other","ftile","wpb","hubT","val","t_base_ms","t_split_ms","speedup","max_delta","allclose"]]
    best=(0.0,None)

    for ft,wpb,hubT in combos:
        fn=lambda: torch.ops.autosage.spmm_csr_split(crow,col,val,x,ft,wpb,hubT)
        ts,ys=bench(fn,iters=args.iters)
        yb=base()[1] if isinstance(base, tuple) else None
        yb = torch.ops.autosage.spmm_csr(crow,col,val,x)
        maxd=float((yb-ys).abs().max().item())
        ok=bool(torch.allclose(yb, ys, rtol=1e-4, atol=3e-3))
        spd=float(tb/ts) if ts>0 else float("inf")
        rows.append([N,F,nnz,args.deg_hub,args.deg_other,ft,wpb,hubT,args.val,f"{tb:.4f}",f"{ts:.4f}",f"{spd:.4f}",f"{maxd:.6f}",ok])
        if spd>best[0]: best=(spd,(ft,wpb,hubT))

    with open(args.out,"w",newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[done] wrote {args.out}")
    if best[1]:
        print(f"[best] speedup {best[0]:.3f}x with ftile={best[1][0]} wpb={best[1][1]} hubT={best[1][2]}")

if __name__=="__main__":
    main()
