#!/usr/bin/env python3
import os, sys, csv, json, argparse, statistics as stats
import torch

# Load compiled extension once
torch.ops.load_library(os.path.abspath("cmake-build/libautosage_cuda.so"))

# ------------------------------------------------------------
# Optional deps: PyG, OGB, DGL
# ------------------------------------------------------------

# --- PyTorch 2.6 safe-unpickling: allowlist common PyG classes used in processed files
# --- PyTorch 2.6 safe-unpickling: allowlist common PyG classes used in processed files
try:
    import inspect
    import torch.serialization as _ts
    _allow = []
    # Import PyG modules lazily; skip if PyG isn't installed.
    try:
        from torch_geometric.data import data as _pyg_data_mod
        from torch_geometric.data import storage as _pyg_storage_mod
        for _mod in (_pyg_data_mod, _pyg_storage_mod):
            for _name, _obj in vars(_mod).items():
                if inspect.isclass(_obj) and (
                    _name == "Data" or
                    _name.endswith("Attr") or
                    _name.endswith("Storage")
                ):
                    _allow.append(_obj)
    except Exception:
        pass
    if _allow and hasattr(_ts, "add_safe_globals"):
        # IMPORTANT: pass an iterable of classes/functions, not a dict
        _ts.add_safe_globals(_allow)
except Exception:
    # If anything about this probing fails, just continue without the allowlist.
    pass



def try_imports():
    pyg_ok = ogb_ok = dgl_ok = True
    try:
        from torch_geometric.datasets import Reddit  # noqa: F401
    except Exception:
        pyg_ok = False
    try:
        from ogb.nodeproppred import PygNodePropPredDataset  # noqa: F401
    except Exception:
        ogb_ok = False
    try:
        import dgl  # noqa: F401
        from dgl.data import RedditDataset  # noqa: F401
    except Exception:
        dgl_ok = False
    return pyg_ok, ogb_ok, dgl_ok


PYG_OK, OGB_OK, DGL_OK = try_imports()

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def edge_index_to_csr(edge_index: torch.Tensor, num_nodes: int, device=None):
    """Build CSR (crow, col) where rows are dst nodes (output rows), cols are src."""
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    device = device or edge_index.device
    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    perm = dst.argsort(stable=True)
    dst = dst[perm]
    src = src[perm]
    crow = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    crow.index_add_(0, dst + 1, torch.ones_like(dst, dtype=torch.long))
    crow = crow.cumsum(0)
    return crow, src


def baseline_spmm(crow, col, val, x):
    """Try cuSPARSE wrapper -> torch_sparse -> compiled autosage spmm."""
    if hasattr(torch.ops.autosage, "cusparse_spmm_csr"):
        return torch.ops.autosage.cusparse_spmm_csr(crow, col, val if val is not None else None, x)
    try:
        import torch_sparse
        row = torch.repeat_interleave(
            torch.arange(crow.numel() - 1, device=crow.device, dtype=torch.long),
            (crow[1:] - crow[:-1]),
        )
        coo = torch.stack([row, col], dim=0)
        val_local = (
            torch.ones(col.numel(), device=x.device, dtype=x.dtype)
            if val is None
            else val.to(x.dtype)
        )
        return torch_sparse.spmm(coo, val_local, crow.numel() - 1, x.size(0), x)
    except Exception:
        pass
    from autosage.ops import spmm_csr
    return spmm_csr(crow, col, val, x)


def gb_moved(nnz, N, F, dtype_bytes=4):
    """
    Approx bytes moved for CSR SpMM (message passing):
      - read neighbor features: nnz * F * elem
      - write output features:  N * F * elem
    """
    return ((nnz + N) * F * dtype_bytes) / 1e9


def time_op(fn, iters=10, warmup=3):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        t1.synchronize()
        times.append(t0.elapsed_time(t1))
    return stats.median(times), stats.mean(times), min(times), max(times)


# ------------------------------------------------------------
# Datasets (reddit via DGL/PyG, OGB-products, and synthetic)
# ------------------------------------------------------------
def load_dataset(
    dataset,
    device,
    prefer="auto",
    pyg_root="data/pyg/Reddit",
    edge_index_tensor=None,
    num_nodes=None,
    x_tensor=None,
):
    # Bypass: user-provided tensors
    if edge_index_tensor:
        edge = torch.load(edge_index_tensor, map_location=device)
        assert (
            num_nodes is not None
        ), "--num-nodes is required with --edge-index-tensor"
        crow, col = edge_index_to_csr(edge, int(num_nodes), device=device)
        X = torch.load(x_tensor, map_location=device) if x_tensor else None
        return "custom", int(num_nodes), crow, col, X

    name = dataset.lower()

    # Reddit
    if name in ("reddit", "reddit-dgl", "reddit-pyg"):
        if (prefer in ("auto", "dgl") and DGL_OK) or name == "reddit-dgl":
            from dgl.data import RedditDataset

            ds = RedditDataset(raw_dir="data/dgl")
            g = ds[0]
            N = int(g.num_nodes())
            src, dst = g.edges()
            edge = torch.stack([src.to(device), dst.to(device)], dim=0)
            crow, col = edge_index_to_csr(edge, N, device=device)
            X = g.ndata.get("feat", None)
            X = X.to(device) if X is not None else None
            return "reddit(dgl)", N, crow, col, X

        if (prefer in ("auto", "pyg") and PYG_OK) or name == "reddit-pyg":
            from torch_geometric.datasets import Reddit

            ds = Reddit(root=pyg_root)
            data = ds[0]
            N = int(data.num_nodes)
            edge = data.edge_index.to(device)
            crow, col = edge_index_to_csr(edge, N, device=device)
            X = data.x.to(device) if getattr(data, "x", None) is not None else None
            return "reddit(pyg)", N, crow, col, X

        raise RuntimeError("Neither DGL nor PyG Reddit loaders are available.")

    # OGBN-Products
    if name in ("ogbn-products", "products"):
        if not OGB_OK:
            raise RuntimeError("ogb/pyg not available")
        from ogb.nodeproppred import PygNodePropPredDataset

        ds = PygNodePropPredDataset(name="ogbn-products", root="data/ogb")
        data = ds[0]
        N = int(data.num_nodes)
        edge = data.edge_index.to(device)
        crow, col = edge_index_to_csr(edge, N, device=device)
        X = data.x.to(device) if getattr(data, "x", None) is not None else None
        return "ogbn-products", N, crow, col, X

    # Synthetic ER(N,p) — memory safe (no NxN mask)
    if name.startswith("synth:"):
        parts = name.split(":")
        N = int(parts[1]) if len(parts) > 1 else 1_000_000
        p = float(parts[2]) if len(parts) > 2 else 1e-5
        import numpy as np

        seed_local = int(os.environ.get("AUTOSAGE_SYNTH_SEED", "0"))
        rng = np.random.default_rng(seed_local)

        # Expected edges; sample exactly M with replacement, dedup afterward.
        M = max(1, int(p * N * N))
        row = rng.integers(0, N, size=M, dtype=np.int64)
        col = rng.integers(0, N, size=M, dtype=np.int64)

        # Deduplicate and sort by row
        pair = row * np.int64(N) + col
        pair = np.unique(pair)
        row = (pair // np.int64(N)).astype(np.int64)
        col = (pair % np.int64(N)).astype(np.int64)
        perm = np.argsort(row, kind="mergesort")
        row = row[perm]
        col = col[perm]

        import numpy as _np
        crow = _np.zeros(N + 1, dtype=_np.int64)
        _np.add.at(crow, row + 1, 1)
        _np.cumsum(crow, out=crow)

        crow = torch.from_numpy(crow).to(device=device, dtype=torch.long)
        col = torch.from_numpy(col).to(device=device, dtype=torch.long)
        return f"synth:{N}:{p}", N, crow, col, None

    # Skewed synthetic: synthhub:N:k:hfrac  (k hubs, each connects to ~hfrac*N targets)
    if name.startswith("synthhub"):
        parts = name.split(":")
        N = int(parts[1])
        k = int(parts[2])
        hfrac = float(parts[3]) if len(parts) > 3 else 0.10
        import numpy as np

        seed_local = int(os.environ.get("AUTOSAGE_SYNTH_SEED", "0"))
        rng = np.random.default_rng(seed_local)

        # Hubs are first k nodes; each connects to ~hfrac*N random distinct targets
        targets_per_hub = max(1, int(hfrac * N))
        rows, cols = [], []
        for h in range(k):
            tgt = rng.choice(N, size=targets_per_hub, replace=False)
            rows.append(np.full(tgt.shape, h, dtype=np.int64))
            cols.append(tgt.astype(np.int64))

        # Tiny background edges so non-hubs aren’t empty, but O(N) (not O(N^2))
        bg_per_node = float(os.environ.get("AUTOSAGE_SYNTH_BG_PER_NODE", "1.0"))
        M = max(0, int(bg_per_node * N))
        M = min(M, int(os.environ.get("AUTOSAGE_SYNTH_BG_CAP", "2000000")))  # hard cap
        if M > 0:
            row_bg = rng.integers(0, N, size=M, dtype=np.int64)
            col_bg = rng.integers(0, N, size=M, dtype=np.int64)
            rows.append(row_bg)
            cols.append(col_bg)

        row = np.concatenate(rows) if rows else np.empty((0,), dtype=np.int64)
        col = np.concatenate(cols) if cols else np.empty((0,), dtype=np.int64)

        # Dedup and sort by row
        pair = row * np.int64(N) + col
        pair = np.unique(pair)
        row = (pair // np.int64(N)).astype(np.int64)
        col = (pair % np.int64(N)).astype(np.int64)
        perm = np.argsort(row, kind="mergesort")
        row = row[perm]
        col = col[perm]

        crow = np.zeros(N + 1, dtype=np.int64)
        np.add.at(crow, row + 1, 1)
        np.cumsum(crow, out=crow)

        crow = torch.from_numpy(crow).to(device=device, dtype=torch.long)
        col = torch.from_numpy(col).to(device=device, dtype=torch.long)
        return f"synthhub:{N}:{k}:{hfrac}", N, crow, col, None

    raise ValueError(f"Unknown dataset {dataset}")


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def run_one(
    dataset,
    F_list,
    out_csv,
    repeats,
    guardrail,
    use_real_x,
    seed,
    verbose,
    prefer,
    pyg_root,
    edge_index_tensor,
    num_nodes,
    x_tensor,
    calibrate_full=False,
):
    device = "cuda"
    torch.manual_seed(seed)

    name, N, crow, col, X_real = load_dataset(
        dataset, device, prefer, pyg_root, edge_index_tensor, num_nodes, x_tensor
    )
    nnz, rows = int(col.numel()), int(crow.numel() - 1)

    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "N",
                "nnz",
                "F",
                "path",
                "choice",
                "median_ms",
                "mean_ms",
                "min_ms",
                "max_ms",
                "approx_GBps",
            ]
        )
        for F in F_list:
            x = (
                X_real[:, : F].contiguous()
                if (use_real_x and X_real is not None and X_real.size(1) >= F)
                else torch.randn(rows, F, device=device, dtype=torch.float32).contiguous()
            )
            val = None

            # Prepare cache for this (graph, F)
            if calibrate_full:
                from autosage._auto import calibrate_full as _cal

                ch, info = _cal(crow, col, val, x, guardrail=guardrail)
                print(
                    f"[calibrate-full] F={F} choice={ch} tb={info['tb_ms']:.3f}ms "
                    f"ta={info['ta_ms']:.3f}ms speedup={info['speedup']:.3f}x"
                )
                cached_choice = ch
            else:
                from autosage._auto import spmm_csr_auto

                _ = spmm_csr_auto(
                    crow, col, val, x, guardrail=guardrail, verbose=False
                )
                from autosage._cache import ScheduleCache, _device_sig, _graph_sig

                _rec = ScheduleCache().lookup(
                    _device_sig(), _graph_sig(crow, col, int(x.size(1)))
                )
                cached_choice = _rec.get("result", {}).get("choice") if _rec else "baseline"

            print(f"[auto-cache] choice={cached_choice} for F={F}")

            # Bind chosen path directly to the cached decision to avoid _auto overhead in timing
            from autosage._auto import _spmm as autosage_kernel

            def fn_base():
                return baseline_spmm(crow, col, val, x)

            if cached_choice == "autosage":
                def fn_chosen():
                    return autosage_kernel(crow, col, val, x)
            else:
                def fn_chosen():
                    return baseline_spmm(crow, col, val, x)

            mb, ab, mib, mab = time_op(fn_base, iters=repeats)
            ma, aa, mia, maa = time_op(fn_chosen, iters=repeats)

            GB = gb_moved(nnz, rows, F, 4)
            w.writerow(
                [
                    name,
                    rows,
                    nnz,
                    F,
                    "baseline",
                    "baseline",
                    f"{mb:.4f}",
                    f"{ab:.4f}",
                    f"{mib:.4f}",
                    f"{mab:.4f}",
                    f"{GB/(mb/1e3):.3f}",
                ]
            )
            w.writerow(
                [
                    name,
                    rows,
                    nnz,
                    F,
                    "chosen",
                    cached_choice,
                    f"{ma:.4f}",
                    f"{aa:.4f}",
                    f"{mia:.4f}",
                    f"{maa:.4f}",
                    f"{GB/(ma/1e3):.3f}",
                ]
            )
            f.flush()
            print(
                f"[{name}] F={F:4d}  baseline={mb:.3f} ms  chosen({cached_choice})={ma:.3f} ms  speedup={mb/ma:.3f}x"
            )

    # Sidecar metadata for reproducibility
    dev = torch.cuda.get_device_properties(0)
    meta = {
        "gpu_name": dev.name,
        "sm": f"{dev.major}.{dev.minor}",
        "total_mem_bytes": int(dev.total_memory),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "env": {k: os.environ.get(k) for k in [
            "AUTOSAGE_CACHE","AUTOSAGE_CACHE_DIR","AUTOSAGE_REPLAY_ONLY",
            "AUTOSAGE_PROBE_FRAC","AUTOSAGE_PROBE_MIN_ROWS",
            "AUTOSAGE_SYNTH_BG_PER_NODE","AUTOSAGE_SYNTH_BG_CAP"
        ]},
        "dataset": dataset,
        "out_csv": out_csv,
    }
    with open(out_csv.replace(".csv", ".meta.json"), "w") as g:
        json.dump(meta, g, indent=2)


def _force_exit():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vec4", action="store_true", help="Disable vec4 fast path via AUTOSAGE_VEC4=0")
    ap.add_argument(
        "--dataset",
        default="reddit",
        help="reddit | reddit-dgl | reddit-pyg | ogbn-products | synth:N:p | synthhub:N:k:hfrac",
    )
    ap.add_argument("--F", default="64,128,256", help="comma-separated feature sizes")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--guardrail", type=float, default=0.95)
    ap.add_argument("--use-real-x", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/p4_results.csv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--prefer",
        choices=["auto", "dgl", "pyg"],
        default="auto",
        help="prefer which loader for reddit",
    )
    ap.add_argument("--pyg-root", default="data/pyg/Reddit", help="PyG dataset root")
    ap.add_argument(
        "--edge-index-tensor",
        default="",
        help="Path to a saved edge_index.pt (2,E) tensor (bypass dataset)",
    )
    ap.add_argument(
        "--num-nodes", type=int, default=None, help="Required with --edge-index-tensor"
    )
    ap.add_argument("--x-tensor", default="", help="Optional path to saved X.pt (N,F)")
    ap.add_argument(
        "--calibrate-full",
        action="store_true",
        help="Time full graph once to set cache (no probe bias)",
    )
    ap.add_argument(
        "--force-exit",
        action="store_true",
        help="Hard-exit at end to avoid hangs from lib threads",
    )
    args = ap.parse_args()
    if args.no_vec4:
        os.environ["AUTOSAGE_VEC4"] = "0"
    F_list = [int(x) for x in args.F.split(",")]

    if not torch.cuda.is_available():
        print("CUDA required for P4 benchmarks.", file=sys.stderr)
        sys.exit(1)

    run_one(
        args.dataset,
        F_list,
        args.out,
        args.repeats,
        args.guardrail,
        args.use_real_x,
        args.seed,
        args.verbose,
        args.prefer,
        args.pyg_root,
        args.edge_index_tensor or None,
        args.num_nodes,
        args.x_tensor or None,
        calibrate_full=args.calibrate_full,
    )

    if args.force_exit:
        _force_exit()
