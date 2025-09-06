#!/usr/bin/env bash
set -euo pipefail

OUTDIR=results/p4
mkdir -p "$OUTDIR"

export AUTOSAGE_CACHE_DIR=.autosage-cache-p4
export AUTOSAGE_SYNTH_SEED=0

# (Optional) enforce timing to follow cached decisions only:
# export AUTOSAGE_REPLAY_ONLY=1

# 1) Reddit (PyG)
python scripts/p4_bench.py --dataset reddit --prefer pyg --use-real-x \
  --F 64,128,256 --repeats 15 --calibrate-full --force-exit \
  --out "$OUTDIR/reddit.csv"

# 2) OGBN-Products (only if OGB is available); auto-confirm prompts.
python - <<'PY' || true
import os, sys
try:
    from ogb.nodeproppred import PygNodePropPredDataset  # noqa
except Exception:
    print("[skip] ogb not available"); sys.exit(0)

cmd = (
    "yes y | python scripts/p4_bench.py "
    "--dataset ogbn-products --use-real-x "
    "--F 64,128,256 --repeats 15 --calibrate-full --force-exit "
    "--out results/p4/products.csv"
)
os.system(cmd)
PY

# 3) Memory-safe ER synthetic
python scripts/p4_bench.py --dataset synth:200000:2e-5 \
  --F 64,128,256 --repeats 10 --calibrate-full --force-exit \
  --out "$OUTDIR/synth_er.csv"

# 4) Skewed synthetic (hubs)
export AUTOSAGE_SYNTH_BG_PER_NODE=0.5
export AUTOSAGE_SYNTH_BG_CAP=1500000
python scripts/p4_bench.py --dataset synthhub:200000:4:0.15 \
  --F 64,128,256 --repeats 10 --calibrate-full --force-exit \
  --out "$OUTDIR/synth_hub.csv"
