#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-results/reproduced}"
mkdir -p "$OUTPUT_ROOT/spmm" "$OUTPUT_ROOT/ablations" "$OUTPUT_ROOT/attention" "$OUTPUT_ROOT/probing"

export AUTOSAGE_SYNTH_SEED=0
export AUTOSAGE_SYNTH_BG_PER_NODE=0.5
export AUTOSAGE_SYNTH_BG_CAP=1500000

run_spmm() {
  local dataset="$1"
  local features="$2"
  local output="$3"
  shift 3
  AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/${output%.csv}" \
    python3 scripts/benchmark.py --dataset "$dataset" --features "$features" \
    --repeats 15 --calibrate-full --output "$OUTPUT_ROOT/spmm/$output" "$@"
}

run_spmm reddit 64,128,256 reddit.csv --use-real-features
run_spmm ogbn-products 64,128,256 products.csv --use-real-features
run_spmm synth:200000:2e-5 64,128,256 synth_er.csv
run_spmm synthhub:200000:4:0.15 64,128,256 synth_hub.csv
run_spmm reddit 32,64,96,128,192,256,512 reddit_wide.csv --use-real-features
run_spmm ogbn-products 32,64,96,128,192,256,512 products_wide.csv --use-real-features

AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/reddit_guardrail_098" \
  python3 scripts/benchmark.py --dataset reddit --features 64,128,256 --repeats 15 \
  --guardrail 0.98 --calibrate-full --use-real-features \
  --output "$OUTPUT_ROOT/spmm/reddit_guardrail_098.csv"

for setting in 0 1; do
  AUTOSAGE_VEC4="$setting" AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/vec4_er_$setting" \
    python3 scripts/benchmark.py --dataset synth:200000:2e-5 --features 64,128,256 \
    --repeats 15 --calibrate-full \
    --output "$OUTPUT_ROOT/ablations/vec4_er_$setting.csv"
  AUTOSAGE_VEC4="$setting" AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/vec4_reddit_$setting" \
    python3 scripts/benchmark.py --dataset reddit --features 64 --repeats 15 \
    --calibrate-full --use-real-features \
    --output "$OUTPUT_ROOT/ablations/vec4_reddit_$setting.csv"
done

AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/attention" \
  python3 scripts/benchmark.py --dataset ogbn-products --features 64,128 --mode attention \
  --repeats 12 --use-real-features --output "$OUTPUT_ROOT/attention/products_cached.csv"

AUTOSAGE_CACHE=0 AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/attention_uncached" \
  python3 scripts/benchmark.py --dataset ogbn-products --features 64,128 --mode attention \
  --repeats 3 --use-real-features --output "$OUTPUT_ROOT/attention/products_uncached.csv"

AUTOSAGE_REPLAY_ONLY=1 AUTOSAGE_CACHE_DIR="$OUTPUT_ROOT/cache/attention" \
  python3 scripts/benchmark.py --dataset ogbn-products --features 64,128 --mode attention \
  --repeats 12 --use-real-features --output "$OUTPUT_ROOT/attention/products_replay.csv"

python3 scripts/sweep_hubs.py --hub-degree 5000 --other-degree 64 \
  --output "$OUTPUT_ROOT/ablations/hub_sweep_5000.csv"
python3 scripts/sweep_hubs.py --hub-degree 12000 --other-degree 32 \
  --output "$OUTPUT_ROOT/ablations/hub_sweep_12000.csv"

python3 scripts/probe_overhead.py --dataset reddit --features 64 --use-real-features \
  --output "$OUTPUT_ROOT/probing/reddit.csv"

python3 scripts/environment.py > "$OUTPUT_ROOT/environment.json"
