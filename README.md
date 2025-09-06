# AutoSAGE — Input‑Aware Auto‑Scheduler for Sparse GNN Aggregation

AutoSAGE is a CUDA‑only runtime that chooses between a vendor baseline (cuSPARSE / torch‑sparse) and a custom SpMM kernel **per (graph, feature width)** using a hybrid **cost‑model + micro‑probe** decision. It stores the winner in a persistent cache and **replays decisions deterministically**.

**Status:** P1–P4 complete (kernel, auto path, cache+replay, benchmark harness). P5 provides paper + artifact packaging.

---

## Contents

- `autosage/` — CUDA ops, auto‑scheduler (`_auto.py`), cache (`_cache.py`), helpers.
- `scripts/`
  - `p4_bench.py` — P4 benchmark driver.
  - `p4_run_all.sh` — convenience script to run datasets.
  - `p4_summarize.py` — prints a speedup table from CSVs.
  - *(optional)* `p5_plots.py` — generate PDF plots from CSVs (see below).
- `results/p4/` — CSVs + `.meta.json` sidecars (GPU/SM, Torch/CUDA, env).
- `cmake-build/` — contains `libautosage_cuda.so` (built CUDA extension).
- `paper/` — *(optional)* LaTeX paper for P5.

---

## Quick Start

### 1) Build the CUDA extension
```bash
# Your usual CMake build; result should be:
cmake --preset release && cmake --build --preset release
# -> cmake-build/libautosage_cuda.so
```

### 2) (Optional) Install dataset loaders
```bash
pip install torch-geometric ogb dgl
# torch-sparse is optional; see FAQ if building wheels is tricky
```

### 3) Run the benchmark battery
```bash
bash scripts/p4_run_all.sh
```

### 4) Summarize results
```bash
python scripts/p4_summarize.py results/p4/*.csv
```

Example output:
```
dataset                   F     choice   baseline(ms)   chosen(ms)   speedup
reddit(pyg)              64   autosage         16.330       15.036     1.086
reddit(pyg)             128   baseline         38.137       38.034     1.003
...
```

---

## What P4 Implements

- **Trifecta timing:** `baseline`, `raw autosage kernel`, and `chosen` (scheduler decision).
- **Full‑graph calibration:** `--calibrate-full` sets the cache without probe bias.
- **Cache & replay:** per `(GPU, graph signature, F)` under `~/.cache/autosage` or `$AUTOSAGE_CACHE_DIR`.
- **Reproducibility:** CSVs + `*.meta.json` sidecars with GPU/SM, Torch/CUDA, and relevant env.
- **Datasets:** Reddit (PyG/DGL), OGBN‑Products, memory‑safe synthetics:
  - `synth:N:p` (Erdős–Rényi)
  - `synthhub:N:k:hfrac` (hub‑skew with optional background edges)

---

## Reproducing the Reported Numbers

### Reddit (PyG)
```bash
python scripts/p4_bench.py --dataset reddit --prefer pyg --use-real-x \
  --F 64,128,256 --repeats 15 --calibrate-full \
  --out results/p4/reddit.csv --force-exit
```

### OGBN‑Products
```bash
python scripts/p4_bench.py --dataset ogbn-products --use-real-x \
  --F 64,128,256 --repeats 15 --calibrate-full \
  --out results/p4/products.csv --force-exit
```

### Synthetic ER
```bash
python scripts/p4_bench.py --dataset synth:200000:2e-5 \
  --F 64,128,256 --repeats 10 --calibrate-full \
  --out results/p4/synth_er.csv --force-exit
```

### Synthetic hub‑skew
```bash
export AUTOSAGE_SYNTH_BG_PER_NODE=0.5
export AUTOSAGE_SYNTH_BG_CAP=1500000
python scripts/p4_bench.py --dataset synthhub:200000:4:0.15 \
  --F 64,128,256 --repeats 10 --calibrate-full \
  --out results/p4/synth_hub.csv --force-exit
```

---

## Environment Knobs

- `AUTOSAGE_CACHE_DIR` — cache location (default `~/.cache/autosage`).
- `AUTOSAGE_REPLAY_ONLY=1` — run chosen path only (no probe).
- `AUTOSAGE_PROBE_FRAC` (default `0.02`) — induced‑subgraph row fraction.
- `AUTOSAGE_PROBE_MIN_ROWS` (default `256`).
- `AUTOSAGE_PROBE_ITERS` (default `5`).
- Synthetic: `AUTOSAGE_SYNTH_BG_PER_NODE`, `AUTOSAGE_SYNTH_BG_CAP`, `AUTOSAGE_SYNTH_SEED`.

---

## Guardrail Sensitivity and Wide‑F Sweep *(optional)*

**Guardrail sensitivity:**
```bash
python scripts/p4_bench.py --dataset reddit --prefer pyg --use-real-x \
  --F 64,128,256 --repeats 15 --calibrate-full --guardrail 0.98 \
  --out results/p4/reddit_g098.csv --force-exit
```

**Wide F sweep:**
```bash
python scripts/p4_bench.py --dataset reddit --prefer pyg --use-real-x \
  --F 32,64,96,128,192,256,512 --repeats 12 --calibrate-full \
  --out results/p4/reddit_wideF.csv --force-exit
```

---

## FAQ / Troubleshooting

**`torch-sparse` fails to build wheels.**  
Use PyG’s wheel index to get a prebuilt binary matching your Torch/CUDA:
```bash
python - <<'PY'
import torch
print(torch.__version__.split('+')[0], 'cu'+(torch.version.cuda or 'cpu').replace('.',''))
PY
# Suppose it prints: 2.6.0 cu124
pip install "torch-sparse==0.6.18" -f "https://data.pyg.org/whl/torch-2.6.0+cu124.html"
```
Or skip `torch-sparse` entirely — the benchmark will use the cuSPARSE wrapper if present.

**DGL version is ancient.**  
If you need DGL, upgrade to a modern build (CPU or CUDA‑matching).

**PyTorch 2.6 safe‑unpickling (PyG processed files).**  
We include a safe allowlist in `p4_bench.py` for PyG classes. If you still see errors, delete and re‑process datasets, or update PyG.

---

## Results (from our runs)

- **Reddit (PyG):**  
  F=64 → 16.33→15.04 ms (**1.09×**),  
  F=128 → baseline chosen (~**1.00×**),  
  F=256 → baseline chosen (~**1.00×**).

- **OGBN‑Products:**  
  F=64/128/256 → baseline chosen (~**1.00×**).

- **Synthetic ER (N=200k, p=2e‑5):**  
  F=64/128/256 → **4.70× / 3.09× / 2.11×**.

- **Synthetic hub‑skew (N=200k, k=4, h=0.15):**  
  F=64/128 → **1.13× / 1.07×**, F=256 → baseline chosen.

> **Interpretation:** On real graphs, SpMM is bandwidth‑bound at higher F — the guardrail rightly keeps baseline. Under sparse or skewed regimes, the AutoSAGE kernel wins.

---

## Optional Plots for the Paper

Create `scripts/p5_plots.py`:
```python
#!/usr/bin/env python3
import sys, csv, pathlib
import matplotlib.pyplot as plt

def load(path):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows

def speedup_curve(paths, title, outpdf):
    # expects rows with path in {baseline, chosen}
    byF={}
    for p in paths:
        for row in load(p):
            F=int(row["F"]); path=row["path"]; med=float(row["median_ms"])
            byF.setdefault(F,{}); byF[F][path]=med
    Fs=sorted(k for k,v in byF.items() if "baseline" in v and "chosen" in v)
    sp=[byF[F]["baseline"]/byF[F]["chosen"] for F in Fs]
    plt.figure()
    plt.plot(Fs, sp, marker="o")
    plt.xlabel("Feature width F")
    plt.ylabel("Speedup (baseline / chosen)")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(outpdf)

if __name__=="__main__":
    speedup_curve(["results/p4/reddit.csv"], "Reddit (PyG)", "results/p4/fig_reddit_speedup.pdf")
    speedup_curve(["results/p4/products.csv"], "OGBN-Products", "results/p4/fig_products_speedup.pdf")
    speedup_curve(["results/p4/synth_er.csv"], "ER N=200k p=2e-5", "results/p4/fig_er_speedup.pdf")
    speedup_curve(["results/p4/synth_hub.csv"], "Hub-skew N=200k k=4 h=0.15", "results/p4/fig_hub_speedup.pdf")
```
Run:
```bash
python scripts/p5_plots.py
# -> results/p4/fig_*.pdf (embed into paper)
```

---

## License & Citation

TBD. Please add your license of choice and (optionally) a BibTeX entry for citing AutoSAGE.
