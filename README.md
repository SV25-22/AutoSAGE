# AutoSAGE

AutoSAGE is an input-aware CUDA auto-scheduler for CSR SpMM, SDDMM, and sparse attention in graph neural networks.

Accompanying preprint: **“AutoSAGE: Input-Aware CUDA Scheduling for Sparse GNN Aggregation (SpMM/SDDMM) and CSR Attention,”** arXiv (cs.LG, cs.PF), submitted 17 November 2025.

[Local six-page manuscript](paper/AutoSAGE.pdf) · [arXiv:2511.17594](https://arxiv.org/abs/2511.17594) · [DOI: 10.48550/arXiv.2511.17594](https://doi.org/10.48550/arXiv.2511.17594)

## Scope and limitations

AutoSAGE provides:

- scalar and vec4 warp-per-row SpMM kernels;
- a CTA-per-hub SpMM path for skewed degree distributions;
- row-wise and CTA-per-hub SDDMM kernels;
- an SDDMM → CSR softmax → SpMM attention pipeline;
- per-device and per-graph schedule caching; and
- a reproducible benchmark, sweep, summary, and plotting harness.

The scheduler and native kernels are CUDA-only. The public `spmm_csr` operation also has a PyTorch CPU implementation so imports, examples, and unit tests work without a GPU. CUDA kernels currently support float32, square CSR graphs, and a narrow SpMM/SDDMM/attention operator set; performance depends on the GPU, CUDA/PyTorch versions, sparsity pattern, and feature width. The sampled guardrail is not a formal full-graph non-regression guarantee, and the retained A800 results should not be read as end-to-end GNN training results or as performance guarantees on other systems.

## Quick start

### Requirements

The reference experiments used Ubuntu 22.04, Python 3.12, PyTorch 2.8.0+cu128, CUDA Toolkit 12.4, and an NVIDIA A800-SXM4-40GB GPU. Building the native extension requires:

- a CUDA-capable NVIDIA GPU;
- a CUDA-enabled PyTorch installation;
- CMake 3.24 or newer;
- Ninja; and
- a C++17/CUDA 17 toolchain.

The supplied build and reproduction entry points require Bash. Install a CUDA-enabled PyTorch build appropriate for the host from the official [PyTorch installer](https://pytorch.org/get-started/locally/), then install AutoSAGE and its optional development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,benchmark]"
```

Build the CUDA extension:

```bash
bash scripts/build.sh
```

The build produces `build/libautosage_cuda.so`. Override `PYTHON`, `BUILD_JOBS`, or `CMAKE_CUDA_ARCHITECTURES` when needed.

### Minimal Python example

```python
import torch
from autosage import spmm_csr_auto

crow = torch.tensor([0, 2, 3], dtype=torch.long, device="cuda")
col = torch.tensor([0, 1, 0], dtype=torch.long, device="cuda")
x = torch.randn(2, 64, device="cuda")

output, schedule = spmm_csr_auto(crow, col, None, x)
print(schedule["choice"], schedule.get("candidate"))
```

`AutoSAGEConv` accepts PyG-style `edge_index` with source nodes in row 0 and destination nodes in row 1. CSR rows consistently represent destinations.

```python
from autosage import AutoSAGEConv

layer = AutoSAGEConv(64, 128)
output, schedule = layer(x, edge_index, return_info=True)
```

## Scheduler behavior and configuration

For a new `(device, graph, operation, feature width)` key, AutoSAGE:

1. derives degree quantiles and ranks a small set of effective kernel configurations;
2. constructs an induced row sample;
3. measures the PyTorch CSR baseline and shortlisted native variants;
4. accepts a native variant when its sampled latency satisfies the guardrail; and
5. stores the decision for deterministic replay.

The default guardrail is `0.95`, meaning a native candidate must be at least 5% faster on the sampled probe. `calibrate_full` is available when a one-time full-graph comparison is appropriate. The guardrail and shortlist size are the `guardrail` and `k` Python arguments; they are not environment variables.

Scheduler configuration is read when each call begins. `AUTOSAGE_NATIVE_PATH` is consulted only when the process first attempts to load the native library.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `AUTOSAGE_CACHE_DIR` | `~/.cache/autosage` | Schedule and probe-log directory |
| `AUTOSAGE_CACHE` | `1` | Enable schedule lookup and storage |
| `AUTOSAGE_REPLAY_ONLY` | `0` | Use cached schedules without probing |
| `AUTOSAGE_LOG_PROBES` | `1` | Append probe telemetry |
| `AUTOSAGE_PROBE_FRAC` | `0.03` | Fraction of rows sampled |
| `AUTOSAGE_PROBE_MIN_ROWS` | `512` | Minimum sampled rows |
| `AUTOSAGE_PROBE_ITERS` | `7` | Maximum timing iterations per candidate |
| `AUTOSAGE_PROBE_CAP_MS` | `1.0` | Candidate-probe time budget in milliseconds |
| `AUTOSAGE_MIN_PROBE_ROWS` | `64` | Minimum sampled rows required to accept a native candidate |
| `AUTOSAGE_MIN_PROBE_NNZ` | `512` | Minimum sampled nonzeros required to accept a native candidate |
| `AUTOSAGE_HUB_CTA` | `1` | Enable CTA-per-hub candidates |
| `AUTOSAGE_VEC4` | `1` | Enable aligned vec4 kernels |
| `AUTOSAGE_FTILE` | unset | Restrict SpMM feature tile to 64 or 128 |
| `AUTOSAGE_WPB` | unset | Restrict hub CTA warps to 2, 4, or 8 |
| `AUTOSAGE_HUB_T` | unset | Override the hub degree threshold |
| `AUTOSAGE_VERBOSE` | `0` | Print one-line scheduler telemetry |
| `AUTOSAGE_NATIVE_PATH` | unset | Explicit native-library path used by the loader |

Cache keys include the CUDA device signature, operation, graph structure summary, feature width, dtype, and whether edge weights are present. Normal cache hits must also match the tuning configuration. With `AUTOSAGE_REPLAY_ONLY=1`, a matching cached decision is replayed without probing; a cache miss falls back to the PyTorch baseline. Leave `AUTOSAGE_CACHE=1` when using replay-only mode.

## Reproduction

Run the complete benchmark battery on a CUDA machine:

```bash
bash scripts/reproduce.sh
```

The full battery downloads Reddit and OGBN-Products on first use and can require substantial GPU memory, storage, and runtime. Set `OUTPUT_ROOT` to change the default `results/reproduced` destination. Individual examples:

```bash
python scripts/benchmark.py \
  --dataset reddit \
  --features 64,128,256 \
  --repeats 15 \
  --calibrate-full \
  --use-real-features \
  --output results/reddit.csv

python scripts/summarize.py results/reddit.csv

python scripts/plot.py results/reddit.csv \
  --title "Reddit" \
  --output results/reddit_speedup.pdf
```

The retained data in [artifacts/](artifacts/) are historical measurements used for the preprint. They predate the cleaned benchmark methodology and are preserved for provenance rather than presented as a new validation of this revision. See [artifacts/README.md](artifacts/README.md) for the table-to-file map and limitations.

## Tests

```bash
pytest
ruff check autosage scripts tests
ruff format --check autosage scripts tests
cffconvert --validate
```

CPU tests cover imports, CSR orientation, weighted aggregation, cache behavior, scheduling configuration, SDDMM reference behavior, and the layer API. CUDA correctness tests run automatically when a CUDA-enabled PyTorch installation and the native extension are available. Performance tests remain explicit benchmark commands rather than CI assertions.

## Repository layout

```text
autosage/       Python API, scheduler, cache, and layer
csrc/           CUDA kernels and PyTorch operator bindings
include/        Native declarations
scripts/        Build and evaluation entry points
tests/          CPU and CUDA correctness tests
artifacts/      Curated historical measurements and provenance
paper/          Local manuscript and publication notes
```

## Citation

```bibtex
@misc{stankovic2025autosage,
  title         = {AutoSAGE: Input-Aware CUDA Scheduling for Sparse GNN Aggregation (SpMM/SDDMM) and CSR Attention},
  author        = {Stankovic, Aleksandar},
  year          = {2025},
  eprint        = {2511.17594},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  doi           = {10.48550/arXiv.2511.17594},
  url           = {https://arxiv.org/abs/2511.17594}
}
```

Machine-readable citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

AutoSAGE is released under the [MIT License](LICENSE).
