# Historical paper artifacts

This directory contains the minimal retained measurements used to assemble the AutoSAGE preprint. The files were produced on an NVIDIA A800-SXM4-40GB system with PyTorch 2.8.0+cu128. They are preserved byte-for-byte apart from descriptive filenames and directory placement.

## Paper mapping

| Paper content | Retained input |
| --- | --- |
| Table II, Reddit | `results/spmm/reddit.csv` |
| Table III, OGBN-Products | `results/spmm/products.csv` |
| Table IV, Erdős–Rényi | `results/spmm/synth_er.csv` |
| Table V, hub-skew | `results/spmm/synth_hub.csv` |
| Table VI, guardrail 0.98 | `results/spmm/reddit_guardrail_098.csv` |
| Table VII, Reddit width sweep | `results/spmm/reddit_wide.csv` |
| Table VIII, Products width sweep | `results/spmm/products_wide.csv` |
| Table IX, vec4 ablation | `results/vec4/` |
| Table X, hub split sweeps | `results/hub/` |
| Cached/uncached attention discussion | `results/attention/` |

The metadata sidecars record the environment variables and software information captured at execution time. `environment/original_revision.txt` is the revision recorded by the old artifact script; it is not sufficient proof that every retained file was produced from that exact revision. The direct package versions are also captured in `environment/requirements-paper.txt`.

## Provenance limitations

These measurements were created by an earlier benchmark implementation. That implementation did not record which fallback supplied the SpMM baseline and included baseline preparation inside the timed call. The old probe logs also did not retain the complete sampling overhead used for the paper's probe-overhead percentages. Those logs were therefore removed rather than converted into unsupported data.

The cleaned harness prepares the PyTorch CSR baseline before timing, records the baseline name and source revision, includes sample construction in probe telemetry, and writes a consistent schema. New measurements should be written below the ignored root `results/` directory with `scripts/reproduce.sh`; they should not overwrite this historical snapshot.

SHA-256 digests for every retained measurement, metadata file, environment record, and the paper PDF are stored in `SHA256SUMS`.
