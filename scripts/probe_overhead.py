#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from autosage import load_native, native_available, spmm_csr_auto
from autosage._auto import _prepare_spmm_baseline, _time_cuda
from benchmark import features, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--use-real-features", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    load_native()
    if not native_available():
        raise SystemExit("AutoSAGE native extension is unavailable")

    device = torch.device("cuda")
    dataset, crow, col, real_features = load_dataset(args.dataset, device, args.seed)
    x = features(
        real_features,
        crow.numel() - 1,
        args.features,
        device,
        args.use_real_features,
    )
    baseline, baseline_name = _prepare_spmm_baseline(crow, col, None, x)
    baseline_ms = _time_cuda(baseline, warmup=3, iterations=10)["best_ms"]
    configurations = ((0.03, 1.0, 7), (0.02, 0.5, 5))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset",
                "F",
                "probe_fraction",
                "probe_cap_ms",
                "probe_iterations",
                "probe_rows",
                "probe_nonzeros",
                "sampling_ms",
                "probe_ms",
                "baseline_ms",
                "overhead_percent",
                "choice",
                "baseline",
            ]
        )
        for fraction, cap_ms, iterations in configurations:
            os.environ["AUTOSAGE_CACHE"] = "0"
            os.environ["AUTOSAGE_PROBE_FRAC"] = str(fraction)
            os.environ["AUTOSAGE_PROBE_CAP_MS"] = str(cap_ms)
            os.environ["AUTOSAGE_PROBE_ITERS"] = str(iterations)
            _, info = spmm_csr_auto(crow, col, None, x, seed=args.seed)
            writer.writerow(
                [
                    dataset,
                    args.features,
                    fraction,
                    cap_ms,
                    iterations,
                    info["probe_rows"],
                    info["probe_nonzeros"],
                    f"{info['sampling_ms']:.4f}",
                    f"{info['probe_ms']:.4f}",
                    f"{baseline_ms:.4f}",
                    f"{100 * info['probe_ms'] / baseline_ms:.3f}",
                    info["choice"],
                    baseline_name,
                ]
            )


if __name__ == "__main__":
    main()
