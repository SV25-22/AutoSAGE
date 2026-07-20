#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch

from autosage import load_native, native_available
from autosage._auto import _prepare_spmm_baseline


def make_graph(rows: int, hub_degree: int, other_degree: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    degrees = torch.full((rows,), other_degree, device="cuda", dtype=torch.long)
    degrees[0] = hub_degree
    crow = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.long), degrees.cumsum(0))
    )
    col = torch.randint(
        0, rows, (int(crow[-1].item()),), generator=generator, device="cuda"
    )
    return crow, col


def time_best(function, iterations: int = 15):
    for _ in range(3):
        function()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return min(samples), output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--hub-degree", type=int, required=True)
    parser.add_argument("--other-degree", type=int, required=True)
    parser.add_argument("--feature-tiles", default="64,128")
    parser.add_argument("--warps", default="2,4,8")
    parser.add_argument("--thresholds", default="256,512,1024,2048")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    load_native()
    if not native_available():
        raise SystemExit("AutoSAGE native extension is unavailable")

    crow, col = make_graph(args.rows, args.hub_degree, args.other_degree, args.seed)
    x = torch.randn(args.rows, args.features, device="cuda")
    baseline, _ = _prepare_spmm_baseline(crow, col, None, x)
    baseline_ms, baseline_output = time_best(baseline)
    split = torch.ops.autosage.spmm_csr_split
    combinations = itertools.product(
        map(int, args.feature_tiles.split(",")),
        map(int, args.warps.split(",")),
        map(int, args.thresholds.split(",")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "N",
                "F",
                "nnz",
                "hub_degree",
                "other_degree",
                "feature_tile",
                "warps_per_block",
                "hub_threshold",
                "baseline_ms",
                "split_ms",
                "speedup",
                "max_error",
                "allclose",
            ]
        )
        for feature_tile, warps, threshold in combinations:
            split_ms, output = time_best(
                lambda t=feature_tile, w=warps, h=threshold: split(
                    crow, col, None, x, t, w, h
                )
            )
            error = float((baseline_output - output).abs().max().item())
            writer.writerow(
                [
                    args.rows,
                    args.features,
                    col.numel(),
                    args.hub_degree,
                    args.other_degree,
                    feature_tile,
                    warps,
                    threshold,
                    f"{baseline_ms:.4f}",
                    f"{split_ms:.4f}",
                    f"{baseline_ms / split_ms:.4f}",
                    f"{error:.6f}",
                    torch.allclose(baseline_output, output, rtol=1e-4, atol=3e-3),
                ]
            )


if __name__ == "__main__":
    main()
