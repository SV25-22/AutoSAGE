#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Callable, Optional

import torch

from autosage import (
    calibrate_full,
    csr_attention_forward,
    load_native,
    native_available,
    spmm_csr_auto,
)
from autosage._auto import _prepare_spmm_baseline, _run_spmm_choice
from autosage.utils.csr import edge_index_to_csr


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def time_cuda(
    function: Callable[[], torch.Tensor], repeats: int, warmup: int = 3
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _numpy_csr(row, col, rows: int, device: torch.device):
    import numpy as np

    pair = np.unique(row * np.int64(rows) + col)
    row = pair // np.int64(rows)
    col = pair % np.int64(rows)
    order = np.argsort(row, kind="mergesort")
    row = row[order]
    col = col[order]
    crow = np.zeros(rows + 1, dtype=np.int64)
    np.add.at(crow, row + 1, 1)
    np.cumsum(crow, out=crow)
    return torch.from_numpy(crow).to(device), torch.from_numpy(col).to(device)


def load_dataset(name: str, device: torch.device, seed: int):
    normalized = name.lower()
    if normalized in {"reddit", "reddit-pyg"}:
        from torch_geometric.datasets import Reddit

        data = Reddit(root="data/pyg/Reddit")[0]
        edge_index = data.edge_index.to(device)
        crow, col = edge_index_to_csr(edge_index, int(data.num_nodes), rows_by="dst")
        return "reddit(pyg)", crow, col, data.x.to(device)

    if normalized in {"products", "ogbn-products"}:
        from ogb.nodeproppred import PygNodePropPredDataset

        data = PygNodePropPredDataset(name="ogbn-products", root="data/ogb")[0]
        edge_index = data.edge_index.to(device)
        crow, col = edge_index_to_csr(edge_index, int(data.num_nodes), rows_by="dst")
        return "ogbn-products", crow, col, data.x.to(device)

    if normalized.startswith("synth:"):
        import numpy as np

        _, rows_text, probability_text = normalized.split(":")
        rows = int(rows_text)
        probability = float(probability_text)
        generator = np.random.default_rng(seed)
        edge_count = max(1, int(probability * rows * rows))
        row = generator.integers(0, rows, size=edge_count, dtype=np.int64)
        col = generator.integers(0, rows, size=edge_count, dtype=np.int64)
        crow, col_tensor = _numpy_csr(row, col, rows, device)
        return f"synth:{rows}:{probability}", crow, col_tensor, None

    if normalized.startswith("synthhub:"):
        import numpy as np

        _, rows_text, hubs_text, fraction_text = normalized.split(":")
        rows = int(rows_text)
        hubs = int(hubs_text)
        fraction = float(fraction_text)
        generator = np.random.default_rng(seed)
        per_hub = max(1, int(rows * fraction))
        row_parts = []
        col_parts = []
        for hub in range(hubs):
            targets = generator.choice(rows, size=per_hub, replace=False)
            row_parts.append(np.full(targets.shape, hub, dtype=np.int64))
            col_parts.append(targets.astype(np.int64))
        background = min(
            int(float(os.getenv("AUTOSAGE_SYNTH_BG_PER_NODE", "0.5")) * rows),
            int(os.getenv("AUTOSAGE_SYNTH_BG_CAP", "1500000")),
        )
        if background:
            row_parts.append(
                generator.integers(0, rows, size=background, dtype=np.int64)
            )
            col_parts.append(
                generator.integers(0, rows, size=background, dtype=np.int64)
            )
        crow, col_tensor = _numpy_csr(
            np.concatenate(row_parts), np.concatenate(col_parts), rows, device
        )
        return f"synthhub:{rows}:{hubs}:{fraction}", crow, col_tensor, None

    raise ValueError(f"unknown dataset: {name}")


def features(
    source: Optional[torch.Tensor],
    rows: int,
    width: int,
    device: torch.device,
    use_real: bool,
) -> torch.Tensor:
    if use_real and source is not None and source.size(1) >= width:
        return source[:, :width].float().contiguous()
    return torch.randn(rows, width, dtype=torch.float32, device=device)


def write_spmm_rows(
    writer: csv.writer,
    dataset: str,
    crow: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    repeats: int,
    guardrail: float,
    calibrate: bool,
    seed: int,
) -> str:
    if calibrate:
        choice, info = calibrate_full(crow, col, None, x, guardrail=guardrail)
    else:
        _, info = spmm_csr_auto(crow, col, None, x, guardrail=guardrail, seed=seed)
        choice = str(info["choice"])
    baseline, baseline_name = _prepare_spmm_baseline(crow, col, None, x)
    if choice == "autosage":

        def chosen():
            return _run_spmm_choice(choice, info.get("candidate"), crow, col, None, x)
    else:
        chosen = baseline
    baseline_timing = time_cuda(baseline, repeats)
    chosen_timing = (
        time_cuda(chosen, repeats) if choice == "autosage" else baseline_timing
    )
    rows = crow.numel() - 1
    nonzeros = col.numel()
    moved_gb = ((nonzeros + rows) * x.size(1) * x.element_size()) / 1e9
    for path, timing in (("baseline", baseline_timing), ("chosen", chosen_timing)):
        writer.writerow(
            [
                dataset,
                rows,
                nonzeros,
                x.size(1),
                path,
                "baseline" if path == "baseline" else choice,
                f"{timing['median']:.4f}",
                f"{timing['mean']:.4f}",
                f"{timing['minimum']:.4f}",
                f"{timing['maximum']:.4f}",
                f"{moved_gb / (timing['median'] / 1000):.3f}",
            ]
        )
    print(
        f"[{dataset}] F={x.size(1)} baseline={baseline_timing['median']:.3f} ms "
        f"chosen({choice})={chosen_timing['median']:.3f} ms "
        f"speedup={baseline_timing['median'] / chosen_timing['median']:.3f}x"
    )
    return baseline_name


def write_attention_row(
    writer: csv.writer,
    dataset: str,
    crow: torch.Tensor,
    col: torch.Tensor,
    x: torch.Tensor,
    repeats: int,
    guardrail: float,
    seed: int,
) -> dict[str, str]:
    query = x
    key = torch.randn_like(x)
    value = torch.randn_like(x)
    last_info = {}

    def run():
        nonlocal last_info
        output, last_info = csr_attention_forward(
            crow, col, query, key, value, guardrail=guardrail, seed=seed
        )
        return output

    timing = time_cuda(run, repeats)
    rows = crow.numel() - 1
    nonzeros = col.numel()
    moved_gb = (
        ((3 * nonzeros + rows) * x.size(1) + 2 * nonzeros) * x.element_size() / 1e9
    )
    writer.writerow(
        [
            dataset,
            rows,
            nonzeros,
            x.size(1),
            "attention",
            last_info["sddmm"]["choice"],
            last_info["spmm"]["choice"],
            f"{timing['median']:.4f}",
            f"{timing['mean']:.4f}",
            f"{timing['minimum']:.4f}",
            f"{timing['maximum']:.4f}",
            f"{moved_gb / (timing['median'] / 1000):.3f}",
        ]
    )
    return {
        "sddmm": str(last_info["sddmm"]["baseline"]),
        "spmm": str(last_info["spmm"]["baseline"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--features", default="64,128,256")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--guardrail", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("spmm", "attention"), default="spmm")
    parser.add_argument("--use-real-features", action="store_true")
    parser.add_argument("--calibrate-full", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA-capable PyTorch installation is required")
    load_native()
    if not native_available():
        raise SystemExit(
            "AutoSAGE native extension is unavailable; run scripts/build.sh"
        )

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    dataset, crow, col, real_features = load_dataset(args.dataset, device, args.seed)
    widths = [int(value) for value in args.features.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_name: Optional[str | dict[str, str]] = None
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if args.mode == "spmm":
            writer.writerow(
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
            for width in widths:
                x = features(
                    real_features,
                    crow.numel() - 1,
                    width,
                    device,
                    args.use_real_features,
                )
                baseline_name = write_spmm_rows(
                    writer,
                    dataset,
                    crow,
                    col,
                    x,
                    args.repeats,
                    args.guardrail,
                    args.calibrate_full,
                    args.seed,
                )
        else:
            writer.writerow(
                [
                    "dataset",
                    "N",
                    "nnz",
                    "F",
                    "path",
                    "sddmm_choice",
                    "spmm_choice",
                    "median_ms",
                    "mean_ms",
                    "min_ms",
                    "max_ms",
                    "approx_GBps",
                ]
            )
            for width in widths:
                x = features(
                    real_features,
                    crow.numel() - 1,
                    width,
                    device,
                    args.use_real_features,
                )
                baseline_name = write_attention_row(
                    writer,
                    dataset,
                    crow,
                    col,
                    x,
                    args.repeats,
                    args.guardrail,
                    args.seed,
                )

    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "schema_version": 2,
        "git_commit": git_commit(),
        "dataset": args.dataset,
        "mode": args.mode,
        "baseline": baseline_name,
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory": properties.total_memory,
        "python_packages": {
            name: package_version(name)
            for name in ("torch", "torch-geometric", "ogb", "numpy")
        },
        "torch_cuda": torch.version.cuda,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("AUTOSAGE_")
        },
        "arguments": vars(args) | {"output": str(args.output)},
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
