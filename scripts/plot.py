#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from summarize import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = summarize(args.csv)
    feature_widths = [int(row["F"]) for row in rows]
    speedups = [float(row["speedup"]) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(5.2, 3.2))
    axes.plot(feature_widths, speedups, marker="o")
    axes.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes.set(
        xlabel="Feature width F", ylabel="Speedup (baseline / chosen)", title=args.title
    )
    axes.grid(linestyle=":", alpha=0.6)
    figure.tight_layout()
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
