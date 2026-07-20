#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def summarize(paths: list[Path]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, object]] = defaultdict(dict)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("path") not in {"baseline", "chosen"}:
                    continue
                key = (row["dataset"], int(row["F"]))
                grouped[key][row["path"]] = float(row["median_ms"])
                if row["path"] == "chosen":
                    grouped[key]["choice"] = row.get("choice", "unknown")

    output: list[dict[str, object]] = []
    for (dataset, feature_width), values in sorted(grouped.items()):
        if "baseline" not in values or "chosen" not in values:
            continue
        baseline = float(values["baseline"])
        chosen = float(values["chosen"])
        output.append(
            {
                "dataset": dataset,
                "F": feature_width,
                "choice": values.get("choice", "unknown"),
                "baseline_ms": baseline,
                "chosen_ms": chosen,
                "speedup": baseline / chosen,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        f"{'dataset':<26} {'F':>5} {'choice':>10} {'baseline':>11} {'chosen':>11} {'speedup':>9}"
    )
    for row in summarize(args.csv):
        print(
            f"{row['dataset']:<26} {row['F']:>5} {row['choice']:>10} "
            f"{row['baseline_ms']:>11.3f} {row['chosen_ms']:>11.3f} {row['speedup']:>8.3f}x"
        )


if __name__ == "__main__":
    main()
