#!/usr/bin/env python3
import sys, csv, collections

def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def summarize(paths):
    by = collections.defaultdict(lambda: {"baseline": None, "chosen": None, "choice": "?"})
    keys = []
    for p in paths:
        rows = load_rows(p)
        for row in rows:
            key = (row["dataset"], int(row["F"]))
            keys.append(key)
            if row["path"] in ("baseline","chosen"):
                by[key][row["path"]] = float(row["median_ms"])
            if "choice" in row and row["path"] == "chosen":
                by[key]["choice"] = row["choice"]
    print("\nSpeedup (baseline / chosen):\n")
    print(f"{'dataset':20s} {'F':>6s} {'choice':>10s} {'baseline(ms)':>14s} {'chosen(ms)':>12s} {'speedup':>9s}")
    for ds, F in sorted(set(keys), key=lambda x:(x[0], x[1])):
        rec = by[(ds,F)]
        b, c = rec["baseline"], rec["chosen"]
        if b is None or c is None: continue
        sp = (b/c) if c>0 else float("inf")
        print(f"{ds:20s} {F:6d} {rec['choice']:>10s} {b:14.3f} {c:12.3f} {sp:9.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: scripts/p4_summarize.py results/p4/*.csv")
        sys.exit(1)
    summarize(sys.argv[1:])
