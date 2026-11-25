#!/usr/bin/env python3
"""Summarise results/<run>/<label>/timings.csv into Markdown tables.

Usage: scripts/make_report.py results/20261122_rerun [label1 label2 ...]
Steady-state = rows whose `run` is a number (warmup rows are excluded).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

run_dir = Path(sys.argv[1])
labels = sys.argv[2:] or sorted(p.parent.name for p in run_dir.glob("*/timings.csv"))
CASES = ["t2i", "i2t_1tok", "i2t_64tok", "i2i"]

data = defaultdict(lambda: defaultdict(list))   # label -> case -> [(seconds, tokens)]
warm = defaultdict(lambda: defaultdict(list))
for label in labels:
    path = run_dir / label / "timings.csv"
    if not path.exists():
        continue
    for row in csv.DictReader(path.open()):
        target = data if row["run"].isdigit() else warm
        target[label][row["case"]].append((float(row["seconds"]), row["out_tokens"]))

print("| case | " + " | ".join(labels) + " |")
print("| --- | " + " | ".join("---:" for _ in labels) + " |")
for case in CASES:
    cells = []
    for label in labels:
        v = data[label].get(case)
        if not v:
            cells.append("-")
            continue
        s = [x for x, _ in v]
        toks = sorted({t for _, t in v if t})
        extra = f" [{','.join(toks)} tok]" if toks else ""
        cells.append(f"**{statistics.median(s):.3f}** ({min(s):.3f}-{max(s):.3f}, n={len(s)}){extra}")
    print(f"| {case} | " + " | ".join(cells) + " |")

print("\nFirst warmup request (s):")
print("| case | " + " | ".join(labels) + " |")
print("| --- | " + " | ".join("---:" for _ in labels) + " |")
for case in CASES:
    cells = []
    for label in labels:
        v = warm[label].get(case)
        cells.append(f"{v[0][0]:.2f}" if v else "-")
    print(f"| {case} | " + " | ".join(cells) + " |")
