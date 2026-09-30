"""Core-hour ledger for the exploration phase (one thread per job, so wall time = core time).

Continual runs: results.json wall_time. Single-task runs: frames / cumulative fps on the last metrics.csv row
(the same rule as paper/compute_numbers.py). Analysis jobs append {name, core_h} to explore/budget_extra.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ledger():
    os.chdir(ROOT)
    groups, cl_dirs = defaultdict(float), set()
    for p in glob.glob("explore/runs/**/results.json", recursive=True):
        r = json.load(open(p))
        if "wall_time" in r:
            groups[p.split("/")[2]] += r["wall_time"] / 3600
            cl_dirs.add(os.path.dirname(p))
    for p in glob.glob("explore/runs/**/metrics.csv", recursive=True):
        if any(p.startswith(d + os.sep) for d in cl_dirs):
            continue
        rows = list(csv.DictReader(open(p)))
        if rows and float(rows[-1]["fps"]) > 0:
            groups[p.split("/")[2]] += int(rows[-1]["frames"]) / float(rows[-1]["fps"]) / 3600
    extra = "explore/budget_extra.json"
    for e in (json.load(open(extra)) if os.path.exists(extra) else []):
        groups[e["name"]] += e["core_h"]
    return dict(groups), sum(groups.values())


if __name__ == "__main__":
    g, tot = ledger()
    for k, v in sorted(g.items()):
        print(f"{k:28s} {v:6.2f} core-h")
    print(f"{'TOTAL':28s} {tot:6.2f} core-h (budget about 30)")
