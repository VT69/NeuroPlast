"""Summarise exploration runs next to the frozen capstone references (read-only).

Total memory as in the paper (scripts/memory_fair.py): fp32 parameters + stored replay states at 188 B/state
(CNN/SNN trunks), plus PackNet's owner-map bits. LwF's transient snapshot (one extra network during training) is
reported as a separate peak column.

    python explore/analyze.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BPS = 188


def load(root, suite):
    g = defaultdict(list)
    for p in glob.glob(f"{root}/{suite}_*_s*/results.json"):
        r = json.load(open(p))
        if not r.get("metrics"):
            continue
        name = os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0][len(suite) + 1:]
        g[name].append(r)
    return g


def mem(r, T):
    buf = r.get("method_kwargs", {}).get("buffer_per_task", 5000) if r["method"] in ("replay", "sleep") else 0
    persistent = r["params"] * 4 + r.get("extra_bits", 0) / 8 + buf * T * BPS
    peak = persistent + (r["params_per_task"] * 4 if r["method"] == "lwf" else 0)
    return persistent / 1e6, peak / 1e6


def table(suite, T, roots):
    print(f"\n== {suite} (CNN) ==")
    print(f"{'arm':28s} {'n':>2s} {'ACC':>13s} {'FORGET':>7s} {'mem MB':>7s} {'peak':>6s}  source")
    for label, root in roots:
        for name, runs in sorted(load(root, suite).items()):
            acc = np.array([r["metrics"]["ACC"] for r in runs])
            fg = np.mean([r["metrics"]["FORGET"] for r in runs])
            m, pk = mem(runs[0], T)
            sd = acc.std(ddof=1) if len(acc) > 1 else 0.0
            seeds = sorted(r["seed"] for r in runs)
            print(f"{name:28s} {len(acc):2d} {acc.mean():.3f} ± {sd:.3f} {fg:7.3f} {m:7.2f} {pk:6.2f}  {label} seeds {seeds}")


if __name__ == "__main__":
    table("fetch3", 3, [("frozen", "runs/continual"), ("frozen", "runs/continual_budget"), ("explore", "explore/runs/cl")])
    table("fetch5", 5, [("frozen", "runs/continual5_cnn"), ("explore", "explore/runs/cl")])
    for p in sorted(glob.glob("explore/runs/mem/*/eval.json")):
        e = json.load(open(p))
        print(f"{os.path.basename(os.path.dirname(p)):22s} success {e['success']:.3f} return {e['return_mean']:.3f} "
              f"len {e['len_mean']:.1f} final-train-return {e['final_train_return']:.3f}")
