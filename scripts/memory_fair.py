"""Memory-fair RQ5 (literature doc section 38.3): ACC vs *total* memory for every continual run.

Total memory = fp32 parameters + stored replay states at their measured size.
Measured TaskBuffer cost per stored state (see PROGRESS.md session 3):
    obs uint8 7x7x3 per frame (147 B x window K) + mask (K B) + task id int64 (8 B)
    + teacher logits 7 x fp32 (28 B) + teacher value fp32 (4 B)
    = 188 B/state (K=1, CNN/SNN trunks), 632 B/state (K=4, hybrid).
The implementation also keeps a concatenated copy of the buffer (2x RAM); that is an
implementation artefact, reported separately ("as implemented").
The buffer counted is what the agent must keep to continue learning after the sequence:
buffer_per_task x number of tasks (end_task stores the last task too).
Sleep additionally holds, *during training only*, a reservoir of <= 5000 current-task states
and a frozen snapshot of the network; reported as a transient peak, not persistent memory.

Writes results/rq5_memory.md and results/figs/rq5_memory_<suite>.png.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze import C, INK2, style  # noqa: E402

ROOTS = {"runs/continual": "CNN", "runs/continual_budget": "CNN", "runs/continual_snn": "SNN",
         "runs/continual_snn_budget": "SNN", "runs/continual_hybrid": "Hybrid",
         "runs/continual5_cnn": "CNN", "runs/continual5_snn": "SNN",
         "runs/continual5_snn_450k": "SNN"}
OBS_B, FIXED_B = 147, 40          # per frame; mask/window adds K bytes
RESERVOIR = 5000                  # sleep's current-task reservoir (neuroplast/sleep/sleep.py)


def bytes_per_state(window):
    return OBS_B * window + window + FIXED_B


def load():
    groups = defaultdict(list)
    for root, trunk in ROOTS.items():
        for p in glob.glob(os.path.join(root, "*", "results.json")):
            r = json.load(open(p))
            if not r.get("metrics"):
                continue
            method = r["method"]
            name = os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0]
            cfg = r["cfg"]
            window = cfg.get("window", 1) if cfg.get("memory") else 1
            T = len(r["tasks"])
            mk = r.get("method_kwargs", {}) or {}
            buf_per_task = mk.get("buffer_per_task", 5000) if method in ("replay", "sleep") else 0
            if method == "sleep" and mk.get("distill") is False:
                buf_per_task = 0
            bps = bytes_per_state(window)
            params_mb = r["params"] * 4 / 1e6
            buf_mb = buf_per_task * T * bps / 1e6
            peak_extra = (min(RESERVOIR, 10**9) * bps + r["params"] * 4) / 1e6 if method == "sleep" else 0.0
            name = f"{name} @{cfg['total_frames'] // 1000}k"  # 150k vs 450k fetch5 runs share dir names
            groups[(r["suite"], trunk, name)].append(dict(
                acc=r["metrics"]["ACC"], params_mb=params_mb, buf_mb=buf_mb, total_mb=params_mb + buf_mb,
                impl_mb=params_mb + 2 * buf_mb, peak_mb=params_mb + buf_mb + peak_extra,
                buf_per_task=buf_per_task, method=method, bps=bps, T=T, frames=cfg["total_frames"]))
    return groups


def main():
    g = load()
    lines = ["# RQ5, memory-fair: ACC vs total memory (parameters + replay buffer)\n",
             "Total memory = fp32 parameters + stored replay states at the measured minimum size "
             "(188 B/state for CNN/SNN trunks, 632 B/state for the hybrid's 4-frame windows). "
             "'as implemented' doubles the buffer (the code caches a concatenated copy). "
             "Sleep's transient training peak adds a <=5000-state current-task reservoir + one network snapshot. "
             "Buffer counted = buffer_per_task x tasks (what's needed to keep learning). "
             "ACC/MB = ACC per MB of total memory.\n"]
    for suite in sorted({k[0] for k in g}):
        for trunk in ("CNN", "SNN", "Hybrid"):
            rows = [(k[2], v) for k, v in g.items() if k[0] == suite and k[1] == trunk]
            if not rows:
                continue
            lines.append(f"\n## {suite}, {trunk} trunk\n")
            lines.append("| arm | n | frames/task | buffer/task | params MB | buffer MB | **total MB** | as impl. MB | sleep peak MB | ACC | ACC per MB |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for name, runs in sorted(rows, key=lambda kv: np.mean([r["total_mb"] for r in kv[1]])):
                a = np.array([r["acc"] for r in runs])
                r0 = runs[0]
                sd = f" ± {a.std(ddof=1):.3f}" if len(a) > 1 else ""
                lines.append(f"| {name} | {len(a)} | {r0['frames'] // 1000}k | {r0['buf_per_task']} | {r0['params_mb']:.2f} | "
                             f"{r0['buf_mb']:.2f} | **{r0['total_mb']:.2f}** | {r0['impl_mb']:.2f} | "
                             f"{r0['peak_mb']:.2f} | {a.mean():.3f}{sd} | {a.mean() / r0['total_mb']:.3f} |"
                             if r0["method"] == "sleep" else
                             f"| {name} | {len(a)} | {r0['frames'] // 1000}k | {r0['buf_per_task']} | {r0['params_mb']:.2f} | "
                             f"{r0['buf_mb']:.2f} | **{r0['total_mb']:.2f}** | {r0['impl_mb']:.2f} | - | "
                             f"{a.mean():.3f}{sd} | {a.mean() / r0['total_mb']:.3f} |")
        plot(suite, {k: v for k, v in g.items() if k[0] == suite})
    os.makedirs("results", exist_ok=True)
    with open("results/rq5_memory.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(suite, g):
    os.makedirs("results/figs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = {"CNN": "o", "SNN": "s", "Hybrid": "^"}
    methods = ["isolation", "sleep", "replay", "naive", "ewc"]
    seen = set()
    for (s, trunk, name), runs in g.items():
        m = runs[0]["method"]
        if m not in methods:
            continue
        x = np.mean([r["total_mb"] for r in runs])
        y = np.mean([r["acc"] for r in runs])
        col = C[methods.index(m)]
        lab = m if m not in seen else None
        seen.add(m)
        ax.scatter(x, y, s=45, marker=markers[trunk], color=col, label=lab, edgecolor="white", linewidth=1.5, zorder=3)
        if m in ("sleep", "replay") and runs[0]["buf_per_task"] != 5000:
            ax.annotate(f"{trunk} buf {runs[0]['buf_per_task']}", (x, y), textcoords="offset points", xytext=(5, -3),
                        fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    style(ax, f"{suite}: ACC vs total memory (params + buffer)", "total memory, MB (log)", "final ACC")
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker=mk, ls="", color=INK2) for mk in markers.values()]
    l += [f"{t} trunk" for t in markers]
    ax.legend(h, l, frameon=False, fontsize=7, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"results/figs/rq5_memory_{suite}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
