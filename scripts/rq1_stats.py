"""RQ1 pre-registered comparisons (session 3), DoorKey-6x6, 400k frames (runs/rq1_v2).

Arms: bp (backprop), homeo (bp + homeostasis), tfs0.03 (bp + stabilised three-factor STDP
alpha 0.03, which includes homeostasis), rands0.03 (bp + random update of matched RMS at
alpha 0.03 + homeostasis). Primary statistic: solve rate (final eval return > 0.5), Fisher's
exact test, two-sided. Secondary: AUC (mean training return over the run) and final eval
return, Welch t-test. Holm correction across the four primary contrasts.
Writes results/rq1_stats.md.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroplast.eval.metrics import auc  # noqa: E402

ROOT = "runs/rq1_v2"
ARMS = ["bp", "homeo", "tfs0.03", "rands0.03"]
CONTEXT = ["frozen", "rand0.3", "tfs0.3", "hebs0.03", "dfae", "dfae_homeo", "dfae_tfs0.03", "dfae_rands0.03"]
CONTRASTS = [("tfs0.03", "homeo", "(a) does STDP add beyond homeostasis?"),
             ("tfs0.03", "rands0.03", "(b) does STDP add beyond matched random perturbation?"),
             ("homeo", "bp", "(c) does homeostasis help backprop (not only DFA)?"),
             ("rands0.03", "homeo", "(d) does generic perturbation add beyond homeostasis?")]


def load(arm):
    out = {}
    for p in glob.glob(os.path.join(ROOT, f"rq1v2_{arm}_s*", "eval.json")):
        name = os.path.basename(os.path.dirname(p))
        if name.rsplit("_s", 1)[0] != f"rq1v2_{arm}":
            continue
        seed = int(name.rsplit("_s", 1)[1])
        d = pd.read_csv(p.replace("eval.json", "metrics.csv"))
        ev = json.load(open(p))["return_mean"]
        f09 = d.frames[d.ep_return >= 0.9].min() if (d.ep_return >= 0.9).any() else np.nan
        out[seed] = dict(eval=ev, solved=ev > 0.5, auc=auc(d.ep_return), f09=f09)
    return out


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else (f"{x.mean():.3f}" if len(x) else "-")


def main():
    data = {a: load(a) for a in ARMS + CONTEXT}
    lines = ["# RQ1 statistics (pre-registered, session 3): DoorKey-6x6, 400k frames\n",
             "solved = final eval return > 0.5 (outcomes are bimodal). Fisher = two-sided Fisher's exact test on "
             "solve counts; Welch = two-sided Welch t-test; Holm = Holm-adjusted p across the four primary contrasts "
             "(computed separately for each statistic). frames_to_0.9 is over solving seeds only.\n",
             "## Arms\n", "| arm | n | solved | solve rate | AUC | final eval | frames to 0.9 (solvers) |", "|---|---|---|---|---|---|---|"]
    for a in ARMS + ["---"] + CONTEXT:
        if a == "---":
            lines.append("| *context arms (not part of the pre-registered test)* | | | | | | |")
            continue
        d = data[a]
        if not d:
            continue
        v = list(d.values())
        s = sum(x["solved"] for x in v)
        f = [x["f09"] for x in v if x["solved"] and not np.isnan(x["f09"])]
        lines.append(f"| {a} | {len(v)} | {s}/{len(v)} | {s / len(v):.2f} | {fmt([x['auc'] for x in v])} | "
                     f"{fmt([x['eval'] for x in v])} | {np.mean(f) / 1e3:.0f}k |" if f else
                     f"| {a} | {len(v)} | {s}/{len(v)} | {s / len(v):.2f} | {fmt([x['auc'] for x in v])} | "
                     f"{fmt([x['eval'] for x in v])} | - |")
    rows, pf, pa, pe = [], [], [], []
    for x, y, q in CONTRASTS:
        A, B = list(data[x].values()), list(data[y].values())
        if len(A) < 2 or len(B) < 2:
            continue
        sa, sb = sum(r["solved"] for r in A), sum(r["solved"] for r in B)
        p_f = stats.fisher_exact([[sa, len(A) - sa], [sb, len(B) - sb]]).pvalue
        p_a = stats.ttest_ind([r["auc"] for r in A], [r["auc"] for r in B], equal_var=False).pvalue
        p_e = stats.ttest_ind([r["eval"] for r in A], [r["eval"] for r in B], equal_var=False).pvalue
        d_auc = np.mean([r["auc"] for r in A]) - np.mean([r["auc"] for r in B])
        rows.append((q, x, y, f"{sa}/{len(A)} vs {sb}/{len(B)}", d_auc))
        pf.append(p_f), pa.append(p_a), pe.append(p_e)
    if rows:
        hf, ha, he = holm(pf), holm(pa), holm(pe)
        lines += ["\n## Pre-registered contrasts\n",
                  "| question | contrast | solved | Fisher p (Holm) | ΔAUC | Welch AUC p (Holm) | Welch eval p (Holm) |",
                  "|---|---|---|---|---|---|---|"]
        for i, (q, x, y, sv, d) in enumerate(rows):
            lines.append(f"| {q} | {x} vs {y} | {sv} | {pf[i]:.3f} ({hf[i]:.3f}) | {d:+.3f} | "
                         f"{pa[i]:.3f} ({ha[i]:.3f}) | {pe[i]:.3f} ({he[i]:.3f}) |")
    os.makedirs("results", exist_ok=True)
    with open("results/rq1_stats.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
