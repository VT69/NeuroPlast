"""Phase C secondary analyses. NOT pre-registered: written after seeing the primary results, so exploratory only.

    python explore/confirm_secondary.py    # prints and writes explore/confirm_secondary.md
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy import stats

from confirm_stats import runs, welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    p = 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1) if se > 0 else 1.0
    return d.mean(), d.mean() - h, d.mean() + h, p


def collapsed(r):  # an old task ends below 0.5 although it was learned (>= 0.5 right after its own training)
    T = len(r["R"])
    return any(r["R"][i][i] >= 0.5 and r["R"][-1][i] < 0.5 for i in range(T - 1))


out = ["# Phase C secondary analyses (exploratory; not pre-registered)", ""]

# C3: bimodality of sleep's failure, and the plasticity cost of interleaved replay
sl = runs("explore/runs/confirm", "fetch3", "sleep_rb14080", range(201, 209))
rp = runs("explore/runs/confirm", "fetch3", "replay_b3", range(201, 209))
cs, cr = sum(collapsed(r) for r in sl.values()), sum(collapsed(r) for r in rp.values())
_, pf = stats.fisher_exact([[cs, len(sl) - cs], [cr, len(rp) - cr]])
t0s = [r["R"][-1][0] for r in sl.values()]
t0r = [r["R"][-1][0] for r in rp.values()]
lasts = [r["R"][-1][-1] for r in sl.values()]
lastr = [r["R"][-1][-1] for r in rp.values()]
d, lo, hi, p = welch(lasts, lastr)
out += ["## C3 (CNN fetch3, ~14k replayed samples)", "",
        f"- Runs where a learned old task collapsed below 0.5: sleep {cs}/{len(sl)}, replay {cr}/{len(rp)} "
        f"(Fisher exact p = {pf:.3g}).",
        f"- Task 0 at the end: sleep {', '.join(f'{x:.2f}' for x in t0s)}; replay {', '.join(f'{x:.2f}' for x in t0r)}.",
        f"- Last task (no forgetting possible yet; measures plasticity): sleep {np.mean(lasts):.3f} ± "
        f"{np.std(lasts, ddof=1):.3f}, replay {np.mean(lastr):.3f} ± {np.std(lastr, ddof=1):.3f}; "
        f"difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}], Welch p = {p:.3g}.",
        f"- Replayed samples: sleep {np.mean([r['replay_samples'] for r in sl.values()]):,.0f}, replay "
        f"{np.mean([r['replay_samples'] for r in rp.values()]):,.0f} (range "
        f"{min(r['replay_samples'] for r in rp.values()):,}-{max(r['replay_samples'] for r in rp.values()):,}).", ""]

# C2: paired by seed (both arms share the seed's task-0 environment draws and initialisation)
lw = runs("explore/runs/confirm_snn", "fetch3", "lwf_int8", range(201, 209))
rb = runs("explore/runs/confirm_snn", "fetch3", "replay_buf1380", range(201, 209))
seeds = sorted(set(lw) & set(rb))
out += ["## C2 (SNN fetch3), paired by seed", ""]
for metric in ("ACC", "FORGET"):
    a = [lw[s]["metrics"][metric] for s in seeds]
    b = [rb[s]["metrics"][metric] for s in seeds]
    d, lo, hi, p = paired(a, b)
    out.append(f"- {metric}: LwF-int8 − replay@1380 = {d:+.3f} [{lo:+.3f}, {hi:+.3f}], paired t p = {p:.3g} "
               f"(n = {len(seeds)} seeds).")
acc_l = np.array([lw[s]["metrics"]["ACC"] for s in seeds])
acc_r = np.array([rb[s]["metrics"]["ACC"] for s in seeds])
out += [f"- Seed-level correlation of ACC across arms: r = {np.corrcoef(acc_l, acc_r)[0, 1]:.2f}. Seed 204 fails "
        f"task 0 in both arms (0.47 right after its own training), so the seed, not the method, decides it.", ""]

# C1: spread and worst case
l5 = runs("explore/runs/confirm", "fetch5", "lwf_int8", range(201, 211))
r5 = runs("explore/runs/confirm", "fetch5", "replay_buf973", range(201, 211))
a = [r["metrics"]["ACC"] for r in l5.values()]
b = [r["metrics"]["ACC"] for r in r5.values()]
lev = stats.levene(a, b, center="median")
d, lo, hi, p = paired([l5[s]["metrics"]["ACC"] for s in sorted(l5)], [r5[s]["metrics"]["ACC"] for s in sorted(l5)])
out += ["## C1 (CNN fetch5)", "",
        f"- Spread of final ACC: LwF-int8 SD {np.std(a, ddof=1):.3f} (min {min(a):.3f}), replay@973 SD "
        f"{np.std(b, ddof=1):.3f} (min {min(b):.3f}); Brown-Forsythe p = {lev.pvalue:.3g}.",
        f"- Paired by seed: LwF-int8 − replay@973 = {d:+.3f} [{lo:+.3f}, {hi:+.3f}], paired t p = {p:.3g}.",
        f"- LwF-int8 is above replay@973 on {sum(l5[s]['metrics']['ACC'] > r5[s]['metrics']['ACC'] for s in l5)}"
        f"/{len(l5)} seeds.", ""]

text = "\n".join(out)
print(text)
open("explore/confirm_secondary.md", "w").write(text)
