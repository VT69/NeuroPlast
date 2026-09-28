"""RQ5 significance: shared-weight sleep vs parameter isolation, per trunk.

For each continual root that has both `sleep` and `isolation` runs:
  - final ACC (mean return over all tasks after the sequence)
  - ACC per 100k parameters (RQ5's efficiency axis)
  - FWT of sleep (isolation's from-scratch curves are the zero reference, so
    this is a one-sample test of FWT > 0)
with three tests each, because n is small (3 seeds per arm in session 1):
  * Welch t-test (two-sided)
  * exact permutation test on the difference of means (two-sided; with 3 vs 3
    the smallest attainable p is 2/20 = 0.10, so it can never reach 0.05)
  * 95% bootstrap CI of the difference (10k resamples within each arm)
Seeds are *not* paired (isolation initialises 3 fresh networks per seed), so all
tests are unpaired. Re-run after merging more seeds:  python scripts/rq5_stats.py
Writes results/rq5_stats.md.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroplast.eval.metrics import auc  # noqa: E402

ROOTS = {("runs/continual", "fetch3"): "CNN trunk, fetch3", ("runs/continual_snn", "fetch3"): "SNN trunk, fetch3",
         ("runs/continual_hybrid", "fetch3"): "Full hybrid (SNN + Transformer), fetch3",
         ("runs/continual5_cnn", "fetch5"): "CNN trunk, fetch5", ("runs/continual5_snn", "fetch5"): "SNN trunk, fetch5"}


def load(root, method, suite="fetch3"):
    out = {}
    for p in glob.glob(os.path.join(root, f"{suite}_{method}_s*", "results.json")):
        name = os.path.basename(os.path.dirname(p))
        if name.rsplit("_s", 1)[0] != f"{suite}_{method}":
            continue
        r = json.load(open(p))
        if r.get("metrics"):
            out[r["seed"]] = r
    return out


def perm_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n, cnt, tot = len(a), 0, 0
    for idx in itertools.combinations(range(len(pooled)), n):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        cnt += abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12
        tot += 1
    return cnt / tot


def boot_ci(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def fmt(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"


def main():
    lines = ["# RQ5 statistics: shared-weight sleep vs parameter isolation\n",
             "Unpaired tests (isolation builds fresh networks, so seeds aren't paired). Welch = two-sided "
             "Welch t-test; perm = exact two-sided permutation test (min attainable p with 3 vs 3 is 0.10); "
             "CI = 95% bootstrap CI of (sleep − isolation). FWT row: one-sample t-test of sleep's FWT against 0 "
             "(isolation's from-scratch curves are the reference, FWT_isolation ≡ 0).\n"]
    for (root, suite), title in ROOTS.items():
        sl, iso = load(root, "sleep", suite), load(root, "isolation", suite)
        if len(sl) < 2 or len(iso) < 2:
            continue
        acc_s = [r["metrics"]["ACC"] for r in sl.values()]
        acc_i = [r["metrics"]["ACC"] for r in iso.values()]
        eff_s = [r["metrics"]["ACC"] / (r["params"] / 1e5) for r in sl.values()]
        eff_i = [r["metrics"]["ACC"] / (r["params"] / 1e5) for r in iso.values()]
        fwt = []
        for seed, r in sl.items():
            if seed in iso:
                ref = iso[seed]["curves"]
                fwt.append(np.mean([(auc(r["curves"][j]) - auc(ref[j])) / (1 - auc(ref[j]))
                                    for j in range(1, len(r["curves"]))]))
        lines.append(f"\n## {title} (`{root}`): sleep n={len(sl)}, isolation n={len(iso)}\n")
        lines.append("| metric | sleep | isolation | diff | Welch p | perm p | 95% CI of diff |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, a, b in (("ACC", acc_s, acc_i), ("ACC / 100k params", eff_s, eff_i)):
            ci = boot_ci(a, b)
            lines.append(f"| {name} | {fmt(a)} | {fmt(b)} | {np.mean(a) - np.mean(b):+.3f} | "
                         f"{stats.ttest_ind(a, b, equal_var=False).pvalue:.3g} | {perm_p(a, b):.2f} | "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        if len(fwt) > 1:
            p1 = stats.ttest_1samp(fwt, 0).pvalue
            lines.append(f"| FWT (sleep vs 0) | {fmt(fwt)} | 0 (reference) | {np.mean(fwt):+.3f} | "
                         f"{p1:.3g} (1-sample) | - | - |")
        # per-seed detail so outliers are visible
        lines.append("\nper-seed ACC: sleep " + ", ".join(f"s{k}={v['metrics']['ACC']:.3f}" for k, v in sorted(sl.items()))
                     + "; isolation " + ", ".join(f"s{k}={v['metrics']['ACC']:.3f}" for k, v in sorted(iso.items())))
    os.makedirs("results", exist_ok=True)
    with open("results/rq5_stats.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
