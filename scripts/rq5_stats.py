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
         ("runs/continual5_cnn", "fetch5"): "CNN trunk, fetch5", ("runs/continual5_snn", "fetch5"): "SNN trunk, fetch5 (150k/task)",
         ("runs/continual5_snn_450k", "fetch5"): "SNN trunk, fetch5, 450k/task (session 3 isolation budget)"}


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
    lines += memory_fair_section()
    os.makedirs("results", exist_ok=True)
    with open("results/rq5_stats.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


MEM_ROOTS = [("runs/continual_budget", "runs/continual", "CNN"), ("runs/continual_snn_budget", "runs/continual_snn", "SNN")]
BYTES_PER_STATE = 188  # measured TaskBuffer cost, K=1 trunks (scripts/memory_fair.py)


def total_mb(r, buf_per_task):
    return (r["params"] * 4 + buf_per_task * len(r["tasks"]) * BYTES_PER_STATE) / 1e6


def memory_fair_section(suite="fetch3"):
    """Pre-registered session-3 test: shared weights at small buffers vs isolation, on ACC and ACC per MB of
    total memory (params + replay buffer)."""
    out = ["\n## Memory-fair RQ5 (session 3): small-buffer shared weights vs isolation, fetch3\n",
           "Total memory = fp32 params + buffer_per_task x tasks x 188 B. Same tests as above; "
           "diff = arm − isolation.\n",
           "| trunk | arm | n | total MB (arm vs iso) | ACC arm | ACC iso | ACC diff | Welch p | perm p | 95% CI | "
           "ACC/MB arm | ACC/MB iso | ACC/MB Welch p |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for broot, iroot, trunk in MEM_ROOTS:
        iso = load(iroot, "isolation", suite)
        if len(iso) < 2:
            continue
        acc_i = [r["metrics"]["ACC"] for r in iso.values()]
        mem_i = total_mb(next(iter(iso.values())), 0)
        eff_i = [a / mem_i for a in acc_i]
        for buf in (50, 200, 2000):
            for method in ("sleep", "replay"):
                arm = load(broot, f"{method}_buf{buf}", suite)
                if len(arm) < 2:
                    continue
                acc_a = [r["metrics"]["ACC"] for r in arm.values()]
                mem_a = total_mb(next(iter(arm.values())), buf)
                eff_a = [a / mem_a for a in acc_a]
                ci = boot_ci(acc_a, acc_i)
                out.append(f"| {trunk} | {method} @{buf}/task | {len(arm)} vs {len(iso)} | {mem_a:.2f} vs {mem_i:.2f} | "
                           f"{fmt(acc_a)} | {fmt(acc_i)} | {np.mean(acc_a) - np.mean(acc_i):+.3f} | "
                           f"{stats.ttest_ind(acc_a, acc_i, equal_var=False).pvalue:.3g} | {perm_p(acc_a, acc_i):.2f} | "
                           f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {np.mean(eff_a):.3f} | {np.mean(eff_i):.3f} | "
                           f"{stats.ttest_ind(eff_a, eff_i, equal_var=False).pvalue:.3g} |")
    return out


if __name__ == "__main__":
    main()
