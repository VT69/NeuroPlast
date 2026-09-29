"""Session 4 pre-registered tests (PROGRESS.md "Session 4 ... Pre-registration"). Writes results/s4_stats.md.

Block 1: full hybrid, fetch3: sleep_matched (sleep with replay_budget = same-seed replay run's replay_samples)
         vs replay. Planned test: final ACC, two-sided Welch. n = 8 vs 8 if seeds 7-8 of both arms exist, else
         seeds 1-6 (stopping rule fixed in advance).
Block 2: DoorKey-6x6, DFA on the SNN encoder vs + homeostasis, 10 seeds each. Planned: Fisher exact on solve rate
         (eval return > 0.5) and Welch on AUC, Holm across the two.
Block 3: memory benchmark. Planned: eval success, SNN+Transformer vs SNN frame stack, two-sided Welch, n = 3 vs 3.
Everything else in the file is descriptive.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rq1_stats import fmt, holm, load as load_rq1  # noqa: E402
from scripts.rq5_stats import boot_ci, perm_p  # noqa: E402

HYB = "runs/continual_hybrid"
MEM = "runs/s4_mem"


def cl_runs(name):
    out = {}
    for p in glob.glob(os.path.join(HYB, f"fetch3_{name}_s*", "results.json")):
        d = os.path.basename(os.path.dirname(p))
        if d.rsplit("_s", 1)[0] != f"fetch3_{name}":
            continue
        r = json.load(open(p))
        if r.get("metrics"):
            out[r["seed"]] = r
    return out


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def block1():
    L = ["## Block 1: full hybrid, sleep vs replay with replayed samples matched exactly (RQ2)\n"]
    sm, rp, su = cl_runs("sleep_matched"), cl_runs("replay"), cl_runs("sleep")
    seeds = [s for s in range(1, 9) if s in sm and s in rp]
    use = seeds if all(s in seeds for s in (7, 8)) else [s for s in seeds if s <= 6]
    if len(use) < 2:
        return L + ["(not enough runs yet)\n"]
    L.append(f"Seeds analysed: {use} (stopping rule: 8 vs 8 if seeds 7-8 finished, else 1-6).\n")
    L.append("| seed | replay: replayed states | sleep_matched: replayed states | match | replay ACC | "
             "sleep_matched ACC | unmatched sleep ACC (1.43M states) |")
    L.append("|---|---|---|---|---|---|---|")
    for s in use:
        a, b = rp[s]["replay_samples"], sm[s]["replay_samples"]
        L.append(f"| {s} | {a:,} | {b:,} | {'exact' if a == b else 'MISMATCH'} | {rp[s]['metrics']['ACC']:.3f} | "
                 f"{sm[s]['metrics']['ACC']:.3f} | {su[s]['metrics']['ACC']:.3f} |" if s in su else
                 f"| {s} | {a:,} | {b:,} | {'exact' if a == b else 'MISMATCH'} | {rp[s]['metrics']['ACC']:.3f} | "
                 f"{sm[s]['metrics']['ACC']:.3f} | - |")
    L.append("\n| metric | sleep_matched | replay | diff | Welch p | perm p | 95% bootstrap CI |")
    L.append("|---|---|---|---|---|---|---|")
    for k, planned in (("ACC", True), ("FORGET", False), ("BWT", False)):
        a = [sm[s]["metrics"][k] for s in use]
        b = [rp[s]["metrics"][k] for s in use]
        ci = boot_ci(a, b)
        tag = " **(planned test)**" if planned else " (descriptive)"
        L.append(f"| {k}{tag} | {fmt(a)} | {fmt(b)} | {np.mean(a) - np.mean(b):+.3f} | {welch(a, b):.3f} | "
                 f"{perm_p(a, b):.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    both = [s for s in use if s in su]
    if len(both) >= 2:
        a = [sm[s]["metrics"]["ACC"] for s in both]
        b = [su[s]["metrics"]["ACC"] for s in both]
        L.append(f"\nDescriptive: matched vs unmatched sleep, same seeds {both}: {np.mean(a):.3f} vs {np.mean(b):.3f} "
                 f"(paired diff {np.mean(np.subtract(a, b)):+.3f}, paired t p={stats.ttest_rel(a, b).pvalue:.3f}).")
    return L


def block2():
    L = ["\n## Block 2: DoorKey-6x6, DFA on the SNN encoder with vs without homeostasis\n"]
    h, d = load_rq1("dfae_homeo"), load_rq1("dfae")
    if len(h) < 2 or len(d) < 2:
        return L + ["(not enough runs yet)\n"]
    sh, sd = sum(v["solved"] for v in h.values()), sum(v["solved"] for v in d.values())
    pf = stats.fisher_exact([[sh, len(h) - sh], [sd, len(d) - sd]]).pvalue
    ah, ad = [v["auc"] for v in h.values()], [v["auc"] for v in d.values()]
    pw = welch(ah, ad)
    adj = holm([pf, pw])
    L.append("| arm | n | solved | AUC | final eval | frames to 0.9 (solvers) |")
    L.append("|---|---|---|---|---|---|")
    for name, r in (("DFA + homeostasis", h), ("DFA", d)):
        f09 = [v["f09"] for v in r.values() if not np.isnan(v["f09"])]
        L.append(f"| {name} | {len(r)} | {sum(v['solved'] for v in r.values())}/{len(r)} | "
                 f"{fmt([v['auc'] for v in r.values()])} | {fmt([v['eval'] for v in r.values()])} | "
                 f"{f'{np.mean(f09) / 1e3:.0f}k' if f09 else '-'} |")
    L.append("\n| planned test | p | Holm (2 tests) |")
    L.append("|---|---|---|")
    L.append(f"| solve rate, Fisher exact | {pf:.4f} | {adj[0]:.4f} |")
    L.append(f"| AUC, Welch (diff {np.mean(ah) - np.mean(ad):+.3f}) | {pw:.2e} | {adj[1]:.2e} |")
    L.append("\nper-seed eval: DFA+homeo " + ", ".join(f"s{k}={v['eval']:.2f}" for k, v in sorted(h.items()))
             + "; DFA " + ", ".join(f"s{k}={v['eval']:.2f}" for k, v in sorted(d.items())))
    return L


def block3():
    L = ["\n## Block 3: memory benchmark\n"]
    probe = sorted(glob.glob("runs/s4_mem_probe/*/eval.json"))
    if probe:
        L.append("Probe (sizing only, 1 seed, lr 3e-4, 2M frames):\n")
        L.append("| run | eval success | eval return | mean episode length |")
        L.append("|---|---|---|---|")
        for p in probe:
            e = json.load(open(p))
            L.append(f"| {os.path.basename(os.path.dirname(p))} | {e['success']:.3f} | {e['return_mean']:.3f} | "
                     f"{e['len_mean']:.1f} |")
    arms = {}
    for p in glob.glob(os.path.join(MEM, "*", "eval.json")):
        name = os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0]
        arms.setdefault(name, []).append(json.load(open(p))["success"])
    if not arms:
        return L
    L.append("\nMain runs (eval success, 200 episodes, sampled policy):\n")
    L.append("| arm | n | success | per seed |")
    L.append("|---|---|---|---|")
    for k in sorted(arms):
        L.append(f"| {k} | {len(arms[k])} | {fmt(arms[k])} | {', '.join(f'{x:.3f}' for x in arms[k])} |")
    tf = next((k for k in arms if k.endswith("snn_tf12")), None)
    fs = next((k for k in arms if k.endswith("snn_fs12")), None)
    if tf and fs and len(arms[tf]) >= 2 and len(arms[fs]) >= 2:
        a, b = arms[tf], arms[fs]
        L.append(f"\n**Planned test** ({tf} vs {fs}): diff {np.mean(a) - np.mean(b):+.3f}, Welch p = {welch(a, b):.4f}"
                 f" (exact permutation p = {perm_p(a, b):.2f}; minimum attainable with 3 vs 3 is 0.10).")
    return L


def main():
    L = ["# Session 4 pre-registered tests\n", "Planned tests are labelled; everything else is descriptive. "
         "Welch = two-sided Welch t-test; perm = exact two-sided permutation test; CI = bootstrap 95% CI.\n"]
    L += block1() + block2() + block3()
    os.makedirs("results", exist_ok=True)
    with open("results/s4_stats.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
