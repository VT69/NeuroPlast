"""Build demo/index.html: a single self-contained page from the frozen results (no network, no external libraries).

    python demo/build_dashboard.py

Everything on the page is computed here from runs/ and results/ (via the paper's own numbers pipeline, so the
scorecard matches the paper exactly) and embedded into the HTML: the data as JSON, the GIFs as base64.
"""
from __future__ import annotations

import base64
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper"))
import compute_numbers as cn  # noqa: E402  (the paper's numbers pipeline)
from scripts.memory_fair import load as load_mem  # noqa: E402


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def numbers():
    for fn in (cn.protocol, cn.rq3, cn.rq1, cn.dfa, cn.rq2, cn.rq5, cn.memory, cn.rq4, cn.demo, cn.seeds, cn.compute_budget):
        fn()
    # typographic minus for negative numbers ("-0.167" -> "−0.167"); ranges already use an en dash
    return {k: re.sub(r"(?<![\w.])-(?=\d)", "−", v[1].replace("$\\pm$", "±").replace("--", "–")) for k, v in cn.N.items()}


SEQUENCES = [("cnn3", "CNN · 3 tasks", "runs/continual", "fetch3", "CNN"),
             ("snn3", "SNN · 3 tasks", "runs/continual_snn", "fetch3", "SNN"),
             ("hyb3", "SNN+Transformer · 3 tasks", "runs/continual_hybrid", "fetch3", "Hybrid"),
             ("cnn5", "CNN · 5 tasks", "runs/continual5_cnn", "fetch5", "CNN")]
METHODS = [("naive", "naive"), ("ewc_lam100", "EWC (λ=100)"), ("replay", "replay"), ("sleep", "sleep"),
           ("sleep_matched", "sleep, replay-matched"), ("isolation", "isolation")]


def forgetting():
    out = {}
    for key, label, root, suite, _ in SEQUENCES:
        ser = []
        for m, mlab in METHODS:
            runs = cn.cl(root, m, suite)
            if len(runs) < 2:
                continue
            R = np.array([r["R"] for r in runs.values()])
            T = R.shape[1]
            first = R[:, :, 0]
            seen = np.array([[R[s, i, :i + 1].mean() for i in range(T)] for s in range(len(R))])
            pts = lambda y: [dict(x=i + 1, mean=round(float(y[:, i].mean()), 4), ci=round(ci95(y[:, i]), 4))  # noqa: E731
                             for i in range(T)]
            ser.append(dict(key=m, label=mlab, n=len(runs), first=pts(first), seen=pts(seen)))
        out[key] = dict(label=label, T=int(R.shape[1]), series=ser,
                        source=f"{root}/{suite}_<method>_s*/results.json")
    return out


def memory():
    g = load_mem()
    pts = []
    for (suite, trunk, name), runs in g.items():
        m = runs[0]["method"]
        base = name.split(" @")[0]
        if m not in ("isolation", "sleep", "replay"):
            continue
        if base != f"{suite}_{m}" and not base.startswith(f"{suite}_{m}_buf"):
            continue
        if suite == "fetch5" and trunk == "SNN" and "@150k" in name:
            continue
        seq = {("fetch3", "CNN"): "cnn3", ("fetch3", "SNN"): "snn3", ("fetch3", "Hybrid"): "hyb3",
               ("fetch5", "CNN"): "cnn5", ("fetch5", "SNN"): "snn5"}[(suite, trunk)]
        acc = [r["acc"] for r in runs]
        pts.append(dict(seq=seq, trunk=trunk, suite=suite, method=m, buf=runs[0]["buf_per_task"],
                        frames=runs[0]["frames"], mb=round(runs[0]["total_mb"], 3), acc=round(float(np.mean(acc)), 4),
                        sd=round(float(np.std(acc, ddof=1)), 4) if len(acc) > 1 else 0.0, n=len(acc)))
    return sorted(pts, key=lambda p: (p["seq"], p["mb"]))


def demo_agents():
    out = {}
    rows = [line.split("|") for line in open("demo/assets/index.md") if line.startswith("| [fetch3_")]
    eps = {}
    for r in rows:  # "| [gif](gif) | method | task | ✓ 0.99 (4 steps), ✗ 0.00 (256 steps), ... | score |"
        m, task = r[2].strip(), int(r[3].strip().split(":")[0])
        eps.setdefault(m, {})[task] = [dict(ok=e.strip().startswith("✓"), ret=float(e.split()[1]), steps=int(e.split("(")[1].split()[0]))
                                       for e in r[4].split(",")]
    for m in ("naive", "sleep"):
        r = json.load(open(f"runs/demo_ckpt/fetch3_{m}_s1/results.json"))
        seeds = cn.cl("runs/continual", m)
        accs = [x["metrics"]["ACC"] for x in seeds.values()]
        out[m] = dict(R=[[round(v, 3) for v in row] for row in r["R"]], acc=round(r["metrics"]["ACC"], 3),
                      seedAcc=[round(a, 3) for a in accs], seedMean=round(float(np.mean(accs)), 3),
                      episodes=[eps[m][k] for k in range(3)],
                      gifs=[f"data:image/gif;base64,{base64.b64encode(open(f'demo/assets/fetch3_{m}_task{k}.gif', 'rb').read()).decode()}"
                            for k in range(3)])
    return out


def key_memory(N, mem):
    """The annotated result: SNN fetch3, sleep with the smallest buffer vs isolation (paper Section 5, RQ5)."""
    a = next(p for p in mem if p["seq"] == "snn3" and p["method"] == "sleep" and p["buf"] == 200)
    b = next(p for p in mem if p["seq"] == "snn3" and p["method"] == "isolation")
    return dict(seq="snn3", buf=200, bigBuf=N["bufferPerTask"], sleepAcc=f"{a['acc']:.3f}", isoAcc=f"{b['acc']:.3f}",
                sleepMb=f"{a['mb']:.2f}", isoMb=f"{b['mb']:.2f}", ratioPct=round(100 * a["mb"] / b["mb"]),
                diffCi=N["snnSleepTwoHundredVsIsoCi"], p=N["snnSleepTwoHundredVsIsoP"], n=min(a["n"], b["n"]))


def scorecard(N):
    """Plain-language version of the paper's conclusions. kind: helps | tradeoff | none | untested."""
    return [
        dict(mech="Spiking encoder", kind="tradeoff", verdict="Fewer operations, not more accuracy",
             q="Does a spiking encoder reach the same accuracy with fewer operations than a pruned ordinary network?",
             num=f"Works at {N['rqThreeSnnTwoLowOps']} operations per frame (accuracy {N['rqThreeSnnTwoLowAcc']}); the pruned "
                 f"CNN stops working below {N['rqThreeCnnFloorOps']}. Costs {N['snnCpuSlowdown']}× more CPU time per training step.",
             ev=f"{N['rqThreeSeeds']} seeds per point; energy is an estimate, not measured on hardware"),
        dict(mech="Local learning rule (STDP)", kind="none", verdict="No effect",
             q="Does a brain-like local learning rule (reward-modulated STDP) help, compared with a random update of the same size?",
             num=f"Difference in learning speed vs the random update: ΔAUC {N['rqOneContrBAucCi']}, p = {N['rqOneContrBAucP']}.",
             ev=f"pre-registered, {N['rqOneBpN']} seeds per arm, Holm-corrected"),
        dict(mech="Homeostasis, without backprop", kind="helps", verdict="Faster learning, not more reliable",
             q="Does homeostasis (keeping firing rates near a target) help SNNs trained without backprop (DFA)?",
             num=f"Learning speed ΔAUC {N['dfaAucCi']} (Holm p = {N['dfaAucHolm']}). Solved {N['rqOneDfaHomeoSolved']} vs "
                 f"{N['rqOneDfaSolved']} runs, not significant (Holm p = {N['dfaFisherHolm']}).",
             ev=f"pre-registered, {N['rqOneDfaN']} seeds per arm"),
        dict(mech="Homeostasis, with backprop", kind="none", verdict="Not significant after correction",
             q="Does homeostasis help SNNs trained with ordinary backprop?",
             num=f"ΔAUC {N['rqOneContrCAucCi']}; raw p = {N['rqOneContrCAucP']}, Holm-corrected p = {N['rqOneContrCAucHolm']}.",
             ev=f"pre-registered, {N['rqOneBpN']} seeds per arm"),
        dict(mech="Sleep-like consolidation", kind="none", verdict="Ties replay",
             q="Does offline, sleep-like consolidation reduce forgetting more than replay, with the same number of replayed memories?",
             num=f"Sleep minus replay, final accuracy: {N['matchAccCi']}, p = {N['matchAccP']}. Forgetting "
                 f"{N['matchForgetSleep']} vs {N['matchForgetReplay']}.",
             ev=f"pre-registered, replayed samples matched exactly, {N['matchN']} seeds per arm; both beat naive fine-tuning"),
        dict(mech="One shared network", kind="helps", verdict="Only with small replay buffers",
             q="Is one shared network more memory-efficient than a separate network per task (isolation)?",
             num=f"SNN: sleep with 200 states per task ties isolation ({N['smallSnnTwoHundredAcc']} vs {N['snnIsoAcc']}) in "
                 f"{N['smallSnnTwoHundredMb']} vs {N['memSnnThreeIsoMb']} MB. At 5,000 states per task isolation gets "
                 f"{N['perMbRatioRange']}× more accuracy per MB.",
             ev="3 seeds per arm; per parameter the shared network always wins"),
        dict(mech="Transformer working memory", kind="untested", verdict="Untested: no usable benchmark",
             q="Does Transformer working memory help on a task that needs memory?",
             num=f"On MemoryS11/S13 a memoryless CNN ({N['probeElevenOne']}, {N['probeThirteenOne']}) and a "
                 f"{N['memWindow']}-frame stack ({N['probeElevenTwelve']}, {N['probeThirteenTwelve']}) both stay at chance, "
                 f"so the comparison could not be run.",
             ev="pre-registered rule skipped the comparison"),
    ]


def main():
    N = numbers()
    mem = memory()
    arch = dict(T=N["snnT"], window=N["hybridWindow"], buffer=N["bufferPerTask"], period=N["sleepPeriod"],
                steps=N["sleepSteps"])
    data = dict(forgetting=forgetting(), memory=mem, demo=demo_agents(), scorecard=scorecard(N), arch=arch,
                keyMem=key_memory(N, mem), seedRange=N["seedRange"], generated="2026-09-30")
    html = open("demo/dashboard_template.html").read().replace("/*__DATA__*/null", json.dumps(data))
    with open("demo/index.html", "w") as f:
        f.write(html)
    print(f"wrote demo/index.html ({os.path.getsize('demo/index.html') / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
