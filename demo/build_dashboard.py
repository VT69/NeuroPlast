"""Build demo/index.html: a single self-contained page from the frozen results (no network, no external libraries).

    python demo/build_dashboard.py

Everything on the page is computed here from runs/ and results/, plus the labelled post-freeze follow-up in
explore/runs/confirm/ (via the paper's own numbers pipeline, so the scorecard matches the paper exactly), and embedded
into the HTML: the data as JSON, the GIFs as base64.
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
    for fn in (cn.protocol, cn.rq3, cn.rq1, cn.dfa, cn.rq2, cn.rq5, cn.memory, cn.rq4, cn.demo, cn.seeds, cn.compute_budget, cn.postfreeze):
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
    """Plain-language version of the paper's conclusion. kind: survived | conditional | notsurvived | unresolved.
    lab: evidential status of the key number (confirmatory = prospectively specified in a version-controlled
    experiment log; exploratory; descriptive)."""
    return [
        dict(mech="Spiking encoder", kind="conditional", verdict="Fewer operations in an estimate; higher measured CPU cost",
             q="Does a spiking encoder reach the same accuracy with fewer operations than a sparsified ordinary network?",
             num=f"Works at {N['rqThreeSnnTwoLowOps']} operations per frame (accuracy {N['rqThreeSnnTwoLowAcc']}); the sparsified "
                 f"CNN stops working below {N['rqThreeCnnFloorOps']}. Measured cost in our CPU implementation: "
                 f"{N['msSnn']} vs {N['msCnn']} ms per training step ({N['snnCpuSlowdown']}×).",
             ev=f"descriptive; {N['rqThreeSeeds']} seeds per point; energy is an estimated arithmetic energy proxy, not measured "
                "on hardware"),
        dict(mech="Local learning rule (STDP)", kind="notsurvived", verdict="Not distinguishable from a same-size random update",
             q=f"Does the tested rule (stabilised reward-modulated STDP, α = {N['stdpAlpha']}, added to backprop) help on "
               "DoorKey-6x6, compared with a random update of the same size?",
             num=f"Difference in learning speed vs the random update: ΔAUC {N['rqOneContrBAucCi']}, p = {N['rqOneContrBAucP']}. "
                 "The interval is too wide to exclude a large effect either way.",
             ev=f"confirmatory, {N['rqOneBpN']} seeds per arm (same seeds), Holm-corrected"),
        dict(mech="Homeostasis, backprop-free encoder", kind="survived", verdict="Faster learning, not more reliable",
             q="Does homeostasis (keeping firing rates near a target) help an SNN whose encoder credit assignment is "
               "backprop-free (DFA; the heads use exact gradients)?",
             num=f"Learning speed ΔAUC {N['dfaAucCi']} (Holm p = {N['dfaAucHolm']}). Solved {N['rqOneDfaHomeoSolved']} vs "
                 f"{N['rqOneDfaSolved']} runs, not significant (Holm p = {N['dfaFisherHolm']}).",
             ev=f"confirmatory, {N['rqOneDfaN']} seeds per arm (same seeds)"),
        dict(mech="Homeostasis, with backprop", kind="notsurvived", verdict="Not significant after correction",
             q="Does homeostasis help SNNs trained with ordinary backprop?",
             num=f"ΔAUC {N['rqOneContrCAucCi']}; raw p = {N['rqOneContrCAucP']}, Holm-corrected p = {N['rqOneContrCAucHolm']}.",
             ev=f"confirmatory, {N['rqOneBpN']} seeds per arm (same seeds)"),
        dict(mech="Sleep-like consolidation", kind="notsurvived", verdict="No detectable difference from replay at a matched budget",
             q="Does offline, sleep-like consolidation beat replay when both replay the same number of stored states?",
             num=f"Sleep minus replay, final accuracy: {N['matchAccCi']}, p = {N['matchAccP']} (a narrow interval). "
                 f"Forgetting {N['matchForgetSleep']} vs {N['matchForgetReplay']} (descriptive).",
             ev=f"confirmatory (accuracy; forgetting descriptive): replayed samples matched exactly, {N['matchN']} seeds per arm "
                "(same seeds); "
                "both beat naive fine-tuning"),
        dict(mech="Shared trunk with per-task heads", kind="conditional", verdict="Only with small buffers and full-width isolation",
             q="Is one shared trunk with a head per task more memory-efficient than a separate network per task (isolation)?",
             num=f"SNN: sleep with 200 states per task reaches a similar mean accuracy to full-width isolation "
                 f"({N['smallSnnTwoHundredMean']} vs {N['snnIsoMean']}; difference {N['snnSleepTwoHundredVsIsoCi']}) in "
                 f"{N['smallSnnTwoHundredMb']} vs {N['memSnnThreeIsoMb']} MB. Post-freeze (CNN, 5 tasks): quarter-width isolation "
                 f"{N['postNarrowMean']} at {N['postNarrowMb']} MB beat the shared trunk with LwF ({N['postLwfMean']}, "
                 f"{N['postLwfMb']} MB) and with replay ({N['postReplayMean']}, {N['postReplayMb']} MB).",
             ev=f"confirmatory: SNN {N['smallSnnTwoHundredN']} seeds per arm, wide CI; post-freeze {N['postN']} seeds per arm. "
                "The shared trunk was not shrunk the same way; the SNN case is untested"),
        dict(mech="Transformer working memory", kind="unresolved", verdict="Inconclusive: benchmark validity not established",
             q="Does Transformer working memory help on a task that needs memory?",
             num=f"On MemoryS11/S13 a memoryless CNN ({N['probeElevenOne']}, {N['probeThirteenOne']}) and a "
                 f"{N['memWindow']}-frame stack ({N['probeElevenTwelve']}, {N['probeThirteenTwelve']}) both stay at chance, "
                 f"so the comparison could not be run.",
             ev="confirmatory (decision rule): the prospectively specified rule skipped the comparison"),
    ]


def postfreeze(N):
    """The post-freeze follow-up C4 (explore/EXPLORE.md), for the memory section."""
    arms = [("isolation, quarter width", "Narrow", "isolation"), ("shared trunk + LwF (int8 teacher)", "Lwf", "lwf"),
            (f"shared trunk + replay ({N['postReplayBuf']}/task)", "Replay", "replay"), ("isolation, full width", "Iso", "isolation")]
    return dict(n=N["postN"], seeds=N["postSeeds"], arms=[dict(label=l, method=m, acc=N[f"post{k}Acc"], mb=N[f"post{k}Mb"])
                                                           for l, k, m in arms],
                lwfCi=N["postHEightCi"], lwfHolm=N["postHEightHolm"], replayCi=N["postHNineCi"], replayHolm=N["postHNineHolm"],
                isoCi=N["postHTenCi"])


def whats_next(N):
    """The paper's future-work items (plans and open questions, not results); no timeline here."""
    return dict(
        why="The main limitation is scale: every experiment ran on a 4-core CPU, and the bottleneck was CPU-bound "
            "environment stepping, not model size.",
        plan="Port the agent, the mechanisms and the baselines to JAX and run them on GPUs with XLand-MiniGrid, a GPU-native "
             "MiniGrid (reported to simulate millions of environment steps per second on one GPU) with a rule-and-goal system "
             "for generating many related tasks.",
        validate="Replicate the headline results (sleep vs replay at a matched replay budget, STDP vs a "
                 "same-size random update, the isolation-width result) on XLand-MiniGrid's MiniGrid ports before trusting "
                 "anything new.",
        questions=["Longer sequences of 10–50 related tasks where transfer matters, with full accuracy-vs-memory curves in "
                   "which every method shrinks along its own size setting (isolation width, shared-trunk width, buffer size)",
                   "10–20 seeds per arm, with power analysis and equivalence tests, so that “no large effect” nulls become "
                   "informative",
                   "A memory benchmark agents can actually learn; then Transformer vs GRU vs frame stack, and a stateful SNN",
                   f"The SNN version of the isolation-width comparison at an adequate budget ({N['framesPerTaskFiveFair']} frames "
                   "per task)",
                   "Whether the conclusions change with network size",
                   "A task-agnostic setting with observable goals"],
        smaller=["replay-budget sweep for sleep vs interleaved replay (motivated by an exploratory result only)",
                 "sleep split into offline replay alone vs replay plus self-distillation",
                 "plots of the STDP runaway mechanism", "a stronger CNN sparsifier (k-winners-take-all)",
                 "compressed or generative replay to shrink the buffer", "a PackNet variant that can regrow capacity",
                 "one standard benchmark, such as Continual World"],
        method="Same methodology throughout: tests prospectively specified in a version-controlled experiment log, fresh "
               "seeds for confirmation, matched resources, every failure logged.")


def main():
    N = numbers()
    mem = memory()
    arch = dict(T=N["snnT"], window=N["hybridWindow"], buffer=N["bufferPerTask"], period=N["sleepPeriod"],
                steps=N["sleepSteps"])
    data = dict(forgetting=forgetting(), memory=mem, demo=demo_agents(), scorecard=scorecard(N), arch=arch,
                keyMem=key_memory(N, mem), post=postfreeze(N), next=whats_next(N), seedRange=N["seedRange"],
                generated="2026-09-30")
    html = open("demo/dashboard_template.html").read().replace("/*__DATA__*/null", json.dumps(data))
    with open("demo/index.html", "w") as f:
        f.write(html)
    print(f"wrote demo/index.html ({os.path.getsize('demo/index.html') / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
