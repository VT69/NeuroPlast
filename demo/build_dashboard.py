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
    for fn in (cn.protocol, cn.rq3, cn.rq1, cn.dfa, cn.rq2, cn.rq5, cn.memory, cn.rq4, cn.demo, cn.seeds):
        fn()
    return {k: v[1].replace("$\\pm$", "±").replace("--", "–") for k, v in cn.N.items()}


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
    for m in ("naive", "sleep"):
        r = json.load(open(f"runs/demo_ckpt/fetch3_{m}_s1/results.json"))
        seeds = cn.cl("runs/continual", m)
        accs = [x["metrics"]["ACC"] for x in seeds.values()]
        out[m] = dict(R=[[round(v, 3) for v in row] for row in r["R"]], acc=round(r["metrics"]["ACC"], 3),
                      seedAcc=[round(a, 3) for a in accs], seedMean=round(float(np.mean(accs)), 3),
                      gifs=[f"data:image/gif;base64,{base64.b64encode(open(f'demo/assets/fetch3_{m}_task{k}.gif', 'rb').read()).decode()}"
                            for k in range(3)])
    idx = open("demo/assets/index.md").read()
    out["episodes"] = [line.split("|")[4].strip() for line in idx.splitlines() if line.startswith("| [fetch3_")]
    return out


def scorecard(N):
    return [
        dict(mech="Spiking encoder", q="Better accuracy-vs-operations than a sparsified CNN?", icon="◐",
             verdict="Lower-ops regime, not more accuracy",
             num=f"works at {N['rqThreeSnnTwoLowOps']} ops/frame (acc {N['rqThreeSnnTwoLowAcc']}) where the CNN stops at "
                 f"{N['rqThreeCnnFloorOps']}; {N['snnCpuSlowdown']}× CPU cost per training step",
             ev=f"{N['rqThreeSeeds']} seeds per point; energy only a proxy"),
        dict(mech="STDP (reward-modulated)", q="Helps beyond homeostasis and a same-size random update?", icon="○",
             verdict="No effect", num=f"vs random update: ΔAUC {N['rqOneContrBAucCi']}, p = {N['rqOneContrBAucP']}",
             ev=f"pre-registered, {N['rqOneBpN']} seeds per arm, Holm-corrected"),
        dict(mech="Homeostasis", q="Helps a weight-transport-free learner (DFA)?", icon="◐",
             verdict="Faster learning, not reliable learning",
             num=f"ΔAUC {N['dfaAucCi']} (Holm p = {N['dfaAucHolm']}); solved {N['rqOneDfaHomeoSolved']} vs "
                 f"{N['rqOneDfaSolved']} (Holm p = {N['dfaFisherHolm']})",
             ev=f"pre-registered, {N['rqOneDfaN']} seeds per arm"),
        dict(mech="Homeostasis with backprop", q="Helps ordinary backprop?", icon="○", verdict="Not significant",
             num=f"ΔAUC {N['rqOneContrCAucCi']}, raw p = {N['rqOneContrCAucP']}, Holm p = {N['rqOneContrCAucHolm']}",
             ev=f"pre-registered, {N['rqOneBpN']} seeds per arm"),
        dict(mech="Sleep-like consolidation", q="Less forgetting than replay at an equal replay budget?", icon="○",
             verdict="Equal to replay",
             num=f"ΔACC {N['matchAccCi']}, p = {N['matchAccP']}; FORGET {N['matchForgetSleep']} vs {N['matchForgetReplay']}",
             ev=f"pre-registered, replayed samples matched exactly, {N['matchN']} seeds per arm"),
        dict(mech="Shared weights vs isolation", q="More performance per unit of total memory?", icon="◐",
             verdict="Only with small replay buffers",
             num=f"at 5,000 states/task isolation is {N['perMbRatioRange']}× more ACC per MB; SNN sleep at 200/task "
                 f"ties isolation ({N['smallSnnTwoHundredAcc']} vs {N['snnIsoAcc']}) in {N['smallSnnTwoHundredMb']} vs "
                 f"{N['memSnnThreeIsoMb']} MB",
             ev="3 seeds per arm; per parameter shared weights always win"),
        dict(mech="Transformer working memory", q="Helps on a task that needs memory?", icon="?", verdict="Untested",
             num=f"no usable benchmark: on MemoryS11/S13 a memoryless CNN ({N['probeElevenOne']}, {N['probeThirteenOne']}) "
                 f"and a {N['memWindow']}-frame stack ({N['probeElevenTwelve']}, {N['probeThirteenTwelve']}) are both at chance",
             ev="pre-registered rule skipped the comparison"),
    ]


def main():
    N = numbers()
    data = dict(forgetting=forgetting(), memory=memory(), demo=demo_agents(), scorecard=scorecard(N),
                seedRange=N["seedRange"], generated="2026-09-30")
    html = open("demo/dashboard_template.html").read().replace("/*__DATA__*/null", json.dumps(data))
    with open("demo/index.html", "w") as f:
        f.write(html)
    print(f"wrote demo/index.html ({os.path.getsize('demo/index.html') / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
