"""Every number quoted in the paper, computed from the frozen result files (runs/ and results/).

    python paper/compute_numbers.py        # writes paper/numbers.tex and paper/NUMBERS.md

Each number becomes a LaTeX macro (letters only) used in main.tex, and a row in NUMBERS.md with its description
and source file(s). Nothing in main.tex should be a hand-typed result: if a number is not a macro, it is a protocol
constant (also listed here) or a literature value (labelled as such in the text).

Statistics: difference of means with a 95% Welch (Welch-Satterthwaite) confidence interval and two-sided Welch p;
solve rates with Fisher's exact test and a Newcombe (Wilson score) 95% CI for the difference of proportions;
Holm correction where a pre-registered family has several tests. n is small everywhere (3-10 seeds), so the CIs
are wide and are the main thing to read.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import yaml
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from neuroplast.eval.metrics import auc  # noqa: E402
from scripts.memory_fair import load as load_mem  # noqa: E402
from scripts.rq1_stats import holm, load as load_rq1  # noqa: E402

N = {}  # macro -> (latex value, plain value, description, source)


def put(name, value, desc, src, plain=None):
    assert name.isalpha(), name
    assert name not in N, f"duplicate macro {name}"
    v = str(value)
    N[name] = (v.replace("-", "\\ensuremath{-}") if v.startswith("-") or " -" in v or "[-" in v else v,
               plain if plain is not None else v, desc, src)


# ---------------------------------------------------------------- formatting / statistics helpers
def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def sgn(x, d=3):
    return f"{x:+.{d}f}"


def pf(p):
    if p < 0.001:
        return "<0.001" if p >= 1e-9 else "<10^{-9}"
    return f"{p:.3f}"


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, d, 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se, stats.ttest_ind(a, b, equal_var=False).pvalue


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue


def ci_str(d, lo, hi, dd=3):
    return f"{sgn(d, dd)} [{sgn(lo, dd)}, {sgn(hi, dd)}]"


def msd(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def cl(root, name, suite="fetch3"):
    """{seed: results.json} for runs/<root>/<suite>_<name>_s<seed> with final metrics."""
    out = {}
    for p in glob.glob(os.path.join(root, f"{suite}_{name}_s*", "results.json")):
        d = os.path.basename(os.path.dirname(p))
        if d.rsplit("_s", 1)[0] != f"{suite}_{name}":
            continue
        r = json.load(open(p))
        if r.get("metrics"):
            out[r["seed"]] = r
    return dict(sorted(out.items()))


def fwt(run, ref):
    return float(np.mean([(auc(run["curves"][j]) - auc(ref["curves"][j])) / (1 - auc(ref["curves"][j]))
                          for j in range(1, len(run["curves"]))]))


W = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
     10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen"}


# ---------------------------------------------------------------- protocol constants (runs/*/config or results.json cfg)
def protocol():
    c = json.load(open("runs/continual/fetch3_sleep_s1/results.json"))
    h = json.load(open("runs/continual_hybrid/fetch3_sleep_s1/results.json"))
    f5 = json.load(open("runs/continual5_snn_450k/fetch5_sleep_s1/results.json"))
    dk = yaml.safe_load(open("runs/rq1_v2/rq1v2_bp_s1/config.yaml"))
    src_c, src_h = "runs/continual/fetch3_sleep_s1/results.json (cfg)", "runs/continual_hybrid/fetch3_sleep_s1/results.json (cfg)"
    put("framesPerTaskFlat", f"{c['cfg']['total_frames'] // 1000}k", "frames per task, CNN/SNN fetch3 and fetch5 protocol", src_c)
    put("framesPerTaskHybrid", f"{h['cfg']['total_frames'] // 1000}k", "frames per task, full hybrid protocol", src_h)
    put("framesPerTaskFiveFair", f"{f5['cfg']['total_frames'] // 1000}k", "frames per task, fetch5 SNN fair-budget runs",
        "runs/continual5_snn_450k/fetch5_sleep_s1/results.json (cfg)")
    put("framesDoorKey", f"{dk['total_frames'] // 1000}k", "training frames, DoorKey-6x6 (RQ1)", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("lrFlat", f"{c['cfg']['lr']:g}", "Adam learning rate, CNN/SNN trunks", src_c)
    put("lrHybrid", f"{h['cfg']['lr']:g}", "Adam learning rate, full hybrid", src_h)
    put("numEnvs", str(c["cfg"]["num_envs"]), "parallel environments", src_c)
    put("numSteps", str(c["cfg"]["num_steps"]), "rollout length per env per PPO update", src_c)
    put("updateEpochs", str(c["cfg"]["update_epochs"]), "PPO epochs per update", src_c)
    put("minibatch", str(dk["minibatch_size"]), "PPO minibatch size", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("targetKl", f"{c['cfg']['target_kl']:g}", "PPO early-stop KL threshold", src_c)
    put("clipCoef", f"{dk['clip_coef']:g}", "PPO clip coefficient", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("gammaVal", f"{dk['gamma']:g}", "discount", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("gaeLambda", f"{dk['gae_lambda']:g}", "GAE lambda", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("snnT", str(dk["enc_kwargs"]["T"]), "SNN timesteps per environment step", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("featDim", str(dk["feat_dim"]), "encoder feature dimension", "runs/rq1_v2/rq1v2_bp_s1/config.yaml")
    put("hybridWindow", str(h["cfg"]["window"]), "Transformer context window (frames), hybrid", src_h)
    put("paramsCnnShared", f"{c['params']:,}", "parameters, shared CNN agent (3 task heads)", src_c)
    iso = json.load(open("runs/continual/fetch3_isolation_s1/results.json"))
    put("paramsCnnPerTask", f"{iso['params_per_task']:,}", "parameters, one isolated CNN network", "runs/continual/fetch3_isolation_s1/results.json")
    put("paramsHybridShared", f"{h['params']:,}", "parameters, shared full-hybrid agent", src_h)
    put("bufferPerTask", f"{c['method_kwargs'].get('buffer_per_task', 5000):,}", "replay states stored per task (default)",
        "runs/continual/fetch3_sleep_s1/results.json (method_kwargs; default of Sleep/Replay)")
    put("sleepPeriod", "25", "PPO updates between sleep phases (Sleep default)", "neuroplast/sleep/sleep.py (protocol constant)")
    put("sleepSteps", "700", "gradient steps per sleep phase (Sleep default)", "neuroplast/sleep/sleep.py (protocol constant)")
    put("evalEpisodesCl", "100", "evaluation episodes per task (continual)", "continual.py --eval-episodes default (protocol constant)")
    put("evalEpisodesSingle", "200", "evaluation episodes (single-task runs)", "train.py --eval-episodes default (protocol constant)")
    ho = yaml.safe_load(open("runs/rq1_v2/rq1v2_homeo_s1/config.yaml"))["stdp"]
    tf = yaml.safe_load(open("runs/rq1_v2/rq1v2_tfs0.03_s1/config.yaml"))["stdp"]
    put("homeoTarget", f"{ho['homeo']:g}", "homeostasis target firing rate", "runs/rq1_v2/rq1v2_homeo_s1/config.yaml")
    put("stdpAlpha", f"{tf['alpha']:g}", "local-term weight alpha, stabilised STDP and random control", "runs/rq1_v2/rq1v2_tfs0.03_s1/config.yaml")
    put("stdpAlphaLarge", "0.3", "larger alpha tested (context arm tfs0.3)", "runs/rq1_v2/rq1v2_tfs0.3_s1/config.yaml")
    put("homeoLr", "0.01", "homeostasis bias learning rate (HybridSTDP default)", "neuroplast/learning/hybrid.py (protocol constant)")
    put("convChannels", "16, 32 and 64", "conv channels (2x2 kernels) of both encoders", "neuroplast/models/encoders.py (protocol constant)")
    put("lifBeta", "0.9", "initial LIF leak (learnable)", "neuroplast/models/encoders.py (protocol constant)")
    put("surrogateSlope", "25", "surrogate-gradient slope", "neuroplast/models/encoders.py (protocol constant)")
    put("solveThreshold", "0.5", "solved = final eval return above this", "scripts/rq1_stats.py (protocol constant)")
    pr = yaml.safe_load(open("runs/s4_mem_probe/probe_S11_cnn_fs12_s1/config.yaml"))
    put("memWindow", str(pr["frame_stack"]), "frame-stack length / context window in the memory probes", "runs/s4_mem_probe/probe_S11_cnn_fs12_s1/config.yaml")
    pi = yaml.safe_load(open("runs/s3_mem_pilot/mem_cnn_fs1_s1/config.yaml"))
    put("sSevenFrames", f"{pi['total_frames'] / 1e6:.0f}M", "MemoryS7 pilot training frames", "runs/s3_mem_pilot/mem_cnn_fs1_s1/config.yaml")
    put("pjMac", "4.6", "energy per 32-bit MAC used in the proxy (literature value, 45 nm)", "results/rq3_doorkey6.md (header)")
    put("pjAc", "0.9", "energy per accumulate used in the proxy (literature value, 45 nm)", "results/rq3_doorkey6.md (header)")
    # buffer bytes per state (measured, results/rq5_memory.md header)
    put("bytesPerState", "188", "replay-buffer bytes per stored state, CNN/SNN trunks", "results/rq5_memory.md (header); scripts/memory_fair.py")
    put("bytesPerStateHybrid", "632", "replay-buffer bytes per stored state, hybrid (4-frame window)", "results/rq5_memory.md (header); scripts/memory_fair.py")


# ---------------------------------------------------------------- RQ3: operation-count frontier (SNN encoder)
def rq3():
    src = "runs/rq3_doorkey6/results.jsonl"
    g = defaultdict(list)
    for line in open(src):
        r = json.loads(line)
        g[(r["kind"], r["T"], r["lam"])].append(r)

    def agg(kind, T, lam, k):
        return float(np.mean([r[k] for r in g[(kind, T, lam)]]))

    ops = lambda kind, T, lam: agg(kind, T, lam, "sparse_ops") / 1e3  # noqa: E731
    acc = lambda kind, T, lam: agg(kind, T, lam, "acc")  # noqa: E731
    en = lambda kind, T, lam: agg(kind, T, lam, "energy_sparse_pj") / 1e3  # noqa: E731
    put("rqThreeSeeds", str(len(g[("cnn", 1, 0.0)])), "seeds per RQ3 point", src)
    put("rqThreeCnnDenseOps", f"{ops('cnn', 1, 0.0):.0f}k", "CNN event-driven ops/frame, no sparsity penalty", src)
    put("rqThreeCnnDenseAcc", f3(acc("cnn", 1, 0.0)), "CNN BC test accuracy, no penalty", src)
    put("rqThreeCnnDenseEnergy", f"{agg('cnn', 1, 0.0, 'energy_pj') / 1e3:.0f}", "CNN dense energy proxy (nJ/frame), 4.6 pJ/MAC", src)
    put("rqThreeCnnFloorOps", f"{ops('cnn', 1, 10.0):.0f}k", "lowest-ops CNN that still works (lam 10)", src)
    put("rqThreeCnnFloorAcc", f3(acc("cnn", 1, 10.0)), "its accuracy", src)
    put("rqThreeCnnCollapseOps", f"{ops('cnn', 1, 20.0):.1f}k", "CNN at lam 20: whole net silent", src)
    put("rqThreeCnnCollapseAcc", f3(acc("cnn", 1, 20.0)), "its accuracy (= majority-action rate)", src)
    put("rqThreeSnnTwoLowOps", f"{ops('snn', 2, 0.03):.1f}k", "SNN T=2 at lam 0.03: ops/frame", src)
    put("rqThreeSnnTwoLowAcc", f3(acc("snn", 2, 0.03)), "SNN T=2 at lam 0.03: accuracy", src)
    put("rqThreeSnnTwoLowRet", f3(agg("snn", 2, 0.03, "return_mean")), "SNN T=2 at lam 0.03: closed-loop return", src)
    put("rqThreeSnnTwoLowEnergy", f"{en('snn', 2, 0.03):.1f}", "SNN T=2 lam 0.03 energy proxy (nJ/frame), 0.9 pJ/SynOp", src)
    put("rqThreeSnnFourLowOps", f"{ops('snn', 4, 0.1):.1f}k", "SNN T=4 at lam 0.1: ops/frame", src)
    put("rqThreeSnnFourLowAcc", f3(acc("snn", 4, 0.1)), "SNN T=4 at lam 0.1: accuracy", src)
    put("rqThreeCnnFloorEnergy", f"{en('cnn', 1, 10.0):.0f}", "CNN lam 10 event-driven energy proxy (nJ/frame)", src)
    put("rqThreeOpsRatio", f"{ops('cnn', 1, 10.0) / ops('snn', 2, 0.03):.1f}", "CNN floor ops / SNN T=2 low-ops point", src)
    # CPU cost (results/compute.json)
    comp = {c["variant"]: c for c in json.load(open("results/compute.json"))}
    cnn_ms = comp["CNN"]["train_step_ms_per_256"]
    snn_ms = comp["A: SNN e2e (surrogate BPTT)"]["train_step_ms_per_256"]
    hyb_ms = comp["A+mem: SNN + Transformer (window 4)"]["train_step_ms_per_256"]
    put("msCnn", f"{cnn_ms:.0f}", "CPU ms per PPO minibatch step (256), CNN", "results/compute.json")
    put("msSnn", f"{snn_ms:.0f}", "CPU ms per PPO minibatch step, SNN T=4", "results/compute.json")
    put("msHybrid", f"{hyb_ms:.0f}", "CPU ms per PPO minibatch step, SNN+Transformer", "results/compute.json")
    put("snnCpuSlowdown", f"{snn_ms / cnn_ms:.1f}", "SNN / CNN CPU training cost per step", "results/compute.json")


# ---------------------------------------------------------------- RQ1: local plasticity (DoorKey-6x6, n=10, pre-registered)
def rq1():
    src = "runs/rq1_v2/rq1v2_<arm>_s*/{eval.json,metrics.csv} (scripts/rq1_stats.py; results/rq1_stats.md)"
    A = {k: load_rq1(k) for k in ("bp", "homeo", "tfs0.03", "rands0.03", "dfae", "dfae_homeo", "frozen", "tfs0.3", "hebs0.03")}
    tag = {"bp": "Bp", "homeo": "Homeo", "tfs0.03": "Stdp", "rands0.03": "Rand", "dfae": "Dfa", "dfae_homeo": "DfaHomeo",
           "frozen": "Frozen", "tfs0.3": "StdpLarge", "hebs0.03": "Heb"}
    for k, r in A.items():
        n, s = len(r), sum(v["solved"] for v in r.values())
        put(f"rqOne{tag[k]}N", str(n), f"seeds, arm {k}", src)
        put(f"rqOne{tag[k]}Solved", f"{s}/{n}", f"solved (final eval return > 0.5), arm {k}", src)
        put(f"rqOne{tag[k]}Auc", msd([v["auc"] for v in r.values()]), f"AUC (mean train return) mean $\\pm$ SD, arm {k}", src)
        put(f"rqOne{tag[k]}Eval", f3(np.mean([v["eval"] for v in r.values()])), f"mean final eval return, arm {k}", src)
    contr = [("Stdp", "tfs0.03", "homeo", "A"), ("Rand", "tfs0.03", "rands0.03", "B"), ("Homeo", "homeo", "bp", "C"),
             ("Pert", "rands0.03", "homeo", "D")]
    pf_, pa_ = [], []
    res = {}
    for _, a, b, lab in contr:
        ra, rb = A[a], A[b]
        sa, sb = sum(v["solved"] for v in ra.values()), sum(v["solved"] for v in rb.values())
        p_f = fisher(sa, len(ra), sb, len(rb))
        d, lo, hi, p_a = welch([v["auc"] for v in ra.values()], [v["auc"] for v in rb.values()])
        pf_.append(p_f)
        pa_.append(p_a)
        res[lab] = (d, lo, hi, p_f, p_a, newcombe(sa, len(ra), sb, len(rb)))
    hf, ha = holm(pf_), holm(pa_)
    for i, (_, a, b, lab) in enumerate(contr):
        d, lo, hi, p_f, p_a, (pd, plo, phi) = res[lab]
        put(f"rqOneContr{lab}AucCi", ci_str(d, lo, hi), f"contrast ({lab.lower()}) {a} vs {b}: AUC diff [95% Welch CI]", src)
        put(f"rqOneContr{lab}AucP", pf(p_a), f"contrast ({lab.lower()}) AUC Welch p (raw)", src)
        put(f"rqOneContr{lab}AucHolm", pf(ha[i]), f"contrast ({lab.lower()}) AUC Welch p, Holm over 4 contrasts", src)
        put(f"rqOneContr{lab}FisherP", pf(p_f), f"contrast ({lab.lower()}) solve-rate Fisher p (raw)", src)
        put(f"rqOneContr{lab}FisherHolm", pf(hf[i]), f"contrast ({lab.lower()}) Fisher p, Holm", src)
        put(f"rqOneContr{lab}SolveCi", ci_str(pd, plo, phi, 2), f"contrast ({lab.lower()}) solve-rate difference [95% Newcombe CI]", src)
    h3 = [v["auc"] for k, v in A["homeo"].items() if k <= 3]
    b5 = [v["auc"] for k, v in A["bp"].items() if k <= 5]  # backprop had seeds 1-5 at the time (jobs/rq1_v2*.txt)
    put("homeoEarlySeeds", f"{len(h3)} vs {len(b5)} seeds", "session-2 snapshot seed counts (homeo vs bp)", src)
    put("homeoEarlyP", pf(welch(h3, b5)[3]), "session-2 snapshot: homeo (seeds 1-3) vs bp (seeds 1-5) AUC Welch p", src)
    put("homeoEarlySolved", f"{sum(A['homeo'][k]['solved'] for k in (1, 2, 3))}/3", "session-2 snapshot: homeo solved, seeds 1-3", src)
    # session 1, DoorKey-5x5: vanilla (unstabilised) STDP
    s1 = "runs/rq1/rq1_<arm>_s*/{eval.json,metrics.csv}; results/rq1.md"

    def load5(arm):
        out = {}
        for p in glob.glob(f"runs/rq1/rq1_{arm}_s*/eval.json"):
            dname = os.path.basename(os.path.dirname(p))
            if dname.rsplit("_s", 1)[0] != f"rq1_{arm}":
                continue
            rows = list(csv.DictReader(open(p.replace("eval.json", "metrics.csv"))))
            out[dname] = dict(auc=auc([float(x["ep_return"]) for x in rows]), eval=json.load(open(p))["return_mean"],
                              act=[float(x["enc_activity"]) for x in rows])
        return out
    bp5, tf5 = load5("bp"), load5("tf1")
    d, lo, hi, p = welch([v["auc"] for v in tf5.values()], [v["auc"] for v in bp5.values()])
    put("vanillaStdpAucCi", ci_str(d, lo, hi), "DoorKey-5x5: vanilla STDP (alpha 1) minus backprop, AUC [95% CI]", s1)
    put("vanillaStdpP", pf(p), "its Welch p", s1)
    put("vanillaStdpSolved", f"{sum(v['eval'] > 0.5 for v in tf5.values())}/{len(tf5)}", "vanilla STDP solved (5x5)", s1)
    put("vanillaBpSolved", f"{sum(v['eval'] > 0.5 for v in bp5.values())}/{len(bp5)}", "backprop solved (5x5)", s1)
    act_tf = np.mean([np.mean(v["act"][-20:]) for v in tf5.values()])
    act_bp = np.mean([np.mean(v["act"][-20:]) for v in bp5.values()])
    put("vanillaStdpRate", f2(act_tf), "vanilla STDP: encoder activity (spike rate) over the last 20 updates", s1)
    put("bpRate", f2(act_bp), "backprop: encoder activity over the last 20 updates", s1)


# ---------------------------------------------------------------- DFA +/- homeostasis (session 4, block 2)
def dfa():
    src = "runs/rq1_v2/rq1v2_dfae{,_homeo}_s*/ (scripts/s4_stats.py; results/s4_stats.md)"
    h, d = load_rq1("dfae_homeo"), load_rq1("dfae")
    sh, sd = sum(v["solved"] for v in h.values()), sum(v["solved"] for v in d.values())
    p_f = fisher(sh, len(h), sd, len(d))
    dd, lo, hi, p_a = welch([v["auc"] for v in h.values()], [v["auc"] for v in d.values()])
    adj = holm([p_f, p_a])
    pd, plo, phi = newcombe(sh, len(h), sd, len(d))
    put("dfaSolveCi", ci_str(pd, plo, phi, 2), "DFA+homeo minus DFA solve rate [95% Newcombe CI]", src)
    put("dfaFisherP", pf(p_f), "Fisher p (raw)", src)
    put("dfaFisherHolm", pf(adj[0]), "Fisher p, Holm over the 2 planned tests", src)
    put("dfaAucCi", ci_str(dd, lo, hi), "DFA+homeo minus DFA AUC [95% Welch CI]", src)
    put("dfaAucP", pf(p_a), "AUC Welch p (raw)", src)
    put("dfaAucHolm", pf(adj[1]), "AUC Welch p, Holm", src)
    put("dfaAucRatio", f"{np.mean([v['auc'] for v in h.values()]) / np.mean([v['auc'] for v in d.values()]):.1f}",
        "AUC ratio DFA+homeo / DFA", src)
    # earlier small-n snapshot (session 2 seeds: homeo s1-3, DFA s1-5)
    h3 = {k: v for k, v in h.items() if k <= 3}
    d5 = {k: v for k, v in d.items() if k <= 5}
    put("dfaEarlySolvedHomeo", f"{sum(v['solved'] for v in h3.values())}/{len(h3)}", "session-2 snapshot: DFA+homeo solved (seeds 1-3)", src)
    put("dfaEarlySolvedDfa", f"{sum(v['solved'] for v in d5.values())}/{len(d5)}", "session-2 snapshot: DFA solved (seeds 1-5)", src)
    put("dfaEarlyAucP", pf(welch([v["auc"] for v in h3.values()], [v["auc"] for v in d5.values()])[3]),
        "session-2 snapshot AUC Welch p", src)
    # continual SNN with DFA+homeostasis
    r = cl("runs/continual_snn", "sleep_dfaeh")
    put("dfaContinualAcc", msd([x["metrics"]["ACC"] for x in r.values()]), "SNN fetch3 continual ACC, DFA+homeo encoder + sleep",
        "runs/continual_snn/fetch3_sleep_dfaeh_s*/results.json")


# ---------------------------------------------------------------- RQ2: sleep vs replay / EWC / naive
def rq2():
    trunks = {"Cnn": ("runs/continual", "fetch3"), "Snn": ("runs/continual_snn", "fetch3"),
              "Hyb": ("runs/continual_hybrid", "fetch3"), "CnnFive": ("runs/continual5_cnn", "fetch5")}
    methods = {"naive": "Naive", "ewc_lam100": "Ewc", "replay": "Replay", "sleep": "Sleep", "isolation": "Iso",
               "sleep_matched": "SleepMatched"}
    for t, (root, suite) in trunks.items():
        for m, M in methods.items():
            r = cl(root, m, suite)
            if len(r) < 2:
                continue
            src = f"{root}/{suite}_{m}_s*/results.json (results/continual.md)"
            put(f"acc{t}{M}", msd([x["metrics"]["ACC"] for x in r.values()]), f"{t} {suite} {m}: ACC mean $\\pm$ SD", src)
            put(f"forget{t}{M}", f3(np.mean([x["metrics"]["FORGET"] for x in r.values()])), f"{t} {suite} {m}: mean FORGET", src)
            put(f"n{t}{M}", str(len(r)), f"{t} {suite} {m}: seeds", src)
        s, rp = cl(root, "sleep", suite), cl(root, "replay", suite)
        if len(s) > 1 and len(rp) > 1:
            d, lo, hi, p = welch([x["metrics"]["ACC"] for x in s.values()], [x["metrics"]["ACC"] for x in rp.values()])
            put(f"sleepVsReplay{t}Ci", ci_str(d, lo, hi), f"{t}: sleep minus replay ACC [95% Welch CI] (unmatched budget)",
                f"{root}/{suite}_{{sleep,replay}}_s*/results.json")
            put(f"sleepVsReplay{t}P", pf(p), f"{t}: its Welch p", f"{root}/{suite}_{{sleep,replay}}_s*/results.json")
    # EWC lambda sweep on CNN
    for lam, L in ((10, "Ten"), (100, "Hundred"), (1000, "Thousand")):
        r = cl("runs/continual", f"ewc_lam{lam}")
        src = f"runs/continual/fetch3_ewc_lam{lam}_s*/results.json"
        put(f"ewcAcc{L}", msd([x["metrics"]["ACC"] for x in r.values()]), f"CNN EWC lambda={lam}: ACC", src)
        put(f"ewcForget{L}", f3(np.mean([x["metrics"]["FORGET"] for x in r.values()])), f"CNN EWC lambda={lam}: FORGET", src)
        # plasticity cost: return on the last task right after training it
        put(f"ewcLastTask{L}", f3(np.mean([x["R"][-1][-1] for x in r.values()])),
            f"CNN EWC lambda={lam}: return on the last task after training it", src)
    r = cl("runs/continual", "naive")
    put("naiveLastTaskCnn", f3(np.mean([x["R"][-1][-1] for x in r.values()])), "CNN naive: return on the last task", "runs/continual/fetch3_naive_s*/results.json")
    # block 1: exact replay-sample matching (hybrid, n=8)
    src = "runs/continual_hybrid/fetch3_{sleep_matched,replay,sleep}_s*/results.json (scripts/s4_stats.py; results/s4_stats.md)"
    sm, rp, su = cl("runs/continual_hybrid", "sleep_matched"), cl("runs/continual_hybrid", "replay"), cl("runs/continual_hybrid", "sleep")
    seeds = [s for s in range(1, 9) if s in sm and s in rp]
    assert all(sm[s]["replay_samples"] == rp[s]["replay_samples"] for s in seeds)
    put("matchN", str(len(seeds)), "block 1 seeds per arm", src)
    put("matchSamplesMin", f"{min(rp[s]['replay_samples'] for s in seeds) / 1e6:.2f}M", "smallest per-run replay count", src)
    put("matchSamplesMax", f"{max(rp[s]['replay_samples'] for s in seeds) / 1e6:.2f}M", "largest per-run replay count", src)
    put("unmatchedSleepSamples", f"{su[1]['replay_samples'] / 1e6:.2f}M", "replay states consumed by an unmatched hybrid sleep run", src)
    extra = np.mean([su[s]["replay_samples"] for s in su]) / np.mean([rp[s]["replay_samples"] for s in su]) - 1
    put("unmatchedExtraPct", f"{100 * extra:.0f}", "% more replayed states for unmatched sleep vs replay (same seeds 1-6)", src)
    for k, K in (("ACC", "Acc"), ("FORGET", "Forget")):
        a, b = [sm[s]["metrics"][k] for s in seeds], [rp[s]["metrics"][k] for s in seeds]
        d, lo, hi, p = welch(a, b)
        put(f"match{K}Sleep", msd(a), f"block 1 sleep_matched {k}", src)
        put(f"match{K}Replay", msd(b), f"block 1 replay {k}", src)
        put(f"match{K}Ci", ci_str(d, lo, hi), f"block 1 sleep_matched minus replay {k} [95% Welch CI]", src)
        put(f"match{K}P", pf(p), f"block 1 {k} Welch p", src)
    both = [s for s in seeds if s in su]
    a = [sm[s]["metrics"]["FORGET"] for s in both]
    b = [su[s]["metrics"]["FORGET"] for s in both]
    put("unmatchedForgetSameSeeds", f3(np.mean(b)), "unmatched sleep FORGET, seeds 1-6", src)
    put("matchedForgetSameSeeds", f3(np.mean(a)), "matched sleep FORGET, seeds 1-6", src)
    a2 = [sm[s]["metrics"]["ACC"] for s in both]
    b2 = [su[s]["metrics"]["ACC"] for s in both]
    t = stats.ttest_rel(a2, b2)
    dd = np.subtract(a2, b2)
    half = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / math.sqrt(len(dd))
    put("matchedMinusUnmatchedAccCi", ci_str(dd.mean(), dd.mean() - half, dd.mean() + half),
        "matched minus unmatched sleep ACC, same seeds [95% paired-t CI] (descriptive)", src)
    put("matchedMinusUnmatchedAccP", pf(t.pvalue), "its paired t p", src)
    # history of the hybrid sleep-vs-replay claim (seeds 1-3, 1-6)
    for n, N_ in ((3, "Three"), (6, "Six")):
        a = [su[s]["metrics"]["ACC"] for s in range(1, n + 1)]
        b = [rp[s]["metrics"]["ACC"] for s in range(1, n + 1)]
        d, lo, hi, p = welch(a, b)
        put(f"hybSleepReplayN{N_}Ci", ci_str(d, lo, hi), f"hybrid unmatched sleep minus replay ACC, seeds 1-{n} [95% CI]", src)
        put(f"hybSleepReplayN{N_}P", pf(p), f"its Welch p (seeds 1-{n})", src)
        put(f"hybSleepReplayN{N_}Seeds", f"seeds 1--{n}", f"seed set of the snapshot", src)
    # STDP inside sleep (SNN trunk)
    for m, M in (("sleep_stdp", "SleepVanillaStdp"), ("sleep_stdps0.003", "SleepStabStdp"), ("sleep_homeo", "SleepHomeo")):
        r = cl("runs/continual_snn", m)
        put(f"snn{M}Acc", msd([x["metrics"]["ACC"] for x in r.values()]), f"SNN fetch3 {m}: ACC", f"runs/continual_snn/fetch3_{m}_s*/results.json")
        put(f"snn{M}N", str(len(r)), f"SNN fetch3 {m}: seeds", f"runs/continual_snn/fetch3_{m}_s*/results.json")


# ---------------------------------------------------------------- RQ5: shared weights vs isolation, memory-fair
def rq5():
    g = load_mem()
    srcm = "runs/<root>/<suite>_<arm>_s*/results.json via scripts/memory_fair.py (results/rq5_memory.md)"

    def grp(suite, trunk, name):
        runs = g[(suite, trunk, name)]
        return np.array([r["acc"] for r in runs]), runs[0]
    rows = [("CnnThree", "fetch3", "CNN", "fetch3_isolation @150k", "fetch3_sleep @150k"),
            ("SnnThree", "fetch3", "SNN", "fetch3_isolation @150k", "fetch3_sleep @150k"),
            ("HybThree", "fetch3", "Hybrid", "fetch3_isolation @400k", "fetch3_sleep @400k"),
            ("CnnFive", "fetch5", "CNN", "fetch5_isolation @150k", "fetch5_sleep @150k"),
            ("SnnFiveFair", "fetch5", "SNN", "fetch5_isolation @450k", "fetch5_sleep @450k")]
    for tag, suite, trunk, iso, slp in rows:
        ai, ri = grp(suite, trunk, iso)
        as_, rs = grp(suite, trunk, slp)
        put(f"mem{tag}IsoMb", f2(ri["total_mb"]), f"{trunk} {suite}: isolation total MB", srcm)
        put(f"mem{tag}SleepMb", f2(rs["total_mb"]), f"{trunk} {suite}: sleep (5000/task) total MB", srcm)
        put(f"mem{tag}IsoPerMb", f2(ai.mean() / ri["total_mb"]), f"{trunk} {suite}: isolation ACC per MB", srcm)
        put(f"mem{tag}SleepPerMb", f2(as_.mean() / rs["total_mb"]), f"{trunk} {suite}: sleep ACC per MB", srcm)
        put(f"mem{tag}IsoAcc", f3(ai.mean()), f"{trunk} {suite}: isolation ACC", srcm)
        put(f"mem{tag}SleepAcc", f3(as_.mean()), f"{trunk} {suite}: sleep ACC", srcm)
        # per-parameter efficiency
        pi = np.array([r["acc"] / (r["params_mb"] * 1e6 / 4 / 1e5) for r in g[(suite, trunk, iso)]])
        ps = np.array([r["acc"] / (r["params_mb"] * 1e6 / 4 / 1e5) for r in g[(suite, trunk, slp)]])
        put(f"perParam{tag}Iso", f2(pi.mean()), f"{trunk} {suite}: isolation ACC per 100k params", srcm)
        put(f"perParam{tag}Sleep", f2(ps.mean()), f"{trunk} {suite}: sleep ACC per 100k params", srcm)
    ratios = []
    for tag, suite, trunk, iso, slp in rows[:4]:
        pi = np.mean([r["acc"] / (r["params_mb"] * 1e6 / 4 / 1e5) for r in g[(suite, trunk, iso)]])
        ps = np.mean([r["acc"] / (r["params_mb"] * 1e6 / 4 / 1e5) for r in g[(suite, trunk, slp)]])
        ratios.append(ps / pi)
    put("perParamRatioRange", f"{min(ratios):.1f}--{max(ratios):.1f}", "sleep / isolation ACC per parameter, range over CNN3, SNN3, hybrid3, CNN5", srcm)
    mb_ratios = [(np.mean([r["acc"] for r in g[(suite, trunk, iso)]]) / g[(suite, trunk, iso)][0]["total_mb"]) /
                 (np.mean([r["acc"] for r in g[(suite, trunk, slp)]]) / g[(suite, trunk, slp)][0]["total_mb"]) for _, suite, trunk, iso, slp in rows[:4]]
    put("perMbRatioRange", f"{min(mb_ratios):.1f}--{max(mb_ratios):.1f}", "isolation / sleep(5000) ACC per MB, range over CNN3, SNN3, hybrid3, CNN5", srcm)
    # small buffers
    for tag, trunk, name in (("CnnTwoHundred", "CNN", "fetch3_sleep_buf200 @150k"), ("SnnTwoHundred", "SNN", "fetch3_sleep_buf200 @150k"),
                             ("SnnTwoThousand", "SNN", "fetch3_sleep_buf2000 @150k"),
                             ("SnnReplayTwoThousand", "SNN", "fetch3_replay_buf2000 @150k"),
                             ("SnnReplayTwoHundred", "SNN", "fetch3_replay_buf200 @150k")):
        a, r = grp("fetch3", trunk, name)
        put(f"small{tag}Acc", msd(a), f"{trunk} fetch3 {name}: ACC", srcm)
        put(f"small{tag}Mb", f2(r["total_mb"]), f"{trunk} fetch3 {name}: total MB", srcm)
        put(f"small{tag}PerMb", f2(a.mean() / r["total_mb"]), f"{trunk} fetch3 {name}: ACC per MB", srcm)
    # tests vs isolation (Welch CI) for SNN small/matched buffers and fetch5 fair
    iso_snn = np.array([x["metrics"]["ACC"] for x in cl("runs/continual_snn", "isolation").values()])
    for tag, name in (("TwoHundred", "sleep_buf200"), ("TwoThousand", "sleep_buf2000")):
        a = [x["metrics"]["ACC"] for x in cl("runs/continual_snn_budget", name).values()]
        d, lo, hi, p = welch(a, iso_snn)
        put(f"snnSleep{tag}VsIsoCi", ci_str(d, lo, hi), f"SNN {name} minus isolation ACC [95% CI]",
            f"runs/continual_snn_budget/fetch3_{name}_s*, runs/continual_snn/fetch3_isolation_s* (results/rq5_stats.md)")
        put(f"snnSleep{tag}VsIsoP", pf(p), "its Welch p", "results/rq5_stats.md")
    put("snnIsoAcc", msd(iso_snn), "SNN fetch3 isolation ACC", "runs/continual_snn/fetch3_isolation_s*/results.json")
    # mean-only values, seed counts and memory ratio for the small-buffer wording (SNN, 200 states/task vs isolation)
    a200, r200 = grp("fetch3", "SNN", "fetch3_sleep_buf200 @150k")
    _, riso = grp("fetch3", "SNN", "fetch3_isolation @150k")
    put("smallSnnTwoHundredMean", f3(np.mean(a200)), "SNN fetch3 sleep_buf200: mean ACC", srcm)
    put("snnIsoMean", f3(iso_snn.mean()), "SNN fetch3 isolation: mean ACC", "runs/continual_snn/fetch3_isolation_s*/results.json")
    put("smallSnnTwoHundredN", str(min(len(a200), len(iso_snn))), "seeds per arm, SNN sleep_buf200 vs isolation", srcm)
    put("smallSnnTwoHundredMemPct", f"{100 * r200['total_mb'] / riso['total_mb']:.0f}",
        "SNN sleep_buf200 total memory as % of isolation's", srcm)
    # fetch5 SNN: short budget (n=2) vs fair budget (n=3)
    for tag, root in (("Short", "runs/continual5_snn"), ("Fair", "runs/continual5_snn_450k")):
        s, i = cl(root, "sleep", "fetch5"), cl(root, "isolation", "fetch5")
        a, b = [x["metrics"]["ACC"] for x in s.values()], [x["metrics"]["ACC"] for x in i.values()]
        d, lo, hi, p = welch(a, b)
        put(f"fiveSnn{tag}Sleep", msd(a), f"fetch5 SNN ({tag}) sleep ACC", f"{root}/fetch5_sleep_s*/results.json")
        put(f"fiveSnn{tag}Iso", msd(b), f"fetch5 SNN ({tag}) isolation ACC", f"{root}/fetch5_isolation_s*/results.json")
        put(f"fiveSnn{tag}Ci", ci_str(d, lo, hi), f"fetch5 SNN ({tag}) sleep minus isolation [95% CI]", root)
        put(f"fiveSnn{tag}P", pf(p), f"fetch5 SNN ({tag}) Welch p", root)
        put(f"fiveSnn{tag}N", f"{len(a)}", f"fetch5 SNN ({tag}) seeds per arm", root)
        if tag == "Fair":
            thr = 0.8  # pre-registered "task learned" threshold (PROGRESS.md, session 3 pre-registration)
            put("learnThreshold", f"{thr:g}", "task counted as learned if R[k][k] >= this (pre-registered)", "PROGRESS.md session-3 pre-registration (protocol constant)")
            learned_s = sum(sum(x["R"][k][k] >= thr for k in range(5)) for x in s.values())
            learned_i = sum(sum(x["R"][k][k] >= thr for k in range(5)) for x in i.values())
            put("fiveSnnFairLearnedSleep", f"{learned_s}/{5 * len(s)}", "fetch5 SNN fair: tasks learned to >= 0.8, sleep", root)
            put("fiveSnnFairLearnedIso", f"{learned_i}/{5 * len(i)}", "fetch5 SNN fair: tasks learned to >= 0.8, isolation", root)
            f = [fwt(s[k], i[k]) for k in s if k in i]
            put("fiveSnnFairFwt", msd(f), "fetch5 SNN fair: sleep FWT vs isolation curves", root)
            put("fiveSnnFairFwtP", pf(stats.ttest_1samp(f, 0).pvalue), "one-sample t p (FWT vs 0)", root)
    # FWT on 3-task trunks
    for tag, root in (("Cnn", "runs/continual"), ("Snn", "runs/continual_snn"), ("Hyb", "runs/continual_hybrid")):
        s, i = cl(root, "sleep"), cl(root, "isolation")
        f = [fwt(s[k], i[k]) for k in s if k in i]
        put(f"fwt{tag}", msd(f), f"{tag} fetch3 sleep FWT (vs same-seed isolation curves)", f"{root}/fetch3_{{sleep,isolation}}_s*/results.json (results/rq5_stats.md)")
        put(f"fwt{tag}P", pf(stats.ttest_1samp(f, 0).pvalue), "one-sample t p", "results/rq5_stats.md")
    s, i = cl("runs/continual5_cnn", "sleep", "fetch5"), cl("runs/continual5_cnn", "isolation", "fetch5")
    put("fwtCnnFive", msd([fwt(s[k], i[k]) for k in s if k in i]), "CNN fetch5 sleep FWT", "runs/continual5_cnn/*")


# ---------------------------------------------------------------- Transformer / memory benchmark
def memory():
    for n, tag in (("mem_cnn_fs1", "CnnOne"), ("mem_cnn_fs4", "CnnFour"), ("mem_cnn_fs8", "CnnEight"),
                   ("mem_snn_fs8", "SnnEight"), ("mem_snn_tf8", "SnnTf")):
        p = f"runs/s3_mem_pilot/{n}_s1/eval.json"
        put(f"sSeven{tag}", f3(json.load(open(p))["success"]), f"MemoryS7 pilot {n}: eval success (1 seed)", p)
    lr_tf = yaml.safe_load(open("runs/s3_mem_pilot/mem_snn_tf8_s1/config.yaml"))["lr"]
    lr_fs = yaml.safe_load(open("runs/s3_mem_pilot/mem_snn_fs8_s1/config.yaml"))["lr"]
    put("sSevenLrTf", f"{lr_tf:g}", "S7 pilot SNN+Transformer lr", "runs/s3_mem_pilot/mem_snn_tf8_s1/config.yaml")
    put("sSevenLrOther", f"{lr_fs:g}", "S7 pilot other arms lr", "runs/s3_mem_pilot/mem_snn_fs8_s1/config.yaml")
    for n, tag in (("probe_S11_cnn_fs1", "ElevenOne"), ("probe_S11_cnn_fs12", "ElevenTwelve"),
                   ("probe_S13_cnn_fs1", "ThirteenOne"), ("probe_S13_cnn_fs12", "ThirteenTwelve"),
                   ("probe_S11_cnn_fs12_6M", "ElevenTwelveLong")):
        p = f"runs/s4_mem_probe/{n}_s1/eval.json"
        e = json.load(open(p))
        put(f"probe{tag}", f3(e["success"]), f"{n}: eval success", p)
        put(f"probe{tag}Len", f"{e['len_mean']:.0f}", f"{n}: mean episode length", p)
    rows = list(csv.DictReader(open("runs/s4_mem_probe/probe_S11_cnn_fs12_6M_s1/metrics.csv")))
    fr = np.array([int(x["frames"]) for x in rows])
    ret = np.array([float(x["ep_return"]) for x in rows])
    lo = min(ret[max(0, np.searchsorted(fr, q * 10 ** 6) - 20):np.searchsorted(fr, q * 10 ** 6)].mean() for q in range(1, 7))
    hi = max(ret[max(0, np.searchsorted(fr, q * 10 ** 6) - 20):np.searchsorted(fr, q * 10 ** 6)].mean() for q in range(1, 7))
    put("probeLongRange", f"{lo:.2f}--{hi:.2f}", "S11 fs12 6M: range of training return at 1..6M frames", "runs/s4_mem_probe/probe_S11_cnn_fs12_6M_s1/metrics.csv")
    put("probeFrames", f"{json.load(open('runs/s4_mem_probe/probe_S11_cnn_fs1_s1/eval.json'))['train_frames'] / 1e6:.0f}M",
        "probe training frames", "runs/s4_mem_probe/probe_S11_cnn_fs1_s1/eval.json")
    put("probeLongFrames", f"{json.load(open('runs/s4_mem_probe/probe_S11_cnn_fs12_6M_s1/eval.json'))['train_frames'] / 1e6:.0f}M",
        "exploratory long probe: training frames", "runs/s4_mem_probe/probe_S11_cnn_fs12_6M_s1/eval.json")
    put("probeLr", f"{yaml.safe_load(open('runs/s4_mem_probe/probe_S11_cnn_fs1_s1/config.yaml'))['lr']:g}", "probe lr",
        "runs/s4_mem_probe/probe_S11_cnn_fs1_s1/config.yaml")


# ---------------------------------------------------------------- RQ4: per-compute robustness (results/rq4.md)
def rq4():
    src = "results/rq4.md"
    rows = {}
    for line in open(src):
        if line.startswith("| ") and not line.startswith("| variant"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells
    get = lambda k: rows[k][-1]  # noqa: E731
    put("rqFourCnn", get("CNN (reference)"), "continual ACC per unit training compute, CNN", src)
    put("rqFourSnn", get("A: SNN, surrogate-gradient BPTT"), "same, SNN + sleep", src)
    put("rqFourSnnHomeo", get("A + local homeostasis"), "same, SNN + homeostasis", src)
    put("rqFourHybrid", get("Full hybrid: A + Transformer memory"), "same, full hybrid", src)
    put("rqFourStdp", get("A + stabilised 3-factor STDP"), "same, stabilised STDP", src)
    put("rqFourDfaHomeo", get("DFA on SNN + homeostasis (heads exact)"), "same, DFA + homeostasis", src)
    put("rqFourRatio", f"{float(get('CNN (reference)')) / float(get('A: SNN, surrogate-gradient BPTT')):.1f}",
        "CNN / best spiking variant, ACC per unit compute", src)
    put("rqFourLocalOnly", rows["B: local-only SNN encoder"][1], "local-only (STDP-trained) encoder solve rate", src)


# ---------------------------------------------------------------- demo agents (seed-1 reruns)
def demo():
    for m in ("naive", "sleep"):
        r = json.load(open(f"runs/demo_ckpt/fetch3_{m}_s1/results.json"))
        put(f"demo{m.capitalize()}Row", ", ".join(f3(x) for x in r["R"][-1]), f"demo {m} seed 1: final accuracy-matrix row",
            f"runs/demo_ckpt/fetch3_{m}_s1/results.json")


def seeds():
    """Seed-count range over the main (non-pilot) arms that enter the paper's comparisons."""
    arms = [len(load_rq1(k)) for k in ("bp", "homeo", "tfs0.03", "rands0.03", "dfae", "dfae_homeo")]
    for root, suite, names in (("runs/continual", "fetch3", ("naive", "ewc_lam100", "replay", "sleep", "isolation")),
                               ("runs/continual_snn", "fetch3", ("naive", "ewc_lam100", "replay", "sleep", "isolation")),
                               ("runs/continual_hybrid", "fetch3", ("naive", "replay", "sleep", "sleep_matched", "isolation")),
                               ("runs/continual5_cnn", "fetch5", ("naive", "replay", "sleep", "isolation")),
                               ("runs/continual5_snn_450k", "fetch5", ("sleep", "isolation")),
                               ("runs/continual_snn_budget", "fetch3", ("sleep_buf200", "replay_buf200", "sleep_buf2000", "replay_buf2000"))):
        arms += [len(cl(root, n, suite)) for n in names]
    put("seedRange", f"{min(arms)}--{max(arms)}", "seeds per arm, range over the main arms", "runs/ (counted by paper/compute_numbers.py)")


def compute_budget():
    """Total CPU time of the frozen PPO runs (one thread each): continual runs log wall_time in results.json;
    single-task runs log cumulative fps in metrics.csv, so time = frames / fps on the last row."""
    import csv as _csv
    cont, single, n_cont, n_single, cl_dirs = 0.0, 0.0, 0, 0, set()
    for p in glob.glob("runs/**/results.json", recursive=True):
        r = json.load(open(p))
        if "wall_time" in r:
            cont, n_cont = cont + r["wall_time"], n_cont + 1
            cl_dirs.add(os.path.dirname(p))
    for p in glob.glob("runs/**/metrics.csv", recursive=True):
        if any(p.startswith(d + os.sep) for d in cl_dirs):
            continue  # task sub-runs of a continual run, already inside its wall_time
        rows = list(_csv.DictReader(open(p)))
        if rows and float(rows[-1]["fps"]) > 0:
            single, n_single = single + int(rows[-1]["frames"]) / float(rows[-1]["fps"]), n_single + 1
    src = "runs/**/results.json (wall_time) + runs/**/metrics.csv (frames / cumulative fps)"
    put("computeCoreHours", f"{round((cont + single) / 3600, -1):,.0f}",
        "total single-thread CPU hours of all frozen PPO runs (continual + single-task; excludes RQ3 behaviour cloning)", src)
    put("computeRuns", f"{n_cont + n_single:,}", "number of frozen PPO runs counted in computeCoreHours", src)


def main():
    for fn in (protocol, rq3, rq1, dfa, rq2, rq5, memory, rq4, demo, seeds, compute_budget):
        fn()
    with open("paper/numbers.tex", "w") as f:
        f.write("% AUTO-GENERATED by paper/compute_numbers.py from runs/ and results/. Do not edit by hand.\n")
        for k, (v, _, d, _) in N.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}} % {d}\n")
    with open("paper/NUMBERS.md", "w") as f:
        f.write("# Numbers in the paper and where they come from\n\n"
                "Generated by `python paper/compute_numbers.py` from the frozen files in `runs/` and `results/`; every "
                "result number in `main.tex` is one of these macros. Intervals are 95% CIs: Welch (difference of "
                "means), Newcombe/Wilson (difference of solve rates) or paired t where stated. Protocol constants "
                "that are not stored in `runs/` are marked *(protocol constant)* with the code file that sets them.\n\n"
                "| macro | value | what it is | source |\n|---|---|---|---|\n")
        pm = "$\\pm$"
        for k, (_, plain, d, src) in N.items():
            val = plain.replace(pm, "±").replace("|", "/")
            f.write("| `\\" + k + "` | " + val + " | " + d.replace(pm, "±") + " | `" + src + "` |\n")
    print(f"{len(N)} numbers written to paper/numbers.tex and paper/NUMBERS.md")


if __name__ == "__main__":
    main()
