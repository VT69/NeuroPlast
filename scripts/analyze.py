"""Aggregate experiment outputs into results/*.md tables and results/figs/*.png.

    python scripts/analyze.py [rq3|continual|rq1|all]
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
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroplast.eval.metrics import auc  # noqa: E402

OUT = "results"
FIG = os.path.join(OUT, "figs")
# categorical slots in fixed order (reference dataviz palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def welch_p(a, b):
    """Two-sided Welch t-test p-value (nan if either side has < 2 samples)."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}" if len(x) else "-"


# ----------------------------------------------------------------------------- RQ3
def rq3(path="runs/rq3_encoders/results.jsonl", tag=""):
    if not os.path.exists(path):
        return
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    df["arm"] = np.where(df.kind == "cnn", "CNN", "SNN T=" + df["T"].astype(str))
    g = df.groupby(["arm", "lam"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), ret=("return_mean", "mean"),
                                       v_r2=("v_r2", "mean") if "v_r2" in df else ("acc", lambda x: np.nan),
                                       energy_nj=("energy_pj", lambda x: x.mean() / 1e3),
                                       energy_sparse_nj=("energy_sparse_pj", lambda x: x.mean() / 1e3),
                                       ops_k=("sparse_ops", lambda x: x.mean() / 1e3),
                                       rate=("rates", lambda r: np.mean([np.mean(v) for v in r])),
                                       n=("seed", "count")).reset_index()
    lines = [f"# RQ3 encoder comparison {tag}\n", f"source: `{path}`\n",
             "energy_nj: CNN = dense MACs x 4.6 pJ; SNN = SynOps x 0.9 pJ. energy_sparse_nj: CNN counting only "
             "nonzero-activation MACs (event-driven CNN). ops_k: event-driven ops (thousands) per frame. "
             "rate: mean fraction of active units (spike rate / nonzero ReLU).\n",
             g.to_markdown(index=False, floatfmt=".4g")]
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(OUT, f"rq3{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    has_v = g["v_r2"].notna().any()
    panels = [("ops_k", "acc", "Action accuracy vs ops", "test accuracy (BC of PPO teacher)")]
    if has_v:
        panels.append(("ops_k", "v_r2", "Value R\u00b2 vs ops", "test R\u00b2 (teacher value)"))
    panels.append(("energy_sparse_nj", "acc", "Action accuracy vs energy", "test accuracy"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    arms = sorted(g.arm.unique(), key=lambda a: (a != "CNN", a))
    for k, (xcol, ycol, title, yl) in enumerate(panels):
        ax = axes[k]
        for i, a in enumerate(arms):
            d = g[(g.arm == a) & (g.acc > 0.6)].sort_values(xcol)  # drop fully collapsed nets
            ax.plot(d[xcol], d[ycol], "-o", color=C[i], lw=2, ms=5, label=a)
        ax.set_xscale("log")
        xl = ("event-driven ops per frame, thousands (log)" if xcol == "ops_k"
              else "energy per frame, nJ (log; CNN event-driven MACs, SNN SynOps)")
        style(ax, title, xl, yl)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"rq3{tag}.png"), dpi=130)
    plt.close(fig)
    print(open(os.path.join(OUT, f"rq3{tag}.md")).read())


# ----------------------------------------------------------------------- continual
CL_ROOTS = {"runs/continual": "CNN trunk", "runs/continual_snn": "SNN trunk (T=4)",
            "runs/continual_budget": "CNN trunk, small replay buffers",
            "runs/continual_hybrid": "Full hybrid: SNN (T=4) + Transformer memory (window 4)",
            "runs/continual5_cnn": "CNN trunk, 5-task sequence",
            "runs/continual5_snn": "SNN trunk, 5-task sequence",
            "runs/continual_snn_budget": "SNN trunk, small replay buffers (session 3)",
            "runs/continual5_snn_450k": "SNN trunk, 5-task sequence, 450k frames/task (session 3 isolation budget)"}


def continual():
    lines = ["# Continual learning results (RQ2 / RQ5)\n",
             "R[i][j] = mean return on task j after training task i (100 eval episodes, held-out seeds). "
             "ACC = final mean return over all tasks; BWT = mean(R[T-1][j] - R[j][j]); FORGET = mean(max_i R[i][j] - R[T-1][j]); "
             "FWT = normalised AUC gain of each task's training curve (tasks 2..T) vs the isolation arm's from-scratch curve on the same task and seed "
             "(budget root: vs the CNN isolation runs). ACC/100k params = parameter efficiency (RQ5). "
             "replay_samples = replayed states consumed (budget matching). p(ACC vs replay) = two-sided Welch t-test "
             "against the replay arm with the same buffer setting.\n"]
    iso_ref = {}
    for root, title in CL_ROOTS.items():
        lines += continual_root(root, title, iso_ref)
    with open(os.path.join(OUT, "continual.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def continual_root(root, title, iso_ref):
    res = {}
    for p in glob.glob(os.path.join(root, "*", "results.json")):
        r = json.load(open(p))
        if r.get("metrics") is None:
            continue
        key = (r["suite"], r["method"] + r.get("tag", ""))
        res.setdefault(key, {})[r["seed"]] = r
    if not res:
        return []
    lines = [f"\n# {title} (`{root}`)"]
    for suite in sorted({k[0] for k in res}):
        iso = res.get((suite, "isolation")) or iso_ref.get(suite, {})
        iso_ref.setdefault(suite, iso)
        table = []
        for (s, m), seeds in sorted(res.items()):
            if s != suite:
                continue
            acc, bwt, fg, fwt, eff, rs = [], [], [], [], [], []
            for seed, r in seeds.items():
                mt = r["metrics"]
                acc.append(mt["ACC"]), bwt.append(mt["BWT"]), fg.append(mt["FORGET"])
                eff.append(mt["ACC"] / (r["params"] / 1e5))
                rs.append(r.get("replay_samples", 0))
                if seed in iso:
                    ref = iso[seed]["curves"]
                    f = []
                    for j in range(1, len(r["curves"])):
                        a, ar = auc(r["curves"][j]), auc(ref[j])
                        f.append((a - ar) / (1 - ar))
                    fwt.append(np.mean(f))
            ref_name = next((k[1] for k in res if k[0] == suite and k[1].startswith("replay")
                             and k[1].replace("replay", "") == m.replace("sleep", "").replace("replay", "")), None)
            ref = res.get((suite, ref_name)) if ref_name and ref_name != m else None
            p_vs = welch_p(acc, [r["metrics"]["ACC"] for r in ref.values()]) if ref else float("nan")
            table.append(dict(method=m, seeds=len(seeds), ACC=ms(acc), **{"p(ACC vs replay)": f"{p_vs:.3f}"},
                              BWT=ms(bwt), FORGET=ms(fg),
                              FWT=ms(fwt) if fwt else "-", params=int(np.mean([r["params"] for r in seeds.values()])),
                              **{"ACC/100k params": ms(eff)}, replay_samples=int(np.mean(rs))))
        lines += [f"\n## suite `{suite}`\n", pd.DataFrame(table).to_markdown(index=False)]
        # mean R matrices
        lines.append("\nMean R matrices (rows: after training task i; cols: task j):\n")
        for (s, m), seeds in sorted(res.items()):
            if s != suite:
                continue
            R = np.mean([r["R"] for r in seeds.values()], 0)
            lines.append(f"`{m}`\n```\n" + "\n".join("  ".join(f"{x:.2f}" for x in row) for row in R) + "\n```")
        plot_curves(suite, {m: seeds for (s, m), seeds in res.items() if s == suite}, os.path.basename(root))
    return lines


def plot_curves(suite, by_method, rootname):
    """Per-task eval return over the whole sequence: R[i][j] across i, per method."""
    os.makedirs(FIG, exist_ok=True)
    methods = sorted(by_method)
    T = len(next(iter(next(iter(by_method.values())).values()))["R"])
    fig, axes = plt.subplots(1, T, figsize=(3.4 * T, 3.2), sharey=True)
    for j in range(T):
        ax = axes[j]
        for i, m in enumerate(methods):
            Rs = np.array([r["R"] for r in by_method[m].values()])  # seeds x T x T
            y = Rs[:, :, j].mean(0)
            ax.plot(range(T), y, "-o", color=C[i % len(C)], lw=2, ms=5, label=m)
        style(ax, f"task {j}", "after training task", "eval return" if j == 0 else None)
        ax.set_xticks(range(T))
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=INK2, loc="center left", bbox_to_anchor=(0.995, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{rootname}_{suite}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- RQ1
def rq1(root="runs/rq1", tag="", title="RQ1: hybrid STDP + backprop (SNN encoder, PPO, DoorKey-5x5)"):
    groups = defaultdict(list)
    for p in glob.glob(os.path.join(root, "*", "metrics.csv")):
        name = os.path.basename(os.path.dirname(p))
        arm = name.rsplit("_s", 1)[0].split("_", 1)[1]
        d = pd.read_csv(p)
        ev = os.path.join(os.path.dirname(p), "eval.json")
        if not os.path.exists(ev):
            continue  # still running
        groups[arm].append((d, json.load(open(ev))))
    if not groups:
        return
    rows = []
    bp_auc = [auc(d.ep_return) for d, _ in groups.get("bp", [])]
    for arm, runs in sorted(groups.items()):
        f09 = [d.frames[d.ep_return >= 0.9].min() if (d.ep_return >= 0.9).any() else np.nan for d, _ in runs]
        solved = sum(e["return_mean"] > 0.5 for _, e in runs)
        rows.append(dict(arm=arm, seeds=len(runs), solved=f"{solved}/{len(runs)}",
                         eval_return=ms([e["return_mean"] for _, e in runs if e]),
                         AUC=ms([auc(d.ep_return) for d, _ in runs]),
                         **{"p(AUC vs bp)": f"{welch_p([auc(d.ep_return) for d, _ in runs], bp_auc):.3f}"},
                         frames_to_0_9=(f"{np.nanmean(f09) / 1e3:.0f}k" if not np.all(np.isnan(f09)) else "-")
                         + f" ({int(np.sum(np.isnan(f09)))} never)"))
    t = pd.DataFrame(rows)
    with open(os.path.join(OUT, f"rq1{tag}.md"), "w") as f:
        f.write(f"# {title}\n\n"
                "AUC = mean training return over the run (sample efficiency); frames_to_0_9 = first update "
                "where the 100-episode moving train return >= 0.9 (mean over seeds that reach it). solved = seeds whose "
                "final eval return > 0.5 (outcomes are bimodal on DoorKey-6x6). p = two-sided Welch t-test of AUC vs `bp` "
                "(no multiple-comparison correction).\n\n" + t.to_markdown(index=False) + "\n")
    print(t.to_markdown(index=False))
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (arm, runs) in enumerate(sorted(groups.items())):
        n = min(len(d) for d, _ in runs)
        y = np.mean([d.ep_return.values[:n] for d, _ in runs], 0)
        ax.plot(runs[0][0].frames.values[:n] / 1e3, y, color=C[i % len(C)], lw=2, label=arm)
    style(ax, f"{title.split(':')[0]}: training return (mean over seeds)", "frames (thousands)", "train return (100-ep mean)")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"rq1{tag}.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- RQ4
def _arm_runs(root, arm):
    out = []
    for p in glob.glob(os.path.join(root, "*", "eval.json")):
        name = os.path.basename(os.path.dirname(p))
        if name.rsplit("_s", 1)[0].split("_", 1)[1] == arm:
            d = pd.read_csv(os.path.join(os.path.dirname(p), "metrics.csv"))
            out.append((json.load(open(p))["return_mean"], auc(d.ep_return)))
    return out


def _cl_acc(root, method):
    accs = []
    for p in glob.glob(os.path.join(root, f"fetch3_{method}_s*", "results.json")):
        if os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0] != f"fetch3_{method}":
            continue  # e.g. "sleep_s*" must not match "sleep_stdp_s1"
        r = json.load(open(p))
        if r.get("metrics"):
            accs.append(r["metrics"]["ACC"])
    return accs


def _cl_frames(root, method):
    for p in glob.glob(os.path.join(root, f"fetch3_{method}_s*", "results.json")):
        if os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0] == f"fetch3_{method}":
            return json.load(open(p))["cfg"]["total_frames"]
    return None


def rq4():
    """Robustness per unit of compute, per architecture variant (joins RQ1/continual/compute outputs)."""
    if not os.path.exists(os.path.join(OUT, "compute.json")):
        return
    comp = {r["variant"].split(":")[0]: r for r in json.load(open(os.path.join(OUT, "compute.json")))}
    V = [  # variant key in compute.json, label, RQ1 (5x5) arm, RQ1 v2 (6x6) arm, continual root, CL method
        ("CNN", "CNN (reference)", "cnn", None, "runs/continual", "sleep"),
        ("A", "A: SNN, surrogate-gradient BPTT", "bp", "bp", "runs/continual_snn", "sleep"),
        ("A", "A + local homeostasis", None, "homeo", "runs/continual_snn", "sleep_homeo"),
        ("A+mem", "Full hybrid: A + Transformer memory", "HYBRID", None, "runs/continual_hybrid", "sleep"),
        ("A+STDP", "A + stabilised 3-factor STDP", None, "tfs0.03", "runs/continual_snn", "sleep_stdps0.003"),
        ("B", "B: local-only SNN encoder", "tf1_local", "tfs0.3_local", None, None),
        ("Cenc", "DFA on SNN synapses (heads exact)", "dfae", "dfae", None, None),
        ("Cenc", "DFA on SNN + homeostasis (heads exact)", None, "dfae_homeo", "runs/continual_snn", "sleep_dfaeh"),
        ("Cmlp", "C: DFA everywhere, MLP heads (no backprop)", "dfaall", None, None, None),
        ("C", "C: DFA, linear heads (session 1)", "dfa", None, None, None),
    ]
    cnn_ms = comp["CNN"]["train_step_ms_per_256"]
    rows = []
    for key, label, a5, a6, root, clm in V:
        c = comp.get(key, {})
        if a5 == "HYBRID":  # single-task runs outside the RQ1 sweep (runs/hybrid_doorkey5_s*)
            r5 = []
            for p in glob.glob("runs/hybrid_doorkey5_s*/eval.json"):
                d = pd.read_csv(p.replace("eval.json", "metrics.csv"))
                r5.append((json.load(open(p))["return_mean"], auc(d.ep_return[d.frames <= 300_000])))  # same budget as RQ1
        else:
            r5 = _arm_runs("runs/rq1", a5) if a5 else []
        r6 = _arm_runs("runs/rq1_v2", a6) if a6 else []
        cl = _cl_acc(root, clm) if root else []
        fpt = _cl_frames(root, clm) if root else None
        solve = [e > 0.5 for e, _ in r5 + r6]
        step_ms = c.get("train_step_ms_per_256", float("nan"))
        rows.append({"variant": label,
                     "solve rate (5x5+6x6 seeds)": f"{sum(solve)}/{len(solve)}" if solve else "-",
                     "AUC 5x5": ms([a for _, a in r5]) if r5 else "-",
                     "AUC 6x6": ms([a for _, a in r6]) if r6 else "-",
                     "continual ACC (fetch3)": (ms(cl) + f" [{clm}]") if cl else "-",
                     "train ms/step (CPU)": f"{step_ms:.0f}",
                     "params": c.get("params", "-"),
                     "continual train compute (x CNN@150k)": (f"{step_ms * fpt / (cnn_ms * 150_000):.1f}" if cl else "-"),
                     "continual ACC per unit compute": (f"{np.mean(cl) / (step_ms * fpt / (cnn_ms * 150_000)):.3f}"
                                                        if cl else "-")})
    with open(os.path.join(OUT, "rq4.md"), "w") as f:
        f.write("# RQ4: robustness per unit of compute\n\nJoins runs/rq1 (DoorKey-5x5), runs/rq1_v2 (DoorKey-6x6), "
                "the continual sweeps and results/compute.json (CPU process time per PPO step, batch 256, 1 thread). "
                "'continual train compute' = per-step cost x frames per task, relative to the CNN at 150k frames/task "
                "(the full hybrid uses 400k/task, the others 150k); replay/sleep overheads are not included. "
                "Inference energy per decision is in results/rq3_doorkey6.md (SNN: 15-200 nJ depending on sparsity, CNN: "
                "1653 nJ dense / 130-940 nJ event-driven). 'continual ACC per unit compute' = final continual ACC / that relative compute "
                "(higher = more robust per unit of training compute).\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("rq3", "all"):
        rq3()
        rq3("runs/rq3_doorkey6/results.jsonl", "_doorkey6")
    if what in ("continual", "all"):
        continual()
    if what in ("rq1", "all"):
        rq1()
        rq1("runs/rq1_v2", "_v2", "RQ1 v2: stabilised STDP + backprop (SNN encoder, PPO, DoorKey-6x6)")
    if what in ("rq4", "all"):
        rq4()
