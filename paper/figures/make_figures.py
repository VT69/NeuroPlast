"""Paper figures, generated only from the frozen result files (runs/, results/).

    python paper/figures/make_figures.py      # writes paper/figures/*.pdf (+ .png previews)

Figures:
  fig_forgetting   retention of task 0 and average return over seen tasks, after each task, by method (4 trunks)
  fig_memory       final ACC vs total memory (fp32 parameters + replay buffer), by method and trunk
  fig_rq1          per-seed AUC for the RQ1 / DFA arms (DoorKey-6x6), mean and 95% CI
  fig_frontier     accuracy and closed-loop return vs event-driven operations per frame (RQ3)
  fig_budget       hybrid sleep vs replay with replayed samples matched exactly (session 4, block 1)
  fig_probe        memory-benchmark probes: training return vs frames on MemoryS11/S13

Colour follows the method and matches the dashboard (validated palette: slots 1-5 and 7 of the reference categorical
palette; passes the adjacent CVD/normal-vision gates; low-contrast slots are relieved by direct labels and markers).
Series that are not continual-learning methods (encoders, probe runs, RQ1 arms) use grey and violet, so blue, orange
and green always mean sleep, replay and isolation. Text is set in Computer Modern to match the paper body, and each
figure is drawn at the width it is printed at (6.5 in = TMLR text width, or less), so font sizes are the printed sizes.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from scripts.memory_fair import load as load_mem  # noqa: E402
from scripts.rq1_stats import load as load_rq1  # noqa: E402

OUT = "paper/figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
NEUTRAL, VIOLET, VIOLET2 = "#52514e", "#4a3aa7", "#9085e9"  # non-method series
M = {  # method -> (label, colour, marker, linestyle)
    "sleep": ("sleep", "#2a78d6", "o", "-"),
    "replay": ("replay", "#eb6834", "s", "-"),
    "isolation": ("isolation", "#1baf7a", "D", "-"),
    "ewc_lam100": ("EWC ($\\lambda$=100)", "#eda100", "^", "--"),
    "naive": ("naive", "#e87ba4", "v", ":"),
    "sleep_matched": ("sleep, replay-matched", "#4a3aa7", "P", "-"),
}
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    # Computer Modern, as in the paper body (matplotlib ships cmr10; cmr10 has no minus glyph, so ticks use mathtext)
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "lines.dash_capstyle": "round",
})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def cl(root, name, suite="fetch3"):
    out = {}
    for p in glob.glob(os.path.join(root, f"{suite}_{name}_s*", "results.json")):
        if os.path.basename(os.path.dirname(p)).rsplit("_s", 1)[0] != f"{suite}_{name}":
            continue
        r = json.load(open(p))
        if r.get("metrics"):
            out[r["seed"]] = r
    return out


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.pdf")


# ------------------------------------------------------------------ forgetting curves
def fig_forgetting():
    panels = [("CNN, 3 tasks", "runs/continual", "fetch3"), ("SNN, 3 tasks", "runs/continual_snn", "fetch3"),
              ("SNN+Transformer, 3 tasks", "runs/continual_hybrid", "fetch3"),
              ("CNN, 5 tasks", "runs/continual5_cnn", "fetch5")]
    dodge = {"naive": -0.15, "ewc_lam100": -0.09, "replay": -0.03, "sleep": 0.03, "sleep_matched": 0.09, "isolation": 0.15}
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.6), sharey=True)
    for c, (title, root, suite) in enumerate(panels):
        for m in ("naive", "ewc_lam100", "replay", "sleep", "sleep_matched", "isolation"):
            runs = cl(root, m, suite)
            if len(runs) < 2:
                continue
            lab, col, mk, ls = M[m]
            R = np.array([r["R"] for r in runs.values()])  # seeds x T x T
            T = R.shape[1]
            x = np.arange(1, T + 1)
            first = R[:, :, 0]  # return on task 0 after each stage
            seen = np.array([[R[s, i, :i + 1].mean() for i in range(T)] for s in range(len(R))])
            for row, y in ((0, first), (1, seen)):
                ax = axes[row, c]
                mu = y.mean(0)
                e = np.array([ci95(y[:, i]) for i in range(T)])
                xd = x + dodge[m]
                ax.errorbar(xd, mu, yerr=e, fmt="none", ecolor=col, elinewidth=0.7, alpha=0.6, capsize=0)
                ax.plot(xd, mu, color=col, marker=mk, ms=4, lw=1.5, ls=ls, label=f"{lab} (n={len(runs)})",
                        markeredgecolor="white", markeredgewidth=0.6)
        for row in (0, 1):
            ax = axes[row, c]
            style(ax)
            T = 3 if suite == "fetch3" else 5
            ax.set_xticks(range(1, T + 1))
            ax.set_ylim(-0.05, 1.05)
            if row == 0:
                ax.set_title(title)
            else:
                ax.set_xlabel("tasks trained so far")
        axes[0, 0].set_ylabel("return on task 1\n(first task)")
        axes[1, 0].set_ylabel("mean return over\ntasks seen so far")
    h, l = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            key = ll.split(" (n=")[0]
            if key not in [x.split(" (n=")[0] for x in l]:
                h.append(hh)
                l.append(key)
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.04), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_forgetting")


# ------------------------------------------------------------------ ACC vs total memory
def fig_memory():
    from matplotlib.ticker import FuncFormatter, LogLocator
    g = load_mem()
    panels = [("CNN, 3 tasks", "fetch3", {"CNN"}), ("SNN, 3 tasks", "fetch3", {"SNN"}),
              ("SNN+Transformer, 3 tasks", "fetch3", {"Hybrid"}), ("5 tasks, CNN and SNN", "fetch5", {"CNN", "SNN"})]
    ticks = [[0.5, 1, 2, 3], [1, 2, 3], [3, 5, 10], [3, 4, 5, 6]]
    tm = {"CNN": "o", "SNN": "s", "Hybrid": "^"}
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    for k, (ax, (title, suite, trunks)) in enumerate(zip(axes, panels)):
        labels = {}
        for (s, trunk, name), runs in g.items():
            m = runs[0]["method"]
            base = name.split(" @")[0]
            if s != suite or trunk not in trunks or m not in ("isolation", "sleep", "replay"):
                continue
            if base != f"{suite}_{m}" and not base.startswith(f"{suite}_{m}_buf"):
                continue  # mechanism variants (sleep_homeo, sleep_stdp, ...) are not method arms
            if suite == "fetch5" and trunk == "SNN" and "@150k" in name:
                continue  # short-budget fetch5 SNN runs: isolation under-trained; plot the fair 450k runs
            acc = np.array([r["acc"] for r in runs])
            x, mu = runs[0]["total_mb"], acc.mean()
            jit = x * (1 + np.linspace(-0.04, 0.04, len(acc)) + (0.015 if m == "replay" else -0.015 if m == "sleep" else 0))
            ax.scatter(jit, acc, s=7, color=M[m][1], alpha=0.45, linewidth=0, zorder=2)
            ax.plot(x, mu, tm[trunk], color=M[m][1], ms=6, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            if m != "isolation":
                labels[(round(x, 1), runs[0]["buf_per_task"])] = x
        for i, ((_, bpt), x) in enumerate(sorted(labels.items(), key=lambda kv: kv[1])):
            ax.text(x, 0.515 + 0.035 * (i % 2), f"{bpt}", ha="center", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(ticks[k])
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=()))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        style(ax, "both")
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0.5, 1.02)
        ax.margins(x=0.12)
    axes[0].set_ylabel("final ACC")
    fig.supxlabel("total memory, MB (fp32 parameters + replay buffer; log scale); numbers above the axis = replay states per task",
                  fontsize=8, color=INK2, y=-0.01)
    h = [plt.Line2D([], [], ls="", marker="o", color=M[m][1], label=M[m][0]) for m in ("isolation", "sleep", "replay")]
    h += [plt.Line2D([], [], ls="", marker=v, color=INK2, markerfacecolor="none", label="SNN+Transformer" if k == "Hybrid" else k)
          for k, v in tm.items()]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.06), labelcolor=INK2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.18)
    save(fig, "fig_memory")


# ------------------------------------------------------------------ RQ1 arms
def fig_rq1():
    arms = [("bp", "backprop"), ("homeo", "+ homeo-\nstasis"), ("tfs0.03", "+ stabilised\nSTDP"), ("rands0.03", "+ matched\nrandom"),
            ("dfae", "DFA"), ("dfae_homeo", "DFA + homeo-\nstasis")]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    xs = [0, 1, 2, 3, 4.8, 5.8]
    col = NEUTRAL
    for x, (k, lab) in zip(xs, arms):
        r = load_rq1(k)
        a = np.array([v["auc"] for v in r.values()])
        solved = np.array([v["solved"] for v in r.values()])
        jit = np.linspace(-0.18, 0.18, len(a))
        ax.scatter(x + jit[solved], a[solved], s=14, color=col, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(x + jit[~solved], a[~solved], s=14, facecolor="white", edgecolor=col, linewidth=0.9, zorder=3)
        ax.errorbar(x + 0.32, a.mean(), yerr=ci95(a), fmt="_", color=INK, ms=9, elinewidth=1.2, capsize=0, zorder=4)
        ax.text(x, 1.0, f"{solved.sum()}/{len(a)}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.axvline(3.9, color=GRID, lw=1)
    ax.text(1.5, 1.17, "global signal: backprop", ha="center", fontsize=8.5, color=INK)
    ax.text(5.3, 1.17, "global signal: DFA", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in arms])
    ax.set_ylim(-0.05, 1.24)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("AUC (mean training return)")
    style(ax)
    h = [plt.Line2D([], [], ls="", marker="o", color=col, label="seed, solved"),
         plt.Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=col, label="seed, not solved"),
         plt.Line2D([], [], ls="", marker="_", color=INK, ms=9, label="mean, 95% CI")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    save(fig, "fig_rq1")


# ------------------------------------------------------------------ RQ3 frontier
def fig_frontier():
    g = defaultdict(list)
    for line in open("runs/rq3_doorkey6/results.jsonl"):
        r = json.loads(line)
        g[(r["kind"], r["T"])].append(r)
    series = [(("cnn", 1), "CNN (L1-sparsified)", NEUTRAL, "o", "-"), (("snn", 2), "SNN, $T=2$", VIOLET, "s", "-"),
              (("snn", 4), "SNN, $T=4$", VIOLET2, "D", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for key, lab, col, mk, ls in series:
        by = defaultdict(list)
        for r in g[key]:
            by[r["lam"]].append(r)
        lams = sorted(by)
        ops = np.array([np.mean([r["sparse_ops"] for r in by[l]]) / 1e3 for l in lams])
        for ax, k in ((axes[0], "acc"), (axes[1], "return_mean")):
            y = np.array([np.mean([r[k] for r in by[l]]) for l in lams])
            o = np.argsort(ops)
            ax.plot(ops[o], y[o], color=col, marker=mk, ms=3.8, lw=1.4, ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.5)
    from matplotlib.ticker import FuncFormatter
    for ax, yl in ((axes[0], "behaviour-cloning test accuracy"), (axes[1], "closed-loop return of cloned policy")):
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("event-driven operations per frame (thousands, log)")
        ax.set_ylabel(yl)
        style(ax, "both")
    axes[0].axhline(0.5244, color=MUTED, lw=0.8, ls="--")
    axes[0].text(8, 0.55, "majority action", fontsize=7.5, color=INK2)
    axes[0].legend(frameon=False, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    save(fig, "fig_frontier")


# ------------------------------------------------------------------ block 1: replay budget
def fig_budget():
    sm, rp, su = cl("runs/continual_hybrid", "sleep_matched"), cl("runs/continual_hybrid", "replay"), cl("runs/continual_hybrid", "sleep")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
    arms = [("replay", rp), ("sleep_matched", sm), ("sleep", su)]
    extra = 100 * (np.mean([su[k]["replay_samples"] for k in su]) / np.mean([rp[k]["replay_samples"] for k in su]) - 1)
    labels = ["replay", "sleep\n(matched)", f"sleep\n(+{extra:.0f}% replay)"]
    for ax, k, yl in ((axes[0], "ACC", "final ACC"), (axes[1], "FORGET", "FORGET")):
        for s in range(1, 9):
            ys = [(i, d[s]["metrics"][k]) for i, (_, d) in enumerate(arms) if s in d]
            ax.plot([p[0] for p in ys], [p[1] for p in ys], color=GRID, lw=0.8, zorder=1)
        for i, (m, d) in enumerate(arms):
            v = np.array([r["metrics"][k] for r in d.values()])
            ax.scatter(np.full(len(v), i), v, s=14, color=M[m][1], edgecolor="white", linewidth=0.5, zorder=3)
            ax.errorbar(i + 0.22, v.mean(), yerr=ci95(v), fmt="_", color=INK, ms=9, elinewidth=1.2, zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.4, 2.6)
        ax.set_ylabel(yl)
        style(ax)
    axes[0].set_title("SNN+Transformer, fetch3: accuracy", loc="left")
    axes[1].set_title("forgetting (lower is better)", loc="left")
    fig.tight_layout()
    save(fig, "fig_budget")


# ------------------------------------------------------------------ memory probes
def fig_probe():
    runs = [("probe_S11_cnn_fs1_s1", "S11, single frame", NEUTRAL, "-"), ("probe_S11_cnn_fs12_s1", "S11, 12-frame stack", VIOLET, "-"),
            ("probe_S13_cnn_fs1_s1", "S13, single frame", NEUTRAL, ":"), ("probe_S13_cnn_fs12_s1", "S13, 12-frame stack", VIOLET, ":"),
            ("probe_S11_cnn_fs12_6M_s1", "S11, 12-frame stack, 6M frames (exploratory)", VIOLET2, "-")]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    for d, lab, col, ls in runs:
        rows = list(csv.DictReader(open(f"runs/s4_mem_probe/{d}/metrics.csv")))
        f = np.array([int(x["frames"]) for x in rows]) / 1e6
        r = np.array([float(x["ep_return"]) for x in rows])
        k = 25
        sm = np.convolve(r, np.ones(k) / k, mode="valid")
        ax.plot(f[k - 1:], sm, color=col, lw=1.3, ls=ls, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.text(5.95, 0.545, "chance at the junction", fontsize=7.5, color=INK2, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("training frames (millions)")
    ax.set_ylabel("training return (25-update mean)")
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2, labelcolor=INK2)
    fig.tight_layout()
    save(fig, "fig_probe")


if __name__ == "__main__":
    for fn in (fig_forgetting, fig_memory, fig_rq1, fig_frontier, fig_budget, fig_probe):
        fn()
