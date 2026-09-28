"""Generate the job files in jobs/ (one line = one run; see scripts/run_queue.py).

    python scripts/make_jobs.py                 # default seeds 1-3 (what session 1 ran)
    python scripts/make_jobs.py --seeds 4-10 --suffix _more   # extra seeds -> jobs/*_more.txt

Seeds only change the run names/markers, so extra-seed files can be run on
another machine and merged into runs/ afterwards (analysis globs by directory).
"""
from __future__ import annotations

import argparse
import os

RQ1_ARMS = {
    "bp": "",
    "tf0.1": '"stdp={alpha: 0.1, mode: three_factor}"',
    "tf1": '"stdp={alpha: 1.0, mode: three_factor}"',
    "heb0.1": '"stdp={alpha: 0.1, mode: hebbian}"',
    "heb1": '"stdp={alpha: 1.0, mode: hebbian}"',
    "rand1": '"stdp={alpha: 1.0, mode: random}"',
    "tf1_local": '"stdp={alpha: 1.0, mode: three_factor, beta: 0}"',
    "frozen": '"stdp={alpha: 0.0, mode: three_factor, beta: 0}"',
    "dfa": "dfa=true hidden=0",
    "bp_linear": "hidden=0",
    "cnn": "encoder=cnn",
}


STAB = "center: true, homeo: 0.15"
RQ1_V2_ARMS = {
    "bp": "",
    "frozen": '"stdp={alpha: 0.0, beta: 0}"',
    "tfs0.03": f'"stdp={{alpha: 0.03, mode: three_factor, {STAB}}}"',
    "tfs0.3": f'"stdp={{alpha: 0.3, mode: three_factor, {STAB}}}"',
    "hebs0.03": f'"stdp={{alpha: 0.03, mode: hebbian, {STAB}}}"',
    "hebs0.3": f'"stdp={{alpha: 0.3, mode: hebbian, {STAB}}}"',
    "rand0.3": '"stdp={alpha: 0.3, mode: random}"',
    "tfs0.3_local": f'"stdp={{alpha: 0.3, mode: three_factor, beta: 0, {STAB}}}"',
}


def rq1_v2(seeds):
    out = ["# RQ1 v2: stabilised STDP (covariance + homeostasis) + backprop, DoorKey-6x6, 400k frames"]
    for s in seeds:
        for arm, st in RQ1_V2_ARMS.items():
            name = f"rq1v2_{arm}_s{s}"
            out.append(f"runs/rq1_v2/{name}/eval.json :: python train.py --config configs/rq1_v2.yaml "
                       f"--set seed={s} run_name={name} out_dir=runs/rq1_v2 {st}".rstrip())
    return out


def rq1(seeds):
    out = ["# RQ1 (+RQ4 variants): learning rules on the SNN encoder, DoorKey-5x5, 300k frames"]
    for s in seeds:
        for arm, st in RQ1_ARMS.items():
            name = f"rq1_{arm}_s{s}"
            out.append(f"runs/rq1/{name}/eval.json :: python train.py --config configs/rq1_snn.yaml "
                       f"--set seed={s} run_name={name} out_dir=runs/rq1 {st}".rstrip())
    return out


def cl(seeds, trunk):
    cfg = f"configs/cl_{trunk}.yaml"
    root = "runs/continual" if trunk == "cnn" else f"runs/continual_{trunk}"
    arms = [("naive", "", ""), ("isolation", "", ""), ("replay", "", ""), ("sleep", "", "")]
    if trunk == "cnn":
        arms += [("ewc", f"lam={l}", f"_lam{l}") for l in (10, 100, 1000)]
    else:
        # (session 1 also ran vanilla-STDP sleep, alpha=1 "_stdp"/"_dream", seed 1 only:
        #  catastrophic, so extra seeds use the stabilised rule instead)
        arms += [("sleep", "stdp_alpha=0.01 stdp_center=true stdp_homeo=0.15", "_stdps0.01"),
                 ("sleep", "stdp_alpha=0.003 stdp_center=true stdp_homeo=0.15", "_stdps0.003"),
                 ("ewc", "lam=100", "_lam100"),
                 ("naive", "", "_wakestdp")]
    out = [f"# continual sweep, {trunk} trunk, fetch3 (150k frames/task)"]
    for s in seeds:
        for method, mk, tag in arms:
            name = f"fetch3_{method}{tag}_s{s}"
            extra = (f" --mk {mk}" if mk else "") + (f" --tag {tag}" if tag else "")
            if tag == "_wakestdp":
                extra += ' --set "stdp={alpha: 1.0, mode: three_factor}"'
            out.append(f"{root}/{name}/results.json :: python continual.py --suite fetch3 --method {method} "
                       f"--config {cfg} --seed {s} --out {root}{extra}")
    return out


def cl_hybrid(seeds):
    """Session 2, priority 1: the full hybrid (SNN + Transformer memory) on fetch3.
    Backprop (surrogate BPTT) is the main rule; `_wtfs` adds the best-mean wake-time
    local rule from RQ1 v2 (stabilised three-factor STDP, alpha 0.03) on top of sleep."""
    root = "runs/continual_hybrid"
    arms = [("naive", "", "", ""), ("isolation", "", "", ""), ("replay", "", "", ""), ("sleep", "", "", ""),
            ("sleep", "", "_wtfs", ' --set "stdp={alpha: 0.03, mode: three_factor, center: true, homeo: 0.15}"'),
            # session 2: controls showed homeostasis (not STDP) is the active ingredient, and wtfs
            # runs cost ~3 h each under load, so wtfs ran for seed 1 only and _whomeo replaced it
            ("sleep", "", "_whomeo", ' --set "stdp={alpha: 0.0, homeo: 0.15}"')]
    out = ["# continual sweep, FULL HYBRID (SNN T=4 + Transformer window 4), fetch3 (150k frames/task)"]
    for s in seeds:
        for method, mk, tag, extra in arms:
            name = f"fetch3_{method}{tag}_s{s}"
            out.append(f"{root}/{name}/results.json :: python continual.py --suite fetch3 --method {method} "
                       f"--config configs/cl_hybrid_v2.yaml --seed {s} --out {root}"
                       + (f" --mk {mk}" if mk else "") + (f" --tag {tag}" if tag else "") + extra)
    return out


STAB_TF = "mode: three_factor, center: true, homeo: 0.15"


def rq1_dfa(seeds5x5, seeds6x6):
    """Session 2, priority 3: DFA as the global term of the hybrid rule.
    dfae = DFA on the SNN synapses only (heads by exact gradients) -> dW_snn = beta*DFA (+ alpha*STDP);
    dfaall = DFA everywhere (heads' hidden layers too) = fully backprop-free variant C with MLP heads."""
    out = ["# RQ1/RQ4 with DFA as the global signal (session 2)"]
    for s in seeds5x5:  # DoorKey-5x5, comparable to runs/rq1
        for arm, st in (("dfae", "dfa=true dfa_scope=encoder"), ("dfaall", "dfa=true dfa_scope=all")):
            name = f"rq1_{arm}_s{s}"
            out.append(f"runs/rq1/{name}/eval.json :: python train.py --config configs/rq1_snn.yaml "
                       f"--set seed={s} run_name={name} out_dir=runs/rq1 {st}")
    for s in seeds6x6:  # DoorKey-6x6, comparable to runs/rq1_v2 (bp, tfs0.03 have 5 seeds)
        for arm, st in (("dfae", ""), ("dfae_tfs0.03", f'"stdp={{alpha: 0.03, {STAB_TF}}}"'),
                        ("dfae_tfs0.3", f'"stdp={{alpha: 0.3, {STAB_TF}}}"')):
            name = f"rq1v2_{arm}_s{s}"
            out.append(f"runs/rq1_v2/{name}/eval.json :: python train.py --config configs/rq1_v2.yaml "
                       f"--set seed={s} run_name={name} out_dir=runs/rq1_v2 dfa=true dfa_scope=encoder {st}".rstrip())
    return out


def rq1_dfa_ctrl(seeds):
    """Controls for 'DFA + stabilised STDP rescues DFA' (session 2): is it the homeostasis,
    the STDP direction, or both? Also backprop + homeostasis only."""
    arms = (("dfae_homeo", "dfa=true dfa_scope=encoder", '"stdp={alpha: 0.0, homeo: 0.15}"'),
            ("dfae_rands0.03", "dfa=true dfa_scope=encoder", '"stdp={alpha: 0.03, mode: random, homeo: 0.15}"'),
            ("homeo", "", '"stdp={alpha: 0.0, homeo: 0.15}"'))
    out = ["# RQ1 v2 controls for the DFA+STDP interaction (session 2), DoorKey-6x6"]
    for s in seeds:
        for arm, base, st in arms:
            name = f"rq1v2_{arm}_s{s}"
            out.append(f"runs/rq1_v2/{name}/eval.json :: python train.py --config configs/rq1_v2.yaml "
                       f"--set seed={s} run_name={name} out_dir=runs/rq1_v2 {base} {st}".replace("  ", " "))
    return out


def cl_dfa(seeds):
    """RQ4: SNN trunk on fetch3 + sleep, trained by DFA + homeostasis (the configuration that
    works on DoorKey-6x6; DFA alone fails there) vs backprop + homeostasis (separates the two)."""
    root = "runs/continual_snn"
    homeo = '"stdp={alpha: 0.0, homeo: 0.15}"'
    out = ["# continual, SNN trunk + sleep: DFA+homeostasis vs backprop+homeostasis, fetch3 (session 2)"]
    for s in seeds:
        for tag, extra in (("_dfaeh", f"dfa=true dfa_scope=encoder {homeo}"), ("_homeo", homeo)):
            name = f"fetch3_sleep{tag}_s{s}"
            out.append(f"{root}/{name}/results.json :: python continual.py --suite fetch3 --method sleep "
                       f"--config configs/cl_snn.yaml --seed {s} --out {root} --tag {tag} --set {extra}")
    return out


def cl_fetch5(seeds, trunk):
    """Session 2, priority 4: 5-task sequence (fetch5) to test whether RQ2/RQ5 hold beyond 3 tasks."""
    root = f"runs/continual5_{trunk}"
    out = [f"# continual sweep, {trunk} trunk, fetch5 (5 tasks x 150k frames)"]
    for s in seeds:
        for method in ("naive", "isolation", "sleep", "replay"):
            name = f"fetch5_{method}_s{s}"
            out.append(f"{root}/{name}/results.json :: python continual.py --suite fetch5 --method {method} "
                       f"--config configs/cl_{trunk}.yaml --seed {s} --out {root}")
    return out


def s3_queue1():
    """Session 3 queue 1: P3a isolation-budget probes, P1 RQ1 controls to 10 seeds, P2 SNN small buffers."""
    out = ["# session 3 queue 1 (see PROGRESS.md 'Session 3' plan)",
           "# P3a: single-task SNN on hard fetch5 tasks, 600k frames, to size the isolation budget"]
    for task, seed in ((0, 1), (2, 1), (3, 1), (4, 2)):
        name = f"probe5_task{task}_s{seed}"
        out.append(f"runs/s3_probes/{name}/eval.json :: python train.py --config configs/cl_snn.yaml "
                   f"--set seed={seed} run_name={name} out_dir=runs/s3_probes env_id=FetchObj5-{task} total_frames=600000")
    out.append("# P1: RQ1 DoorKey-6x6 controls to 10 seeds (seed-major so partial results stay balanced)")
    arms = {"bp": "", "homeo": '"stdp={alpha: 0.0, homeo: 0.15}"',
            "tfs0.03": '"stdp={alpha: 0.03, mode: three_factor, center: true, homeo: 0.15}"',
            "rands0.03": '"stdp={alpha: 0.03, mode: random, homeo: 0.15}"'}
    for s_ in range(1, 11):
        for arm, st in arms.items():
            name = f"rq1v2_{arm}_s{s_}"
            out.append(f"runs/rq1_v2/{name}/eval.json :: python train.py --config configs/rq1_v2.yaml "
                       f"--set seed={s_} run_name={name} out_dir=runs/rq1_v2 {st}".rstrip())
    out.append("# P2: memory-fair RQ5, SNN trunk fetch3, small buffers (2000/task ~ memory-matched to isolation)")
    for s_ in (1, 2, 3):
        for buf in (200, 2000):
            for method in ("sleep", "replay"):
                name = f"fetch3_{method}_buf{buf}_s{s_}"
                out.append(f"runs/continual_snn_budget/{name}/results.json :: python continual.py --suite fetch3 "
                           f"--method {method} --config configs/cl_snn.yaml --seed {s_} --out runs/continual_snn_budget "
                           f"--mk buffer_per_task={buf} --tag _buf{buf}")
    return out


def cl_budget(seeds):
    """RQ2 in the regime where replay is NOT at ceiling: tiny per-task buffers."""
    out = ["# RQ2 budget sweep: replay vs sleep with small per-task buffers, CNN trunk"]
    for s in seeds:
        for buf in (50, 200):
            for method in ("replay", "sleep"):
                tag = f"_buf{buf}"
                name = f"fetch3_{method}{tag}_s{s}"
                out.append(f"runs/continual_budget/{name}/results.json :: python continual.py --suite fetch3 "
                           f"--method {method} --config configs/cl_cnn.yaml --seed {s} --out runs/continual_budget "
                           f"--mk buffer_per_task={buf} --tag {tag}")
    return out


def parse_seeds(x):
    if "-" in x:
        a, b = x.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in x.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1-3")
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    seeds = parse_seeds(a.seeds)
    os.makedirs("jobs", exist_ok=True)
    files = {"rq1": rq1(seeds), "cl_cnn": cl(seeds, "cnn"), "cl_snn": cl(seeds, "snn"),
             "cl_budget": cl_budget(seeds), "rq1_v2": rq1_v2(seeds),
             "cl_hybrid": cl_hybrid(seeds), "rq1_dfa": rq1_dfa(seeds, sorted(set(seeds) | {4, 5})),
             "cl_dfa": cl_dfa(seeds), "rq1_dfa_ctrl": rq1_dfa_ctrl(seeds), "cl5_cnn": cl_fetch5(seeds, "cnn"), "cl5_snn": cl_fetch5(seeds, "snn"),
             "s3_q1": s3_queue1()}
    for name, lines in files.items():
        path = f"jobs/{name}{a.suffix}.txt"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(path, len(lines) - 1, "jobs")


if __name__ == "__main__":
    main()
