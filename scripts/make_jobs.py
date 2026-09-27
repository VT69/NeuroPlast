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
             "cl_budget": cl_budget(seeds), "rq1_v2": rq1_v2(seeds)}
    for name, lines in files.items():
        path = f"jobs/{name}{a.suffix}.txt"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(path, len(lines) - 1, "jobs")


if __name__ == "__main__":
    main()
