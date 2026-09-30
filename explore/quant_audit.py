"""Idea 3: int8 storage audit of the memory frontier (read-only use of frozen checkpoints).

For each frozen checkpoint, evaluate the network in fp32 and with every weight matrix (ndim >= 2) replaced by its
symmetric per-tensor int8 quantisation (biases and other 1-D parameters stay fp32). Evaluation: the task's
100 held-out episodes with the same environment seeds and the same action-sampling RNG state, so the two
evaluations are paired. Checkpoints: runs/<root>/<run>/task<i>/final.pt. For isolation, task i's network is
evaluated on task i. For shared agents, the last task's checkpoint is evaluated on every task. For sleep, that
checkpoint precedes the final end-of-task sleep phase, which is fine for a paired fp32-vs-int8 drop.

Writes explore/quant/quant_audit.json and prints a summary.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
import time
from dataclasses import fields

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from neuroplast.envs.continual import SUITES  # noqa: E402
from train import Config, build_agent, evaluate  # noqa: E402

NAMES = {f.name for f in fields(Config)}


def load(path):
    ck = torch.load(path, map_location="cpu")
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in NAMES})
    agent = build_agent(cfg)
    agent.load_state_dict(ck["agent"])
    return agent.eval(), cfg


def quantise(agent):
    q = copy.deepcopy(agent)
    with torch.no_grad():
        for p in q.parameters():
            if p.ndim >= 2:
                s = p.abs().max() / 127.0
                if s > 0:
                    p.copy_((p / s).round().clamp(-127, 127) * s)
    return q.eval()


def ev(agent, env_id, task):
    torch.manual_seed(0)
    return evaluate(agent, env_id, episodes=100, task=task)["return_mean"]


def main():
    t0 = time.time()
    torch.set_num_threads(1)
    jobs = [("CNN fetch3", "runs/continual", "fetch3", ["isolation", "replay", "sleep"]),
            ("CNN fetch3 small buffer", "runs/continual_budget", "fetch3", ["sleep_buf200"]),
            ("SNN fetch3", "runs/continual_snn", "fetch3", ["isolation", "sleep"]),
            ("SNN fetch3 small buffer", "runs/continual_snn_budget", "fetch3", ["sleep_buf200"]),
            ("CNN fetch5", "runs/continual5_cnn", "fetch5", ["isolation", "sleep"])]
    out = []
    for label, root, suite, methods in jobs:
        tasks = SUITES[suite]
        for m in methods:
            for run in sorted(glob.glob(f"{root}/{suite}_{m}_s[0-9]*")):
                if os.path.basename(run).rsplit("_s", 1)[0] != f"{suite}_{m}":
                    continue
                fp, q8 = [], []
                for j, env_id in enumerate(tasks):
                    if m == "isolation":
                        a, _ = load(f"{run}/task{j}/final.pt")
                        task = None
                    else:
                        if j == 0:
                            shared, _ = load(f"{run}/task{len(tasks) - 1}/final.pt")
                            shared_q = quantise(shared)
                        a, task = shared, j
                    fp.append(ev(a, env_id, task))
                    q8.append(ev(quantise(a) if m == "isolation" else shared_q, env_id, task))
                rec = dict(label=label, run=run, method=m, fp32=fp, int8=q8, acc_fp32=float(np.mean(fp)),
                           acc_int8=float(np.mean(q8)))
                out.append(rec)
                print(f"{label:24s} {os.path.basename(run):24s} ACC fp32 {rec['acc_fp32']:.3f} int8 {rec['acc_int8']:.3f} "
                      f"drop {rec['acc_fp32'] - rec['acc_int8']:+.3f}", flush=True)
    json.dump(out, open("explore/quant/quant_audit.json", "w"), indent=1)
    core_h = (time.time() - t0) / 3600
    extra = "explore/budget_extra.json"
    led = json.load(open(extra)) if os.path.exists(extra) else []
    led.append(dict(name="quant_audit", core_h=core_h))
    json.dump(led, open(extra, "w"), indent=1)
    print(f"done in {core_h:.2f} core-h")


if __name__ == "__main__":
    main()
