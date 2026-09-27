"""Continual RL runner: train one agent on a task sequence, measure forgetting.

    python continual.py --suite fetch3 --method naive --config configs/cl_cnn.yaml --seed 1

Methods (see neuroplast/baselines and neuroplast/sleep):
  naive      sequential fine-tuning (no protection)
  ewc        elastic weight consolidation (online EWC, Fisher from policy + value)
  replay     interleaved experience replay (CLEAR-style behaviour cloning on stored old-task states)
  sleep      periodic offline sleep phases: replay + distillation + optional STDP (neuroplast/sleep)
  isolation  one freshly initialised network per task (zero forgetting, linear params)

Outputs runs/<out>/<suite>_<method>_s<seed>/: results.json with the accuracy matrix
R[i][j] (return on task j after training task i), training curves per task,
metrics (ACC/BWT/FORGET), parameter counts; per-task metrics.csv + final.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import torch

from neuroplast.baselines import make_method
from neuroplast.envs.continual import SUITES
from neuroplast.eval.metrics import cl_metrics
from train import build_agent, evaluate, load_config, train


def n_params(agents):
    return int(sum(p.numel() for a in agents for p in a.parameters()))


def run(suite, method, cfg, seed, out_root="runs/continual", eval_episodes=100, method_kwargs=None, tag=""):
    torch.set_num_threads(cfg.threads)
    tasks = SUITES[suite]
    T = len(tasks)
    name = f"{suite}_{method}{tag}_s{seed}"
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    isolation = method == "isolation"
    if isolation:
        base = replace(cfg, seed=seed, n_tasks=1, multihead=False)
        agents = []
        for i in range(T):
            torch.manual_seed(seed * 100 + i)
            agents.append(build_agent(base))
        agent_for = lambda j: agents[j]
        task_arg = lambda j: None
    else:
        base = replace(cfg, seed=seed, n_tasks=T, multihead=True)
        torch.manual_seed(seed)
        shared = build_agent(base)
        agent_for = lambda j: shared
        task_arg = lambda j: j
    m = make_method(method, **(method_kwargs or {}))
    R, curves, rows_succ, t0 = [], [], [], time.time()
    init_row = [evaluate(agent_for(j), tasks[j], episodes=eval_episodes, task=task_arg(j))["return_mean"]
                for j in range(T)]
    for i, env_id in enumerate(tasks):
        agent = agent_for(i)
        tcfg = replace(base, env_id=env_id, task_id=i if not isolation else 0, run_name=f"{name}/task{i}")
        hooks = m.hooks(agent, i, tcfg) if hasattr(m, "hooks") else []
        if base.stdp:  # wake-time hybrid rule (RQ1) composes with any CL method
            from neuroplast.learning.hybrid import HybridSTDP
            hooks = [HybridSTDP(**base.stdp)] + list(hooks)
        _, hist = train(tcfg, agent, hooks=hooks, run_dir=os.path.join(out, f"task{i}"), log_prefix=f"[{name} t{i}] ")
        curves.append([h["ep_return"] for h in hist])
        if hasattr(m, "end_task"):
            m.end_task(agent, i, tcfg)
        evals = [evaluate(agent_for(j), tasks[j], episodes=eval_episodes, task=task_arg(j)) for j in range(T)]
        R.append([e["return_mean"] for e in evals])
        rows_succ.append([e["success"] for e in evals])
        print(f"[{name}] after task {i}: R = {[round(x, 3) for x in R[-1]]}", flush=True)
        res = dict(suite=suite, method=method, tag=tag, seed=seed, tasks=tasks, R=R, success=rows_succ, init=init_row,
                   curves=curves, metrics=cl_metrics(R) if len(R) == T else None,
                   params=n_params(agents if isolation else [shared]),
                   params_per_task=n_params([agents[0]]) if isolation else n_params([shared]),
                   extra_params=getattr(m, "extra_params", 0), replay_samples=getattr(m, "replay_samples", 0),
                   sleep_phases=getattr(m, "n_phases", 0), wall_time=time.time() - t0,
                   cfg=asdict(base), method_kwargs=method_kwargs or {})
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(res, f, indent=1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="fetch3")
    p.add_argument("--method", default="naive")
    p.add_argument("--config", default="configs/cl_cnn.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--mk", nargs="*", default=[], help="method kwargs key=value")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="runs/continual")
    p.add_argument("--eval-episodes", type=int, default=100)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    import yaml
    mk = {k: yaml.safe_load(v) for k, v in (x.split("=", 1) for x in a.mk)}
    res = run(a.suite, a.method, cfg, a.seed, a.out, a.eval_episodes, mk, a.tag)
    print("FINAL", json.dumps(res["metrics"]), "params", res["params"])


if __name__ == "__main__":
    main()
