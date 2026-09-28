"""RQ4 compute side: cost of each architecture variant, measured the same way.

For every variant: parameter count, inference ops/energy per decision (the
encoder op accounting from neuroplast/models/encoders.py, on real DoorKey-6x6
observations), and CPU time (process time, 1 thread, so it's robust to other
jobs sharing the machine) for one rollout forward (batch 16) and one PPO
training step (forward+backward, minibatch 256), plus the local-rule overhead
where the variant has one. Writes results/compute.md and results/compute.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroplast.envs.minigrid_env import VecEnv  # noqa: E402
from neuroplast.learning.hybrid import HybridSTDP  # noqa: E402
from train import Config, build_agent  # noqa: E402

VARIANTS = {
    "CNN": dict(encoder="cnn"),
    "A: SNN e2e (surrogate BPTT)": dict(encoder="snn", enc_kwargs={"T": 4}),
    "A+mem: SNN + Transformer (window 4)": dict(encoder="snn", enc_kwargs={"T": 4}, memory=True, window=4),
    "A+STDP: hybrid rule (stabilised, alpha 0.3)": dict(encoder="snn", enc_kwargs={"T": 4},
                                                        stdp={"alpha": 0.3, "mode": "three_factor", "center": True, "homeo": 0.15}),
    "B: local-only SNN encoder + backprop heads": dict(encoder="snn", enc_kwargs={"T": 4},
                                                       stdp={"alpha": 0.3, "mode": "three_factor", "beta": 0, "center": True, "homeo": 0.15}),
    "C: DFA SNN + linear readouts (no backprop)": dict(encoder="snn", enc_kwargs={"T": 4}, dfa=True, hidden=0),
    "Cmlp: DFA SNN + MLP heads (no backprop)": dict(encoder="snn", enc_kwargs={"T": 4}, dfa=True, dfa_scope="all"),
    "Cenc: DFA on SNN synapses + homeostasis, heads exact": dict(encoder="snn", enc_kwargs={"T": 4}, dfa=True,
                                                                 dfa_scope="encoder",
                                                                 stdp={"alpha": 0.0, "homeo": 0.15}),
}


def cpu_time(fn, reps):
    fn()  # warm-up
    t = time.process_time()
    for _ in range(reps):
        fn()
    return (time.process_time() - t) / reps


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    envs = VecEnv("MiniGrid-DoorKey-6x6-v0", 16, seed=0, history=4)
    o, m = envs.reset()
    obs, masks = [], []
    for _ in range(16):  # 256 real observation windows
        obs.append(torch.as_tensor(o)), masks.append(torch.as_tensor(m))
        o, m, *_ = envs.step(torch.randint(0, 7, (16,)).numpy())
    obs, masks = torch.cat(obs), torch.cat(masks)
    rows = []
    for name, kw in VARIANTS.items():
        stdp = kw.pop("stdp", None)
        cfg = Config(**kw)
        agent = build_agent(cfg)
        hook = None
        if stdp:
            hook = HybridSTDP(**stdp)
            hook.setup(agent)
        opt = torch.optim.Adam(agent.parameters(), 1e-3)
        n_params = sum(p.numel() for p in agent.parameters())
        # inference ops / energy per decision (encoder; x window frames for memory agents)
        agent.encoder.track = True
        with torch.no_grad():
            agent(obs[:64], masks[:64])
        st = dict(agent.encoder.stats)
        agent.encoder.track = False
        frames_per_decision = 1  # memory agents reuse cached features for older frames

        pf = torch.zeros(len(obs), agent.window, agent.feat_dim) if agent.use_memory else None

        def rollout():
            with torch.no_grad():
                agent(obs[:16], masks[:16], past_feats=pf[:16] if pf is not None else None)

        def train_step():
            logits, v = agent(obs, masks, past_feats=pf)
            loss = logits.logsumexp(-1).mean() + v.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hook is not None:
                hook.after_step(agent, dict(adv=torch.randn(len(obs)), mask=masks, idx=torch.arange(len(obs))), 0)

        r = dict(variant=name, params=n_params,
                 enc_ops_k_per_decision=st["sparse_ops"] * frames_per_decision / 1e3,
                 enc_energy_nj_per_decision=st["energy_pj"] * frames_per_decision / 1e3,
                 rollout_ms_per_16=1e3 * cpu_time(rollout, 20),
                 train_step_ms_per_256=1e3 * cpu_time(train_step, 5))
        print(json.dumps(r), flush=True)
        rows.append(r)
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/compute.json", "w"), indent=1)
    import pandas as pd
    with open("results/compute.md", "w") as f:
        f.write("# RQ4 compute per variant\n\nCPU process time, 1 thread; encoder energy uses the RQ3 "
                "accounting (CNN: dense MACs x 4.6 pJ; SNN: SynOps x 0.9 pJ) at initialisation "
                "(trained SNNs are sparser, see results/rq3*.md). Timings were measured while other jobs ran; they vary by ~15% with machine load. Memory agents encode one new frame per decision "
                "(cached features for the rest of the window); their train step re-encodes only the newest frame too.\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g") + "\n")


if __name__ == "__main__":
    main()
