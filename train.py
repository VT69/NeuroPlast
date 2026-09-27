"""CleanRL-style single-file PPO for MiniGrid, shared by every experiment.

Usage:
    python train.py --config configs/cnn_doorkey.yaml [--set key=value ...]

Writes to runs/<run_name>/: config.yaml, metrics.csv (one row per update),
ckpt.pt (latest, overwritten every `ckpt_every` updates) and final.pt.

The training loop is exposed as `train(cfg, agent=None, ...)` so the continual
learning scripts can call it task after task on the same agent, and so plug-ins
(STDP hybrid rule, EWC penalty, sleep phases) can hook in via `hooks`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import yaml

from neuroplast.envs.minigrid_env import VecEnv
from neuroplast.models.agent import Agent, FeatureCache


@dataclass
class Config:
    run_name: str = "debug"
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    env_kwargs: dict = field(default_factory=dict)
    seed: int = 1
    total_frames: int = 300_000
    num_envs: int = 16
    num_steps: int = 128
    lr: float = 1e-3
    anneal_lr: bool = True
    target_kl: float | None = 0.03  # early-stop PPO epochs when approx KL exceeds this
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    act_reg: float = 0.0          # sparsity penalty on encoder activity (spike rate / L1)
    # agent
    encoder: str = "cnn"
    enc_kwargs: dict = field(default_factory=dict)
    memory: bool = False
    window: int = 1
    mem_kwargs: dict = field(default_factory=dict)
    mem_cache: bool = True        # reuse rollout-time features for older window frames (see Agent.features)
    feat_dim: int = 128
    n_tasks: int = 1
    task_id: int = 0
    multihead: bool = False
    hidden: int = 64              # head MLP width; 0 = linear readouts
    dfa: bool = False             # variant C: DFA/e-prop-style training, no backprop between layers
    # hybrid learning rule (RQ1): see neuroplast/learning/hybrid.py
    stdp: dict = field(default_factory=dict)
    # bookkeeping
    threads: int = 1
    ckpt_every: int = 20
    out_dir: str = "runs"


def load_config(path=None, overrides=()):
    d = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    for ov in overrides:
        k, v = ov.split("=", 1)
        d[k] = yaml.safe_load(v)
    return Config(**d)


def build_agent(cfg: Config) -> Agent:
    return Agent(encoder=cfg.encoder, enc_kwargs=cfg.enc_kwargs, memory=cfg.memory, window=cfg.window,
                 mem_kwargs=cfg.mem_kwargs, feat_dim=cfg.feat_dim, n_tasks=cfg.n_tasks,
                 multihead=cfg.multihead, hidden=cfg.hidden, dfa=cfg.dfa)


def train(cfg: Config, agent: Agent | None = None, hooks=(), log_prefix="", run_dir=None, quiet=False):
    """Train `agent` (built from cfg if None) with PPO. Returns (agent, history).

    hooks: objects with optional methods
        extra_loss(agent, batch) -> tensor     (added to the PPO loss)
        after_step(agent, batch, update)       (after each optimizer step)
        after_update(agent, update, stats)     (after each PPO update)
    """
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_dir = run_dir or os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(asdict(cfg), f)

    agent = agent or build_agent(cfg)
    for h in hooks:
        if hasattr(h, "setup"):
            h.setup(agent)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    K = max(cfg.window if cfg.memory else 1, 1)
    envs = VecEnv(cfg.env_id, cfg.num_envs, seed=cfg.seed * 10_000, history=K, env_kwargs=cfg.env_kwargs)
    N, S = cfg.num_envs, cfg.num_steps
    obs_buf = torch.zeros((S, N, K, 7, 7, 3), dtype=torch.uint8)
    mask_buf = torch.zeros((S, N, K), dtype=torch.bool)
    act_buf = torch.zeros((S, N), dtype=torch.long)
    logp_buf = torch.zeros((S, N))
    rew_buf = torch.zeros((S, N))
    done_buf = torch.zeros((S, N))
    val_buf = torch.zeros((S, N))
    task = cfg.task_id if cfg.n_tasks > 1 else None
    cache = FeatureCache(N, K, agent.feat_dim) if (agent.use_memory and cfg.mem_cache) else None
    feat_buf = torch.zeros((S, N, K, agent.feat_dim)) if cache is not None else None

    o, m = envs.reset()
    next_obs, next_mask = torch.as_tensor(o), torch.as_tensor(m)
    num_updates = max(cfg.total_frames // (N * S), 1)
    history = []
    recent = []  # finished episodes (return, length)
    t0 = time.time()
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = None

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = cfg.lr * (1 - (update - 1) / num_updates)
        agent.eval()
        for step in range(S):
            obs_buf[step], mask_buf[step] = next_obs, next_mask
            with torch.no_grad():
                a, logp, _, v, _ = agent.get_action_and_value(
                    next_obs, next_mask, task=task, past_feats=cache.f if cache else None)
            if cache is not None:
                cache.write_newest(agent.last_frame_feat)
                feat_buf[step] = cache.f
            act_buf[step], logp_buf[step], val_buf[step] = a, logp, v
            o, m, r, d, fin = envs.step(a.numpy())
            if cache is not None:
                cache.advance(d)
            rew_buf[step], done_buf[step] = torch.as_tensor(r), torch.as_tensor(d)
            next_obs, next_mask = torch.as_tensor(o), torch.as_tensor(m)
            recent.extend(fin)

        # GAE (done_buf[t] = episode ended at step t -> no bootstrap from t+1)
        with torch.no_grad():
            _, _, _, next_v, _ = agent.get_action_and_value(next_obs, next_mask, task=task,
                                                             past_feats=cache.f if cache else None)
            adv = torch.zeros_like(rew_buf)
            last = 0.0
            for t in reversed(range(S)):
                nv = next_v if t == S - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * nv * nonterm - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                adv[t] = last
            ret = adv + val_buf

        b_obs, b_mask = obs_buf.flatten(0, 1), mask_buf.flatten(0, 1)
        b_feat = feat_buf.flatten(0, 1) if cache is not None else None
        b_act, b_logp, b_adv, b_ret, b_val = (x.flatten() for x in (act_buf, logp_buf, adv, ret, val_buf))
        B = N * S
        agent.train()
        clipfracs, stats = [], {}
        stop = False
        for epoch in range(cfg.update_epochs):
            if stop:
                break
            perm = torch.randperm(B)
            for start in range(0, B, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                _, newlogp, ent, newv, _ = agent.get_action_and_value(
                    b_obs[idx], b_mask[idx], b_act[idx], task=task, past_feats=b_feat[idx] if cache else None)
                logratio = newlogp - b_logp[idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                mb_adv = b_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_loss = 0.5 * ((newv - b_ret[idx]) ** 2).mean()
                ent_loss = ent.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss
                if cfg.act_reg:
                    loss = loss + cfg.act_reg * agent.encoder.act_reg
                batch = dict(idx=idx, obs=b_obs[idx], mask=b_mask[idx], adv=mb_adv, task=task)
                for h in hooks:
                    if hasattr(h, "extra_loss"):
                        loss = loss + h.extra_loss(agent, batch)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
                for h in hooks:
                    if hasattr(h, "after_step"):
                        h.after_step(agent, batch, update)
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    stop = True
                    break

        frames = update * B
        rets = [e[0] for e in recent[-100:]]
        stats = dict(update=update, frames=frames, fps=int(frames / (time.time() - t0)),
                     ep_return=float(np.mean(rets)) if rets else float("nan"),
                     ep_len=float(np.mean([e[1] for e in recent[-100:]])) if rets else float("nan"),
                     n_episodes=len(recent), pg_loss=pg_loss.item(), v_loss=v_loss.item(),
                     entropy=ent_loss.item(), approx_kl=approx_kl.item(), clipfrac=float(np.mean(clipfracs)),
                     enc_activity=float(agent.encoder.act_reg.detach()))
        recent = recent[-100:]
        for h in hooks:
            if hasattr(h, "after_update"):
                h.after_update(agent, update, stats)
        history.append(stats)
        if writer is None:
            writer = csv.DictWriter(csv_f, fieldnames=list(stats.keys()))
            writer.writeheader()
        writer.writerow({k: stats.get(k) for k in writer.fieldnames})
        csv_f.flush()
        if not quiet and (update % 5 == 0 or update == num_updates):
            print(f"{log_prefix}[{cfg.run_name}] upd {update}/{num_updates} frames {frames} fps {stats['fps']} "
                  f"ret {stats['ep_return']:.3f} len {stats['ep_len']:.1f} ent {stats['entropy']:.3f} "
                  f"act {stats['enc_activity']:.3f}", flush=True)
        if update % cfg.ckpt_every == 0:
            torch.save({"agent": agent.state_dict(), "update": update, "cfg": asdict(cfg)},
                       os.path.join(run_dir, "ckpt.pt"))
    csv_f.close()
    torch.save({"agent": agent.state_dict(), "cfg": asdict(cfg)}, os.path.join(run_dir, "final.pt"))
    if os.path.exists(os.path.join(run_dir, "ckpt.pt")):
        os.remove(os.path.join(run_dir, "ckpt.pt"))  # final.pt supersedes it
    return agent, history


@torch.no_grad()
def evaluate(agent: Agent, env_id: str, episodes=100, seed=12345, task=None, greedy=False,
             env_kwargs=None, num_envs=16):
    """Mean return / success over `episodes` fresh episodes (held-out seeds)."""
    agent.eval()
    K = agent.window
    envs = VecEnv(env_id, num_envs, seed=seed, history=K, env_kwargs=env_kwargs)
    cache = FeatureCache(num_envs, K, agent.feat_dim) if agent.use_memory else None
    o, m = envs.reset()
    done_eps = []
    while len(done_eps) < episodes:
        a, *_ = agent.get_action_and_value(torch.as_tensor(o), torch.as_tensor(m), task=task, greedy=greedy,
                                           past_feats=cache.f if cache else None)
        if cache is not None:
            cache.write_newest(agent.last_frame_feat)
        o, m, _, d, fin = envs.step(a.numpy())
        if cache is not None:
            cache.advance(d)
        done_eps.extend(fin)
    rets = np.array([e[0] for e in done_eps[:episodes]])
    return dict(return_mean=float(rets.mean()), success=float((rets > 0).mean()),
                len_mean=float(np.mean([e[1] for e in done_eps[:episodes]])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML-parsed)")
    p.add_argument("--eval-episodes", type=int, default=200)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    hooks = []
    if cfg.stdp:
        from neuroplast.learning.hybrid import HybridSTDP
        hooks.append(HybridSTDP(**cfg.stdp))
    agent, hist = train(cfg, hooks=hooks)
    res = evaluate(agent, cfg.env_id, episodes=args.eval_episodes, task=cfg.task_id if cfg.n_tasks > 1 else None,
                   env_kwargs=cfg.env_kwargs)
    res["train_frames"] = hist[-1]["frames"]
    res["final_train_return"] = hist[-1]["ep_return"]
    print("EVAL", json.dumps(res))
    with open(os.path.join(cfg.out_dir, cfg.run_name, "eval.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
