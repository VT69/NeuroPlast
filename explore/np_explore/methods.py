"""New continual-learning arms for the exploration phase (ideas 1 and 2).

PackNet (Mallya & Lazebnik 2018), adapted to our multi-head PPO agent
  * Only the shared encoder is packed; heads and task-embedding rows are already per task.
  * Each encoder weight tensor (ndim >= 2) gets an owner map: 0 = free, k = owned by task k-1.
  * Task t trains the free weights for the first `prune_frac` of its updates. Then, per layer, it keeps the
    largest-magnitude free weights: an equal share of what is left, F / (T - t), so every task gets about 1/T of
    the encoder. The rest are zeroed and stay free, and the kept ones are retrained for the remaining updates.
    Pruning and retraining happen inside the task's normal frame budget; no extra frames are used.
  * Weights owned by earlier tasks are frozen, and encoder biases (and any other non-matrix encoder parameters)
    are frozen after task 0, as in the paper. Frozen values are restored after every optimizer step, so Adam
    momentum cannot move them.
  * Task j is evaluated with the weights owned by tasks <= j only (free weights are zero outside training).
  * The last task keeps all remaining free weights and is not pruned.
  * Storage: one network + ceil(log2(T+1)) bits per packed weight for the owner map (`extra_bits`).

LwF (Li & Hoiem 2016/2017), "buffer-free distillation"
  * At the start of task t >= 1, freeze a snapshot of the whole network (all old heads).
  * On every PPO minibatch, take `batch` of the *current-task* states and give each a uniformly random old task
    j < t. Then add the same loss replay uses (KL(snapshot_j || student_j) + 0.5 * value MSE) to the PPO loss.
  * No states are stored. The replayed-sample counter matches replay's: `batch` per minibatch.
  * Persistent memory = one network; peak memory during training = two networks (student + snapshot).
"""
from __future__ import annotations

import math

import torch

from neuroplast.baselines import make_method as _base_make_method
from neuroplast.baselines.replay import distill_loss
from neuroplast.models.agent import snapshot


# ---------------------------------------------------------------- PackNet
class PackNet:
    def __init__(self, prune_frac=0.75, prune_start=None, prune_steps=1):
        """prune_steps=1: one-shot pruning at prune_frac of the task's updates (pilot round 1; this collapsed task 0
        on fetch3 s101). prune_steps>1: gradual magnitude pruning (Zhu & Gupta 2017) in prune_steps equal steps
        between prune_start and prune_frac; the final capacity split is the same."""
        self.prune_frac, self.prune_start, self.prune_steps = prune_frac, prune_start, prune_steps
        self.owner = None      # name -> int8 owner map (0 = free)
        self.T = None
        self.current = 0
        self.extra_bits = 0
        self.kept = []         # per task: fraction of packed weights it owns

    def _packed(self, agent):
        return {n: p for n, p in agent.encoder.named_parameters() if p.ndim >= 2}

    def _others(self, agent):
        return {n: p for n, p in agent.encoder.named_parameters() if p.ndim < 2}

    def _install(self, agent, T):
        self.T = T
        self.owner = {n: torch.zeros_like(p, dtype=torch.int8) for n, p in self._packed(agent).items()}
        n_w = sum(o.numel() for o in self.owner.values())
        self.extra_bits = n_w * math.ceil(math.log2(T + 1))
        packnet, fwd = self, agent.forward

        def forward(obs, mask, task=None, past_feats=None):
            # evaluate an *earlier* task with only the weights it owned (later tasks' weights masked out)
            if isinstance(task, int) and task < packnet.current:
                params = packnet._packed(agent)
                saved = {n: p.data.clone() for n, p in params.items()}
                with torch.no_grad():
                    for n, p in params.items():
                        own = packnet.owner[n]
                        p.data.mul_(((own >= 1) & (own <= task + 1)).to(p.dtype))
                try:
                    return fwd(obs, mask, task, past_feats)
                finally:
                    for n, p in params.items():
                        p.data.copy_(saved[n])
            return fwd(obs, mask, task, past_feats)
        agent.forward = forward

    def hooks(self, agent, task_idx, cfg):
        if self.owner is None:
            self._install(agent, cfg.n_tasks)
        self.current = task_idx
        return [_PackNetHook(self, agent, task_idx, cfg)]

    def end_task(self, agent, task_idx, cfg):
        if task_idx == self.T - 1:  # last task keeps every remaining free weight
            for n, o in self.owner.items():
                o[o == 0] = task_idx + 1
        n_w = sum(o.numel() for o in self.owner.values())
        self.kept.append(sum(int((o == task_idx + 1).sum()) for o in self.owner.values()) / n_w)
        self.current = task_idx + 1


class _PackNetHook:
    def __init__(self, pn, agent, t, cfg):
        self.pn, self.t = pn, t
        self.num_updates = max(cfg.total_frames // (cfg.num_envs * cfg.num_steps), 1)
        last = t == pn.T - 1
        if last:
            self.schedule = []
        elif pn.prune_steps <= 1:
            self.schedule = [max(1, round(pn.prune_frac * self.num_updates))]
        else:
            a, b = pn.prune_start * self.num_updates, pn.prune_frac * self.num_updates
            self.schedule = [max(1, round(a + (b - a) * i / (pn.prune_steps - 1))) for i in range(pn.prune_steps)]
        self.cand = {n: (pn.owner[n] == 0).clone() for n in pn._packed(agent)}   # weights this task may keep
        self.n_cand = {n: int(c.sum()) for n, c in self.cand.items()}
        self._freeze(agent)

    def _freeze(self, agent):
        self.frozen = {}
        for n, p in self.pn._packed(agent).items():
            train = self.pn.owner[n] == 0
            self.frozen[n] = (train, p.data.clone())
        for n, p in self.pn._others(agent).items():
            train = torch.full_like(p, self.t == 0, dtype=torch.bool)
            self.frozen[n] = (train, p.data.clone())

    @torch.no_grad()
    def after_step(self, agent, batch, update):
        params = dict(agent.encoder.named_parameters())
        for n, (train, val) in self.frozen.items():
            p = params[n]
            p.data.copy_(torch.where(train, p.data, val))

    @torch.no_grad()
    def after_update(self, agent, update, stats):
        if update not in self.schedule:
            return
        step = self.schedule.index(update) + 1
        final = step == len(self.schedule)
        remaining = self.pn.T - self.t
        for n, p in self.pn._packed(agent).items():
            cand = self.cand[n]
            target = self.n_cand[n] / remaining                     # final share: equal split of what was free
            k = int(round(self.n_cand[n] - (self.n_cand[n] - target) * step / len(self.schedule)))
            mag = torch.where(cand, p.data.abs(), torch.full_like(p.data, -1.0)).flatten()
            keep = torch.zeros_like(cand).flatten()
            keep[mag.topk(k).indices] = True
            keep = keep.view_as(cand) & cand
            p.data[cand & ~keep] = 0.0
            cand &= keep
            if final:
                self.pn.owner[n][keep] = self.t + 1
        # free weights not in cand are pruned: frozen at zero; kept candidates stay trainable
        self.frozen = {}
        for n, p in self.pn._packed(agent).items():
            self.frozen[n] = (self.cand[n].clone(), p.data.clone())
        for n, p in self.pn._others(agent).items():
            self.frozen[n] = (torch.full_like(p, self.t == 0, dtype=torch.bool), p.data.clone())


# ---------------------------------------------------------------- LwF
class LwF:
    def __init__(self, batch=128, coef=1.0, teacher_int8=False):
        """teacher_int8: store the snapshot with every weight matrix quantised to symmetric per-tensor int8
        (biases fp32), so the transient peak is about 1.25 networks instead of 2 (int8 storage audit: no
        measurable ACC change on these networks)."""
        self.batch, self.coef, self.teacher_int8 = batch, coef, teacher_int8
        self.replay_samples = 0

    def hooks(self, agent, task_idx, cfg):
        if task_idx == 0:
            return []
        teacher = snapshot(agent)
        if self.teacher_int8:
            with torch.no_grad():
                for p in teacher.parameters():
                    if p.ndim >= 2:
                        sc = p.abs().max() / 127.0
                        if sc > 0:
                            p.copy_((p / sc).round().clamp(-127, 127) * sc)
        return [_LwFHook(self, teacher, task_idx)]


class _LwFHook:
    def __init__(self, owner, teacher, t):
        self.owner, self.teacher, self.t = owner, teacher, t

    def extra_loss(self, agent, batch):
        obs, mask = batch["obs"], batch["mask"]
        n = min(self.owner.batch, len(obs))
        idx = torch.randperm(len(obs))[:n]
        tasks = torch.randint(0, self.t, (n,))
        with torch.no_grad():
            logits_t, value_t = self.teacher(obs[idx], mask[idx], tasks)
        self.owner.replay_samples += n
        return self.owner.coef * distill_loss(agent, obs[idx], mask[idx], tasks, logits_t, value_t)


def make_method(name, **kw):
    if name == "packnet":
        return PackNet(**kw)
    if name == "lwf":
        return LwF(**kw)
    return _base_make_method(name, **kw)
