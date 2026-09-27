"""Interleaved experience replay for continual PPO (CLEAR-style, Rolnick 2019).

PPO is on-policy, so old-task transitions can't be replayed through the PPO
loss directly. Following CLEAR, replay = behaviour cloning on stored old-task
states: KL(pi_stored || pi_current) + value regression to the stored values,
added to every PPO minibatch while learning later tasks.

Budget: `buffer_per_task` states stored per task, one replay minibatch of
`batch` states per PPO minibatch (these numbers are matched by the sleep arm).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from neuroplast.baselines.buffer import TaskBuffer, collect_states


def distill_loss(agent, obs, mask, task, logits_t, value_t, vf_coef=0.5):
    logits, value = agent(obs, mask, task)
    kl = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(logits_t, -1), log_target=True,
                  reduction="batchmean")
    return kl + vf_coef * F.mse_loss(value, value_t)


class _ReplayHook:
    def __init__(self, owner):
        self.owner = owner
        self.buf, self.batch, self.coef = owner.buf, owner.batch, owner.coef

    def extra_loss(self, agent, batch):
        self.owner.replay_samples += self.batch
        o, m, t, l, v = self.buf.sample(self.batch)
        return self.coef * distill_loss(agent, o, m, t, l, v)


class Replay:
    def __init__(self, buffer_per_task=5000, batch=128, coef=1.0):
        self.buf = TaskBuffer()
        self.buffer_per_task, self.batch, self.coef = buffer_per_task, batch, coef
        self.replay_samples = 0

    def hooks(self, agent, task_idx, cfg):
        return [_ReplayHook(self)] if len(self.buf) else []

    def end_task(self, agent, task_idx, cfg):
        o, m, l, v = collect_states(agent, cfg.env_id, task_idx, self.buffer_per_task)
        self.buf.add(o, m, task_idx, l, v)
