"""Online EWC (Kirkpatrick 2017; online variant of Schwarz 2018).

After each task: diagonal Fisher of the policy, E_s E_{a~pi} [(d log pi(a|s) / d theta)^2],
estimated from `n_samples` states visited by the trained policy. Fishers
accumulate as F <- gamma * F + F_task, anchors are reset to the current
weights. The Fisher is normalised to mean 1 over parameters so `lam` has a
comparable meaning across encoders (SNN vs CNN gradient scales differ a lot).

Penalty: lam/2 * sum_i F_i (theta_i - theta*_i)^2
"""
from __future__ import annotations

import torch

from neuroplast.baselines.buffer import collect_states


class _EWCHook:
    def __init__(self, ewc):
        self.ewc = ewc

    def extra_loss(self, agent, batch):
        loss = 0.0
        for n, p in agent.named_parameters():
            if n in self.ewc.fisher:
                loss = loss + (self.ewc.fisher[n] * (p - self.ewc.anchor[n]) ** 2).sum()
        return 0.5 * self.ewc.lam * loss


class EWC:
    def __init__(self, lam=100.0, gamma=1.0, n_samples=500):
        self.lam, self.gamma, self.n_samples = lam, gamma, n_samples
        self.fisher, self.anchor = {}, {}

    def hooks(self, agent, task_idx, cfg):
        return [_EWCHook(self)] if self.fisher else []

    def end_task(self, agent, task_idx, cfg):
        obs, mask, _, _ = collect_states(agent, cfg.env_id, task_idx, self.n_samples)
        params = {n: p for n, p in agent.named_parameters() if p.requires_grad}
        F = {n: torch.zeros_like(p) for n, p in params.items()}
        agent.eval()
        for i in range(len(obs)):
            logits, _ = agent(obs[i:i + 1], mask[i:i + 1], task_idx)
            logp = torch.log_softmax(logits, -1)
            a = torch.distributions.Categorical(logits=logits.detach()).sample()
            agent.zero_grad()
            logp[0, a].sum().backward()
            for n, p in params.items():
                if p.grad is not None:
                    F[n] += p.grad.detach() ** 2
        total = sum(f.sum() for f in F.values())
        count = sum(f.numel() for f in F.values())
        scale = count / total.clamp_min(1e-12)
        for n in F:
            F[n] = F[n] * scale
            self.fisher[n] = self.gamma * self.fisher.get(n, torch.zeros_like(F[n])) + F[n]
            self.anchor[n] = params[n].detach().clone()
        agent.zero_grad()
