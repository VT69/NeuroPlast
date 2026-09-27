"""Per-task state buffer with the policy's outputs at the end of that task.

Used by interleaved replay (CLEAR-style behaviour cloning) and by the sleep
phase. Stores raw observation windows + masks (so replay re-encodes them with
the *current* encoder), the task id, and the policy logits / values at
storage time (the distillation targets).
"""
from __future__ import annotations

import torch

from neuroplast.envs.minigrid_env import VecEnv


@torch.no_grad()
def collect_states(agent, env_id, task, n, seed=4242, num_envs=16, greedy=False):
    """Run the agent's policy and return n (obs, mask, logits, value) samples."""
    agent.eval()
    envs = VecEnv(env_id, num_envs, seed=seed, history=agent.window)
    o, m = envs.reset()
    O, M, L, V = [], [], [], []
    while sum(len(x) for x in O) < n:
        ot, mt = torch.as_tensor(o), torch.as_tensor(m)
        logits, v = agent(ot, mt, task)
        O.append(ot.clone()), M.append(mt.clone()), L.append(logits), V.append(v)
        a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        o, m, _, _, _ = envs.step(a.numpy())
    cat = lambda xs: torch.cat(xs)[:n]
    return cat(O), cat(M), cat(L), cat(V)


class TaskBuffer:
    def __init__(self):
        self.obs, self.mask, self.task, self.logits, self.value = [], [], [], [], []

    def add(self, obs, mask, task, logits, value):
        self.obs.append(obs), self.mask.append(mask), self.logits.append(logits), self.value.append(value)
        self.task.append(torch.full((len(obs),), task, dtype=torch.long))
        self._cat()

    def _cat(self):
        self.O = torch.cat(self.obs)
        self.M = torch.cat(self.mask)
        self.T = torch.cat(self.task)
        self.L = torch.cat(self.logits)
        self.V = torch.cat(self.value)

    def __len__(self):
        return 0 if not self.obs else len(self.O)

    def sample(self, b):
        idx = torch.randint(0, len(self), (b,))
        return self.O[idx], self.M[idx], self.T[idx], self.L[idx], self.V[idx]
