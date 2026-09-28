"""Full agent: encoder (CNN | SNN) -> [Transformer working memory] -> actor/critic.

Observations arrive as a window of raw uint8 MiniGrid frames (B, K, 7, 7, 3)
plus a validity mask (B, K). Without memory only the newest frame is encoded.

Task conditioning for continual learning: `n_tasks > 1` adds a learned task
embedding (added to the features) and, if `multihead`, one actor/critic head
pair per task (standard task-incremental setup; the shared trunk is what can
forget / transfer).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from neuroplast.envs.minigrid_env import N_ACTIONS, obs_to_onehot
from neuroplast.models.encoders import make_encoder
from neuroplast.models.heads.actor_critic import ActorCriticHeads
from neuroplast.models.memory.transformer import WorkingMemory


class Agent(nn.Module):
    def __init__(self, encoder="cnn", enc_kwargs=None, memory=False, window=1, mem_kwargs=None,
                 feat_dim=128, n_tasks=1, multihead=False, task_embed=True, hidden=64, dfa=False,
                 dfa_scope="all"):
        super().__init__()
        enc_kwargs = dict(enc_kwargs or {})
        enc_kwargs.setdefault("out_dim", feat_dim)
        self.encoder = make_encoder(encoder, **enc_kwargs)
        self.use_memory = memory
        self.feat_dim = feat_dim
        self.window = window if memory else 1
        if memory:
            self.memory = WorkingMemory(feat_dim, window, **(mem_kwargs or {}))
        self.n_tasks = n_tasks
        self.task_emb = nn.Embedding(n_tasks, feat_dim) if (n_tasks > 1 and task_embed) else None
        if self.task_emb is not None:
            nn.init.normal_(self.task_emb.weight, std=0.1)
        self.dfa = None
        if dfa:  # variant C: no backprop between layers (see neuroplast/learning/dfa.py)
            assert encoder == "snn" and not memory, "DFA agent = SNN encoder, no Transformer memory"
            from neuroplast.learning.dfa import DFAContext
            c = self.encoder.convs
            sizes = [c[0].out_channels * 36, c[1].out_channels * 25, c[2].out_channels * 16, feat_dim]
            # dfa_scope "all": every hidden layer (SNN + head MLPs + task-conditioned
            # features) learns from random feedback -> no backprop anywhere.
            # "encoder": only the SNN synapses get the DFA signal (the "global term" of
            # the hybrid rule); the small non-spiking heads keep exact gradients.
            assert dfa_scope in ("all", "encoder")
            self.dfa_scope = dfa_scope
            extra = {}
            if dfa_scope == "all":
                extra = {"ha": hidden, "hc": hidden} if hidden else {}
                if self.task_emb is not None:
                    extra["z"] = feat_dim
            self.dfa = DFAContext(sizes, N_ACTIONS + 1, extra=extra)
            self.dfa.T = self.encoder.T
            object.__setattr__(self.encoder, "dfa", self.dfa)  # plain ref, not a 2nd registration
        n_heads = n_tasks if multihead else 1
        self.heads = nn.ModuleList([ActorCriticHeads(feat_dim, N_ACTIONS, hidden) for _ in range(n_heads)])

    def features(self, obs, mask, task=None, past_feats=None):
        """past_feats (B, K, D): cached features of the window's frames (slot -1
        is ignored and recomputed). When given, only the newest frame goes
        through the encoder (gradients reach the encoder via that frame only);
        older frames use the features computed at rollout time."""
        if not self.use_memory:
            z = self.encoder(obs_to_onehot(obs[:, -1]))
            if self.task_emb is not None and task is not None:
                z = z + self.task_emb(task)
            return z
        obs, mask = obs[:, -self.window:], mask[:, -self.window:]
        B, K = mask.shape
        if past_feats is not None:
            f_new = self.encoder(obs_to_onehot(obs[:, -1]))
            self.last_frame_feat = f_new.detach()
            feats = torch.cat([past_feats[:, :-1] * mask[:, :-1, None], f_new[:, None]], 1)
        else:
            x = obs_to_onehot(obs[mask])  # encode only real frames
            f = self.encoder(x)
            feats = f.new_zeros(B, K, f.shape[-1])
            feats[mask] = f
            self.last_frame_feat = feats[:, -1].detach()
        if self.task_emb is not None and task is not None:
            feats = feats + self.task_emb(task)[:, None]
        return self.memory(feats, mask)

    def _heads(self, z, task, box=None):
        kw = dict(dfa=self.dfa, box=box) if box is not None else {}
        if len(self.heads) == 1:
            return self.heads[0](z, **kw)
        if isinstance(task, int):
            return self.heads[task](z, **kw)
        logits = z.new_zeros(z.shape[0], N_ACTIONS)
        value = z.new_zeros(z.shape[0])
        for t in task.unique().tolist():
            sel = task == t
            rows = sel.nonzero().squeeze(1)
            logits[sel], value[sel] = self.heads[t](z[sel], **(dict(kw, rows=rows) if kw else {}))
        return logits, value

    def forward(self, obs, mask, task=None, past_feats=None):
        task_t = task
        if isinstance(task, int):
            task_t = torch.full((obs.shape[0],), task, dtype=torch.long, device=obs.device)
        if self.dfa is None:
            z = self.features(obs, mask, task_t, past_feats)
            return self._heads(z, task if task is not None else 0)
        box = {}
        self.encoder.dfa_box = box
        z = self.features(obs, mask, task_t, past_feats)
        self.encoder.dfa_box = None
        if self.dfa_scope == "all":
            if self.task_emb is not None and task is not None:
                z = self.dfa.hidden(z, "z", box, 1.0)  # task embedding learns from B_z e, not W^T e
            logits, value = self._heads(z, task if task is not None else 0, box)
        else:  # heads by backprop; their gradient into z is discarded at the SNN's DFA nodes
            logits, value = self._heads(z, task if task is not None else 0)
        y = self.dfa.output(torch.cat([logits, value[:, None]], 1), box)
        return y[:, :-1], y[:, -1]

    def get_action_and_value(self, obs, mask, action=None, task=None, greedy=False, past_feats=None):
        logits, value = self(obs, mask, task, past_feats)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, logits


def snapshot(agent: "Agent") -> "Agent":
    """Frozen deep copy (drops cached non-leaf tensors that block deepcopy)."""
    enc = agent.encoder
    enc.act_reg = enc.act_reg.detach()
    enc.record = []
    agent.__dict__.pop("last_frame_feat", None)
    return copy.deepcopy(agent).eval()


class FeatureCache:
    """Per-env window of cached frame features mirroring VecEnv's frame history."""

    def __init__(self, num_envs, window, dim):
        self.f = torch.zeros(num_envs, window, dim)

    def write_newest(self, feat):
        self.f[:, -1] = feat

    def advance(self, dones):
        """Call after env.step: shift continuing envs, clear envs that reset."""
        self.f[:, :-1] = self.f[:, 1:].clone()
        self.f[:, -1] = 0
        d = torch.as_tensor(dones).bool()
        self.f[d] = 0
