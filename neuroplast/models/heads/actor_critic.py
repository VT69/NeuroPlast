"""Actor / critic MLP heads (optionally multi-head, one pair per task)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCriticHeads(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=64):
        super().__init__()
        if hidden == 0:  # linear readouts (variant C: local delta-rule updates)
            self.actor = layer_init(nn.Linear(in_dim, n_actions), std=0.01)
            self.critic = layer_init(nn.Linear(in_dim, 1), std=1.0)
            return
        self.actor = nn.Sequential(layer_init(nn.Linear(in_dim, hidden)), nn.Tanh(),
                                   layer_init(nn.Linear(hidden, n_actions), std=0.01))
        self.critic = nn.Sequential(layer_init(nn.Linear(in_dim, hidden)), nn.Tanh(),
                                    layer_init(nn.Linear(hidden, 1), std=1.0))

    def forward(self, z, dfa=None, box=None, rows=None):
        if dfa is None or isinstance(self.actor, nn.Linear):
            return self.actor(z), self.critic(z).squeeze(-1)
        # DFA: each hidden layer's gradient is replaced by B e (random feedback);
        # the output layers keep their exact (local, delta-rule) gradients.
        ha = dfa.hidden(self.actor[1](self.actor[0](z)), "ha", box, 1.0, rows)
        hc = dfa.hidden(self.critic[1](self.critic[0](z)), "hc", box, 1.0, rows)
        return self.actor[2](ha), self.critic[2](hc).squeeze(-1)
