"""Leaky integrate-and-fire neurons with surrogate gradients.

Implemented in-repo instead of via snnTorch: it's ~40 lines, lets us expose
exactly the per-timestep pre/post spikes the STDP rule needs, and avoids
snnTorch's stateful-module plumbing inside a vectorized PPO loop.
`tests/test_snn.py` cross-checks the forward dynamics against snntorch.Leaky.

Dynamics (discrete time, reset-by-subtraction):
    v[t] = beta * v[t-1] + I[t] - thr * s[t-1]
    s[t] = H(v[t] - thr)
Backward uses the fast-sigmoid surrogate dS/dv = 1 / (1 + slope*|v - thr|)^2.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_thr, slope):
        ctx.save_for_backward(v_minus_thr)
        ctx.slope = slope
        return (v_minus_thr > 0).to(v_minus_thr.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        grad = grad_out / (1.0 + ctx.slope * x.abs()) ** 2
        return grad, None


def spike_fn(v_minus_thr: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(v_minus_thr, slope)


class LIF(nn.Module):
    """Stateless-per-call LIF: caller threads the membrane state through time.

    beta is optionally learnable (parameterised via a sigmoid so it stays in
    (0, 1)); per-channel if `channels` is given.
    """

    def __init__(self, beta: float = 0.9, threshold: float = 1.0, slope: float = 25.0,
                 learn_beta: bool = False, channels: int | None = None):
        super().__init__()
        self.threshold = threshold
        self.slope = slope
        init = torch.logit(torch.tensor(float(beta)))
        shape = (channels,) if channels else ()
        if learn_beta:
            self.beta_logit = nn.Parameter(init.expand(shape).clone())
        else:
            self.register_buffer("beta_logit", init.expand(shape).clone())

    def beta(self, ndim: int) -> torch.Tensor:
        b = torch.sigmoid(self.beta_logit)
        if b.ndim == 1 and ndim == 4:  # per-channel for conv maps (B, C, H, W)
            b = b.view(1, -1, 1, 1)
        return b

    def forward(self, current: torch.Tensor, v: torch.Tensor | None, s_prev: torch.Tensor | None):
        if v is None:
            v = torch.zeros_like(current)
            s_prev = torch.zeros_like(current)
        # reset term is detached (as in snnTorch's default) for stable gradients
        v = self.beta(current.ndim) * v + current - self.threshold * s_prev.detach()
        s = spike_fn(v - self.threshold, self.slope)
        return s, v
