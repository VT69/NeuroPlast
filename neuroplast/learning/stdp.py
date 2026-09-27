"""Pair-based STDP with exponential traces, eligibility traces and a
third-factor modulator (reward-modulated / three-factor STDP).

Discrete-time trace STDP for a synapse j -> i:

    x_j[t] = lam_pre  * x_j[t-1] + pre_j[t]            (presynaptic trace)
    y_i[t] = lam_post * y_i[t-1] + post_i[t]           (postsynaptic trace)
    stdp_ij[t] = A_plus * post_i[t] * x_j[t]           (LTP: pre at or before post)
               - A_minus * y_i[t-1] * pre_j[t]         (LTD: post strictly before pre)

The LTP term uses x_j[t], which includes the presynaptic spike at the same
step: in our discrete LIF a presynaptic spike at t drives the postsynaptic
membrane at t, so same-step pairs are causal.

Eligibility and third factor:

    e_ij[t] = lam_e * e_ij[t-1] + stdp_ij[t]
    dW_ij    = eta * M * e_ij

`stdp_dense` / `stdp_conv` compute the per-sample eligibility accumulated
over a spike sequence (lam_e = 1 inside a single forward) and contract it
with a per-sample modulator M_b in one einsum, which is what the hybrid rule
uses inside PPO (M_b = normalised advantage). `EligibilityTrace` is the
stateful online version for rules that run step by step across env steps.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def traces(seq: Sequence[torch.Tensor], lam: float) -> list[torch.Tensor]:
    """Exponential trace over a list of spike tensors (inclusive of step t)."""
    out, tr = [], None
    for s in seq:
        tr = s if tr is None else lam * tr + s
        out.append(tr)
    return out


def _mod(mod, B, ref):
    if mod is None:
        return ref.new_ones(B)
    return mod.to(ref.dtype)


def stdp_dense(pre: Sequence[torch.Tensor], post: Sequence[torch.Tensor], modulator: torch.Tensor | None = None,
               a_plus=1.0, a_minus=1.0, lam_pre=0.8, lam_post=0.8, mean=True) -> torch.Tensor:
    """pre[t]: (B, in), post[t]: (B, out) spikes. Returns dW (out, in).

    dW = sum_b M_b sum_t [A+ post_b[t] x_b[t]^T - A- y_b[t-1] pre_b[t]^T]  (/B if mean)
    """
    B = pre[0].shape[0]
    m = _mod(modulator, B, pre[0])[:, None]
    x = traces(pre, lam_pre)
    y = traces(post, lam_post)
    dW = pre[0].new_zeros(post[0].shape[1], pre[0].shape[1])
    for t in range(len(pre)):
        dW += a_plus * (m * post[t]).T @ x[t]
        if t > 0:
            dW -= a_minus * (m * y[t - 1]).T @ pre[t]
    return dW / B if mean else dW


def stdp_conv(pre: Sequence[torch.Tensor], post: Sequence[torch.Tensor], kernel_size, modulator=None,
              a_plus=1.0, a_minus=1.0, lam_pre=0.8, lam_post=0.8, stride=1, padding=0, mean=True) -> torch.Tensor:
    """Conv STDP with shared weights: pre[t] (B,C,H,W), post[t] (B,O,H',W').

    Each output location is a synapse instance of the shared kernel; the
    weight change is summed over locations (and averaged over the batch).
    Returns dW (O, C, kh, kw).
    """
    B, C = pre[0].shape[:2]
    O = post[0].shape[1]
    ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    m = _mod(modulator, B, pre[0])[:, None, None]
    unf = lambda z: F.unfold(z, ks, stride=stride, padding=padding)  # (B, C*kh*kw, L)
    x = traces(pre, lam_pre)
    y = traces(post, lam_post)
    dW = pre[0].new_zeros(O, C * ks[0] * ks[1])
    for t in range(len(pre)):
        p_t = (m * post[t].flatten(2))                      # (B, O, L)
        dW += a_plus * torch.einsum("bol,bkl->ok", p_t, unf(x[t]))
        if t > 0:
            y_t = (m * y[t - 1].flatten(2))
            dW -= a_minus * torch.einsum("bol,bkl->ok", y_t, unf(pre[t]))
    dW = dW.view(O, C, *ks)
    return dW / B if mean else dW


class EligibilityTrace:
    """Online three-factor rule for a dense layer, one step at a time.

    step(pre, post) updates traces and the eligibility e (per-batch-element
    summed); apply(weight, modulator, eta) does W += eta * M * e.
    """

    def __init__(self, n_in, n_out, lam_pre=0.8, lam_post=0.8, lam_e=0.9, a_plus=1.0, a_minus=1.0):
        self.lam_pre, self.lam_post, self.lam_e = lam_pre, lam_post, lam_e
        self.a_plus, self.a_minus = a_plus, a_minus
        self.x = None
        self.y = None
        self.e = torch.zeros(n_out, n_in)

    def reset(self):
        self.x = self.y = None
        self.e.zero_()

    def step(self, pre: torch.Tensor, post: torch.Tensor):
        y_prev = self.y
        self.x = pre.clone() if self.x is None else self.lam_pre * self.x + pre
        self.y = post.clone() if self.y is None else self.lam_post * self.y + post
        d = self.a_plus * post.T @ self.x
        if y_prev is not None:
            d = d - self.a_minus * y_prev.T @ pre
        self.e = self.lam_e * self.e + d
        return d

    @torch.no_grad()
    def apply(self, weight: torch.Tensor, modulator: float, eta: float):
        weight += eta * modulator * self.e
