"""Hybrid learning rule for the SNN encoder: dW = alpha * (local STDP) + beta * (backprop).

Implemented as a `train.py` hook. After every PPO optimizer step (the beta
term: Adam on the PPO loss), the spikes recorded during that minibatch's
forward pass are turned into a local STDP weight change for each chosen SNN
layer and added in place.

Scaling (so alpha is interpretable): the STDP update for each layer is
normalised to unit RMS and multiplied by alpha * lr, i.e. alpha = 1 means the
local term moves each weight by about as much as one Adam step (Adam's
per-weight step is ~lr). beta scales the gradient reaching SNN layers; with
Adam only beta in {0, 1} is meaningful (Adam is invariant to gradient scale),
so beta = 0 means "these SNN layers learn by the local rule only" (a
variant-B-style frontend inside the same agent).

Modes:
  three_factor: modulator M_b = normalised PPO advantage of the sample
                (reward-modulated STDP / R-STDP; same sign convention as the
                policy gradient: strengthen correlations that preceded
                better-than-expected actions)
  hebbian:      M_b = 1 (unsupervised STDP) + synaptic scaling (each
                postsynaptic neuron's weight vector keeps its L2 norm), which
                keeps plain Hebbian growth bounded.
  random:       control: an update with the same per-layer RMS but random
                direction (to separate "STDP helps" from "any noise helps").

Stabilisers (added after vanilla STDP was found to cause runaway excitation,
see PROGRESS.md): `center=True` mean-centres pre/post spikes per neuron
(covariance rule, no drift under independent firing); `homeo=<rate>` adds
local intrinsic plasticity (bias += homeo_lr * (target_rate - rate)).
"""
from __future__ import annotations

import torch

from neuroplast.learning.stdp import stdp_conv, stdp_dense


def _center(seq):
    """Subtract each neuron's (channel's) mean over batch, time [and space]:
    turns STDP into a covariance rule with zero expected drift for
    independent pre/post firing."""
    x = torch.stack(seq)  # (T, B, C[, H, W])
    dims = (0, 1) + tuple(range(3, x.dim()))
    x = x - x.mean(dim=dims, keepdim=True)
    return list(x)


class HybridSTDP:
    def __init__(self, alpha=0.1, beta=1.0, mode="three_factor", layers="all", lr=1e-3,
                 lam_pre=0.8, lam_post=0.8, a_plus=1.0, a_minus=1.0, every=1, center=False,
                 homeo=None, homeo_lr=0.01):
        self.alpha, self.beta, self.mode = alpha, beta, mode
        self.center, self.homeo, self.homeo_lr = center, homeo, homeo_lr
        self.layers = layers
        self.lr = lr
        self.kw = dict(lam_pre=lam_pre, lam_post=lam_post, a_plus=a_plus, a_minus=a_minus)
        self.every = every
        self._n = 0
        self.last_rms = {}

    def setup(self, agent):
        enc = agent.encoder
        assert hasattr(enc, "lifs"), "HybridSTDP needs an SNN encoder"
        enc.record_spikes = True
        self.enc = enc
        idx = range(4) if self.layers == "all" else ([3] if self.layers == "fc" else list(self.layers))
        self.idx = list(idx)
        if self.beta == 0:
            for i in self.idx:
                layer = enc.layers[i]
                for p in (layer.weight, layer.bias):
                    p.register_hook(lambda g: torch.zeros_like(g))

    def _modulator(self, batch, n_frames, window):
        if self.mode != "three_factor":
            return None
        adv = batch["adv"].detach()
        if n_frames == adv.shape[0]:
            return adv
        # memory agent: the encoder saw every valid frame of each sample's window
        counts = batch["mask"][:, -window:].sum(1)
        if int(counts.sum()) != n_frames:
            return "mismatch"  # last recorded forward wasn't this PPO minibatch (e.g. a replay pass)
        return adv.repeat_interleave(counts)

    @torch.no_grad()
    def after_step(self, agent, batch, update):
        self._n += 1
        rec = self.enc.record
        if not rec:
            return
        if self.homeo is not None:
            self._homeostasis(rec)
        if self.alpha == 0 or self._n % self.every:
            return
        n_frames = rec[0][0][0].shape[0]
        mod = self._modulator(batch, n_frames, agent.window)
        if isinstance(mod, str):
            self.skipped = getattr(self, "skipped", 0) + 1
            return
        for i in self.idx:
            layer = self.enc.layers[i]
            pre, post = rec[i]
            if self.center:
                pre, post = _center(pre), _center(post)
            if self.mode == "random":
                dW = torch.randn_like(layer.weight)
            elif i < 3:
                dW = stdp_conv(pre, post, layer.kernel_size, mod, **self.kw)
            else:
                dW = stdp_dense(pre, post, mod, **self.kw)
            rms = dW.pow(2).mean().sqrt()
            self.last_rms[i] = rms.item()
            if rms < 1e-12:
                continue
            if self.mode == "hebbian":
                w = layer.weight
                norms = w.flatten(1).norm(dim=1)
                w += (self.alpha * self.lr / rms) * dW
                scale = norms / w.flatten(1).norm(dim=1).clamp_min(1e-12)
                w *= scale.view(-1, *([1] * (w.dim() - 1)))
            else:
                layer.weight += (self.alpha * self.lr / rms) * dW

    @torch.no_grad()
    def _homeostasis(self, rec):
        """Local intrinsic plasticity: nudge each neuron's bias toward a target rate."""
        for i in self.idx:
            post = torch.stack(rec[i][1])  # (T, B, C[, H, W])
            dims = (0, 1) + tuple(range(3, post.dim()))
            rate = post.mean(dim=dims)
            self.enc.layers[i].bias += self.homeo_lr * (self.homeo - rate)
