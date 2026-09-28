"""Direct Feedback Alignment for the SNN encoder (variant C: no backprop).

Autograd plumbing that turns the ordinary backward pass into DFA:

  * `DFAContext.output(y)` wraps the network output y = [logits, value]
    (identity forward). Its backward stores e = dL/dy (the output error: the
    global learning signal) and passes the gradient on to the linear
    readouts, whose weight gradients are then local delta rules
    (error x presynaptic activity).
  * `DFAContext.hidden(s, l)` wraps a hidden layer's spike output (identity
    forward). Its backward *discards* whatever gradient arrives from
    downstream layers and returns B_l e instead, where B_l is a fixed random
    matrix (no weight transport).

Inside a layer the gradient still passes through the LIF surrogate
derivative to the layer's own weights, so each weight update is
  dW_l ~ sum_t [(B_l e) * sigma'(v_l[t])] x pre_l[t]
i.e. a three-factor rule: presynaptic activity x postsynaptic (surrogate)
derivative x broadcast error, as in e-prop / feedback-alignment SNN work.
The one non-local piece left is the membrane recurrence within a layer over
T steps, which is local in space (same neuron) — standard in e-prop.
"""
from __future__ import annotations

import torch


class _Output(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, box):
        ctx.box = box
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        ctx.box["e"] = g.detach()
        return g, None


class _Hidden(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, B, box, scale, rows):
        ctx.B, ctx.box, ctx.scale, ctx.shape, ctx.rows = B, box, scale, s.shape, rows
        return s.view_as(s)

    @staticmethod
    def backward(ctx, g):
        e = ctx.box.get("e")
        if e is None:
            return torch.zeros_like(g), None, None, None, None
        if ctx.rows is not None:  # this node saw only a subset of the batch (multi-head routing)
            e = e[ctx.rows]
        proj = (e @ ctx.B.T) * ctx.scale      # (N, numel of the layer's per-sample output)
        return proj.view(ctx.shape), None, None, None, None


class DFAContext(torch.nn.Module):
    def __init__(self, layer_sizes, out_dim, seed=0, extra=None):
        """layer_sizes: SNN layers (keys 0..n-1, error spread over T steps);
        extra: {name: size} for non-spiking hidden layers (MLP heads 'ha'/'hc',
        task-conditioned features 'z'); created after the SNN ones so B0..B3 are
        identical with or without them."""
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        for i, n in enumerate(layer_sizes):
            # unit-variance random feedback, scaled by 1/sqrt(out_dim)
            self.register_buffer(f"B{i}", torch.randn(n, out_dim, generator=g) / out_dim ** 0.5)
        for name, n in (extra or {}).items():
            self.register_buffer(f"B{name}", torch.randn(n, out_dim, generator=g) / out_dim ** 0.5)
        self.T = 1

    # `box` is a fresh dict per forward pass, shared by that pass's output and
    # hidden nodes, so several forwards in one loss (e.g. PPO + replay) each
    # use their own error.
    def output(self, y, box):
        return _Output.apply(y, box)

    def hidden(self, s, i, box, scale=None, rows=None):
        B = getattr(self, f"B{i}")
        # SNN layers: the same error is broadcast at every timestep; divide by T so
        # the total signal per frame doesn't scale with simulation length.
        # Non-spiking layers (heads, features) pass scale=1.
        return _Hidden.apply(s, B, box, 1.0 / self.T if scale is None else scale, rows)
