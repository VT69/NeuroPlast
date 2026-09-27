"""Observation encoders: a dense CNN and a topology-matched spiking CNN.

Both map a one-hot MiniGrid view (B, 20, 7, 7) to a feature vector (B, out_dim)
with the same layer shapes:

    conv 2x2 (20->c1) -> conv 2x2 (c1->c2) -> conv 2x2 (c2->c3) -> fc (c3*16 -> out_dim)

The CNN uses ReLU; the SNN replaces every ReLU with a LIF layer, is simulated
for `T` timesteps with the binary frame presented at every step (direct
coding), and reads out the final layer's firing rate.

Op / energy accounting (RQ3). After a forward with `track=True`, `.stats`
holds per-sample averages of:
  - dense_macs: MACs a dense implementation executes (CNN)
  - sparse_ops: ops an event-driven implementation executes, i.e. for every
    layer sum(nonzero inputs x fan-out). For the SNN these are accumulates
    (SynOps), summed over all T timesteps. For the CNN it's the
    activation-sparsity-aware MAC count (a fair "sparse CNN" comparison).
  - rates: per-layer fraction of nonzero activations (spike rate for SNN).
  - energy_pj: 45nm estimate, E_MAC = 4.6 pJ, E_AC = 0.9 pJ (Horowitz 2014,
    the convention used by most SNN energy papers). CNN -> dense MACs x
    E_MAC; SNN -> SynOps x E_AC (+ first-layer input ops, counted the same
    way for both since the binary input frame is identical). The SNN's
    layer-0 current is computed once per frame (static input), so its input
    ops are counted once, not T times. CNN energy_sparse_pj counts layer-0
    input events as accumulates too (the input is binary for both).
`.act_reg` is a differentiable mean activation (spike rate / L1) that can be
added to the loss to trade accuracy for sparsity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplast.models.snn.lif import LIF

E_MAC_PJ = 4.6
E_AC_PJ = 0.9


def _conv_event_ops(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """Per-sample #(nonzero input x fan-out) for a conv layer (exact)."""
    nz = (x != 0).float().sum(1, keepdim=True)  # (B,1,H,W) nonzero count per pixel
    k = torch.ones(1, 1, *conv.kernel_size, device=x.device)
    per_out_pos = F.conv2d(nz, k, stride=conv.stride, padding=conv.padding)
    return per_out_pos.flatten(1).sum(1) * conv.out_channels


def _dense_conv_macs(x_shape, conv: nn.Conv2d) -> int:
    _, _, h, w = x_shape
    kh, kw = conv.kernel_size
    oh, ow = h - kh + 1, w - kw + 1
    return oh * ow * conv.out_channels * conv.in_channels * kh * kw


class _Base(nn.Module):
    def __init__(self, in_ch=20, channels=(16, 32, 64), out_dim=128):
        super().__init__()
        c1, c2, c3 = channels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, c1, 2), nn.Conv2d(c1, c2, 2), nn.Conv2d(c2, c3, 2)])
        self.fc = nn.Linear(c3 * 4 * 4, out_dim)
        self.out_dim = out_dim
        self.track = False
        self.stats: dict = {}
        self.act_reg = torch.zeros(())

    @property
    def layers(self):
        return list(self.convs) + [self.fc]


class CNNEncoder(_Base):
    def forward(self, x):
        acts_in = []  # inputs to each layer (for op counting)
        h = x
        regs = []
        for conv in self.convs:
            acts_in.append(h)
            h = F.relu(conv(h))
            regs.append(h.mean())
        h = h.flatten(1)
        acts_in.append(h)
        out = F.relu(self.fc(h))
        regs.append(out.mean())
        self.act_reg = torch.stack(regs).mean()
        if self.track:
            self._stats(acts_in, out)
        return out

    @torch.no_grad()
    def _stats(self, acts_in, out):
        dense = sum(_dense_conv_macs(a.shape, c) for a, c in zip(acts_in[:3], self.convs))
        dense += self.fc.in_features * self.fc.out_features
        ops0 = _conv_event_ops(acts_in[0], self.convs[0])  # binary input -> accumulates
        sparse = ops0 + sum(_conv_event_ops(a, c) for a, c in zip(acts_in[1:3], self.convs[1:]))
        sparse = sparse + (acts_in[3] != 0).float().sum(1) * self.fc.out_features
        rates = [(a != 0).float().mean().item() for a in acts_in[1:]] + [(out != 0).float().mean().item()]
        self.stats = dict(dense_macs=float(dense), sparse_ops=sparse.mean().item(), rates=rates,
                          energy_pj=float(dense) * E_MAC_PJ,
                          energy_sparse_pj=ops0.mean().item() * E_AC_PJ
                          + (sparse - ops0).mean().item() * E_MAC_PJ)


class SNNEncoder(_Base):
    """Spiking CNN. Surrogate-gradient trainable end to end (variant A).

    record=True keeps per-timestep (pre, post) spike tensors for every layer in
    `.record` (list over layers of (pre[T], post[T])) for the STDP rule.
    """

    def __init__(self, in_ch=20, channels=(16, 32, 64), out_dim=128, T=4, beta=0.9,
                 threshold=1.0, slope=25.0, learn_beta=True, init_gain=2.0, readout="rate"):
        super().__init__(in_ch, channels, out_dim)
        self.T = T
        self.readout = readout
        self.lifs = nn.ModuleList([
            LIF(beta, threshold, slope, learn_beta, channels=c) for c in (*channels, out_dim)])
        self.record_spikes = False
        self.record: list = []
        self.dfa = None       # set by Agent(dfa=True): neuroplast.learning.dfa.DFAContext
        self.dfa_box = None
        # Default PyTorch init gives sub-threshold currents -> silent network.
        # Scale up so initial firing rates are O(10%) (checked in tests).
        with torch.no_grad():
            for layer in self.layers:
                nn.init.kaiming_uniform_(layer.weight, a=0, nonlinearity="relu")
                layer.weight.mul_(init_gain / 2 ** 0.5)
                layer.bias.zero_()

    def forward(self, x):
        B = x.shape[0]
        state = [(None, None)] * 4
        out_sum = 0.0
        spike_sums = [0.0] * 4
        in_sums = None
        ops = torch.zeros(B, device=x.device) if self.track else None
        rec = [([], []) for _ in range(4)] if self.record_spikes else None
        # layer-0 input current is identical at every step (static frame): compute once
        cur0 = self.convs[0](x)
        for t in range(self.T):
            h = x
            for i, layer in enumerate(self.layers):
                if i == 3:
                    h = h.flatten(1)
                if self.track and (i > 0 or t == 0):  # layer-0 current computed once
                    ops = ops + (_conv_event_ops(h, layer) if i < 3 else (h != 0).float().sum(1) * layer.out_features)
                cur = cur0 if i == 0 else layer(h)
                v, s_prev = state[i]
                s, v = self.lifs[i](cur, v, s_prev)
                state[i] = (v, s)
                if self.dfa is not None and self.dfa_box is not None:
                    s = self.dfa.hidden(s, i, self.dfa_box)  # variant C: DFA learning signal
                if rec is not None:
                    rec[i][0].append(h.detach())
                    rec[i][1].append(s.detach())
                spike_sums[i] = spike_sums[i] + s.mean()
                h = s
            out_sum = out_sum + (h if self.readout == "rate" else state[3][0])
        self.record = rec or []
        rates = [s / self.T for s in spike_sums]
        self.act_reg = torch.stack(rates).mean()
        if self.track:
            with torch.no_grad():
                dense = sum(_dense_conv_macs(x.shape if i == 0 else (B, c.in_channels, 7 - i, 7 - i), c)
                            for i, c in enumerate(self.convs)) + self.fc.in_features * self.fc.out_features
                self.stats = dict(dense_macs=float(dense), sparse_ops=ops.mean().item(),
                                  rates=[r.item() for r in rates],
                                  energy_pj=ops.mean().item() * E_AC_PJ,
                                  energy_sparse_pj=ops.mean().item() * E_AC_PJ)
        return out_sum / self.T


def make_encoder(kind: str, **kw) -> nn.Module:
    if kind == "cnn":
        kw = {k: v for k, v in kw.items() if k in ("in_ch", "channels", "out_dim")}
        return CNNEncoder(**kw)
    if kind == "snn":
        return SNNEncoder(**kw)
    raise ValueError(kind)
