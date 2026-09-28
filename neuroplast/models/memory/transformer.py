"""Transformer working memory over a short window of encoded frames.

Input: (B, K, D) per-frame features (oldest first) + (B, K) bool mask of which
slots are real frames. Output: (B, D) = the causal-transformer state at the
newest (last) slot. Padding slots before the episode start are masked out of
attention, so the model never sees frames from a previous episode.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WorkingMemory(nn.Module):
    def __init__(self, dim=128, window=8, layers=1, heads=4, ff_mult=2, dropout=0.0, final_norm=True):
        """final_norm=False returns the (pre-LN, residual) stream at the newest slot, which keeps
        the encoder's feature scale (spike rates in [0, 1]) instead of re-normalising to unit
        variance -- see PROGRESS.md session 2 for why this matters for learning speed."""
        super().__init__()
        self.window = window
        self.pos = nn.Parameter(torch.zeros(1, window, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * ff_mult, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        K = x.shape[1]
        h = x + self.pos[:, -K:]
        # the newest slot is always valid, so no row is fully masked
        out = self.tf(h, src_key_padding_mask=~mask)
        return self.norm(out[:, -1])
