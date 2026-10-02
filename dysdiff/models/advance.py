"""Reverse hold-diffusion step (report Section 3.3).

A TCN predicts, for every frame of an elongated sequence X_t, the probability
that it starts a new source frame (advance) rather than repeating one (hold).
Top-K compression then keeps the K most likely advance frames, in order,
which undoes elongation while keeping the alignment monotonic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock1D, SinusoidalTimeEmbedding, mlp


class AdvancePredictor(nn.Module):
    def __init__(self, dim_in=1024, hidden=256, time_dim=128,
                 dilations=(1, 2, 4, 8, 1, 2, 4, 8), kernel_size=3):
        super().__init__()
        self.in_proj = nn.Conv1d(dim_in, hidden, 1)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = mlp(time_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock1D(hidden, kernel_size, d) for d in dilations])
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x_t, t, T):
        """x_t: (B, L, D) -> advance logits (B, L)."""
        h = self.in_proj(x_t.transpose(1, 2))
        h = h + self.time_mlp(self.time_embed(t.float() / T.float().clamp(min=1.0)))[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(1)


def weighted_bce(logits, target, mask, pos_weight: Optional[float] = None):
    """BCE over valid frames; positives are up-weighted by the hold/advance ratio by default."""
    m = mask.float()
    if pos_weight is None:
        pos = (target * m).sum()
        pos_weight = float(((m.sum() - pos) / pos.clamp(min=1.0)).clamp(min=1.0))
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction="none")
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def topk_compress(x_t: torch.Tensor, adv_prob: torch.Tensor, target_len: int) -> torch.Tensor:
    """Keep the `target_len` frames with highest advance probability, in time order.

    x_t: (L, D), adv_prob: (L,). Frame 0 is always kept.
    """
    L = x_t.shape[0]
    k = int(min(max(target_len, 1), L))
    p = adv_prob.clone()
    p[0] = float("inf")
    idx = torch.topk(p, k).indices.sort().values
    return x_t[idx]


def threshold_compress(x_t: torch.Tensor, adv_prob: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Keep every frame predicted to advance (length is inferred, not scheduled)."""
    keep = adv_prob >= thr
    keep[0] = True
    return x_t[keep]
