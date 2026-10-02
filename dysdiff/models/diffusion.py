"""Conditional diffusion models of the combined pipeline (report 3.5.2, 3.5.4, 3.5.5).

* `NeuralForwardDiffusion`  learns the dysarthric corruption delta at step t,
  conditioned on x_{t-1}, t and the StatsEncoder style vector s.
* `TextEncoder`             character CNN giving a 128-d transcript vector.
* `GatedReverseDenoiser`    predicts a gated correction delta to subtract from
  x_t, conditioned on t and the transcript vector, with a learned per-sample
  scale gate in [0, 1].
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResidualBlock1D, SinusoidalTimeEmbedding, masked_mean, mlp


class _TCNBackbone(nn.Module):
    """1024 -> hidden pointwise projection, dilated residual blocks, additive conditioning."""

    def __init__(self, dim_in=1024, hidden=384, time_dim=128, cond_dim=128,
                 dilations=(1, 2, 4, 1, 2, 4), kernel_size=5):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = mlp(time_dim, hidden)
        self.cond_mlp = mlp(cond_dim, hidden)
        self.in_proj = nn.Conv1d(dim_in, hidden, 1)
        self.blocks = nn.ModuleList([ResidualBlock1D(hidden, kernel_size, d) for d in dilations])
        self.out_proj = nn.Conv1d(hidden, dim_in, 1)

    def hidden(self, x, t, T, cond):
        """x: (B, L, D) -> h: (B, hidden, L)."""
        bias = self.time_mlp(self.time_embed(t.float() / T.float().clamp(min=1.0))) + self.cond_mlp(cond)
        h = self.in_proj(x.transpose(1, 2)) + bias[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        return h


class NeuralForwardDiffusion(_TCNBackbone):
    def forward(self, x_prev, t, T, style):
        """Predict delta_t: (B, L, D). The chain then sets x_t = x_{t-1} + alpha_t * delta_t."""
        return self.out_proj(self.hidden(x_prev, t, T, style)).transpose(1, 2)


# --------------------------------------------------------------------------- #
# Text conditioning
# --------------------------------------------------------------------------- #
VOCAB = "abcdefghijklmnopqrstuvwxyz "  # index 0 is padding -> 28 symbols


def encode_text(texts: List[str], max_len: int = 200) -> torch.Tensor:
    """Lower-case, keep a-z and space, map to ints (pad = 0). Returns (B, L) long."""
    ids = []
    for s in texts:
        s = "".join(ch for ch in s.lower() if ch in VOCAB)[:max_len] or " "
        ids.append([VOCAB.index(ch) + 1 for ch in s])
    L = max(len(i) for i in ids)
    out = torch.zeros(len(ids), L, dtype=torch.long)
    for b, seq in enumerate(ids):
        out[b, : len(seq)] = torch.tensor(seq)
    return out


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = len(VOCAB) + 1, emb_dim: int = 64, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim if i == 0 else hidden, hidden, 3, padding=1) for i in range(3)
        ])
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """char_ids: (B, L) -> (B, out_dim)."""
        mask = char_ids != 0
        h = self.embed(char_ids).transpose(1, 2)
        for conv in self.convs:
            h = F.gelu(conv(h))
        return self.out_proj(masked_mean(h, mask, dim=2))


# --------------------------------------------------------------------------- #
# Reverse model
# --------------------------------------------------------------------------- #
class GatedReverseDenoiser(_TCNBackbone):
    """ReverseArticulationUNetLite with text conditioning and a scale gate."""

    def __init__(self, *args, gate_hidden: int = 96, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.out_proj.in_channels
        self.gate = nn.Sequential(nn.Linear(hidden, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1), nn.Sigmoid())

    def forward(self, x_t, t, T, text_vec, mask: Optional[torch.Tensor] = None):
        """Returns (delta_hat, gate) with delta_hat: (B, L, D), gate: (B, 1)."""
        h = self.hidden(x_t, t, T, text_vec)
        raw = self.out_proj(h).transpose(1, 2)
        g = self.gate(masked_mean(h, mask, dim=2))
        return g[:, :, None] * raw, g


def clip_delta(delta: torch.Tensor, x: torch.Tensor, ratio: float = 0.20) -> torch.Tensor:
    """Limit each frame's correction to `ratio` of that frame's norm (inference safeguard)."""
    limit = ratio * x.norm(dim=-1, keepdim=True)
    norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return delta * torch.clamp(limit / norm, max=1.0)
