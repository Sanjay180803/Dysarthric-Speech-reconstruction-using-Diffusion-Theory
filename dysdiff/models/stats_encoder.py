"""StatsEncoder (report Section 3.5.1).

Maps a dysarthric embedding sequence (B, L, 1024) to a 128-d style vector s
describing how the speaker's embeddings deviate from typical speech.

Layer layout matches `runs/exp1/stats_encoder_best.pt`:
four dilated convs (k = 7, 5, 5, 5; d = 1, 1, 2, 4) with GroupNorm + GELU,
mean+std pooling (256 + 256), linear to 128.

Training uses a proxy objective because TORGO has no paired data: the style
vector must match a fixed projection (`target_head`, 2048 -> 128) of the mean
and std of a *randomly associated clean* utterance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatsEncoder(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 256, out_dim: int = 128, groups: int = 8):
        super().__init__()
        layers = []
        cfg = [(in_dim, 7, 1), (hidden, 5, 1), (hidden, 5, 2), (hidden, 5, 4)]
        for c_in, k, d in cfg:
            layers += [
                nn.Conv1d(c_in, hidden, k, padding=(k - 1) // 2 * d, dilation=d),
                nn.GroupNorm(groups, hidden),
                nn.GELU(),
            ]
        self.conv_stack = nn.Sequential(*layers)
        self.out_proj = nn.Linear(2 * hidden, out_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, L, D); mask: (B, L) bool. Returns s: (B, out_dim)."""
        h = self.conv_stack(x.transpose(1, 2))  # (B, H, L)
        if mask is None:
            mean, std = h.mean(dim=2), h.std(dim=2, unbiased=False)
        else:
            m = mask.float()[:, None, :]
            n = m.sum(dim=2).clamp(min=1.0)
            mean = (h * m).sum(dim=2) / n
            std = (((h - mean[:, :, None]) ** 2 * m).sum(dim=2) / n).clamp(min=1e-8).sqrt()
        return self.out_proj(torch.cat([mean, std], dim=-1))


def utterance_stats(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Concatenated per-dimension mean and std of a (B, L, D) batch -> (B, 2D)."""
    if mask is None:
        return torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=-1)
    m = mask.float()[:, :, None]
    n = m.sum(1).clamp(min=1.0)
    mean = (x * m).sum(1) / n
    std = (((x - mean[:, None]) ** 2 * m).sum(1) / n).clamp(min=1e-8).sqrt()
    return torch.cat([mean, std], dim=-1)


class StatsEncoderTrainer(nn.Module):
    """Wraps the encoder with the proxy-objective target head."""

    def __init__(self, emb_dim: int = 1024, style_dim: int = 128, freeze_target_head: bool = True):
        super().__init__()
        self.encoder = StatsEncoder(emb_dim, out_dim=style_dim)
        self.target_head = nn.Linear(2 * emb_dim, style_dim)
        if freeze_target_head:
            # A fixed random projection of the clean statistics; freezing it
            # stops encoder and head from collapsing to a constant together.
            for p in self.target_head.parameters():
                p.requires_grad_(False)

    def loss(self, x_dys, mask_dys, x_clean, mask_clean):
        s = self.encoder(x_dys, mask_dys)
        target = self.target_head(utterance_stats(x_clean, mask_clean))
        return F.mse_loss(s, target)

    def state_dict_for_release(self):
        return {"encoder": self.encoder.state_dict(), "target_head": self.target_head.state_dict()}


def load_stats_encoder(path: str, map_location="cpu") -> StatsEncoder:
    ckpt = torch.load(path, map_location=map_location)
    enc = StatsEncoder()
    enc.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt)
    return enc.eval()
