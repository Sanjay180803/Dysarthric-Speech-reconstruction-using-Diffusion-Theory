"""Building blocks shared by all models."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Maps a normalised timestep t/T in [0, 1] to a `dim`-d sinusoidal code."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t_norm.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t_norm.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock1D(nn.Module):
    """conv -> GroupNorm -> GELU -> conv -> GroupNorm -> GELU, plus skip."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, groups: int = 8):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = F.gelu(self.norm2(self.conv2(h)))
        return x + h


def mlp(in_dim: int, out_dim: int, act: type = nn.GELU) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), act(), nn.Linear(out_dim, out_dim))


def masked_mean(h: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """Mean over `dim`, ignoring padded positions. h: (B, C, L) or (B, L, C); mask: (B, L)."""
    if mask is None:
        return h.mean(dim=dim)
    m = mask.float()
    m = m[:, None, :] if dim == 2 else m[:, :, None]
    return (h * m).sum(dim=dim) / m.sum(dim=dim).clamp(min=1.0)


def masked_l1_l2(pred, target, mask, l1_weight: float = 1.0, l2_weight: float = 1.0):
    """L1 + L2 over valid frames. pred/target: (B, L, D); mask: (B, L)."""
    m = mask.unsqueeze(-1).float()
    denom = m.sum().clamp(min=1.0) * pred.shape[-1]
    l1 = (torch.abs(pred - target) * m).sum() / denom
    l2 = (((pred - target) ** 2) * m).sum() / denom
    return l1_weight * l1 + l2_weight * l2, l1.detach(), l2.detach()


def masked_mse(pred, target, mask):
    m = mask.unsqueeze(-1).float()
    return (((pred - target) ** 2) * m).sum() / (m.sum().clamp(min=1.0) * pred.shape[-1])


def temporal_smoothness(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared frame-to-frame change over valid frame pairs (B, L, D)."""
    d = x[:, 1:] - x[:, :-1]
    m = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).float() if mask.dtype == torch.bool else (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
    return ((d ** 2) * m).sum() / (m.sum().clamp(min=1.0) * x.shape[-1])
