"""Per-factor reverse denoisers from the exploratory stage.

* `ReverseStepModel` (articulation.ipynb) predicts the forward delta and
  returns it; x_{t-1} = x_t - delta. Parameter names match
  `reverse_articulation_step_model_best.pt`.
* `ReverseUNetLite` (articulation-severe / phonation / pause notebooks)
  returns x_{t-1} directly as x_t + learned correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ResidualBlock1D, SinusoidalTimeEmbedding


class ReverseStepModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, time_dim=128, num_layers=4, kernel_size=5):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            )
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t, t, T):
        """x_t: (B, L, D) -> predicted delta (B, L, D)."""
        t_feat = self.time_proj(self.time_embed(t.float() / T.float().clamp(min=1.0)))
        h = (self.in_proj(x_t) + t_feat[:, None, :]).transpose(1, 2)
        for block in self.convs:
            h = block(h) + h
        return self.out_proj(h.transpose(1, 2))

    def denoise(self, x_t, t, T):
        return x_t - self.forward(x_t, t, T)


class ReverseUNetLite(nn.Module):
    def __init__(self, dim_in=1024, hidden=384, time_dim=128, dilations=(1, 2, 4, 1, 2, 4), kernel_size=5):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Conv1d(dim_in, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualBlock1D(hidden, kernel_size, d) for d in dilations])
        self.out_proj = nn.Conv1d(hidden, dim_in, kernel_size=1)

    def forward(self, x_t, t, T):
        """x_t: (B, L, D) -> x_{t-1} estimate (B, L, D)."""
        t_emb = self.time_mlp(self.time_embed(t.float() / T.float().clamp(min=1.0)))
        h = self.in_proj(x_t.transpose(1, 2)) + t_emb[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        return x_t + self.out_proj(h).transpose(1, 2)

    def denoise(self, x_t, t, T):
        return self.forward(x_t, t, T)


@torch.no_grad()
def run_reverse_chain(model, x_T, T: int, device="cpu"):
    """Apply a per-factor model for t = T..1. x_T: (L, D) numpy array."""
    import numpy as np

    model.eval()
    x = torch.from_numpy(np.asarray(x_T, dtype=np.float32)).unsqueeze(0).to(device)
    for t in range(T, 0, -1):
        tt = torch.tensor([t], device=device)
        x = model.denoise(x, tt, torch.tensor([T], device=device))
    return x.squeeze(0).cpu().numpy()
