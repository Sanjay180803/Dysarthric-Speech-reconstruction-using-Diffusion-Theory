"""Corruption-strength schedules shared by every forward process."""

from __future__ import annotations

import math

import numpy as np


def ramp(t: int, T: int, mode: str = "cosine") -> float:
    """Un-normalised strength s(t) in [0, 1] (0 at t=0, 1 at t=T).

    This is the `sw_t` schedule weight used by the neural forward chain
    (alpha_t = 0.5 * sw_t) and by the articulation drift.
    """
    if T <= 0:
        return 1.0
    frac = min(max(t / T, 0.0), 1.0)
    if mode == "linear":
        return frac
    if mode == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * frac))
    raise ValueError(f"Unknown schedule: {mode}")


def step_weights(T: int, schedule: str = "cosine") -> np.ndarray:
    """Per-step weights that sum to 1 (later steps stronger).

    * ``cosine``      quarter-cosine ramp, used by the "pronounced"
                      articulation, phonation and pause/rhythm modules.
    * ``cosine_full`` half-cosine ramp, used by the budgeted articulation chain.
    * ``linear`` / ``uniform``.
    """
    if schedule == "uniform":
        w = np.ones(T, dtype=np.float32)
    elif schedule == "linear":
        w = np.arange(1, T + 1, dtype=np.float32)
    elif schedule == "cosine":
        t = np.linspace(0, 1, T, dtype=np.float32)
        w = 1.0 - np.cos((t + 1e-8) * np.pi / 2.0)
    elif schedule == "cosine_full":
        w = np.array([ramp(t, T, "cosine") for t in range(1, T + 1)], dtype=np.float32)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    w = np.maximum(w, 1e-8)
    return (w / w.sum()).astype(np.float32)


def hold_strength(t: int, steps: int, schedule: str = "cosine") -> float:
    """Strength used by the hold-warp operator (sin^2 ramp over 0..steps-1)."""
    if steps <= 1:
        return 1.0
    x = t / (steps - 1)
    if schedule == "cosine":
        return math.sin(0.5 * math.pi * x) ** 2
    return x


def reverse_timesteps(T: int, n_steps: int) -> list[int]:
    """Evenly spaced reverse-sampling timesteps, e.g. T=30, n=5 -> [30, 24, 18, 12, 6]."""
    stride = T // n_steps
    return [T - i * stride for i in range(n_steps)]
