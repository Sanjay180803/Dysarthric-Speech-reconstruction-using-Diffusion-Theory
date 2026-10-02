"""Temporal elongation via hold-warp (report Section 3.2).

Each frame i of X_{t-1} is held for R_i ~ Geom(p_t) frames, clipped to
L_max(t). The operator records a pointer map ptr_t (which source frame every
output frame came from) and binary advance labels adv_t, which are the exact
supervision for the reverse advance predictor.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..schedules import hold_strength


def ptr_to_advance(ptr: np.ndarray) -> np.ndarray:
    """adv[0] = 1, adv[j] = 1[ptr[j] != ptr[j-1]]."""
    y = np.zeros_like(ptr, dtype=np.float32)
    if ptr.size == 0:
        return y
    y[0] = 1.0
    y[1:] = (ptr[1:] != ptr[:-1]).astype(np.float32)
    return y


def run_lengths(seq: np.ndarray) -> np.ndarray:
    """Lengths of runs of identical consecutive values."""
    if seq.size == 0:
        return np.array([], dtype=np.int32)
    change = np.flatnonzero(seq[1:] != seq[:-1]) + 1
    bounds = np.concatenate([[0], change, [seq.size]])
    return np.diff(bounds).astype(np.int32)


def run_stats_from_ptr(ptr: np.ndarray) -> Dict[str, float]:
    if ptr.size < 2:
        return {"mean_run": float(ptr.size), "p95_run": float(ptr.size), "p_switch": 0.0, "T": float(ptr.size)}
    runs = run_lengths(ptr).astype(np.float32)
    return {
        "mean_run": float(runs.mean()),
        "p95_run": float(np.quantile(runs, 0.95)),
        "p_switch": float((ptr[1:] != ptr[:-1]).mean()),
        "T": float(ptr.size),
    }


def hold_warp_once(
    x0: np.ndarray,
    t: int,
    steps: int,
    geom_p: float,
    p95_final: int,
    schedule: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Apply W_t to a (T, D) sequence.

    Returns (x_t, ptr, adv, p_t, Lmax_t). At low strength p_t -> 1 (identity);
    at full strength p_t -> geom_p and holds are capped at the p95 run length
    measured on real moderate-dysarthric speech.
    """
    s = hold_strength(t, steps, schedule)
    lmax_t = max(1, 1 + int(s * (p95_final - 1)))
    p_t = float(np.clip((1.0 - s) * 1.0 + s * geom_p, 1e-3, 0.999))

    reps = np.minimum(rng.geometric(p_t, size=x0.shape[0]), lmax_t)
    ptr = np.repeat(np.arange(x0.shape[0], dtype=np.int64), reps)
    x_t = x0[ptr]
    return x_t, ptr, ptr_to_advance(ptr), p_t, lmax_t


def elongate_regions(
    x: np.ndarray,
    rng: np.random.Generator,
    n_regions: Tuple[int, int] = (2, 3),
    max_extra: int = 10,
    region_len: Tuple[int, int] = (3, 8),
) -> Tuple[np.ndarray, np.ndarray]:
    """Local elongation used by the combined forward chain (report Fig. 4, stage A).

    Picks 2-3 non-overlapping regions and stretches each by up to `max_extra`
    frames by holding its frames. Returns (x_elongated, ptr).
    """
    T = x.shape[0]
    reps = np.ones(T, dtype=np.int64)
    n = int(rng.integers(n_regions[0], n_regions[1] + 1))
    taken = np.zeros(T, dtype=bool)
    for _ in range(n):
        L = int(rng.integers(region_len[0], region_len[1] + 1))
        if T <= L:
            break
        start = int(rng.integers(0, T - L))
        if taken[start:start + L].any():
            continue
        taken[start:start + L] = True
        extra = int(rng.integers(1, max_extra + 1))
        idx = rng.choice(np.arange(start, start + L), size=extra, replace=True)
        np.add.at(reps, idx, 1)
    ptr = np.repeat(np.arange(T, dtype=np.int64), reps)
    return x[ptr], ptr
