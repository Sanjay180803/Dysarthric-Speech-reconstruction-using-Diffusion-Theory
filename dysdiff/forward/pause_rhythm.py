"""Pause and rhythm corruption at fixed length (pause.ipynb).

Pause:  flatten selected spans toward a smoothed trajectory and weaken them.
Rhythm: irregular local re-indexing inside spans, cadence-breaking mixing with
        shifted frames, and a small global timing jitter.
"""

from __future__ import annotations

import numpy as np

from ..schedules import step_weights
from ..utils import frame_diff, moving_average_1d


def sample_spans(T, prob, num_spans, len_min, len_max, rng):
    spans = []
    for _ in range(num_spans):
        c = int(rng.choice(T, p=prob))
        L = int(rng.integers(len_min, len_max + 1))
        s = max(0, c - L // 2)
        spans.append((s, min(T, s + L)))
    return spans


def spans_to_mask(T, spans):
    mask = np.zeros(T, dtype=np.float32)
    for s, e in spans:
        mask[s:e] = 1.0
    return mask


def pause_rhythm_forward_step(
    x_prev: np.ndarray,
    step_weight: float,
    pause_density_base: float = 0.015,
    pause_density_scale: float = 0.06,
    pause_len_min: int = 3,
    pause_len_max: int = 10,
    rhythm_density_base: float = 0.020,
    rhythm_density_scale: float = 0.08,
    rhythm_len_min: int = 4,
    rhythm_len_max: int = 14,
    boundary_power: float = 1.4,
    pause_flatten_strength: float = 0.65,
    pause_weaken_strength: float = 0.22,
    pause_smooth_kernel: int = 9,
    rhythm_reindex_strength: float = 0.35,
    rhythm_local_mix_strength: float = 0.30,
    rhythm_shift_choices=(-3, -2, -1, 1, 2, 3),
    global_timing_jitter: float = 0.08,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    x_prev = x_prev.astype(np.float32)
    T, D = x_prev.shape
    x_t = x_prev.copy()
    ramp = 0.35 + 0.65 * step_weight

    fd = frame_diff(x_prev)
    boundary = np.power(fd + 1e-8, boundary_power); boundary /= boundary.sum()
    low = 1.0 / (fd + 1e-4); low /= low.sum()

    # Pause spans: mix of boundary-like and low-transition regions
    p_prob = 0.55 * boundary + 0.45 * low; p_prob /= p_prob.sum()
    n_pause = max(1, int((pause_density_base + pause_density_scale * step_weight) * T))
    pause_mask = spans_to_mask(T, sample_spans(T, p_prob, n_pause, pause_len_min, pause_len_max, rng))[:, None]
    flatten, weaken = pause_flatten_strength * ramp, pause_weaken_strength * ramp
    x_t = x_t + pause_mask * (flatten * (moving_average_1d(x_prev, pause_smooth_kernel) - x_t))
    x_t = x_t * (1.0 - pause_mask * weaken)

    # Rhythm spans: favour low-transition regions
    r_prob = 0.35 * boundary + 0.65 * low; r_prob /= r_prob.sum()
    n_rhythm = max(1, int((rhythm_density_base + rhythm_density_scale * step_weight) * T))
    r_spans = sample_spans(T, r_prob, n_rhythm, rhythm_len_min, rhythm_len_max, rng)
    rhythm_mask = spans_to_mask(T, r_spans)[:, None]
    reindex, local_mix = rhythm_reindex_strength * ramp, rhythm_local_mix_strength * ramp

    for s, e in r_spans:
        L = e - s
        if L <= 2:
            continue
        seg = x_t[s:e].copy()
        idx = np.linspace(0, L - 1, L).astype(np.float32)
        k = min(5, L if L % 2 == 1 else max(1, L - 1))
        noise = moving_average_1d(rng.normal(0, 1, size=L).astype(np.float32)[:, None], k).squeeze(1)
        warp = np.clip(idx + reindex * noise * max(1.0, L / 6.0), 0, L - 1)
        i0 = np.floor(warp).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, L - 1)
        frac = (warp - i0)[:, None]
        x_t[s:e] = ((1 - frac) * seg[i0] + frac * seg[i1]).astype(np.float32)

    x_shift = np.roll(x_prev, shift=int(rng.choice(list(rhythm_shift_choices))), axis=0)
    ref = 0.5 * x_shift + 0.5 * moving_average_1d(x_shift, 5)
    x_t = x_t + rhythm_mask * (local_mix * (ref - x_t))

    # Global timing jitter
    jitter = global_timing_jitter * ramp
    smooth = moving_average_1d(rng.normal(0, 1, size=(T, 1)).astype(np.float32), 7)
    g_ref = moving_average_1d(np.roll(x_prev, 1, axis=0), 5)
    x_t = x_t + jitter * smooth * (g_ref - x_t)

    delta = x_t - x_prev
    meta = {"step_weight": float(step_weight), "pause_fraction": float(pause_mask.mean()),
            "rhythm_fraction": float(rhythm_mask.mean()),
            "delta_mean_l2": float(np.mean(np.linalg.norm(delta, axis=1)))}
    return x_t.astype(np.float32), delta.astype(np.float32), meta


def pause_rhythm_forward_chain(x_clean, T=30, schedule="cosine", seed=42, **cfg):
    states, deltas, metas = [x_clean.astype(np.float32).copy()], [None], [None]
    x = states[0].copy()
    for t, w in enumerate(step_weights(T, schedule), start=1):
        x, d, m = pause_rhythm_forward_step(x, float(w), seed=seed + t, **cfg)
        m["t"] = t
        states.append(x); deltas.append(d); metas.append(m)
    return states, deltas, metas
