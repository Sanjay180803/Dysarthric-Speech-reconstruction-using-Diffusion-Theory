"""Phonation corruption: breathiness and voicing instability (phonation.ipynb).

Unlike articulation, phonation targets *stable* low-transition spans
(sustained voicing) and applies gain flutter, smooth jitter, breathy
weakening toward the local mean, and loss of coherence.
"""

from __future__ import annotations

import numpy as np

from ..schedules import step_weights
from ..utils import frame_diff, moving_average_1d


def phonation_forward_step(
    x_prev: np.ndarray,
    step_weight: float,
    total_scale: float = 0.85,
    noise_smooth_kernel: int = 11,
    gain_smooth_kernel: int = 21,
    stable_span_density_base: float = 0.03,
    stable_span_density_scale: float = 0.10,
    span_len_min: int = 4,
    span_len_max: int = 12,
    stability_power: float = 1.5,
    max_gain_flutter: float = 0.16,
    max_jitter_scale: float = 0.22,
    max_breathy_weakening: float = 0.22,
    max_coherence_loss: float = 0.30,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    x_prev = x_prev.astype(np.float32)
    T, D = x_prev.shape

    score = np.power(1.0 / (frame_diff(x_prev) + 1e-4), stability_power)
    prob = score / score.sum()
    num_spans = max(2, int((stable_span_density_base + stable_span_density_scale * step_weight) * T))
    span = np.zeros(T, dtype=np.float32)
    for _ in range(num_spans):
        c = int(rng.choice(T, p=prob))
        L = int(rng.integers(span_len_min, span_len_max + 1))
        s = max(0, c - L // 2)
        span[s:min(T, s + L)] = 1.0
    m = span[:, None]

    # 1) multiplicative gain flutter
    gain = moving_average_1d(rng.normal(0, 1, size=(T, 1)).astype(np.float32), gain_smooth_kernel)
    gain_flutter = max_gain_flutter * (0.35 + 0.65 * step_weight)
    gain_factor = np.clip(1.0 + total_scale * gain_flutter * gain, 0.70, 1.35)
    x_t = x_prev + m * (x_prev * (gain_factor - 1.0))

    # 2) smooth additive jitter scaled by frame energy
    noise = moving_average_1d(rng.normal(0, 1, size=(T, D)).astype(np.float32), noise_smooth_kernel)
    direction = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8)
    jitter_scale = total_scale * max_jitter_scale * (0.30 + 0.70 * step_weight)
    frame_mag = np.linalg.norm(x_prev, axis=1, keepdims=True) + 1e-8
    x_t = x_t + m * (jitter_scale * 0.08 * frame_mag * direction)

    # 3) breathy weakening toward the local mean
    breathy = max_breathy_weakening * (0.30 + 0.70 * step_weight)
    local_mean = moving_average_1d(x_prev, max(5, gain_smooth_kernel // 2))
    x_t = x_t + m * (breathy * (local_mean - x_t))

    # 4) coherence loss against a shifted, smoothed reference
    coh = max_coherence_loss * (0.30 + 0.70 * step_weight)
    x_shift = np.roll(x_prev, shift=int(rng.choice([-2, -1, 1, 2])), axis=0)
    ref = 0.5 * x_shift + 0.5 * moving_average_1d(x_shift, 7)
    x_t = x_t + m * (coh * (ref - x_t))

    delta = x_t - x_prev
    meta = {"step_weight": float(step_weight), "num_spans": int(num_spans),
            "corrupted_fraction": float(span.mean()),
            "delta_mean_l2": float(np.mean(np.linalg.norm(delta, axis=1)))}
    return x_t.astype(np.float32), delta.astype(np.float32), meta


def phonation_forward_chain(x_clean, T=30, schedule="cosine", seed=42, **cfg):
    states, deltas, metas = [x_clean.astype(np.float32).copy()], [None], [None]
    x = states[0].copy()
    for t, w in enumerate(step_weights(T, schedule), start=1):
        x, d, m = phonation_forward_step(x, float(w), seed=seed + t, **cfg)
        m["t"] = t
        states.append(x); deltas.append(d); metas.append(m)
    return states, deltas, metas
