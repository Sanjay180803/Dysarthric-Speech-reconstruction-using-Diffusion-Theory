"""Articulation corruption: slurring, boundary blur and spectral drift in embedding space.

Three variants:
  * `articulation_forward_chain_budgeted`  - mild, renormalised per-step budget
    (articulation.ipynb). Paired with `ReverseStepModel`.
  * `articulation_forward_chain_pronounced` - transition-targeted spans with
    drift + blur + frame confusion + partial replacement (Artciculation_severe.ipynb).
  * `articulation_drift`                    - the mild drift applied inside the
    combined neural forward chain (report Section 3.5.3).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from ..schedules import step_weights
from ..utils import adjacent_cosine, frame_diff, load_embedding, local_diff_energy, moving_average_1d


# --------------------------------------------------------------------------- #
# Statistics from unpaired clean vs. moderate speech
# --------------------------------------------------------------------------- #
def estimate_articulation_stats(
    clean_files: List[str],
    moderate_files: List[str],
    max_clean: int = 100,
    max_moderate: int = 100,
    seed: int = 42,
) -> Dict:
    """Measure smoothness, boundary blur and residual drift on unpaired data.

    The drift statistics (residual after a 3-frame moving average) set the
    magnitude of the articulation perturbation.
    """
    if not clean_files or not moderate_files:
        raise ValueError("Need at least one clean and one moderate file.")
    rng = random.Random(seed)
    clean = rng.sample(clean_files, min(len(clean_files), max_clean))
    moderate = rng.sample(moderate_files, min(len(moderate_files), max_moderate))

    def collect(files):
        energy, adj, resid, norms = [], [], [], []
        for p in files:
            x = load_embedding(p)
            energy.append(local_diff_energy(x))
            adj.append(adjacent_cosine(x))
            resid.append(np.linalg.norm(x - moving_average_1d(x, 3), axis=1))
            norms.append(np.linalg.norm(x, axis=1))
        return energy, adj, np.concatenate(resid), np.concatenate(norms)

    ce, ca, cr, cn = collect(clean)
    me, ma, mr, mn = collect(moderate)
    return {
        "num_clean_files_used": len(clean),
        "num_moderate_files_used": len(moderate),
        "clean_local_energy_mean": float(np.mean(ce)),
        "moderate_local_energy_mean": float(np.mean(me)),
        "smoothness_ratio": float(np.mean(me) / (np.mean(ce) + 1e-8)),
        "clean_adjacent_cos_mean": float(np.mean(ca)),
        "moderate_adjacent_cos_mean": float(np.mean(ma)),
        "boundary_blur_gain": float(np.mean(ma) - np.mean(ca)),
        "clean_residual_l2_mean": float(np.mean(cr)),
        "moderate_residual_l2_mean": float(np.mean(mr)),
        "drift_l2_mean": float(np.mean(mr)),
        "drift_l2_std": float(np.std(mr)),
        "drift_l2_p95": float(np.percentile(mr, 95)),
        "clean_frame_norm_mean": float(np.mean(cn)),
        "moderate_frame_norm_mean": float(np.mean(mn)),
    }


def utterance_drift_stats(x: np.ndarray) -> Dict[str, float]:
    """Drift statistics of a single utterance (residual after a 3-frame average).

    Used to give each simulated chain the drift profile of the dysarthric
    reference utterance it is paired with, so the style vector s carries
    information the neural forward model can use.
    """
    r = np.linalg.norm(x - moving_average_1d(x, 3), axis=1)
    return {"drift_l2_mean": float(r.mean()), "drift_l2_std": float(r.std()),
            "drift_l2_p95": float(np.percentile(r, 95))}


def _smooth_direction(rng, T, D, kernel):
    noise = rng.normal(0.0, 1.0, size=(T, D)).astype(np.float32)
    noise = moving_average_1d(noise, kernel)
    return noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8)


def _drift_magnitude(rng, stats, T):
    mean = float(stats.get("drift_l2_mean", 1.0))
    std = float(stats.get("drift_l2_std", 0.5))
    p95 = float(stats.get("drift_l2_p95", mean + 2 * std))
    mag = rng.normal(mean, max(std, 1e-6), size=(T, 1)).astype(np.float32)
    return np.clip(mag, 0.0, p95)


# --------------------------------------------------------------------------- #
# Budgeted chain (mild)
# --------------------------------------------------------------------------- #
def articulation_forward_step_budgeted(
    x_prev: np.ndarray,
    stats: Dict,
    step_weight: float,
    total_scale: float = 0.60,
    noise_smooth_kernel: int = 7,
    embed_smooth_kernel: int = 3,
    max_blur_alpha: float = 0.35,
    max_mix_prob: float = 0.25,
    max_mix_lam: float = 0.35,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Smooth drift + boundary blur + neighbour mixing, renormalised so each
    step spends exactly its share of the corruption budget."""
    rng = np.random.default_rng(seed)
    x_prev = x_prev.astype(np.float32)
    T, D = x_prev.shape

    direction = _smooth_direction(rng, T, D, noise_smooth_kernel)
    mag = _drift_magnitude(rng, stats, T) * total_scale * step_weight
    x_t = x_prev + mag * direction

    blur_alpha = max_blur_alpha * step_weight
    if blur_alpha > 0:
        x_t = (1 - blur_alpha) * x_t + blur_alpha * moving_average_1d(x_t, embed_smooth_kernel)

    mix_prob, mix_lam = max_mix_prob * step_weight, max_mix_lam * step_weight
    if mix_prob > 0 and mix_lam > 0:
        x_shift = np.roll(x_prev, shift=int(rng.choice([-1, 1])), axis=0)
        mask = (rng.random((T, 1)) < mix_prob).astype(np.float32)
        x_t = (1 - mask) * x_t + mask * ((1 - mix_lam) * x_t + mix_lam * x_shift)

    delta = x_t - x_prev
    delta = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-8) * mag
    meta = {"step_weight": float(step_weight), "delta_mean_applied": float(np.mean(mag)),
            "blur_alpha": float(blur_alpha), "mix_prob": float(mix_prob), "mix_lam": float(mix_lam)}
    return (x_prev + delta).astype(np.float32), delta.astype(np.float32), meta


def articulation_forward_chain_budgeted(
    x_clean: np.ndarray, stats: Dict, T: int = 30, total_scale: float = 1.5,
    noise_smooth_kernel: int = 5, schedule: str = "cosine_full", seed: int = 42,
):
    """x_0 -> ... -> x_T. Returns (states[0..T], deltas[None, 1..T], metas)."""
    states, deltas, metas = [x_clean.astype(np.float32).copy()], [None], [None]
    x = states[0].copy()
    for t, w in enumerate(step_weights(T, schedule), start=1):
        x, d, m = articulation_forward_step_budgeted(
            x, stats, float(w), total_scale=total_scale,
            noise_smooth_kernel=noise_smooth_kernel, seed=seed + t)
        m["t"] = t
        states.append(x); deltas.append(d); metas.append(m)
    return states, deltas, metas


def oracle_reverse_chain(x_T: np.ndarray, deltas: list) -> list:
    """Sanity check: subtracting the true deltas recovers x_0 exactly."""
    T = len(deltas) - 1
    out = [None] * (T + 1)
    x = x_T.astype(np.float32).copy()
    out[T] = x.copy()
    for t in range(T, 0, -1):
        x = x - deltas[t]
        out[t - 1] = x.copy()
    return out


# --------------------------------------------------------------------------- #
# Pronounced chain (transition-targeted)
# --------------------------------------------------------------------------- #
def articulation_forward_step_pronounced(
    x_prev: np.ndarray,
    stats: Dict,
    step_weight: float,
    total_scale: float = 1.30,
    noise_smooth_kernel: int = 9,
    embed_smooth_kernel: int = 7,
    max_blur_alpha: float = 0.85,
    max_mix_prob: float = 0.65,
    max_mix_lam: float = 0.70,
    span_density_base: float = 0.04,
    span_density_scale: float = 0.16,
    span_len_min: int = 3,
    span_len_max: int = 9,
    hard_replace_prob: float = 0.22,
    hard_replace_lam: float = 0.85,
    transition_power: float = 1.75,
    weakening_strength: float = 0.18,
    seed: int = 42,
):
    """Corrupt short spans centred on phoneme transitions (high frame change)."""
    rng = np.random.default_rng(seed)
    x_prev = x_prev.astype(np.float32)
    T, D = x_prev.shape
    fd = frame_diff(x_prev)

    score = np.power(fd + 1e-8, transition_power)
    prob = score / score.sum()
    num_spans = max(2, int((span_density_base + span_density_scale * step_weight) * T))
    span = np.zeros(T, dtype=np.float32)
    for _ in range(num_spans):
        c = int(rng.choice(T, p=prob))
        L = int(rng.integers(span_len_min, span_len_max + 1))
        s = max(0, c - L // 2)
        span[s:min(T, s + L)] = 1.0
    m = span[:, None]

    ramp = 0.35 + 0.65 * step_weight
    direction = _smooth_direction(rng, T, D, noise_smooth_kernel)
    mag = _drift_magnitude(rng, stats, T) * total_scale * (0.45 + 2.0 * step_weight)
    x_t = x_prev + m * (mag * direction)

    blur_alpha = max_blur_alpha * ramp
    x_t = x_t + m * (blur_alpha * (moving_average_1d(x_t, embed_smooth_kernel) - x_t))

    mix_prob, mix_lam = max_mix_prob * ramp, max_mix_lam * ramp
    x_shift = np.roll(x_prev, shift=int(rng.choice([-3, -2, -1, 1, 2, 3])), axis=0)
    gate = (rng.random((T, 1)) < mix_prob).astype(np.float32) * m
    x_t = x_t + gate * (mix_lam * (x_shift - x_t))

    hr_prob, hr_lam = hard_replace_prob * ramp, hard_replace_lam * ramp
    x_shift2 = np.roll(x_prev, shift=int(rng.choice([-2, -1, 1, 2])), axis=0)
    gate2 = (rng.random((T, 1)) < hr_prob).astype(np.float32) * m
    x_t = x_t + gate2 * (hr_lam * (x_shift2 - x_t))

    weaken = 1.0 - weakening_strength * (fd / (fd.max() + 1e-8))[:, None] * m
    x_t = x_t * weaken + x_prev * (1.0 - weaken) * 0.15

    delta = x_t - x_prev
    meta = {"step_weight": float(step_weight), "num_spans": int(num_spans),
            "corrupted_fraction": float(span.mean()),
            "delta_mean_l2": float(np.mean(np.linalg.norm(delta, axis=1)))}
    return x_t.astype(np.float32), delta.astype(np.float32), meta


def articulation_forward_chain_pronounced(x_clean, stats, T=30, schedule="cosine", seed=42, **cfg):
    states, deltas, metas = [x_clean.astype(np.float32).copy()], [None], [None]
    x = states[0].copy()
    for t, w in enumerate(step_weights(T, schedule), start=1):
        x, d, m = articulation_forward_step_pronounced(x, stats, float(w), seed=seed + t, **cfg)
        m["t"] = t
        states.append(x); deltas.append(d); metas.append(m)
    return states, deltas, metas


# --------------------------------------------------------------------------- #
# Drift inside the combined neural forward chain (report 3.5.3)
# --------------------------------------------------------------------------- #
def articulation_drift(
    x: np.ndarray,
    sw: float,
    stats: Dict,
    rng: np.random.Generator,
    drift_scale: float = 0.06,
    noise_kernel: int = 11,
    blur_kernel: int = 5,
    blur_weight: float = 0.05,
    mix_prob: float = 0.04,
    mix_weight: float = 0.05,
) -> np.ndarray:
    """Mild spectral degradation on top of the neural delta.

    1. smooth unit-norm directional noise, magnitude ~ truncated normal * drift_scale * sw
    2. boundary blur with a `blur_kernel` moving average, weight blur_weight * sw
    3. neighbour mixing on a mix_prob * sw fraction of frames, weight mix_weight * sw
    """
    x = x.astype(np.float32)
    T, D = x.shape
    direction = _smooth_direction(rng, T, D, noise_kernel)
    mag = _drift_magnitude(rng, stats, T) * drift_scale * sw
    x_t = x + mag * direction

    a = blur_weight * sw
    x_t = (1 - a) * x_t + a * moving_average_1d(x_t, blur_kernel)

    p, lam = mix_prob * sw, mix_weight * sw
    x_shift = np.roll(x_t, shift=int(rng.choice([-1, 1])), axis=0)
    gate = (rng.random((T, 1)) < p).astype(np.float32)
    x_t = x_t + gate * lam * (x_shift - x_t)
    return x_t.astype(np.float32)
