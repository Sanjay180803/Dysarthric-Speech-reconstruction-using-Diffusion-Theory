"""Elongation statistics of real speech (analysis.ipynb).

A joint k-means codebook (100 clusters) is fit over clean and moderate
embeddings; runs of identical cluster labels measure how long speakers hold
phone-like units. The pooled moderate statistics (mean run, p95 run,
geometric p = 1 / mean run) parameterise the hold-warp operator.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .forward.hold_warp import run_lengths


def train_joint_kmeans(files_by_group: Dict[str, List[str]], n_clusters: int = 100,
                       batch_frames: int = 4000, max_files_per_group: int = 300, seed: int = 0):
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    sampled = [f for files in files_by_group.values() for f in files[:max_files_per_group]]
    rng.shuffle(sampled)
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=seed,
                         n_init="auto", reassignment_ratio=0.01, max_no_improvement=20)
    used = 0
    for fp in sampled:
        x = np.load(fp)
        if x.ndim != 2 or x.shape[0] == 0:
            continue
        n = min(batch_frames, x.shape[0])
        km.partial_fit(x[rng.choice(x.shape[0], size=n, replace=False)].astype(np.float32))
        used += 1
    if used == 0:
        raise RuntimeError("No usable frames to fit k-means.")
    return km


def gamma_moments_fit(x: np.ndarray):
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 10:
        return float("nan"), float("nan")
    mu, var = float(x.mean()), float(x.var())
    if var <= 1e-12 or mu <= 1e-12:
        return float("nan"), float("nan")
    return mu * mu / var, var / mu


def elongation_stats(files: List[str], km, low_norm_percentile: float = 10.0) -> Dict:
    """Run-length, switch-rate, instability-distance and deletion-proxy statistics."""
    centers = km.cluster_centers_.astype(np.float32)
    xs = [np.load(fp).astype(np.float32) for fp in files]
    xs = [x for x in xs if x.size]
    thr = float(np.percentile(np.concatenate([np.linalg.norm(x, axis=1) for x in xs]), low_norm_percentile))

    runs, switches, dists, n_del, n_frames = [], [], [], 0, 0
    for x in xs:
        labels = km.predict(x)
        runs.append(run_lengths(labels))
        switches.append(float(np.mean(labels[1:] != labels[:-1])) if len(labels) > 1 else 0.0)
        dists.append(np.linalg.norm(x - centers[labels], axis=1))
        n_del += int(np.sum(np.linalg.norm(x, axis=1) < thr))
        n_frames += x.shape[0]

    r = np.concatenate(runs)
    d = np.concatenate(dists)
    k, theta = gamma_moments_fit(d)
    return {
        "run_length": {"mean": float(r.mean()), "p95": float(np.percentile(r, 95)), "geom_p": float(1.0 / r.mean())},
        "switch": {"p_switch": float(np.mean(switches))},
        "instability_distance": {"mean": float(d.mean()), "var": float(d.var()), "gamma_k": k, "gamma_theta": theta},
        "deletion_proxy": {"p_del": n_del / n_frames, "norm_threshold": thr},
        "total_frames": int(n_frames),
        "num_files": len(xs),
    }
