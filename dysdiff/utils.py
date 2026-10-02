"""Shared helpers: smoothing, similarity metrics, seeding, config and file I/O."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #
def moving_average_1d(x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Edge-padded moving average along the time axis of a (T, D) array.

    Vectorised with a cumulative sum; numerically identical to the per-frame
    loop used in the original notebooks (same padding, same window).
    """
    x = np.asarray(x, dtype=np.float32)
    if kernel_size <= 1 or len(x) < 2:
        return x.copy()
    T = x.shape[0]
    pad = kernel_size // 2
    x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge").astype(np.float64)
    c = np.cumsum(x_pad, axis=0)
    c = np.concatenate([np.zeros((1, x.shape[1])), c], axis=0)
    out = (c[kernel_size:] - c[:-kernel_size]) / kernel_size
    return out[:T].astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Frame-wise cosine similarity between two (T, D) arrays."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.sum(a_n * b_n, axis=1)


def seq_cosine_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(cosine_sim(a, b, eps)))


def local_diff_energy(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    d = x[1:] - x[:-1]
    return float(np.mean(np.sum(d * d, axis=1)))


def adjacent_cosine(x: np.ndarray) -> float:
    if len(x) < 2:
        return 1.0
    return float(np.mean(cosine_sim(x[:-1], x[1:])))


def evaluate_pair(clean: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    """Embedding-space distance summary between two equal-length sequences."""
    diff = other - clean
    return {
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_l2_per_frame": float(np.mean(np.linalg.norm(diff, axis=1))),
        "cosine": seq_cosine_mean(clean, other),
    }


def frame_diff(x: np.ndarray) -> np.ndarray:
    """L2 distance between consecutive frames, left-padded to length T."""
    d = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return np.pad(d, (1, 0), mode="edge").astype(np.float32)


# --------------------------------------------------------------------------- #
# Reproducibility / config / I/O
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | os.PathLike) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_frame_files(folder: str | os.PathLike) -> List[str]:
    """All `*_frame.npy` embedding files under a folder (recursive, sorted)."""
    return sorted(str(p) for p in Path(folder).rglob("*_frame.npy"))


def load_embedding(path: str | os.PathLike) -> np.ndarray:
    x = np.load(path)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D (T, D) embedding, got {x.shape} for {path}")
    return x.astype(np.float32)


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | os.PathLike):
    with open(path) as f:
        return json.load(f)
