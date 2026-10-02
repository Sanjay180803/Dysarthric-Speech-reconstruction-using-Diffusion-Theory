"""Helpers shared by the command-line scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dysdiff.data import read_manifest  # noqa: E402
from dysdiff.utils import collect_frame_files, load_config, load_json, seed_everything  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    return p


def setup(args):
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    dev = cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
    return cfg, device


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def group_files(cfg, groups):
    """Frame files for the given groups, from the manifest if present, else by folder."""
    manifest = resolve(cfg["paths"]["manifest"])
    if manifest.exists():
        return [r["frame_path"] for r in read_manifest(str(manifest), groups)]
    root = resolve(cfg["paths"]["embeddings_root"])
    return [f for g in groups for f in collect_frame_files(root / g)]


def load_articulation_stats(cfg):
    p = resolve(cfg["stats"]["articulation_file"])
    if p.exists():
        return load_json(p)
    # measured on TORGO (articulation.ipynb); used if compute_stats.py has not been run
    return {"drift_l2_mean": 1.5735, "drift_l2_std": 1.7193, "drift_l2_p95": 5.3654}


def train_val_split(dataset, val_fraction: float, seed: int):
    n_val = max(1, int(len(dataset) * val_fraction))
    return torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val],
                                         generator=torch.Generator().manual_seed(seed))


def save_ckpt(obj, path):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return path


def cosine_schedule(optimizer, epochs: int, steps_per_epoch: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * steps_per_epoch))


__all__ = ["ROOT", "np", "torch", "base_parser", "setup", "resolve", "group_files",
           "load_articulation_stats", "train_val_split", "save_ckpt", "cosine_schedule"]
