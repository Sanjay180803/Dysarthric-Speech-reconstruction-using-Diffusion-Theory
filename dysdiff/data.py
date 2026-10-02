"""Datasets, manifest handling and padding collate functions."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_embedding, load_json

MANIFEST_FIELDS = ["utt_id", "group", "speaker", "session", "frame_path", "utt_path", "wav_path", "num_frames", "text"]


# --------------------------------------------------------------------------- #
# Manifest (written by scripts/extract_embeddings.py)
# --------------------------------------------------------------------------- #
def read_manifest(path: str, groups: Optional[List[str]] = None) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if groups:
        rows = [r for r in rows if r["group"] in groups]
    return rows


def write_manifest(rows: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


# --------------------------------------------------------------------------- #
# Padding
# --------------------------------------------------------------------------- #
def pad_sequences(seqs: List[np.ndarray]):
    """List of (L_i, D) arrays -> (B, L_max, D) tensor + (B, L_max) bool mask."""
    L = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    out = torch.zeros(len(seqs), L, D)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = torch.from_numpy(np.asarray(s, dtype=np.float32))
        mask[i, : s.shape[0]] = True
    return out, mask


# --------------------------------------------------------------------------- #
# Per-factor reverse training pairs (x_t, x_{t-1})
# --------------------------------------------------------------------------- #
class StepPairDataset(Dataset):
    """Reads the `manifest.json` of .npz pairs written by `scripts/build_factor_chains.py`."""

    def __init__(self, pair_dir: str):
        self.files = load_json(Path(pair_dir) / "manifest.json")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        return {"x_t": z["x_t"].astype(np.float32), "x_prev": z["x_prev"].astype(np.float32),
                "t": int(z["t"][0]), "T": int(z["T"][0])}


def collate_step_pairs(batch):
    x_t, mask = pad_sequences([b["x_t"] for b in batch])
    x_prev, _ = pad_sequences([b["x_prev"] for b in batch])
    return {"x_t": x_t, "x_prev": x_prev, "mask": mask,
            "t": torch.tensor([b["t"] for b in batch]), "T": torch.tensor([b["T"] for b in batch])}


# --------------------------------------------------------------------------- #
# StatsEncoder proxy objective: (dysarthric, randomly associated clean)
# --------------------------------------------------------------------------- #
class UnpairedDysCleanDataset(Dataset):
    def __init__(self, dys_files: List[str], clean_files: List[str], max_len: int = 600, seed: int = 0):
        self.dys, self.clean, self.max_len = dys_files, clean_files, max_len
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dys)

    def _crop(self, x):
        if x.shape[0] <= self.max_len:
            return x
        s = self.rng.randint(0, x.shape[0] - self.max_len)
        return x[s:s + self.max_len]

    def __getitem__(self, idx):
        return {"dys": self._crop(load_embedding(self.dys[idx])),
                "clean": self._crop(load_embedding(self.rng.choice(self.clean)))}


def collate_unpaired(batch):
    dys, dm = pad_sequences([b["dys"] for b in batch])
    clean, cm = pad_sequences([b["clean"] for b in batch])
    return {"dys": dys, "dys_mask": dm, "clean": clean, "clean_mask": cm}


# --------------------------------------------------------------------------- #
# Cached diffusion chains: one .npz per utterance with states (T+1, L, D)
# --------------------------------------------------------------------------- #
class CachedChainDataset(Dataset):
    """Samples (x_t, x_{t-stride}) from cached chains.

    stride = 1 gives one-step targets (neural forward model and the report's
    description). stride = T // n_reverse_steps matches the 5-step sampler.
    Items also carry the style vector and transcript when cached.
    """

    def __init__(self, chain_dir: str, stride: int = 1, items_per_chain: int = 8, seed: int = 0):
        self.files = sorted(str(p) for p in Path(chain_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No cached chains in {chain_dir}")
        self.stride, self.items_per_chain = stride, items_per_chain
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files) * self.items_per_chain

    def __getitem__(self, idx):
        z = np.load(self.files[idx % len(self.files)], allow_pickle=False)
        states = z["states"]
        T = states.shape[0] - 1
        t = self.rng.randint(1, T)
        prev = max(t - self.stride, 0)
        return {
            "x_t": states[t].astype(np.float32),
            "x_prev": states[prev].astype(np.float32),
            "t": t, "T": T,
            "style": z["style"].astype(np.float32) if "style" in z else np.zeros(128, np.float32),
            "text": str(z["text"]) if "text" in z else "",
        }


def collate_chain(batch):
    x_t, mask = pad_sequences([b["x_t"] for b in batch])
    x_prev, _ = pad_sequences([b["x_prev"] for b in batch])
    return {"x_t": x_t, "x_prev": x_prev, "mask": mask,
            "t": torch.tensor([b["t"] for b in batch]), "T": torch.tensor([b["T"] for b in batch]),
            "style": torch.from_numpy(np.stack([b["style"] for b in batch])),
            "text": [b["text"] for b in batch]}


# --------------------------------------------------------------------------- #
# Embedding -> mel pairs
# --------------------------------------------------------------------------- #
class EmbeddingMelDataset(Dataset):
    """(clean embedding, log-mel of the same clean recording). Mels are cached as .npy."""

    def __init__(self, rows: List[Dict], mel_cache_dir: str):
        self.rows = [r for r in rows if r.get("wav_path")]
        self.cache = Path(mel_cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def _mel(self, row):
        path = self.cache / f"{row['utt_id']}_mel.npy"
        if path.exists():
            return np.load(path)
        import soundfile as sf
        import librosa
        from .models.embedding_to_mel import compute_mel

        wav, sr = sf.read(row["wav_path"])
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        mel = compute_mel(torch.tensor(wav, dtype=torch.float32)).numpy()
        np.save(path, mel)
        return mel

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {"emb": load_embedding(row["frame_path"]), "mel": self._mel(row).astype(np.float32)}


def collate_emb_mel(batch):
    """Each item is predicted separately to its own mel length, so keep a list."""
    return batch
