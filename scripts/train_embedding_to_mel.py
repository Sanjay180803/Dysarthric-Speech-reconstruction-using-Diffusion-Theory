"""Train EmbeddingToMel on clean speech: wav2vec2 frames -> HiFi-GAN-compatible log-mels.

Aligned 2 s crops (100 embedding frames = 125 mel frames) are batched; L1 + L2 loss.
"""

from __future__ import annotations

import random

from _common import base_parser, np, resolve, save_ckpt, setup, torch
from torch.utils.data import DataLoader

from dysdiff.data import EmbeddingMelDataset, read_manifest
from dysdiff.models.embedding_to_mel import EMB_HOP_SEC, MEL_CFG, EmbeddingToMel

MEL_PER_EMB = EMB_HOP_SEC * MEL_CFG["sample_rate"] / MEL_CFG["hop_length"]  # 1.25


def make_collate(window: int, seed: int):
    rng = random.Random(seed)
    mel_win = int(round(window * MEL_PER_EMB))

    def collate(batch):
        embs, mels = [], []
        for item in batch:
            e, m = item["emb"], item["mel"]
            n = min(e.shape[0], int(m.shape[1] / MEL_PER_EMB))
            if n < window:  # pad short utterances by edge repetition
                e = np.pad(e[:n], ((0, window - n), (0, 0)), mode="edge")
                m = np.pad(m[:, : int(round(n * MEL_PER_EMB))], ((0, 0), (0, mel_win)), mode="edge")
                s = 0
            else:
                s = rng.randint(0, n - window)
            ms = int(round(s * MEL_PER_EMB))
            embs.append(e[s:s + window]); mels.append(m[:, ms:ms + mel_win])
        return torch.from_numpy(np.stack(embs)), torch.from_numpy(np.stack(mels))

    return collate


def main():
    args = base_parser(__doc__).parse_args()
    cfg, device = setup(args)
    ec = cfg["embedding_to_mel"]
    rows = read_manifest(str(resolve(cfg["paths"]["manifest"])), cfg["clean_groups"])
    random.Random(cfg["seed"]).shuffle(rows)
    n_val = max(1, len(rows) // 20)
    cache = resolve(cfg["paths"]["work_dir"]) / "mel_cache"
    collate = make_collate(ec["window_frames"], cfg["seed"])
    tl = DataLoader(EmbeddingMelDataset(rows[n_val:], cache), ec["batch_size"], shuffle=True, collate_fn=collate)
    vl = DataLoader(EmbeddingMelDataset(rows[:n_val], cache), ec["batch_size"], collate_fn=collate)

    model = EmbeddingToMel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ec["lr"], weight_decay=1e-4)
    mel_win = int(round(ec["window_frames"] * MEL_PER_EMB))

    def run(e, m):
        pred = model(e.to(device), target_len=mel_win)
        m = m.to(device)
        return torch.abs(pred - m).mean() + ((pred - m) ** 2).mean()

    best = float("inf")
    for epoch in range(1, ec["epochs"] + 1):
        model.train()
        tr = []
        for e, m in tl:
            loss = run(e, m)
            opt.zero_grad(); loss.backward(); opt.step(); tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            va = float(np.mean([run(e, m).item() for e, m in vl]))
        print(f"epoch {epoch:02d} train={np.mean(tr):.4f} val={va:.4f}")
        if va < best:
            best = va
            save_ckpt(model.state_dict(), cfg["checkpoints"]["embedding_to_mel"])


if __name__ == "__main__":
    main()
