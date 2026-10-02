"""Train the hold-warp advance predictor (report 3.3).

Pairs are generated on the fly: a clean utterance is hold-warped at a random
step t, and the pointer map gives exact advance/hold labels. Loss: weighted BCE.
"""

from __future__ import annotations

import random

from _common import base_parser, group_files, np, save_ckpt, setup, torch
from torch.utils.data import DataLoader, Dataset

from dysdiff.data import pad_sequences
from dysdiff.forward.hold_warp import hold_warp_once
from dysdiff.models.advance import AdvancePredictor, weighted_bce
from dysdiff.utils import load_embedding


class HoldWarpDataset(Dataset):
    def __init__(self, files, hw, max_len, seed):
        self.files, self.hw, self.max_len = files, hw, max_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x0 = load_embedding(self.files[i])
        if x0.shape[0] > self.max_len // 3:
            s = int(self.rng.integers(0, x0.shape[0] - self.max_len // 3 + 1))
            x0 = x0[s:s + self.max_len // 3]
        t = int(self.rng.integers(1, self.hw["steps"]))
        x_t, _, adv, _, _ = hold_warp_once(x0, t, self.hw["steps"], self.hw["geom_p"], self.hw["p95"],
                                           self.hw["schedule"], self.rng)
        return x_t, adv, t


def collate(batch):
    x, mask = pad_sequences([b[0] for b in batch])
    adv, _ = pad_sequences([b[1][:, None] for b in batch])
    return x, adv.squeeze(-1), mask, torch.tensor([b[2] for b in batch])


def main():
    args = base_parser(__doc__).parse_args()
    cfg, device = setup(args)
    ac, hw = cfg["advance"], cfg["hold_warp"]
    files = group_files(cfg, cfg["clean_groups"])
    random.Random(cfg["seed"]).shuffle(files)
    n_val = max(1, len(files) // 10)
    tl = DataLoader(HoldWarpDataset(files[n_val:], hw, ac["max_len"], cfg["seed"]), ac["batch_size"],
                    shuffle=True, collate_fn=collate)
    vl = DataLoader(HoldWarpDataset(files[:n_val], hw, ac["max_len"], cfg["seed"] + 1), ac["batch_size"],
                    collate_fn=collate)

    model = AdvancePredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ac["lr"], weight_decay=1e-4)
    T = torch.tensor(hw["steps"])

    def run(b):
        x, adv, m, t = (v.to(device) for v in b)
        logits = model(x, t, T.to(device).expand_as(t))
        loss = weighted_bce(logits, adv, m)
        acc = (((logits > 0).float() == adv).float() * m).sum() / m.sum()
        return loss, float(acc)

    best = float("inf")
    for epoch in range(1, ac["epochs"] + 1):
        model.train()
        tr = []
        for b in tl:
            loss, _ = run(b)
            opt.zero_grad(); loss.backward(); opt.step(); tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            res = [run(b) for b in vl]
        va, acc = float(np.mean([r[0].item() for r in res])), float(np.mean([r[1] for r in res]))
        print(f"epoch {epoch:02d} train={np.mean(tr):.4f} val={va:.4f} frame_acc={acc:.3f}")
        if va < best:
            best = va
            save_ckpt(model.state_dict(), cfg["checkpoints"]["advance"])


if __name__ == "__main__":
    main()
