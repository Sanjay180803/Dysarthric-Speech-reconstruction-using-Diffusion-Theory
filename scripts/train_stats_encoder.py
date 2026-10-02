"""Train the StatsEncoder with the unpaired proxy objective (report 3.5.1).

Input: a moderate-dysarthric utterance. Target: a fixed projection of the
per-dimension mean and std of a randomly associated clean utterance.
Saves {"encoder", "target_head"} like runs/exp1/stats_encoder_best.pt.
"""

from __future__ import annotations

from _common import base_parser, group_files, np, save_ckpt, setup, torch, train_val_split
from torch.utils.data import DataLoader

from dysdiff.data import UnpairedDysCleanDataset, collate_unpaired
from dysdiff.models.stats_encoder import StatsEncoderTrainer


def main():
    args = base_parser(__doc__).parse_args()
    cfg, device = setup(args)
    sc = cfg["stats_encoder"]

    ds = UnpairedDysCleanDataset(group_files(cfg, [cfg["dysarthric_group"]]),
                                 group_files(cfg, cfg["clean_groups"]), max_len=sc["max_len"], seed=cfg["seed"])
    train, val = train_val_split(ds, 0.1, cfg["seed"])
    tl = DataLoader(train, sc["batch_size"], shuffle=True, collate_fn=collate_unpaired)
    vl = DataLoader(val, sc["batch_size"], collate_fn=collate_unpaired)

    model = StatsEncoderTrainer(freeze_target_head=sc["freeze_target_head"]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=sc["lr"], weight_decay=1e-4)

    def run(b):
        return model.loss(b["dys"].to(device), b["dys_mask"].to(device), b["clean"].to(device), b["clean_mask"].to(device))

    best = float("inf")
    for epoch in range(1, sc["epochs"] + 1):
        model.train()
        tr = []
        for b in tl:
            loss = run(b)
            opt.zero_grad(); loss.backward(); opt.step(); tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            va = float(np.mean([run(b).item() for b in vl]))
        print(f"epoch {epoch:02d} train={np.mean(tr):.5f} val={va:.5f}")
        if va < best:
            best = va
            save_ckpt(model.state_dict_for_release(), cfg["checkpoints"]["stats_encoder"])


if __name__ == "__main__":
    main()
