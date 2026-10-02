"""Train NeuralForwardDiffusion on cached hand-designed chains (report 3.5.2).

Target: the one-step delta x_t - x_{t-1}; inputs x_{t-1}, t and the style
vector of the chain's dysarthric reference. Loss: L1 + L2.
"""

from __future__ import annotations

from _common import base_parser, cosine_schedule, np, resolve, save_ckpt, setup, torch, train_val_split
from torch.utils.data import DataLoader

from dysdiff.data import CachedChainDataset, collate_chain
from dysdiff.models.diffusion import NeuralForwardDiffusion
from dysdiff.models.layers import masked_l1_l2


def main():
    args = base_parser(__doc__).parse_args()
    cfg, device = setup(args)
    nc = cfg["neural_forward"]

    ds = CachedChainDataset(resolve(cfg["paths"]["work_dir"]) / "chains" / "handcrafted",
                            stride=1, items_per_chain=nc["items_per_chain"], seed=cfg["seed"])
    train, val = train_val_split(ds, 0.1, cfg["seed"])
    tl = DataLoader(train, nc["batch_size"], shuffle=True, collate_fn=collate_chain)
    vl = DataLoader(val, nc["batch_size"], collate_fn=collate_chain)

    model = NeuralForwardDiffusion(**nc["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=nc["lr"], weight_decay=1e-4)
    sched = cosine_schedule(opt, nc["epochs"], len(tl))

    def run(b):
        x_t, x_prev, m = b["x_t"].to(device), b["x_prev"].to(device), b["mask"].to(device)
        pred = model(x_prev, b["t"].to(device), b["T"].to(device), b["style"].to(device))
        loss, _, _ = masked_l1_l2(pred, x_t - x_prev, m)
        return loss

    best = float("inf")
    for epoch in range(1, nc["epochs"] + 1):
        model.train()
        tr = []
        for b in tl:
            loss = run(b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            va = float(np.mean([run(b).item() for b in vl]))
        print(f"epoch {epoch:02d} train={np.mean(tr):.6f} val={va:.6f}")
        if va < best:
            best = va
            save_ckpt(model.state_dict(), cfg["checkpoints"]["neural_forward"])


if __name__ == "__main__":
    main()
