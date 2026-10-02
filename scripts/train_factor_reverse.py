"""Train a per-factor reverse denoiser on pairs from build_factor_chains.py.

articulation_budgeted -> ReverseStepModel (predicts delta; MSE on delta and on x_{t-1})
others                -> ReverseUNetLite  (predicts x_{t-1}; L1 + L2)
Evaluation reports clean->x_T vs clean->recovered cosine and MAE on held-out chains.
"""

from __future__ import annotations

from _common import (base_parser, group_files, load_articulation_stats, np, resolve, save_ckpt, setup, torch,
                     train_val_split)
from build_factor_chains import make_chain
from torch.utils.data import DataLoader

from dysdiff.data import StepPairDataset, collate_step_pairs
from dysdiff.models.layers import masked_l1_l2, masked_mse
from dysdiff.models.reverse_factor import ReverseStepModel, ReverseUNetLite, run_reverse_chain
from dysdiff.utils import load_embedding, seq_cosine_mean


def step_loss(model, batch, device, budgeted):
    x_t, x_prev, mask = batch["x_t"].to(device), batch["x_prev"].to(device), batch["mask"].to(device)
    t, T = batch["t"].to(device), batch["T"].to(device)
    if budgeted:
        delta_hat = model(x_t, t, T)
        return masked_mse(delta_hat, x_t - x_prev, mask) + masked_mse(x_t - delta_hat, x_prev, mask)
    loss, _, _ = masked_l1_l2(model(x_t, t, T), x_prev, mask)
    return loss


def main():
    p = base_parser(__doc__)
    p.add_argument("--factor", required=True, choices=["articulation_budgeted", "articulation", "phonation", "pause_rhythm"])
    p.add_argument("--eval_files", type=int, default=20)
    args = p.parse_args()
    cfg, device = setup(args)
    tc = cfg["factor_training"]
    budgeted = args.factor == "articulation_budgeted"

    ds = StepPairDataset(resolve(cfg["paths"]["work_dir"]) / "pairs" / args.factor)
    train, val = train_val_split(ds, tc["val_fraction"], cfg["seed"])
    tl = DataLoader(train, tc["batch_size"], shuffle=True, collate_fn=collate_step_pairs)
    vl = DataLoader(val, tc["batch_size"], collate_fn=collate_step_pairs)

    model = (ReverseStepModel() if budgeted else ReverseUNetLite()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    ckpt = f"{cfg['paths']['work_dir']}/factor_{args.factor}/best.pt"
    best = float("inf")
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        tr = []
        for b in tl:
            loss = step_loss(model, b, device, budgeted)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            va = float(np.mean([step_loss(model, b, device, budgeted).item() for b in vl]))
        print(f"epoch {epoch:02d} train={np.mean(tr):.6f} val={va:.6f}")
        if va < best:
            best = va
            save_ckpt(model.state_dict(), ckpt)

    # Chain-level evaluation with fresh corruption seeds
    model.load_state_dict(torch.load(resolve(ckpt), map_location=device))
    chain = make_chain(args.factor, cfg, load_articulation_stats(cfg))
    files = group_files(cfg, cfg["clean_groups"])[-args.eval_files:]
    res = []
    for i, f in enumerate(files):
        x0 = load_embedding(f)
        states, _, _ = chain(x0, 10_000 + i)
        rec = run_reverse_chain(model, states[-1], len(states) - 1, device)
        res.append((seq_cosine_mean(x0, states[-1]), seq_cosine_mean(x0, rec),
                    float(np.abs(x0 - states[-1]).mean()), float(np.abs(x0 - rec).mean())))
    r = np.mean(res, axis=0)
    print(f"clean->x_T cos {r[0]:.4f} | clean->recovered cos {r[1]:.4f} | MAE {r[2]:.4f} -> {r[3]:.4f}")


if __name__ == "__main__":
    main()
