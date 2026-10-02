"""Train the gated, text-conditioned reverse denoiser and TextEncoder jointly (report 3.5.4-3.5.5).

Pairs (x_t, x_{t-stride}) come from cached neural chains. With stride = T / n_steps
(6 by default) each training target matches one step of the 5-step sampler.

Loss = 1.0 * L(delta_hat, x_t - x_{t-stride})
     + 0.5 * L(x_t - delta_hat, x_{t-stride})
     + 0.3 * smoothness(x_t - delta_hat)
with L = L1 + L2 and a cosine learning-rate schedule.
"""

from __future__ import annotations

from _common import base_parser, cosine_schedule, np, resolve, save_ckpt, setup, torch, train_val_split
from torch.utils.data import DataLoader

from dysdiff.data import CachedChainDataset, collate_chain
from dysdiff.models.diffusion import GatedReverseDenoiser, TextEncoder, encode_text
from dysdiff.models.layers import masked_l1_l2, temporal_smoothness


def main():
    args = base_parser(__doc__).parse_args()
    cfg, device = setup(args)
    rc = cfg["reverse"]
    w = rc["loss_weights"]

    ds = CachedChainDataset(resolve(cfg["paths"]["work_dir"]) / "chains" / "neural",
                            stride=rc["stride"], items_per_chain=rc["items_per_chain"], seed=cfg["seed"])
    if not any(str(np.load(f)["text"]) for f in ds.files[:50]):
        print("warning: no transcripts in cached chains; text conditioning will be uninformative")
    train, val = train_val_split(ds, 0.1, cfg["seed"])
    tl = DataLoader(train, rc["batch_size"], shuffle=True, collate_fn=collate_chain)
    vl = DataLoader(val, rc["batch_size"], collate_fn=collate_chain)

    model = GatedReverseDenoiser(**rc["model"]).to(device)
    text_enc = TextEncoder().to(device)
    params = list(model.parameters()) + list(text_enc.parameters())
    opt = torch.optim.AdamW(params, lr=rc["lr"], weight_decay=1e-4)
    sched = cosine_schedule(opt, rc["epochs"], len(tl))

    def run(b):
        x_t, x_prev, m = b["x_t"].to(device), b["x_prev"].to(device), b["mask"].to(device)
        tvec = text_enc(encode_text(b["text"]).to(device))
        delta, gate = model(x_t, b["t"].to(device), b["T"].to(device), tvec, m)
        x_hat = x_t - delta
        l_delta, _, _ = masked_l1_l2(delta, x_t - x_prev, m)
        l_prev, _, _ = masked_l1_l2(x_hat, x_prev, m)
        l_smooth = temporal_smoothness(x_hat, m)
        loss = w["delta"] * l_delta + w["prev"] * l_prev + w["smooth"] * l_smooth
        return loss, float(gate.mean())

    best = float("inf")
    for epoch in range(1, rc["epochs"] + 1):
        model.train(); text_enc.train()
        tr, gates = [], []
        for b in tl:
            loss, g = run(b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); tr.append(loss.item()); gates.append(g)
        model.eval(); text_enc.eval()
        with torch.no_grad():
            va = float(np.mean([run(b)[0].item() for b in vl]))
        print(f"epoch {epoch:02d} train={np.mean(tr):.6f} val={va:.6f} mean_gate={np.mean(gates):.3f}")
        if va < best:
            best = va
            save_ckpt({"model": model.state_dict(), "text_encoder": text_enc.state_dict()},
                      cfg["checkpoints"]["reverse"])


if __name__ == "__main__":
    main()
