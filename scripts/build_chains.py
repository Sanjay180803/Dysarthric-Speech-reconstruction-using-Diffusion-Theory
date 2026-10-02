"""Cache full diffusion chains (states x_0..x_T) for the combined pipeline.

--kind handcrafted  articulation -> phonation -> pause/rhythm chain. Each clean
                    utterance is paired with a random moderate-dysarthric
                    reference whose drift statistics set the articulation
                    magnitude; the reference's StatsEncoder style vector is
                    stored with the chain. Training data for the neural forward model.
--kind neural       local elongation + neural forward deltas + articulation
                    drift (report Fig. 4). Training data for the reverse model.

Each chain is one .npz: states (T+1, L, 1024) float16, style (128,), text, ptr.
"""

from __future__ import annotations

import random
from pathlib import Path

from _common import base_parser, group_files, load_articulation_stats, np, resolve, setup, torch

from dysdiff.data import read_manifest
from dysdiff.forward.articulation import utterance_drift_stats
from dysdiff.models.diffusion import NeuralForwardDiffusion
from dysdiff.models.stats_encoder import load_stats_encoder
from dysdiff.pipeline import handcrafted_chain, neural_forward_chain
from dysdiff.utils import load_embedding, seq_cosine_mean


def main():
    p = base_parser(__doc__)
    p.add_argument("--kind", required=True, choices=["handcrafted", "neural"])
    p.add_argument("--max_utterances", type=int, default=None)
    args = p.parse_args()
    cfg, device = setup(args)
    T = cfg["diffusion"]["T"]
    rng = random.Random(cfg["seed"])

    texts = {}
    manifest = resolve(cfg["paths"]["manifest"])
    if manifest.exists():
        texts = {r["frame_path"]: r["text"] for r in read_manifest(str(manifest))}

    clean = group_files(cfg, cfg["clean_groups"])
    n = args.max_utterances or cfg["neural_forward"].get("max_utterances") or len(clean)
    clean = rng.sample(clean, min(n, len(clean)))
    dys = group_files(cfg, [cfg["dysarthric_group"]])

    enc = load_stats_encoder(str(resolve(cfg["checkpoints"]["stats_encoder"])), device).to(device)
    fwd = None
    if args.kind == "neural":
        fwd = NeuralForwardDiffusion(**cfg["neural_forward"]["model"]).to(device)
        fwd.load_state_dict(torch.load(resolve(cfg["checkpoints"]["neural_forward"]), map_location=device))
    global_stats = load_articulation_stats(cfg)

    out_dir = resolve(cfg["paths"]["work_dir"]) / "chains" / args.kind
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(clean, 1):
        x0 = load_embedding(f)
        ref = load_embedding(rng.choice(dys))
        with torch.no_grad():
            style = enc(torch.from_numpy(ref).unsqueeze(0).to(device)).squeeze(0).cpu()
        if args.kind == "handcrafted":
            states = handcrafted_chain(x0, utterance_drift_stats(ref), cfg["factors"], T,
                                       cfg["diffusion"]["schedule"], seed=cfg["seed"] + i)
            ptr = np.arange(x0.shape[0])
        else:
            states, ptr = neural_forward_chain(x0, fwd, style, global_stats, T=T,
                                               alpha_scale=cfg["chain"]["alpha_scale"],
                                               elongation=cfg["chain"]["elongation"],
                                               drift=cfg["chain"]["drift"], seed=cfg["seed"] + i, device=device)
        name = Path(f).name.replace("_frame.npy", "")
        np.savez(out_dir / f"{name}.npz", states=states.astype(np.float16), style=style.numpy(),
                 text=np.array(texts.get(f, "")), ptr=ptr)
        if i % 50 == 0 or i == len(clean):
            print(f"[{i}/{len(clean)}] {name} L={states.shape[1]} "
                  f"cos(x_0, x_T)={seq_cosine_mean(states[0].astype(np.float32), states[-1].astype(np.float32)):.4f}")


if __name__ == "__main__":
    main()
