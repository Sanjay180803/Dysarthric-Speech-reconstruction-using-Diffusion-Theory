"""Run one hand-designed forward chain over clean embeddings and save (x_t, x_{t-1}) pairs.

--factor articulation_budgeted | articulation | phonation | pause_rhythm
Pairs are written as compressed .npz files plus manifest.json, the training
data for scripts/train_factor_reverse.py.
"""

from __future__ import annotations

from pathlib import Path

from _common import base_parser, group_files, load_articulation_stats, np, resolve, setup

from dysdiff.forward.articulation import articulation_forward_chain_budgeted, articulation_forward_chain_pronounced
from dysdiff.forward.pause_rhythm import pause_rhythm_forward_chain
from dysdiff.forward.phonation import phonation_forward_chain
from dysdiff.utils import evaluate_pair, load_embedding, save_json


def make_chain(factor, cfg, stats):
    fcfg = dict(cfg["factors"][factor])
    T, sched = cfg["diffusion"]["T"], cfg["diffusion"]["schedule"]
    if factor == "articulation_budgeted":
        return lambda x, seed: articulation_forward_chain_budgeted(x, stats, T=T, seed=seed, **fcfg)
    if factor == "articulation":
        return lambda x, seed: articulation_forward_chain_pronounced(x, stats, T=T, schedule=sched, seed=seed, **fcfg)
    if factor == "phonation":
        return lambda x, seed: phonation_forward_chain(x, T=T, schedule=sched, seed=seed, **fcfg)
    if factor == "pause_rhythm":
        return lambda x, seed: pause_rhythm_forward_chain(x, T=T, schedule=sched, seed=seed, **fcfg)
    raise ValueError(factor)


def main():
    p = base_parser(__doc__)
    p.add_argument("--factor", required=True, choices=["articulation_budgeted", "articulation", "phonation", "pause_rhythm"])
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()
    cfg, _ = setup(args)

    chain = make_chain(args.factor, cfg, load_articulation_stats(cfg))
    files = group_files(cfg, cfg["clean_groups"])[: args.max_files]
    out_dir = resolve(cfg["paths"]["work_dir"]) / "pairs" / args.factor
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, f in enumerate(files, 1):
        x0 = load_embedding(f)
        states, _, _ = chain(x0, cfg["seed"] + i)
        T = len(states) - 1
        base = Path(f).name.replace("_frame.npy", "")
        for t in range(1, T + 1):
            path = out_dir / f"{base}_t{t:02d}.npz"
            np.savez_compressed(path, x_t=states[t], x_prev=states[t - 1],
                                t=np.array([t]), T=np.array([T]))
            manifest.append(str(path))
        if i % 50 == 0 or i == len(files):
            s = evaluate_pair(x0, states[-1])
            print(f"[{i}/{len(files)}] {base} clean->x_T cos={s['cosine']:.4f} mae={s['mean_abs_error']:.4f}")
    save_json(manifest, out_dir / "manifest.json")
    print(f"{len(manifest)} pairs -> {out_dir}")


if __name__ == "__main__":
    main()
