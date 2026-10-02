"""Apply the hold-warp elongation operator to clean embeddings (proof of concept).

Saves warped *_frame.npy files plus pointer maps (*_ptr.npy) and advance
labels (*_adv.npy) that scripts/make_timewarp_plots.py visualises.
"""

from __future__ import annotations

from pathlib import Path

from _common import base_parser, np, setup

from dysdiff.forward.hold_warp import hold_warp_once, run_stats_from_ptr


def main():
    p = base_parser(__doc__)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_files", type=int, default=30)
    p.add_argument("--t", type=int, default=None, help="timestep (default: steps-1, strongest)")
    args = p.parse_args()
    cfg, _ = setup(args)
    hw = cfg["hold_warp"]

    files = sorted(Path(args.in_dir).rglob("*_frame.npy"))
    if not files:
        raise SystemExit(f"No *_frame.npy under {args.in_dir}")
    rng = np.random.default_rng(cfg["seed"])
    sel = [files[i] for i in rng.permutation(len(files))[: args.num_files]]
    t = hw["steps"] - 1 if args.t is None else args.t

    agg = []
    for n, fp in enumerate(sel, 1):
        x0 = np.load(fp)
        x_t, ptr, adv, p_t, lmax = hold_warp_once(x0, t, hw["steps"], hw["geom_p"], hw["p95"], hw["schedule"], rng)
        out = Path(args.out_dir) / fp.relative_to(args.in_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, x_t.astype(np.float32))
        np.save(str(out).replace("_frame.npy", "_ptr.npy"), ptr)
        np.save(str(out).replace("_frame.npy", "_adv.npy"), adv)
        s = run_stats_from_ptr(ptr)
        agg.append(s)
        print(f"[{n:03d}/{len(sel)}] {fp.name} | T {x0.shape[0]}->{x_t.shape[0]} | p_t={p_t:.3f} Lmax={lmax} "
              f"| mean_run {s['mean_run']:.2f} p95 {s['p95_run']:.1f} p_switch {s['p_switch']:.3f}")

    keys = ["T", "mean_run", "p95_run", "p_switch"]
    print("aggregate:", {k: round(float(np.mean([a[k] for a in agg])), 3) for k in keys})


if __name__ == "__main__":
    main()
