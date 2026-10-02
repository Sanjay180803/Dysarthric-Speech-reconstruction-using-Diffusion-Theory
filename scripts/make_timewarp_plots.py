"""Plot hold-warp statistics: run-length histogram, before/after summary, pointer map, advance labels."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def find_files(root: Path, suffix: str):
    return sorted(root.rglob(f"*{suffix}"))

def run_lengths_from_ptr(ptr: np.ndarray):
    if ptr.size == 0:
        return np.array([], dtype=np.int32)
    # runs of constant ptr
    runs = []
    cur = 1
    for j in range(1, ptr.size):
        if ptr[j] == ptr[j-1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.asarray(runs, dtype=np.int32)

def stats_from_runs(runs: np.ndarray):
    if runs.size == 0:
        return {"mean_run": 0.0, "p95_run": 0.0}
    return {"mean_run": float(runs.mean()), "p95_run": float(np.quantile(runs, 0.95))}

def p_switch_from_ptr(ptr: np.ndarray):
    if ptr.size < 2:
        return 0.0
    return float((ptr[1:] != ptr[:-1]).mean())

def cosine_sim(a: np.ndarray, b: np.ndarray, eps=1e-9):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    # assume a and b have same length; compare frame-to-frame
    return np.sum(a * b, axis=1)

def compute_embedding_run_stats(x: np.ndarray, sim_threshold: float = 0.97):
    """
    Approx run length by counting consecutive frames with cosine similarity >= threshold.
    This is a proxy if you want "embedding-run stats" (like you printed earlier).
    """
    if x.shape[0] < 2:
        return {"mean_run": float(x.shape[0]), "p95_run": float(x.shape[0]), "p_switch": 0.0}
    sims = cosine_sim(x[:-1], x[1:])
    same = (sims >= sim_threshold).astype(np.int32)
    runs = []
    cur = 1
    for v in same:
        if v == 1:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    runs = np.asarray(runs, dtype=np.int32)
    return {
        "mean_run": float(runs.mean()),
        "p95_run": float(np.quantile(runs, 0.95)),
        "p_switch": float(1.0 - same.mean()),  # switch when not-same
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True, type=str, help="Original embeddings dir (has *_frame.npy)")
    ap.add_argument("--warped_dir", required=True, type=str, help="Warped dir (has *_frame.npy, *_ptr.npy, *_adv.npy)")
    ap.add_argument("--num_files", default=50, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--sim_threshold", default=0.97, type=float)
    ap.add_argument("--out_dir", default="docs/figures", type=str)
    args = ap.parse_args()

    clean_dir = Path(args.clean_dir)
    warped_dir = Path(args.warped_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_frames = sorted(clean_dir.rglob("*_frame.npy"))
    if not clean_frames:
        raise RuntimeError(f"No *_frame.npy in {clean_dir}")

    rng = np.random.default_rng(args.seed)
    sel = rng.permutation(len(clean_frames))[: min(args.num_files, len(clean_frames))]
    sel = [clean_frames[i] for i in sel]

    # Collect aggregate distributions
    all_runs = []
    clean_mean, clean_p95, clean_ps = [], [], []
    warp_mean, warp_p95, warp_ps = [], [], []
    T_clean, T_warp = [], []

    example = None

    for fp in sel:
        rel = fp.relative_to(clean_dir)
        warped_fp = warped_dir / rel
        ptr_fp = warped_fp.with_name(warped_fp.name.replace("_frame.npy", "_ptr.npy"))
        adv_fp = warped_fp.with_name(warped_fp.name.replace("_frame.npy", "_adv.npy"))

        if not warped_fp.exists() or not ptr_fp.exists() or not adv_fp.exists():
            # skip if missing
            continue

        x0 = np.load(fp).astype(np.float32)
        xt = np.load(warped_fp).astype(np.float32)
        ptr = np.load(ptr_fp).astype(np.int64)
        adv = np.load(adv_fp).astype(np.float32)

        # PTR-derived runs (ground-truth elongation stats)
        runs = run_lengths_from_ptr(ptr)
        all_runs.append(runs)

        s_ptr = stats_from_runs(runs)
        ps_ptr = p_switch_from_ptr(ptr)

        warp_mean.append(s_ptr["mean_run"])
        warp_p95.append(s_ptr["p95_run"])
        warp_ps.append(ps_ptr)

        # Clean stats: we approximate via embedding similarity runs (no ptr exists for clean)
        s_clean = compute_embedding_run_stats(x0, sim_threshold=args.sim_threshold)
        clean_mean.append(s_clean["mean_run"])
        clean_p95.append(s_clean["p95_run"])
        clean_ps.append(s_clean["p_switch"])

        T_clean.append(x0.shape[0])
        T_warp.append(xt.shape[0])

        # pick one example for visualization
        if example is None:
            example = (x0, xt, ptr, adv, rel.as_posix())

    if not all_runs:
        raise RuntimeError("No matched warped files found. Ensure --save_ptr and --save_adv were used.")

    runs_all = np.concatenate(all_runs)

    # -------------------------
    # Plot 1: Run-length histogram (PTR)
    # -------------------------
    plt.figure()
    bins = np.arange(1, min(int(np.max(runs_all)), 40) + 2) - 0.5
    plt.hist(runs_all, bins=bins, density=True)
    plt.xlabel("Run length (frames held)")
    plt.ylabel("Probability density")
    plt.title("Warped run-length distribution (from ptr)")
    plt.tight_layout()
    plt.savefig(out_dir / "plot1_runlength_hist_ptr.png", dpi=200)
    plt.close()

    # -------------------------
    # Plot 2: Summary bars (before vs after)
    # -------------------------
    def mean(x): return float(np.mean(x)) if len(x) else 0.0

    metrics = ["mean_run", "p95_run", "p_switch", "T"]
    before = [mean(clean_mean), mean(clean_p95), mean(clean_ps), mean(T_clean)]
    after  = [mean(warp_mean),  mean(warp_p95),  mean(warp_ps),  mean(T_warp)]

    plt.figure()
    x = np.arange(len(metrics))
    w = 0.35
    plt.bar(x - w/2, before, width=w, label="Before (clean, approx)")
    plt.bar(x + w/2, after,  width=w, label="After (warped, ptr)")
    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Before vs After summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot2_summary_bars.png", dpi=200)
    plt.close()

    # -------------------------
    # Plot 3: Example pointer + advance/stay
    # -------------------------
    x0, xt, ptr, adv, name = example
    Tshow = min(len(ptr), 400)  # keep plot readable

    plt.figure()
    plt.plot(ptr[:Tshow], linewidth=1)
    plt.xlabel("Warped frame index j")
    plt.ylabel("ptr[j] = original frame index i")
    plt.title(f"Example monotonic alignment (ptr)\n{name}")
    plt.tight_layout()
    plt.savefig(out_dir / "plot3_example_ptr.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(adv[:Tshow], linewidth=1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Warped frame index j")
    plt.ylabel("adv[j] (1=advance, 0=stay)")
    plt.title("Example advance/stay labels (BCE target)")
    plt.tight_layout()
    plt.savefig(out_dir / "plot4_example_adv.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {out_dir.resolve()}")
    print("Files:")
    print(" - plot1_runlength_hist_ptr.png")
    print(" - plot2_summary_bars.png")
    print(" - plot3_example_ptr.png")
    print(" - plot4_example_adv.png")

if __name__ == "__main__":
    main()