"""End-to-end smoke run of every training stage on tiny synthetic embeddings.

    python tests/smoke_pipeline.py

Creates a throwaway workspace, writes a 1-epoch config, and runs the scripts in
order. Audio stages (embedding extraction, EmbeddingToMel, vocoding,
evaluation) need TORGO and pretrained downloads and are not covered.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = Path(tempfile.mkdtemp(prefix="dysdiff_smoke_"))
    rng = np.random.default_rng(0)
    for group, n in [("F_Con", 6), ("M_Con", 4), ("moderate", 6)]:
        for i in range(n):
            spk = {"F_Con": "FC01", "M_Con": "MC01", "moderate": "F03"}[group]
            d = work / "emb" / group / f"wav_headMic_{spk}S01"
            d.mkdir(parents=True, exist_ok=True)
            T = int(rng.integers(40, 90))
            x = np.cumsum(rng.normal(0, 0.3, (T, 1024)), axis=0).astype(np.float32)
            np.save(d / f"wav_headMic_{spk}S01_{i:04d}_frame.npy", x)

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    cfg["device"] = "cpu"
    cfg["paths"] = {"torgo_root": str(work / "none"), "embeddings_root": str(work / "emb"),
                    "manifest": str(work / "manifest.csv"), "work_dir": str(work / "runs")}
    cfg["stats"].update(elongation_file=str(work / "elong.json"), articulation_file=str(work / "art.json"),
                        kmeans_clusters=8)
    cfg["diffusion"]["T"] = 6
    cfg["reverse"].update(stride=2, n_steps=3)
    for k in ("factor_training", "stats_encoder", "neural_forward", "reverse", "advance"):
        cfg[k]["epochs"] = 1
        cfg[k]["batch_size"] = 2
    for k in ("neural_forward", "reverse"):
        cfg[k]["items_per_chain"] = 2
        cfg[k]["model"].update(hidden=64, dilations=[1, 2])
    cfg["checkpoints"] = {k: str(work / "runs" / k / "best.pt") for k in cfg["checkpoints"]}
    cfg_path = work / "config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    steps = [
        ["extract_embeddings.py", "--index_only"],
        ["compute_stats.py"],
        ["build_factor_chains.py", "--factor", "articulation_budgeted", "--max_files", "3"],
        ["train_factor_reverse.py", "--factor", "articulation_budgeted", "--eval_files", "2"],
        ["build_factor_chains.py", "--factor", "pause_rhythm", "--max_files", "2"],
        ["train_factor_reverse.py", "--factor", "pause_rhythm", "--eval_files", "1"],
        ["train_stats_encoder.py"],
        ["build_chains.py", "--kind", "handcrafted"],
        ["train_neural_forward.py"],
        ["build_chains.py", "--kind", "neural"],
        ["train_reverse.py"],
        ["train_advance.py"],
    ]
    for s in steps:
        print(f"\n=== {' '.join(s)}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s[0]), *s[1:], "--config", str(cfg_path)],
                           capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit(f"FAILED: {s}")
    print(f"\nAll stages ran. Workspace: {work}")


if __name__ == "__main__":
    main()
