"""Measure elongation and articulation statistics on unpaired clean vs. moderate speech.

Writes results/moderate_elongation_stats.json (per-session and pooled run-length
statistics from a joint k-means codebook) and results/articulation_stats.json.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from _common import base_parser, group_files, resolve, setup

from dysdiff.forward.articulation import estimate_articulation_stats
from dysdiff.stats import elongation_stats, train_joint_kmeans
from dysdiff.utils import save_json


def main():
    args = base_parser(__doc__).parse_args()
    cfg, _ = setup(args)
    clean_groups, dys = cfg["clean_groups"], cfg["dysarthric_group"]

    files = {g: group_files(cfg, [g]) for g in clean_groups + [dys]}
    for g, f in files.items():
        print(f"{g}: {len(f)} files")

    km = train_joint_kmeans(files, n_clusters=cfg["stats"]["kmeans_clusters"], seed=cfg["seed"])

    sessions = defaultdict(list)
    for f in files[dys]:
        sessions[Path(f).parent.name].append(f)
    result = {dys: {
        "per_session": {s: elongation_stats(fs, km) for s, fs in sorted(sessions.items())},
        "pooled": elongation_stats(files[dys], km),
    }, "per_group": {g: elongation_stats(f, km) for g, f in files.items()}}
    out = resolve(cfg["stats"]["elongation_file"])
    save_json(result, out)
    print("pooled:", result[dys]["pooled"]["run_length"], "->", out)

    clean = [f for g in clean_groups for f in files[g]]
    art = estimate_articulation_stats(clean, files[dys], seed=cfg["seed"])
    out = resolve(cfg["stats"]["articulation_file"])
    save_json(art, out)
    print("articulation drift:", {k: round(v, 4) for k, v in art.items() if k.startswith("drift")}, "->", out)


if __name__ == "__main__":
    main()
