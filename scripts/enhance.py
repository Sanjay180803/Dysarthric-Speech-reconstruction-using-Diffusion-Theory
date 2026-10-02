"""Enhance dysarthric recordings end to end.

    python scripts/enhance.py --wav path/to/utt.wav --text "the transcript" --out out.wav
    python scripts/enhance.py --group moderate --limit 20          # from the manifest

Writes enhanced 16 kHz wavs (and the recovered embeddings) to runs/enhanced/.
"""

from __future__ import annotations

from pathlib import Path

from _common import base_parser, np, resolve, setup

from dysdiff.data import read_manifest
from dysdiff.pipeline import Enhancer


def main():
    p = base_parser(__doc__)
    p.add_argument("--wav")
    p.add_argument("--text", default="")
    p.add_argument("--out")
    p.add_argument("--group", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no_metricgan", action="store_true")
    args = p.parse_args()
    cfg, device = setup(args)
    import soundfile as sf

    cfg["checkpoints"] = {k: str(resolve(v)) for k, v in cfg["checkpoints"].items()}
    enh = Enhancer(cfg, device)
    use_mg = not args.no_metricgan and cfg["enhance"]["metricgan"]

    if args.wav:
        jobs = [(args.wav, args.text, Path(args.out or Path(args.wav).stem + "_enhanced.wav"))]
    else:
        rows = read_manifest(str(resolve(cfg["paths"]["manifest"])), [args.group or cfg["dysarthric_group"]])
        rows = [r for r in rows if r["wav_path"]][: args.limit]
        out_dir = resolve(cfg["paths"]["work_dir"]) / "enhanced"
        jobs = [(r["wav_path"], r["text"], out_dir / f"{r['utt_id']}.wav") for r in rows]

    for i, (wav, text, out) in enumerate(jobs, 1):
        y, x0 = enh.enhance_wav(wav, text, use_metricgan=use_mg)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y, 16000)
        np.save(out.with_suffix(".npy"), x0)
        print(f"[{i}/{len(jobs)}] {Path(wav).name} -> {out}")


if __name__ == "__main__":
    main()
