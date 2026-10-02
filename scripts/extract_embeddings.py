"""Extract wav2vec2 phoneme-model embeddings for TORGO and write the manifest.

Handles both TORGO layouts:
  original:  <root>/F/F03/Session1/wav_headMic/0001.wav   (+ Session1/prompts/0001.txt)
  flattened: <root>/F_Dys/wav_headMic_F03S01/wav_headMic_F03S01_0001.wav

Outputs  <embeddings_root>/<group>/wav_headMic_<SPK>S<NN>/wav_headMic_<SPK>S<NN>_<XXXX>_frame.npy  (T, 1024)
         ..._utt.npy (mean-pooled, 1024)
and a manifest CSV with transcripts where TORGO prompts are available.

--index_only builds the manifest from embeddings that already exist (no model needed).
"""

from __future__ import annotations

import re
from pathlib import Path

from _common import base_parser, np, resolve, setup

from dysdiff.data import write_manifest

FLAT_RE = re.compile(r"wav_(?P<mic>\w+?)_(?P<spk>[FM]C?\d+)S(?P<sess>\d+)_(?P<num>\d+)")
TREE_RE = re.compile(r"(?P<spk>[FM]C?\d+)[/\\]Session(?P<sess>\d+)(?:_\d+)?[/\\]wav_(?P<mic>\w+)[/\\](?P<num>\d+)\.wav$")


def parse(path: Path):
    m = TREE_RE.search(str(path)) or FLAT_RE.search(path.stem)
    if not m:
        return None
    return m["spk"], int(m["sess"]), m["num"].zfill(4), m["mic"]


def find_prompt(wav: Path, spk: str, sess: int, num: str) -> str:
    candidates = [
        wav.parent.parent / "prompts" / f"{num}.txt",                       # original layout
        wav.parent / f"prompts_{spk}S{sess:02d}_{num}.txt",                  # some flattened copies
        wav.with_suffix(".txt"),
    ]
    for c in candidates:
        if c.exists():
            text = c.read_text(errors="ignore").strip()
            # picture-description prompts point at image files; no usable transcript
            return "" if re.search(r"\.(jpg|png)$", text, re.I) else text
    return ""


def main():
    p = base_parser(__doc__)
    p.add_argument("--torgo_root", default=None)
    p.add_argument("--index_only", action="store_true", help="index existing *_frame.npy instead of extracting")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg, device = setup(args)

    spk2group = {s: g for g, spks in cfg["speaker_groups"].items() for s in spks}
    torgo = resolve(args.torgo_root or cfg["paths"]["torgo_root"])
    out_root = resolve(cfg["paths"]["embeddings_root"])
    mic = cfg["features"]["mic"]

    wavs = {}
    if torgo.exists():
        for w in torgo.rglob("*.wav"):
            info = parse(w)
            if info and info[3] == mic:
                wavs[(info[0], info[1], info[2])] = w

    rows = []
    if args.index_only:
        for f in sorted(out_root.rglob("*_frame.npy")):
            info = parse(Path(f.name.replace("_frame.npy", "")))
            if not info:
                continue
            spk, sess, num, _ = info
            wav = wavs.get((spk, sess, num))
            rows.append(dict(
                utt_id=f"{spk}S{sess:02d}_{num}", group=spk2group.get(spk, f.parent.parent.name),
                speaker=spk, session=sess, frame_path=str(f),
                utt_path=str(f).replace("_frame.npy", "_utt.npy"), wav_path=str(wav or ""),
                num_frames=int(np.load(f, mmap_mode="r").shape[0]),
                text=find_prompt(wav, spk, sess, num) if wav else ""))
    else:
        from dysdiff.features import Wav2Vec2Extractor

        if not wavs:
            raise SystemExit(f"No {mic} wavs found under {torgo}")
        ext = Wav2Vec2Extractor(cfg["features"]["model_name"], device)
        for i, ((spk, sess, num), wav) in enumerate(sorted(wavs.items()), 1):
            group = spk2group.get(spk)
            if group is None:
                continue
            name = f"wav_{mic}_{spk}S{sess:02d}"
            frame_path = out_root / group / name / f"{name}_{num}_frame.npy"
            if args.overwrite or not frame_path.exists():
                try:
                    x = ext(str(wav))
                except Exception as e:  # corrupt / empty files exist in TORGO
                    print(f"skip {wav}: {e}")
                    continue
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(frame_path, x)
                np.save(str(frame_path).replace("_frame.npy", "_utt.npy"), x.mean(0))
            rows.append(dict(
                utt_id=f"{spk}S{sess:02d}_{num}", group=group, speaker=spk, session=sess,
                frame_path=str(frame_path), utt_path=str(frame_path).replace("_frame.npy", "_utt.npy"),
                wav_path=str(wav), num_frames=int(np.load(frame_path, mmap_mode="r").shape[0]),
                text=find_prompt(wav, spk, sess, num)))
            if i % 200 == 0:
                print(f"{i}/{len(wavs)}")

    manifest = resolve(cfg["paths"]["manifest"])
    write_manifest(rows, str(manifest))
    by_group = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    print(f"Wrote {len(rows)} rows to {manifest}: {by_group}")
    print(f"With transcripts: {sum(1 for r in rows if r['text'])}")


if __name__ == "__main__":
    main()
