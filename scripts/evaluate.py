"""Evaluate enhanced speech: STOI, PESQ (vs. the dysarthric input, as in the report) and ASR WER/CER.

Note: STOI/PESQ against the *input* measure how much of the input's content
and quality the reconstruction preserves; they are not an intelligibility gain.
The ASR comparison (input vs. reconstructed WER) is the direct test.

    python scripts/evaluate.py --group moderate
Writes results/eval_<group>.json with per-utterance and mean scores.
"""

from __future__ import annotations

from _common import base_parser, np, resolve, setup

from dysdiff.data import read_manifest
from dysdiff.features import load_audio
from dysdiff.metrics import ASR, pesq_score, stoi_score, wer_cer
from dysdiff.utils import save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--group", default=None)
    p.add_argument("--enhanced_dir", default=None)
    args = p.parse_args()
    cfg, device = setup(args)
    group = args.group or cfg["dysarthric_group"]
    enh_dir = resolve(args.enhanced_dir or f"{cfg['paths']['work_dir']}/enhanced")

    rows = [r for r in read_manifest(str(resolve(cfg["paths"]["manifest"])), [group])
            if r["wav_path"] and (enh_dir / f"{r['utt_id']}.wav").exists()]
    if not rows:
        raise SystemExit(f"No enhanced wavs for group '{group}' in {enh_dir}; run scripts/enhance.py first")

    asr = ASR(cfg["evaluate"]["asr_model"], device)
    per_utt, refs, hyp_in, hyp_out = [], [], [], []
    for r in rows:
        x_in = load_audio(r["wav_path"])
        x_out = load_audio(str(enh_dir / f"{r['utt_id']}.wav"))
        item = {"utt_id": r["utt_id"], "stoi": stoi_score(x_in, x_out), "pesq": pesq_score(x_in, x_out)}
        if r["text"]:
            item["asr_input"], item["asr_enhanced"] = asr(x_in), asr(x_out)
            refs.append(r["text"]); hyp_in.append(item["asr_input"]); hyp_out.append(item["asr_enhanced"])
        per_utt.append(item)

    summary = {
        "group": group, "n": len(per_utt),
        "stoi_vs_input": float(np.mean([u["stoi"] for u in per_utt])),
        "pesq_vs_input": float(np.mean([u["pesq"] for u in per_utt])),
    }
    if refs:
        summary["asr_input"] = wer_cer(refs, hyp_in)
        summary["asr_enhanced"] = wer_cer(refs, hyp_out)
        summary["n_with_transcript"] = len(refs)
    out = resolve(f"results/eval_{group}.json")
    save_json({"summary": summary, "per_utterance": per_utt}, out)
    print(summary, "->", out)


if __name__ == "__main__":
    main()
