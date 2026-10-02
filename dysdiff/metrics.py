"""Evaluation metrics: STOI, PESQ, and ASR WER/CER."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

SR = 16000


def _align(ref: np.ndarray, deg: np.ndarray):
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]


def stoi_score(ref: np.ndarray, deg: np.ndarray) -> float:
    from pystoi import stoi

    r, d = _align(ref, deg)
    return float(stoi(r, d, SR, extended=False))


def pesq_score(ref: np.ndarray, deg: np.ndarray) -> float:
    from pesq import pesq

    r, d = _align(ref, deg)
    return float(pesq(SR, r, d, "wb"))


def normalize_text(s: str) -> str:
    return " ".join("".join(ch for ch in s.lower() if ch.isalpha() or ch == " ").split())


def wer_cer(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    import jiwer

    refs = [normalize_text(r) for r in refs]
    hyps = [normalize_text(h) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r]
    refs, hyps = [refs[i] for i in keep], [hyps[i] for i in keep]
    return {"wer": float(jiwer.wer(refs, hyps)), "cer": float(jiwer.cer(refs, hyps))}


class ASR:
    """Greedy CTC transcription with a wav2vec2 ASR model."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = "cpu"):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
        self.device = device

    def __call__(self, wav: np.ndarray) -> str:
        with self.torch.no_grad():
            inp = self.proc(wav, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
            ids = self.model(inp).logits.argmax(-1)
        return self.proc.batch_decode(ids)[0]
