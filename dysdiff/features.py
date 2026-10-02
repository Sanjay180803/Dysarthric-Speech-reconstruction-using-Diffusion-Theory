"""wav2vec2 phoneme-model feature extraction (1024-d frames, 20 ms hop)."""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    import librosa
    import soundfile as sf

    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


class Wav2Vec2Extractor:
    """Last hidden state of the wav2vec2 phoneme model, shape (T, 1024)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        # The feature extractor only normalises audio; unlike the full
        # processor it does not need the phonemizer/espeak tokenizer.
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, wav_or_path) -> np.ndarray:
        wav = load_audio(wav_or_path) if isinstance(wav_or_path, str) else wav_or_path
        inputs = self.fe(wav, sampling_rate=16000, return_tensors="pt")
        out = self.model(inputs.input_values.to(self.device))
        return out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
