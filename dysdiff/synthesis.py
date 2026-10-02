"""Waveform synthesis with pretrained SpeechBrain models (no fine-tuning).

* HiFi-GAN vocoder:   speechbrain/tts-hifigan-libritts-16kHz
* MetricGAN+ post-enhancement: speechbrain/metricgan-plus-voicebank
  (suppresses harmonic artefacts HiFi-GAN produces on off-distribution mels)
"""

from __future__ import annotations

import numpy as np
import torch

HIFIGAN_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
METRICGAN_SOURCE = "speechbrain/metricgan-plus-voicebank"


def load_hifigan(device: str = "cpu", savedir: str = "pretrained_models/hifigan-libritts-16kHz"):
    from speechbrain.inference.vocoders import HIFIGAN

    return HIFIGAN.from_hparams(source=HIFIGAN_SOURCE, savedir=savedir, run_opts={"device": device})


def load_metricgan(device: str = "cpu", savedir: str = "pretrained_models/metricgan-plus-voicebank"):
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    return SpectralMaskEnhancement.from_hparams(source=METRICGAN_SOURCE, savedir=savedir, run_opts={"device": device})


@torch.no_grad()
def vocode(hifigan, mel: torch.Tensor) -> np.ndarray:
    """mel: (1, 80, frames) -> 16 kHz waveform (samples,)."""
    wav = hifigan.decode_batch(mel)
    return wav.squeeze().cpu().numpy().astype(np.float32)


@torch.no_grad()
def metricgan_enhance(enhancer, wav: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(wav).float().unsqueeze(0)
    out = enhancer.enhance_batch(x, lengths=torch.tensor([1.0]))
    return out.squeeze().cpu().numpy().astype(np.float32)
