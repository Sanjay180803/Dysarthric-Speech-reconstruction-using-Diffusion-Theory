"""EmbeddingToMel (report Sections 3.4 and 3.5.6).

1-D CNN mapping 1024-d wav2vec2 frames (20 ms hop) to an 80-bin log-mel
spectrogram in the format expected by SpeechBrain's HiFi-GAN
(`speechbrain/tts-hifigan-libritts-16kHz`: 16 kHz, n_fft 1024, hop 256, fmax 8 kHz).
Output is linearly interpolated to the mel frame count.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mel settings of the pretrained HiFi-GAN vocoder
MEL_CFG = dict(sample_rate=16000, hop_length=256, win_length=1024, n_fft=1024, n_mels=80,
               f_min=0.0, f_max=8000.0, power=1, normalized=False, norm="slaney",
               mel_scale="slaney", compression=True)

EMB_HOP_SEC = 0.02


def mel_frames_for(num_emb_frames: int) -> int:
    """Mel frame count covering the same duration as `num_emb_frames` embeddings."""
    samples = int(round(num_emb_frames * EMB_HOP_SEC * MEL_CFG["sample_rate"]))
    return samples // MEL_CFG["hop_length"] + 1


class EmbeddingToMel(nn.Module):
    def __init__(self, dim_in: int = 1024, hidden: int = 512, n_mels: int = 80, n_layers: int = 3, kernel_size: int = 5):
        super().__init__()
        self.in_proj = nn.Conv1d(dim_in, hidden, 1)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2),
                          nn.BatchNorm1d(hidden), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Conv1d(hidden, n_mels, 1)

    def forward(self, emb: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """emb: (B, L, D) -> mel: (B, n_mels, target_len)."""
        h = self.in_proj(emb.transpose(1, 2))
        for layer in self.layers:
            h = layer(h)
        mel = self.out_proj(h)
        if target_len is None:
            target_len = mel_frames_for(emb.shape[1])
        return F.interpolate(mel, size=target_len, mode="linear", align_corners=False)


def compute_mel(wav: torch.Tensor) -> torch.Tensor:
    """Log-mel of a 16 kHz waveform (T,) -> (80, frames), matching the vocoder's training features."""
    try:  # SpeechBrain >= 1.0
        from speechbrain.lobes.models.HifiGAN import mel_spectrogram
    except ImportError:  # older releases used this spelling
        from speechbrain.lobes.models.HifiGAN import mel_spectogram as mel_spectrogram

    out = mel_spectrogram(audio=wav.squeeze(), **MEL_CFG)
    return out[0] if isinstance(out, tuple) else out
