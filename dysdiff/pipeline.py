"""Combined forward/reverse pipeline (report Section 3.5, Figure 4).

Forward (training-data simulation, on clean speech):
    x_0 --A. local elongation--> x_0'
        --for t = 1..T:  x_t = x_{t-1} + alpha_t * NeuralForward(x_{t-1}, t, s)    (B)
                         x_t = ArticulationDrift(x_t, sw_t)                         (C)
    with alpha_t = 0.5 * sw_t and sw_t the cosine schedule weight, and s the
    StatsEncoder style of a randomly drawn moderate-dysarthric utterance.

The NeuralForward model itself is trained to reproduce the hand-designed
corruption chain (articulation -> phonation -> pause/rhythm) built here by
`handcrafted_chain`.

Reverse (enhancement):
    x_T (dysarthric embedding) --5 gated, text-conditioned steps--> x_0_hat
        --[optional] advance prediction + Top-K compression--
        --3-frame smoothing--> EmbeddingToMel --> HiFi-GAN --> MetricGAN+
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .forward.articulation import articulation_drift, articulation_forward_step_pronounced
from .forward.hold_warp import elongate_regions
from .forward.pause_rhythm import pause_rhythm_forward_step
from .forward.phonation import phonation_forward_step
from .schedules import ramp, reverse_timesteps, step_weights


# --------------------------------------------------------------------------- #
# Hand-designed composite chain (targets for the neural forward model)
# --------------------------------------------------------------------------- #
def handcrafted_step(x, w, stats, cfg: Dict, seed: int):
    """C_rhythm o C_phon o C_art for one step; module strengths from `cfg`."""
    x1, _, _ = articulation_forward_step_pronounced(x, stats, w, seed=seed, **cfg.get("articulation", {}))
    x2, _, _ = phonation_forward_step(x1, w, seed=seed + 7919, **cfg.get("phonation", {}))
    x3, _, _ = pause_rhythm_forward_step(x2, w, seed=seed + 15887, **cfg.get("pause_rhythm", {}))
    return x3


def handcrafted_chain(x0: np.ndarray, stats: Dict, cfg: Dict, T: int = 30, schedule: str = "cosine", seed: int = 0):
    """Returns states (T+1, L, D)."""
    states = [x0.astype(np.float32)]
    for t, w in enumerate(step_weights(T, schedule), start=1):
        states.append(handcrafted_step(states[-1], float(w), stats, cfg, seed * 1000 + t))
    return np.stack(states)


# --------------------------------------------------------------------------- #
# Neural forward chain
# --------------------------------------------------------------------------- #
@torch.no_grad()
def neural_forward_chain(
    x0: np.ndarray,
    forward_model,
    style: torch.Tensor,
    stats: Dict,
    T: int = 30,
    alpha_scale: float = 0.5,
    elongation: Optional[Dict] = None,
    drift: Optional[Dict] = None,
    seed: int = 0,
    device: str = "cpu",
):
    """Simulate a moderate-dysarthric version of a clean utterance.

    Returns (states (T+1, L', D) float32, ptr) where ptr maps each elongated
    frame back to its clean source frame.
    """
    rng = np.random.default_rng(seed)
    if elongation is not None and elongation.get("enabled", True):
        kw = {k: v for k, v in elongation.items() if k != "enabled"}
        x0, ptr = elongate_regions(x0, rng, **kw)
    else:
        ptr = np.arange(x0.shape[0])
    forward_model.eval()
    s = style.view(1, -1).to(device)
    x = torch.from_numpy(x0.astype(np.float32)).unsqueeze(0).to(device)
    states = [x0.astype(np.float32)]
    T_t = torch.tensor([T], device=device)
    for t in range(1, T + 1):
        sw = ramp(t, T, "cosine")
        delta = forward_model(x, torch.tensor([t], device=device), T_t, s)
        x = x + alpha_scale * sw * delta
        x_np = articulation_drift(x.squeeze(0).cpu().numpy(), sw, stats, rng, **(drift or {}))
        states.append(x_np)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    return np.stack(states), ptr


# --------------------------------------------------------------------------- #
# Reverse sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reverse_sample(
    reverse_model,
    text_encoder,
    x_T: np.ndarray,
    text: str,
    T: int = 30,
    n_steps: int = 5,
    clip_ratio: float = 0.20,
    device: str = "cpu",
    return_trajectory: bool = False,
):
    """Run the gated reverse model at evenly spaced steps (30, 24, 18, 12, 6 by default).

    Each step predicts a gated correction, clips it to `clip_ratio` of the frame
    norm, and subtracts it. Fewer, larger steps limit error accumulation.
    """
    from .models.diffusion import clip_delta, encode_text

    reverse_model.eval(); text_encoder.eval()
    x = torch.from_numpy(np.asarray(x_T, dtype=np.float32)).unsqueeze(0).to(device)
    tvec = text_encoder(encode_text([text]).to(device))
    T_t = torch.tensor([T], device=device)
    traj, gates = [x.squeeze(0).cpu().numpy()], []
    for t in reverse_timesteps(T, n_steps):
        delta, g = reverse_model(x, torch.tensor([t], device=device), T_t, tvec)
        if clip_ratio:
            delta = clip_delta(delta, x, clip_ratio)
        x = x - delta
        traj.append(x.squeeze(0).cpu().numpy()); gates.append(float(g))
    out = x.squeeze(0).cpu().numpy()
    return (out, traj, gates) if return_trajectory else out


def temporal_smooth(x: np.ndarray, kernel: int = 3) -> np.ndarray:
    """3-frame average pooling (padding 1) to remove frame jitter before vocoding."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).T.unsqueeze(0)
    y = F.avg_pool1d(t, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    return y.squeeze(0).T.numpy()


@torch.no_grad()
def compress_elongation(advance_model, x: np.ndarray, T: int = 30, mode: str = "threshold",
                        target_ratio: float = 0.91, threshold: float = 0.5, min_ratio: float = 0.6,
                        device: str = "cpu"):
    """Undo holds with the advance predictor (Top-K to a target length, or thresholding).

    Thresholding never keeps fewer than `min_ratio` of the frames; if it would,
    Top-K at `min_ratio` is used instead so a miscalibrated predictor cannot
    collapse the utterance.
    """
    from .models.advance import threshold_compress, topk_compress

    advance_model.eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    logits = advance_model(xt.unsqueeze(0), torch.tensor([T], device=device), torch.tensor([T], device=device))
    prob = torch.sigmoid(logits.squeeze(0))
    if mode == "topk":
        out = topk_compress(xt, prob, int(round(target_ratio * xt.shape[0])))
    else:
        out = threshold_compress(xt, prob, threshold)
        floor = int(np.ceil(min_ratio * xt.shape[0]))
        if out.shape[0] < floor:
            out = topk_compress(xt, prob, floor)
    return out.cpu().numpy()


# --------------------------------------------------------------------------- #
# End-to-end enhancer
# --------------------------------------------------------------------------- #
class Enhancer:
    """wav -> wav2vec2 embedding -> reverse diffusion -> mel -> HiFi-GAN [-> MetricGAN+]."""

    def __init__(self, cfg: Dict, device: str = "cpu"):
        from .models.advance import AdvancePredictor
        from .models.diffusion import GatedReverseDenoiser, TextEncoder
        from .models.embedding_to_mel import EmbeddingToMel

        self.cfg, self.device = cfg, device
        ck = cfg["checkpoints"]
        rev = torch.load(ck["reverse"], map_location=device)
        self.reverse = GatedReverseDenoiser(**cfg["reverse"]["model"]).to(device)
        self.reverse.load_state_dict(rev["model"])
        self.text_encoder = TextEncoder().to(device)
        self.text_encoder.load_state_dict(rev["text_encoder"])
        self.e2m = EmbeddingToMel().to(device)
        self.e2m.load_state_dict(torch.load(ck["embedding_to_mel"], map_location=device))
        self.e2m.eval()
        self.advance = None
        if cfg["enhance"].get("compress_elongation") and ck.get("advance"):
            self.advance = AdvancePredictor().to(device)
            self.advance.load_state_dict(torch.load(ck["advance"], map_location=device))
        self._extractor = self._vocoder = self._metricgan = None

    # lazily loaded pretrained components -------------------------------------
    def extractor(self):
        if self._extractor is None:
            from .features import Wav2Vec2Extractor
            self._extractor = Wav2Vec2Extractor(self.cfg["features"]["model_name"], self.device)
        return self._extractor

    def vocoder(self):
        if self._vocoder is None:
            from .synthesis import load_hifigan
            self._vocoder = load_hifigan(self.device)
        return self._vocoder

    def metricgan(self):
        if self._metricgan is None:
            from .synthesis import load_metricgan
            self._metricgan = load_metricgan(self.device)
        return self._metricgan

    # ------------------------------------------------------------------------
    @torch.no_grad()
    def enhance_embedding(self, x_T: np.ndarray, text: str) -> np.ndarray:
        e = self.cfg["enhance"]
        T = self.cfg["diffusion"]["T"]
        x0 = reverse_sample(self.reverse, self.text_encoder, x_T, text, T=T, n_steps=e["n_steps"],
                            clip_ratio=e["clip_ratio"], device=self.device)
        if self.advance is not None:
            x0 = compress_elongation(self.advance, x0, T=T, mode=e["compress_mode"],
                                     target_ratio=e["compress_target_ratio"],
                                     min_ratio=e.get("compress_min_ratio", 0.6), device=self.device)
        if e.get("smooth_kernel", 3) > 1:
            x0 = temporal_smooth(x0, e["smooth_kernel"])
        return x0

    @torch.no_grad()
    def embedding_to_wav(self, x0: np.ndarray, use_metricgan: bool = True) -> np.ndarray:
        from .synthesis import vocode, metricgan_enhance

        mel = self.e2m(torch.from_numpy(x0).unsqueeze(0).to(self.device))
        wav = vocode(self.vocoder(), mel)
        if use_metricgan:
            wav = metricgan_enhance(self.metricgan(), wav)
        return wav

    def enhance_wav(self, wav_path: str, text: str, use_metricgan: Optional[bool] = None):
        if use_metricgan is None:
            use_metricgan = self.cfg["enhance"].get("metricgan", True)
        x_T = self.extractor()(wav_path)
        x0 = self.enhance_embedding(x_T, text)
        return self.embedding_to_wav(x0, use_metricgan), x0
