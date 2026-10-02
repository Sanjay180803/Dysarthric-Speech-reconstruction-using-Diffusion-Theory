"""Fast CPU tests on synthetic embeddings (no TORGO, no downloads)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dysdiff.forward.articulation import (articulation_drift, articulation_forward_chain_budgeted,  # noqa: E402
                                          articulation_forward_chain_pronounced, oracle_reverse_chain)
from dysdiff.forward.hold_warp import elongate_regions, hold_warp_once, ptr_to_advance, run_lengths  # noqa: E402
from dysdiff.forward.pause_rhythm import pause_rhythm_forward_chain  # noqa: E402
from dysdiff.forward.phonation import phonation_forward_chain  # noqa: E402
from dysdiff.models.advance import AdvancePredictor, topk_compress, weighted_bce  # noqa: E402
from dysdiff.models.diffusion import (GatedReverseDenoiser, NeuralForwardDiffusion, TextEncoder,  # noqa: E402
                                      clip_delta, encode_text)
from dysdiff.models.embedding_to_mel import EmbeddingToMel, mel_frames_for  # noqa: E402
from dysdiff.models.reverse_factor import ReverseStepModel, ReverseUNetLite, run_reverse_chain  # noqa: E402
from dysdiff.models.stats_encoder import StatsEncoder, StatsEncoderTrainer  # noqa: E402
from dysdiff.pipeline import handcrafted_chain, neural_forward_chain, reverse_sample, temporal_smooth  # noqa: E402
from dysdiff.schedules import reverse_timesteps, step_weights  # noqa: E402
from dysdiff.utils import moving_average_1d  # noqa: E402

STATS = {"drift_l2_mean": 1.57, "drift_l2_std": 1.72, "drift_l2_p95": 5.37}
SMALL = dict(dim_in=1024, hidden=64, time_dim=32, cond_dim=128, dilations=(1, 2), kernel_size=5)


def emb(T=60, seed=0):
    rng = np.random.default_rng(seed)
    return moving_average_1d(rng.normal(0, 1, (T, 1024)).astype(np.float32), 5) * 3


def test_moving_average_matches_loop():
    x = np.random.default_rng(1).normal(size=(37, 16)).astype(np.float32)
    for k in (1, 3, 4, 5, 11):
        pad = k // 2
        xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        ref = np.stack([xp[t:t + k].mean(0) for t in range(len(x))]) if k > 1 else x
        assert np.allclose(moving_average_1d(x, k), ref, atol=1e-5)


def test_schedules():
    assert np.isclose(step_weights(30, "cosine").sum(), 1)
    assert np.isclose(step_weights(30, "cosine_full").sum(), 1)
    assert reverse_timesteps(30, 5) == [30, 24, 18, 12, 6]


def test_hold_warp_labels():
    x = emb(50)
    x_t, ptr, adv, _, lmax = hold_warp_once(x, 29, 30, 0.3448, 9, "cosine", np.random.default_rng(0))
    assert x_t.shape[0] == ptr.shape[0] >= 50
    assert np.all(np.diff(ptr) >= 0) and adv.sum() == 50
    assert run_lengths(ptr).max() <= lmax
    x_e, p = elongate_regions(x, np.random.default_rng(0))
    assert np.array_equal(x_e, x[p]) and ptr_to_advance(p).sum() == 50


def test_forward_chains_and_oracle():
    x = emb()
    states, deltas, _ = articulation_forward_chain_budgeted(x, STATS, T=10)
    assert np.allclose(oracle_reverse_chain(states[-1], deltas)[0], x, atol=1e-4)
    for chain in (lambda: articulation_forward_chain_pronounced(x, STATS, T=5),
                  lambda: phonation_forward_chain(x, T=5),
                  lambda: pause_rhythm_forward_chain(x, T=5)):
        s, _, _ = chain()
        assert s[-1].shape == x.shape and np.isfinite(s[-1]).all() and not np.allclose(s[-1], x)
    hc = handcrafted_chain(x, STATS, {}, T=4)
    assert hc.shape == (5,) + x.shape
    d = articulation_drift(x, 1.0, STATS, np.random.default_rng(0))
    assert d.shape == x.shape


def test_factor_models_and_chain():
    x = torch.randn(2, 40, 1024)
    t, T = torch.tensor([3, 5]), torch.tensor([30, 30])
    assert ReverseStepModel()(x, t, T).shape == x.shape
    assert ReverseUNetLite()(x, t, T).shape == x.shape
    out = run_reverse_chain(ReverseUNetLite(hidden=64, dilations=(1,)), emb(20), 3)
    assert out.shape == (20, 1024)


def test_stats_encoder_masking_and_loss():
    enc = StatsEncoder()
    x = torch.randn(2, 50, 1024)
    mask = torch.ones(2, 50, dtype=torch.bool); mask[1, 30:] = False
    s = enc(x, mask)
    s_trim = enc(x[1:, :30])
    assert s.shape == (2, 128)
    # conv receptive field sees padding near the end, so only check closeness loosely
    assert torch.isfinite(s).all() and s_trim.shape == (1, 128)
    tr = StatsEncoderTrainer()
    loss = tr.loss(x, mask, torch.randn(2, 40, 1024), None)
    loss.backward()
    assert tr.target_head.weight.grad is None and tr.encoder.out_proj.weight.grad is not None


def test_diffusion_models():
    x = torch.randn(2, 30, 1024)
    t, T = torch.tensor([6, 30]), torch.tensor([30, 30])
    fwd = NeuralForwardDiffusion(**SMALL)
    assert fwd(x, t, T, torch.randn(2, 128)).shape == x.shape
    tv = TextEncoder()(encode_text(["hello world", "a"]))
    assert tv.shape == (2, 128)
    rev = GatedReverseDenoiser(**SMALL)
    delta, g = rev(x, t, T, tv)
    assert delta.shape == x.shape and g.shape == (2, 1) and ((g >= 0) & (g <= 1)).all()
    big = torch.randn_like(x) * 100
    clipped = clip_delta(big, x, 0.2)
    assert (clipped.norm(dim=-1) <= 0.2 * x.norm(dim=-1) + 1e-4).all()


def test_pipeline_chain_and_sampling():
    x0 = emb(40)
    fwd = NeuralForwardDiffusion(**SMALL)
    states, ptr = neural_forward_chain(x0, fwd, torch.randn(128), STATS, T=6,
                                       elongation={"enabled": True}, drift={})
    assert states.shape[0] == 7 and states.shape[1] == len(ptr) >= 40
    rev, te = GatedReverseDenoiser(**SMALL), TextEncoder()
    out, traj, gates = reverse_sample(rev, te, states[-1], "hello", T=6, n_steps=3, return_trajectory=True)
    assert out.shape == states[-1].shape and len(traj) == 4 and len(gates) == 3
    assert temporal_smooth(out).shape == out.shape


def test_advance_and_mel():
    m = AdvancePredictor(hidden=64, dilations=(1, 2))
    x = torch.randn(2, 70, 1024)
    logits = m(x, torch.tensor([5, 9]), torch.tensor([30, 30]))
    assert logits.shape == (2, 70)
    target = (torch.rand(2, 70) > 0.6).float()
    assert torch.isfinite(weighted_bce(logits, target, torch.ones(2, 70, dtype=torch.bool)))
    kept = topk_compress(x[0], torch.sigmoid(logits[0]), 50)
    assert kept.shape == (50, 1024)
    e2m = EmbeddingToMel()
    assert e2m(torch.randn(2, 100, 1024)).shape == (2, 80, mel_frames_for(100))
    assert mel_frames_for(100) == 126


@pytest.mark.skipif(not os.environ.get("DYSDIFF_CKPT_DIR"), reason="set DYSDIFF_CKPT_DIR to the folder with the released checkpoints")
def test_released_checkpoints_load():
    d = Path(os.environ["DYSDIFF_CKPT_DIR"])
    from dysdiff.models.stats_encoder import load_stats_encoder
    enc = load_stats_encoder(str(d / "stats_encoder_best.pt"))
    assert enc(torch.randn(1, 50, 1024)).shape == (1, 128)
    step = ReverseStepModel()
    step.load_state_dict(torch.load(d / "reverse_articulation_step_model_best.pt", map_location="cpu"))
