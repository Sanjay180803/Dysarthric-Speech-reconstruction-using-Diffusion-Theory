"""The ported forward chains must match the original notebook functions numerically."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dysdiff.forward import articulation as A  # noqa: E402
from dysdiff.forward import pause_rhythm as R  # noqa: E402
from dysdiff.forward import phonation as P  # noqa: E402

PRELUDE = "import os, json, math, random\nimport numpy as np\nfrom typing import Dict, List, Tuple\n"
STATS = {"drift_l2_mean": 1.57, "drift_l2_std": 1.72, "drift_l2_p95": 5.37}


def notebook_functions(nb_name, names):
    """Exec only the notebook cells that define the requested functions; return their namespace."""
    nb = json.load(open(ROOT / "notebooks" / nb_name))
    src = [PRELUDE]
    for name in names:
        cells = ["".join(c["source"]) for c in nb["cells"]
                 if c["cell_type"] == "code" and f"def {name}(" in "".join(c["source"])]
        src.append(cells[-1])  # last definition wins, as when the notebook is run top to bottom
    ns = {}
    exec("\n\n".join(src), ns)
    return ns


@pytest.fixture(scope="module")
def x():
    return (np.random.default_rng(0).normal(size=(80, 1024)) * 3).astype(np.float32)


def test_budgeted_articulation(x):
    ns = notebook_functions("02_articulation_diffusion.ipynb", [
        "moving_average_1d", "get_diffusion_strength", "get_step_weights",
        "articulation_forward_step_budgeted", "articulation_forward_diffusion_chain_budgeted"])
    s1, _, _ = ns["articulation_forward_diffusion_chain_budgeted"](x, STATS, T=30, total_scale=1.5, noise_smooth_kernel=5, seed=7)
    s2, _, _ = A.articulation_forward_chain_budgeted(x, STATS, T=30, total_scale=1.5, noise_smooth_kernel=5,
                                                      schedule="cosine_full", seed=7)
    assert np.abs(s1[-1] - s2[-1]).max() < 1e-4


def test_pronounced_articulation(x):
    ns = notebook_functions("03_articulation_pronounced.ipynb", [
        "moving_average_1d", "get_step_weights", "articulation_forward_step_pronounced",
        "articulation_forward_diffusion_chain_pronounced"])
    s1, _, _ = ns["articulation_forward_diffusion_chain_pronounced"](x, STATS, T=30, seed=3)
    s2, _, _ = A.articulation_forward_chain_pronounced(x, STATS, T=30, seed=3)
    assert np.abs(s1[-1] - s2[-1]).max() < 1e-4


def test_phonation(x):
    ns = notebook_functions("04_phonation_diffusion.ipynb", [
        "moving_average_1d", "get_step_weights", "phonation_forward_step_pronounced",
        "phonation_forward_diffusion_chain_pronounced"])
    s1, _, _ = ns["phonation_forward_diffusion_chain_pronounced"](x, T=30, seed=3)
    s2, _, _ = P.phonation_forward_chain(x, T=30, seed=3)
    assert np.abs(s1[-1] - s2[-1]).max() < 1e-4


def test_pause_rhythm(x):
    ns = notebook_functions("05_pause_rhythm_diffusion.ipynb", [
        "moving_average_1d", "get_step_weights", "sample_spans_from_prob", "spans_to_mask",
        "pause_rhythm_forward_step_pronounced", "pause_rhythm_forward_diffusion_chain_pronounced"])
    s1, _, _ = ns["pause_rhythm_forward_diffusion_chain_pronounced"](x, T=30, seed=3)
    s2, _, _ = R.pause_rhythm_forward_chain(x, T=30, seed=3)
    assert np.abs(s1[-1] - s2[-1]).max() < 1e-4
