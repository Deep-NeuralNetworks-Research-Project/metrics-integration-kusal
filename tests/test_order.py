"""Hand-checked order metrics on a 4×4 pair."""
from __future__ import annotations

import numpy as np

from cdlib.metrics.order import f1_optimal_threshold, order_metrics


def test_swap_flips_every_pixel():
    gt = np.ones((4, 4), dtype=np.float32)
    fwd = np.full((4, 4), 0.9, dtype=np.float32)
    rev = np.full((4, 4), 0.1, dtype=np.float32)
    out = order_metrics(fwd, rev, gt, 0.5)
    assert out["sce_flip"] == 1.0
    assert abs(out["sce_prob"] - 0.8) < 1e-6
    assert out["delta_f1_swap"] == 1.0
    # Both maps are constant, so Spearman is undefined and the convention is 0.
    assert out["spearman_rho"] == 0.0
    assert out["positive_prediction_rate_fwd"] == 1.0
    assert out["positive_prediction_rate_rev"] == 0.0


def test_reversed_ranks_are_minus_one():
    values = np.linspace(0.05, 0.95, 16, dtype=np.float64).reshape(4, 4)
    gt = np.ones((4, 4), dtype=np.float32)
    out = order_metrics(values, 1.0 - values, gt, 0.5)
    assert abs(out["spearman_rho"] + 1.0) < 1e-6


def test_identical_probs_do_not_flip():
    gt = np.ones((4, 4), dtype=np.float32)
    p = np.full((4, 4), 0.8, dtype=np.float32)
    out = order_metrics(p, p, gt, 0.5)
    assert out["sce_flip"] == 0.0
    assert out["sce_prob"] == 0.0
    assert out["spearman_rho"] == 1.0
    assert out["delta_f1_swap"] == 0.0


def test_threshold_sweep_stays_on_the_positive_mass():
    probs = np.array([[0.2, 0.0], [0.2, 0.0]], dtype=np.float32)
    gt = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    t = f1_optimal_threshold(probs, gt, valid)
    assert t <= 0.2
