"""Provisional change-size bins."""
from __future__ import annotations

import numpy as np

from cdlib.metrics.change_size import bin_for_area, change_size_scores


def test_bin_edges():
    assert bin_for_area(4) == "small"
    assert bin_for_area(63) == "small"
    assert bin_for_area(64) == "medium"
    assert bin_for_area(100) == "medium"
    assert bin_for_area(1024) == "medium"
    assert bin_for_area(2000) == "large"


def test_components_land_in_bins():
    h = w = 50
    gt = np.zeros((3, h, w), dtype=np.float32)
    probs = np.full((3, h, w), 0.1, dtype=np.float32)
    gt[0, 0:2, 0:2] = 1
    gt[1, 0:10, 0:10] = 1
    gt[2, 0:40, 0:40] = 1
    probs[gt > 0.5] = 0.9
    out = change_size_scores(probs, gt, probs, threshold=0.5)
    assert out["provisional"] == 1.0
    assert out["n_components_small"] == 1
    assert out["n_components_medium"] == 1
    assert out["n_components_large"] == 1
    assert out["f1_small"] == 1.0
    assert out["f1_medium"] == 1.0
    assert out["f1_large"] == 1.0
    assert out["sce_flip_small"] == 0.0
