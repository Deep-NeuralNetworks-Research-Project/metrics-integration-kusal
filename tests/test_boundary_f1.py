"""Hand-checked boundary F1 at 2 px."""
from __future__ import annotations

import numpy as np

from cdlib.metrics.boundary_f1 import boundary_f1


def test_identical_block_is_one():
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    assert boundary_f1(gt, gt, tolerance_px=2) == 1.0


def test_both_empty_is_one():
    z = np.zeros((4, 4), dtype=np.uint8)
    assert boundary_f1(z, z, tolerance_px=2) == 1.0


def test_one_empty_is_zero():
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[1, 1] = 1
    pred = np.zeros_like(gt)
    assert boundary_f1(gt, pred, tolerance_px=2) == 0.0
    assert boundary_f1(pred, gt, tolerance_px=2) == 0.0


def test_tolerance_two_pixels():
    gt = np.zeros((6, 6), dtype=np.uint8)
    pred_far = np.zeros_like(gt)
    pred_near = np.zeros_like(gt)
    gt[0, 0] = 1
    pred_far[0, 3] = 1
    pred_near[0, 2] = 1
    assert boundary_f1(gt, pred_far, tolerance_px=2) == 0.0
    assert boundary_f1(gt, pred_near, tolerance_px=2) == 1.0
