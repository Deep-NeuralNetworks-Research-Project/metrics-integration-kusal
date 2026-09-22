"""Boundary F1 (BF-score) at a fixed pixel tolerance.

Csurka et al., BMVC 2013. A predicted boundary pixel is a true positive
if it lies within ``tolerance_px`` of a ground-truth boundary pixel, and
the reverse for recall. Headline tolerance is 2 px. This is the paper's
boundary number. Boundary IoU stays a secondary key.

Empty-boundary convention:
- both masks empty → 1
- exactly one mask empty → 0

Corpus score sums boundary hits over the whole split, then computes F1
once (same pooling rule as aggregate overlap F1).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.segmentation import _squeeze_mask

DEFAULT_TOLERANCE_PX = 2.0


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    """1-pixel contour: mask minus its 1-iteration erosion."""
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    return (m > 0) & (eroded == 0)


def boundary_counts(gt: np.ndarray, pred: np.ndarray, tolerance_px: float = DEFAULT_TOLERANCE_PX) -> tuple[int, int, int, int]:
    """Return (pred_hit, pred_boundary, gt_hit, gt_boundary)."""
    gt_b = binary_boundary(gt)
    pr_b = binary_boundary(pred)
    n_pr = int(pr_b.sum())
    n_gt = int(gt_b.sum())
    if n_pr == 0 and n_gt == 0:
        return 0, 0, 0, 0
    if n_pr == 0 or n_gt == 0:
        return 0, n_pr, 0, n_gt
    inv_gt = np.where(gt_b, 0, 1).astype(np.uint8)
    inv_pr = np.where(pr_b, 0, 1).astype(np.uint8)
    precise = int(getattr(cv2, "DIST_MASK_PRECISE", 0))
    dist_gt = cv2.distanceTransform(inv_gt, cv2.DIST_L2, precise)
    dist_pr = cv2.distanceTransform(inv_pr, cv2.DIST_L2, precise)
    pred_hit = int((dist_gt[pr_b] <= tolerance_px).sum())
    gt_hit = int((dist_pr[gt_b] <= tolerance_px).sum())
    return pred_hit, n_pr, gt_hit, n_gt


def boundary_f1_from_counts(pred_hit: int, n_pred: int, gt_hit: int, n_gt: int) -> float:
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    precision = pred_hit / n_pred
    recall = gt_hit / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance_px: float = DEFAULT_TOLERANCE_PX) -> float:
    return boundary_f1_from_counts(*boundary_counts(gt, pred, tolerance_px))


class BoundaryF1Metric(Metric):
    def __init__(self, threshold: float = 0.5, tolerance_px: float = DEFAULT_TOLERANCE_PX):
        self.threshold = float(threshold)
        self.tolerance_px = float(tolerance_px)
        self.reset()

    def reset(self) -> None:
        self._pred_hit = 0
        self._n_pred = 0
        self._gt_hit = 0
        self._n_gt = 0

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        gt_raw = _squeeze_mask(as_numpy(batch["mask"]))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        pred = probs >= self.threshold
        gt_pos = gt_raw > 0.5
        valid = valid_mask(gt_raw)
        for i in range(logits.shape[0]):
            p = np.where(valid[i], pred[i], False)
            g = np.where(valid[i], gt_pos[i], False)
            ph, np_, gh, ng = boundary_counts(g, p, self.tolerance_px)
            self._pred_hit += ph
            self._n_pred += np_
            self._gt_hit += gh
            self._n_gt += ng

    def compute(self) -> dict[str, float]:
        return {
            "boundary_f1": boundary_f1_from_counts(
                self._pred_hit, self._n_pred, self._gt_hit, self._n_gt
            ),
            "boundary_f1_tolerance_px": self.tolerance_px,
        }
