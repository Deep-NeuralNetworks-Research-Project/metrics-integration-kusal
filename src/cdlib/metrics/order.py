"""Order-consistency fields for the frozen eval schema.

Names are provisional until P4 confirms them. Definitions used here:

- ``sce_flip``: fraction of valid pixels whose hard label flips when the
  pair is swapped, at one recorded threshold.
- ``sce_prob``: mean absolute probability difference on valid pixels.
- ``delta_f1_swap``: aggregate F1(forward) minus aggregate F1(swapped).
- ``spearman_rho``: Spearman correlation of the two probability maps.
- positive-prediction rate: fraction of valid pixels predicted changed,
  once per ordering.
- F1-optimal threshold: sweep on validation only, then freeze for test.
"""
from __future__ import annotations

import numpy as np

from cdlib.metrics._tensor import valid_mask
from cdlib.metrics.segmentation import confusion_counts, prf1_from_counts


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 0-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    for i in range(1, x.size + 1):
        if i == x.size or sorted_x[i] != sorted_x[start]:
            if i - start > 1:
                avg = float(ranks[order[start:i]].mean())
                ranks[order[start:i]] = avg
            start = i
    return ranks


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0:
        return 1.0
    # A constant map has no rank variation, so Spearman is undefined.
    # Equal constants → 1. Anything else with a constant map → 0.
    if a.size == 1 or np.allclose(a, a.flat[0]) or np.allclose(b, b.flat[0]):
        return 1.0 if np.allclose(a, b) else 0.0
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float((ra * rb).sum() / denom)


def f1_at_threshold(probs: np.ndarray, gt: np.ndarray, valid: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    gt_pos = gt > 0.5
    tp, fp, fn, _ = confusion_counts(pred, gt_pos, valid)
    return prf1_from_counts(tp, fp, fn)[2]


def f1_optimal_threshold(
    probs: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Sweep thresholds. Call this on the validation split only."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        f1 = f1_at_threshold(probs, gt, valid, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def order_metrics(
    probs_fwd: np.ndarray,
    probs_rev: np.ndarray,
    gt: np.ndarray,
    threshold_fwd: float,
    threshold_rev: float | None = None,
) -> dict[str, float]:
    """All arrays are [N,H,W] or [H,W]. ``gt`` uses -1 as ignore."""
    if probs_fwd.ndim == 2:
        probs_fwd = probs_fwd[None]
        probs_rev = probs_rev[None]
        gt = gt[None]
    if threshold_rev is None:
        threshold_rev = threshold_fwd
    valid = valid_mask(gt)
    pf = probs_fwd[valid]
    pr = probs_rev[valid]
    # Flip and ΔF1 share one threshold (the forward operating point).
    # Each ordering still has its own positive-prediction rate.
    hard_f = pf >= threshold_fwd
    hard_r_same = pr >= threshold_fwd
    hard_r = pr >= threshold_rev
    n = int(valid.sum())
    sce_flip = float((hard_f != hard_r_same).mean()) if n else 0.0
    sce_prob = float(np.abs(pf - pr).mean()) if n else 0.0
    ppr_f = float(hard_f.mean()) if n else 0.0
    ppr_r = float(hard_r.mean()) if n else 0.0
    f1_f = f1_at_threshold(probs_fwd, gt, valid, threshold_fwd)
    f1_r = f1_at_threshold(probs_rev, gt, valid, threshold_fwd)
    return {
        "sce_flip": sce_flip,
        "sce_prob": sce_prob,
        "delta_f1_swap": float(f1_f - f1_r),
        "spearman_rho": spearman_rho(pf, pr) if n else 1.0,
        "positive_prediction_rate_fwd": ppr_f,
        "positive_prediction_rate_rev": ppr_r,
        "f1_fwd": f1_f,
        "f1_rev": f1_r,
    }


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return _sigmoid(logits)
