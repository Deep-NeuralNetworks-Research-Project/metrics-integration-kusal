"""Stratify F1 and SCE_flip by ground-truth change-component size.

Bins are provisional until a LEVIR component-size histogram exists:

- small: area < 64 px
- medium: 64–1024 px
- large: area > 1024 px

``provisional`` is 1.0 until those edges are replaced. F1 is the aggregate
score on pairs whose largest GT component falls in the bin. SCE_flip is
the flip rate on pixels that belong to components in the bin.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.order import probs_from_logits
from cdlib.metrics.region import connected_components
from cdlib.metrics.segmentation import _squeeze_mask, confusion_counts, prf1_from_counts

SMALL_MAX_PX = 64
MEDIUM_MAX_PX = 1024
BINS = ("small", "medium", "large")


def bin_for_area(area: int, small_max: int = SMALL_MAX_PX, medium_max: int = MEDIUM_MAX_PX) -> str:
    if area < small_max:
        return "small"
    if area <= medium_max:
        return "medium"
    return "large"


def change_size_scores(
    probs_fwd: np.ndarray,
    gt: np.ndarray,
    probs_rev: np.ndarray | None = None,
    threshold: float = 0.5,
    small_max: int = SMALL_MAX_PX,
    medium_max: int = MEDIUM_MAX_PX,
) -> dict[str, float]:
    if probs_fwd.ndim == 2:
        probs_fwd = probs_fwd[None]
        gt = gt[None]
        if probs_rev is not None:
            probs_rev = probs_rev[None]
    counts = {b: {"tp": 0, "fp": 0, "fn": 0} for b in BINS}
    flip_hit = {b: 0 for b in BINS}
    flip_n = {b: 0 for b in BINS}
    n_comp = {b: 0 for b in BINS}
    pair_bin: list[str | None] = []
    valid = valid_mask(gt)
    pred = probs_fwd >= threshold
    gt_pos = (gt > 0.5) & valid
    for i in range(gt.shape[0]):
        labels, areas = connected_components(gt_pos[i])
        if areas.size == 0:
            pair_bin.append(None)
            continue
        largest = int(areas.max())
        pair_bin.append(bin_for_area(largest, small_max, medium_max))
        for area in areas:
            n_comp[bin_for_area(int(area), small_max, medium_max)] += 1
        if probs_rev is not None:
            hard_f = pred[i]
            hard_r = probs_rev[i] >= threshold
            flipped = hard_f != hard_r
            for lab in range(1, int(labels.max()) + 1):
                comp = labels == lab
                area = int(comp.sum())
                b = bin_for_area(area, small_max, medium_max)
                flip_hit[b] += int((flipped & comp & valid[i]).sum())
                flip_n[b] += int((comp & valid[i]).sum())
    for i, b in enumerate(pair_bin):
        if b is None:
            continue
        tp, fp, fn, _ = confusion_counts(pred[i], gt_pos[i], valid[i])
        counts[b]["tp"] += tp
        counts[b]["fp"] += fp
        counts[b]["fn"] += fn
    out: dict[str, float] = {
        "provisional": 1.0,
        "small_max_px": float(small_max),
        "medium_max_px": float(medium_max),
    }
    for b in BINS:
        out[f"f1_{b}"] = prf1_from_counts(counts[b]["tp"], counts[b]["fp"], counts[b]["fn"])[2]
        out[f"sce_flip_{b}"] = (flip_hit[b] / flip_n[b]) if flip_n[b] else 0.0
        out[f"n_components_{b}"] = float(n_comp[b])
    return out


class ChangeSizeMetric(Metric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._fwd: list[np.ndarray] = []
        self._rev: list[np.ndarray] = []
        self._gt: list[np.ndarray] = []
        self._has_rev = False

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        logits = _squeeze_mask(as_numpy(outputs["logits"]))
        gt = _squeeze_mask(as_numpy(batch["mask"]))
        self._fwd.append(probs_from_logits(logits))
        self._gt.append(gt)
        if "logits_swapped" in outputs:
            self._has_rev = True
            self._rev.append(probs_from_logits(_squeeze_mask(as_numpy(outputs["logits_swapped"]))))

    def compute(self) -> dict[str, float]:
        if not self._fwd:
            return change_size_scores(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), threshold=self.threshold)
        fwd = np.concatenate(self._fwd, axis=0)
        gt = np.concatenate(self._gt, axis=0)
        rev = np.concatenate(self._rev, axis=0) if self._has_rev else None
        return change_size_scores(fwd, gt, rev, threshold=self.threshold)
