"""Pair-swap consistency for ablation B7.

P4 owns the training loss; P5 runs the *metric* on every model, including
ones not trained with the loss. Score = fraction of valid pixels whose
hard prediction is unchanged under (I1,I2) ↔ (I2,I1).

When ``logits_swapped`` is present, ``compute`` also returns the SCE
fields from ``order.py`` so the registry path and the eval JSON share
one definition.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.base import Metric
from cdlib.metrics.order import order_metrics, probs_from_logits
from cdlib.metrics.segmentation import _squeeze_mask


def swap_agreement(logits_fwd: np.ndarray, logits_rev: np.ndarray, threshold: float = 0.5) -> float:
    pf = (1 / (1 + np.exp(-np.clip(_squeeze_mask(np.asarray(logits_fwd)), -80, 80)))) >= threshold
    pr = (1 / (1 + np.exp(-np.clip(_squeeze_mask(np.asarray(logits_rev)), -80, 80)))) >= threshold
    return float((pf == pr).mean())


class SwapConsistencyMetric(Metric):
    """Requires the caller to pass ``outputs['logits_swapped']`` from a second forward."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._agree = 0
        self._n = 0
        self._fwd: list[np.ndarray] = []
        self._rev: list[np.ndarray] = []
        self._gt: list[np.ndarray] = []

    def update(self, outputs: dict[str, Any], batch: dict[str, Any]) -> None:
        if "logits_swapped" not in outputs:
            raise KeyError("SwapConsistencyMetric needs outputs['logits_swapped']")
        fwd = _squeeze_mask(as_numpy(outputs["logits"]))
        rev = _squeeze_mask(as_numpy(outputs["logits_swapped"]))
        gt = _squeeze_mask(as_numpy(batch["mask"]))
        valid = valid_mask(gt)
        pf = (1 / (1 + np.exp(-np.clip(fwd, -80, 80)))) >= self.threshold
        pr = (1 / (1 + np.exp(-np.clip(rev, -80, 80)))) >= self.threshold
        self._agree += int(((pf == pr) & valid).sum())
        self._n += int(valid.sum())
        self._fwd.append(probs_from_logits(fwd))
        self._rev.append(probs_from_logits(rev))
        self._gt.append(gt)

    def compute(self) -> dict[str, float]:
        out = {"swap_consistency": (self._agree / self._n) if self._n else 1.0}
        if not self._fwd:
            out.update(
                {
                    "sce_flip": 0.0,
                    "sce_prob": 0.0,
                    "delta_f1_swap": 0.0,
                    "spearman_rho": 1.0,
                }
            )
            return out
        order = order_metrics(
            np.concatenate(self._fwd, axis=0),
            np.concatenate(self._rev, axis=0),
            np.concatenate(self._gt, axis=0),
            self.threshold,
        )
        out["sce_flip"] = order["sce_flip"]
        out["sce_prob"] = order["sce_prob"]
        out["delta_f1_swap"] = order["delta_f1_swap"]
        out["spearman_rho"] = order["spearman_rho"]
        return out


def score_model_swap(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor, threshold: float = 0.5) -> float:
    model.eval()
    with torch.no_grad():
        fwd = model(img1, img2)["logits"]
        rev = model(img2, img1)["logits"]
    return swap_agreement(as_numpy(fwd), as_numpy(rev), threshold=threshold)
