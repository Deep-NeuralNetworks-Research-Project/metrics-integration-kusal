"""Build the frozen evaluation JSON. Add keys; do not rename them.

See docs/eval_schema.md. ``calibration`` is an empty object for Member 5.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cdlib.metrics._tensor import as_numpy, valid_mask
from cdlib.metrics.boundary import BoundaryMetric
from cdlib.metrics.boundary_f1 import BoundaryF1Metric
from cdlib.metrics.change_size import change_size_scores
from cdlib.metrics.corruptions import SEVERITY, apply_to_pair, main_suite_names
from cdlib.metrics.order import f1_optimal_threshold, order_metrics, probs_from_logits
from cdlib.metrics.region import connected_components
from cdlib.metrics.robustness import retention
from cdlib.metrics.segmentation import SegmentationMetric, _squeeze_mask

SCHEMA_VERSION = "1"
FALSE_ALERT_MIN_AREA = 20

SCHEMA_KEYS = frozenset(
    {
        "schema_version",
        "split",
        "config",
        "seed",
        "checkpoint_sha256",
        "precision",
        "recall",
        "f1",
        "iou",
        "boundary_f1",
        "boundary_iou",
        "positive_prediction_rate_fwd",
        "positive_prediction_rate_rev",
        "sce_flip",
        "sce_prob",
        "delta_f1_swap",
        "spearman_rho",
        "threshold_f1_optimal_fwd",
        "threshold_f1_optimal_rev",
        "fp_pixels_per_pair",
        "fp_regions_per_pair",
        "false_alert_rate",
        "false_alert_min_area",
        "pi",
        "n_pixels",
        "n_ignored",
        "n_pairs",
        "change_size",
        "calibration",
        "nuisance",
    }
)


def _logits(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(img1, img2)["logits"]
    return _squeeze_mask(as_numpy(out))


def _no_change_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    fp_pixels = []
    fp_regions = []
    alerted = 0
    n_neg = 0
    for i in range(gt.shape[0]):
        gt_pos = (gt[i] > 0.5) & valid[i]
        if gt_pos.any():
            continue
        n_neg += 1
        p = pred[i] & valid[i]
        fp_pixels.append(int(p.sum()))
        _, areas = connected_components(p)
        fp_regions.append(int(areas.size))
        if areas.size and float(areas.max()) >= FALSE_ALERT_MIN_AREA:
            alerted += 1
    if n_neg == 0:
        return {
            "fp_pixels_per_pair": 0.0,
            "fp_regions_per_pair": 0.0,
            "false_alert_rate": 0.0,
        }
    return {
        "fp_pixels_per_pair": float(np.mean(fp_pixels)),
        "fp_regions_per_pair": float(np.mean(fp_regions)),
        "false_alert_rate": alerted / n_neg,
    }


def nuisance_grid(
    model: torch.nn.Module,
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor,
    pair_id: str,
) -> dict[str, float]:
    """Full severity × corruption × direction grid. Same seed per cell across models."""
    a = as_numpy(img1[0] if img1.ndim == 4 else img1).astype(np.float32)
    b = as_numpy(img2[0] if img2.ndim == 4 else img2).astype(np.float32)
    m = as_numpy(mask[0] if mask.ndim == 4 else mask).astype(np.float32)
    if m.ndim == 2:
        m = m[None]

    def predict(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        t1 = torch.as_tensor(x1)[None]
        t2 = torch.as_tensor(x2)[None]
        return _logits(model, t1, t2)

    clean = SegmentationMetric()
    clean.update({"logits": predict(a, b)}, {"mask": m})
    clean_f1 = clean.compute()["f1"]
    rows: dict[str, float] = {"clean_f1": clean_f1}
    for name in main_suite_names():
        for severity in SEVERITY:
            for which in ("t1", "t2"):
                c1, c2 = apply_to_pair(a, b, name, severity, which=which, pair_id=pair_id)
                metric = SegmentationMetric()
                metric.update({"logits": predict(c1, c2)}, {"mask": m})
                f1 = metric.compute()["f1"]
                key = f"{name}/s{severity}/{which}"
                rows[f"{key}/f1"] = f1
                rows[f"{key}/retention"] = retention(f1, clean_f1)
    return rows


def build_eval_report(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    split: str,
    seed: int,
    checkpoint_sha256: str,
    config: str = "unnamed",
    val_batch: dict[str, Any] | None = None,
    robustness: bool = False,
) -> dict[str, Any]:
    img1 = batch["img1"]
    img2 = batch["img2"]
    mask_t = batch["mask"]
    fwd = _logits(model, img1, img2)
    rev = _logits(model, img2, img1)
    gt = _squeeze_mask(as_numpy(mask_t))
    probs_f = probs_from_logits(fwd)
    probs_r = probs_from_logits(rev)
    valid = valid_mask(gt)

    if val_batch is not None:
        v_fwd = _logits(model, val_batch["img1"], val_batch["img2"])
        v_rev = _logits(model, val_batch["img2"], val_batch["img1"])
        v_gt = _squeeze_mask(as_numpy(val_batch["mask"]))
        v_valid = valid_mask(v_gt)
        t_fwd = f1_optimal_threshold(probs_from_logits(v_fwd), v_gt, v_valid)
        t_rev = f1_optimal_threshold(probs_from_logits(v_rev), v_gt, v_valid)
    else:
        t_fwd = t_rev = 0.5

    seg = SegmentationMetric(threshold=t_fwd)
    seg.update({"logits": fwd}, {"mask": gt})
    seg_out = seg.compute()
    bf = BoundaryF1Metric(threshold=t_fwd)
    bf.update({"logits": fwd}, {"mask": gt})
    biou = BoundaryMetric(threshold=t_fwd)
    biou.update({"logits": fwd}, {"mask": gt})
    order = order_metrics(probs_f, probs_r, gt, t_fwd, t_rev)
    pred = probs_f >= t_fwd
    neg = _no_change_stats(pred, gt, valid)
    size = change_size_scores(probs_f, gt, probs_r, threshold=t_fwd)

    n_pairs = int(gt.shape[0])
    n_pixels = int(gt.size)
    n_ignored = int((gt < 0).sum())
    pair_id = "pair"
    meta = batch.get("meta") or {}
    if isinstance(meta, dict) and "pair_id" in meta:
        pair_id = str(meta["pair_id"])

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "config": config,
        "seed": int(seed),
        "checkpoint_sha256": checkpoint_sha256,
        "precision": seg_out["precision"],
        "recall": seg_out["recall"],
        "f1": seg_out["f1"],
        "iou": seg_out["iou"],
        "boundary_f1": bf.compute()["boundary_f1"],
        "boundary_iou": biou.compute()["boundary_iou"],
        "positive_prediction_rate_fwd": order["positive_prediction_rate_fwd"],
        "positive_prediction_rate_rev": order["positive_prediction_rate_rev"],
        "sce_flip": order["sce_flip"],
        "sce_prob": order["sce_prob"],
        "delta_f1_swap": order["delta_f1_swap"],
        "spearman_rho": order["spearman_rho"],
        "threshold_f1_optimal_fwd": float(t_fwd),
        "threshold_f1_optimal_rev": float(t_rev),
        "fp_pixels_per_pair": neg["fp_pixels_per_pair"],
        "fp_regions_per_pair": neg["fp_regions_per_pair"],
        "false_alert_rate": neg["false_alert_rate"],
        "false_alert_min_area": float(FALSE_ALERT_MIN_AREA),
        "pi": seg_out["changed_pixel_ratio"],
        "n_pixels": n_pixels,
        "n_ignored": n_ignored,
        "n_pairs": n_pairs,
        "change_size": size,
        "calibration": {},
        "nuisance": {},
    }
    if robustness:
        report["nuisance"] = nuisance_grid(model, img1, img2, mask_t, pair_id)
    missing = SCHEMA_KEYS - report.keys()
    if missing:
        raise RuntimeError(f"eval schema lost keys: {sorted(missing)}")
    return report
