"""Registry integrity + frozen metric contract on dummy tensors."""
from __future__ import annotations

import torch

from cdlib.metrics.base import Metric
from cdlib.metrics.registry import METRIC_REGISTRY, build_metric


def _dummy(b=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "outputs": {
            "logits": torch.randn(b, 1, h, w, generator=g),
            "confidence": None,
            "aux": {},
        },
        "batch": {
            "img1": torch.rand(b, 3, h, w, generator=g),
            "img2": torch.rand(b, 3, h, w, generator=g),
            "mask": (torch.rand(b, 1, h, w, generator=g) > 0.9).float(),
            "nuisance_label": torch.zeros(b, dtype=torch.int64),
        },
    }


def test_expected_keys_present():
    for key in ("segmentation", "boundary", "region", "calibration", "robustness"):
        assert key in METRIC_REGISTRY


def test_unknown_key_is_helpful():
    try:
        METRIC_REGISTRY.get("not_a_metric")
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "segmentation" in str(e)


def test_every_metric_update_compute_reset():
    dummy = _dummy()
    for key in METRIC_REGISTRY.keys():
        if key == "swap_consistency":
            continue  # needs logits_swapped
        m = build_metric(key)
        assert isinstance(m, Metric)
        m.update(dummy["outputs"], dummy["batch"])
        out = m.compute()
        assert isinstance(out, dict)
        assert out, f"{key} returned empty dict"
        for k, v in out.items():
            assert isinstance(k, str)
            assert isinstance(v, float), f"{key}.{k} is {type(v)}"
        m.reset()
        z = m.compute()
        assert isinstance(z, dict)


def test_swap_consistency_contract():
    dummy = _dummy()
    dummy["outputs"]["logits_swapped"] = dummy["outputs"]["logits"].flip(0)
    m = build_metric("swap_consistency")
    m.update(dummy["outputs"], dummy["batch"])
    out = m.compute()
    assert 0.0 <= out["swap_consistency"] <= 1.0
    for key in ("sce_flip", "sce_prob", "delta_f1_swap", "spearman_rho"):
        assert key in out
        assert isinstance(out[key], float)
    assert 0.0 <= out["sce_flip"] <= 1.0
    assert 0.0 <= out["sce_prob"] <= 1.0
    assert -1.0 <= out["spearman_rho"] <= 1.0


def test_swap_sce_matches_order_module():
    """Registry path and evaluate path share order_metrics."""
    from cdlib.metrics.order import order_metrics, probs_from_logits
    from cdlib.metrics.segmentation import _squeeze_mask

    dummy = _dummy(seed=3)
    dummy["outputs"]["logits_swapped"] = -dummy["outputs"]["logits"]
    m = build_metric("swap_consistency")
    m.update(dummy["outputs"], dummy["batch"])
    out = m.compute()
    fwd = probs_from_logits(_squeeze_mask(dummy["outputs"]["logits"].numpy()))
    rev = probs_from_logits(_squeeze_mask(dummy["outputs"]["logits_swapped"].numpy()))
    gt = _squeeze_mask(dummy["batch"]["mask"].numpy())
    expected = order_metrics(fwd, rev, gt, 0.5)
    assert out["sce_flip"] == expected["sce_flip"]
    assert out["sce_prob"] == expected["sce_prob"]
    assert out["delta_f1_swap"] == expected["delta_f1_swap"]
    assert out["spearman_rho"] == expected["spearman_rho"]
