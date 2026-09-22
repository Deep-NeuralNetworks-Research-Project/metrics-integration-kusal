"""Suite seed is (pair_id, corruption, severity, direction)."""
from __future__ import annotations

import numpy as np

from cdlib.metrics.corruptions import apply_to_pair, load_suite


def _img(seed=0, h=48, w=48):
    return np.random.default_rng(seed).random((3, h, w), dtype=np.float32)


def test_suite_forbids_stacking():
    suite = load_suite()
    assert suite["one_frame_only"] is True
    assert suite["both_directions"] is True
    assert suite["stack_corruptions"] is False
    assert suite["main"]["jpeg"]["quality"] == [90, 70, 50, 30, 10]


def test_same_pair_id_is_the_same_corruption():
    a, b = _img(1), _img(2)
    c1, c2 = apply_to_pair(a, b, "occlusion", 4, which="t2", pair_id="pair-7")
    d1, d2 = apply_to_pair(a, b, "occlusion", 4, which="t2", pair_id="pair-7")
    assert np.allclose(c1, a)
    assert np.allclose(c2, d2)
    e1, e2 = apply_to_pair(a, b, "occlusion", 4, which="t2", pair_id="pair-8")
    assert np.allclose(e1, a)
    assert not np.allclose(e2, c2)
