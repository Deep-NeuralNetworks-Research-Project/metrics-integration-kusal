"""Table I: n=1 has no ±; no p-values. Change-size table from nested JSON."""
from __future__ import annotations

from cdlib.metrics.tables import emit_change_size_table, emit_table_i


def _row(config, f1, seed=0):
    return {
        "config": config,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "iou": f1,
        "boundary_f1": f1,
        "seed": seed,
        "change_size": {
            "provisional": 1.0,
            "f1_small": f1,
            "sce_flip_small": 0.1,
            "f1_medium": f1 * 0.9,
            "sce_flip_medium": 0.2,
            "f1_large": f1 * 0.8,
            "sce_flip_large": 0.3,
        },
    }


def test_single_seed_is_flagged():
    text = emit_table_i([_row("levir", 0.8)])
    body = [line for line in text.splitlines() if line.startswith("| levir")][0]
    assert "±" not in body
    assert "†" in body
    assert "p-value" not in text
    assert "No significance tests" in text


def test_three_seeds_have_std():
    rows = [_row("sysu", v) for v in (0.5, 0.6, 0.7)]
    text = emit_table_i(rows)
    body = [line for line in text.splitlines() if line.startswith("| sysu")][0]
    assert "±" in body
    assert "†" not in body
    assert "| 3 |" in body


def test_change_size_table_has_bins():
    text = emit_change_size_table([_row("pcd", 0.7)])
    assert "| small |" in text
    assert "| medium |" in text
    assert "| large |" in text
    assert "sce_flip" in text
    assert "Bins are provisional" in text
    assert "±" not in [line for line in text.splitlines() if "| pcd" in line][0]
    assert "†" in text
