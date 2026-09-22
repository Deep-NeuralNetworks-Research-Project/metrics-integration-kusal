"""Table I: n=1 has no ±; no p-values."""
from __future__ import annotations

from cdlib.metrics.tables import emit_table_i


def _row(config, f1, seed=0):
    return {
        "config": config,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "iou": f1,
        "boundary_f1": f1,
        "seed": seed,
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
