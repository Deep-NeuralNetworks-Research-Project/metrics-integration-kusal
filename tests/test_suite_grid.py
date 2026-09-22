"""Full nuisance grid from evaluate --robustness suite."""
from __future__ import annotations

import json

from cdlib.cli.evaluate import main as evaluate_main
from cdlib.metrics.corruptions import PAPER_GRID


def test_robustness_suite_writes_grid_and_markdown(tmp_path):
    out = tmp_path / "eval.json"
    rc = evaluate_main(
        [
            "--dummy",
            "--robustness",
            "suite",
            "--out",
            str(out),
            "--img-size",
            "16",
            "--batch-size",
            "1",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    blob = json.loads(out.read_text())
    grid = blob["nuisance"]
    assert "clean_f1" in grid
    for name in PAPER_GRID:
        for which in ("t1", "t2"):
            key = f"{name}/s1/{which}/f1"
            assert key in grid, key
            assert f"{name}/s1/{which}/retention" in grid
    side = out.with_suffix(".nuisance.md")
    assert side.is_file()
    text = side.read_text()
    assert "corruption" in text
    assert "retention" in text
    assert "jpeg" in text
