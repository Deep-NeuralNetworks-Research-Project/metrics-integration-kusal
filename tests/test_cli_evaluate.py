from __future__ import annotations

import json

from cdlib.cli.evaluate import main as evaluate_main
from cdlib.cli.export_masks import main as export_main


def test_evaluate_all_metrics(capsys, tmp_path):
    rc = evaluate_main(["--dummy", "--split", "test", "--img-size", "32", "--batch-size", "2", "--seed", "0"])
    assert rc == 0
    blob = json.loads(capsys.readouterr().out)
    assert blob["split"] == "test"
    assert "f1" in blob
    assert "pi" in blob
    assert blob["calibration"] == {}


def test_export_overlays(tmp_path):
    rc = export_main(
        ["exp_id=demo", "--overlays", "best,median,worst", "--out-dir", str(tmp_path), "--n", "4", "--img-size", "32"]
    )
    assert rc == 0
    files = list(tmp_path.joinpath("demo").glob("*.npy"))
    assert len(files) == 3
