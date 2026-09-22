"""Schema keys are additive. A renamed key fails this test."""
from __future__ import annotations

import json

from cdlib.cli.evaluate import main as evaluate_main
from cdlib.metrics.report import SCHEMA_KEYS


def test_dummy_eval_keeps_every_schema_key(capsys):
    rc = evaluate_main(["--dummy", "--split", "test", "--img-size", "16", "--batch-size", "2", "--seed", "1"])
    assert rc == 0
    blob = json.loads(capsys.readouterr().out)
    missing = SCHEMA_KEYS - blob.keys()
    assert not missing, sorted(missing)
    assert blob["calibration"] == {}
    assert blob["schema_version"] == "1"
    assert len(blob["checkpoint_sha256"]) == 64
    assert isinstance(blob["change_size"], dict)
    assert blob["change_size"]["provisional"] == 1.0
