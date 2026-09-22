"""π is computed from masks on disk, or the script refuses."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compute_pi.py"


def test_missing_root_exits(tmp_path):
    missing = tmp_path / "no-such-datasets"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(missing)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "Refusing to invent" in proc.stderr


def test_npy_masks_give_known_pi(tmp_path):
    for dataset in ("levir", "sysu", "pcd"):
        for split in ("train", "test"):
            folder = tmp_path / dataset / split
            folder.mkdir(parents=True)
            mask = np.zeros((2, 2), dtype=np.float32)
            mask[0, 0] = 1
            mask[0, 1] = -1
            np.save(folder / "m.npy", mask)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # 1 changed, 3 valid (the -1 is ignored) → 1/3
    assert "0.333333" in proc.stdout
