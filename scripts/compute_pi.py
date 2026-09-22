"""Recompute changed-pixel ratio π. Does not invent a number.

Looks under ``--root`` for mask files:

    <root>/<dataset>/{train,test}/**/*.{png,npy}

Datasets: levir, sysu, pcd. A pixel is changed when its value is > 0,
ignored when it is < 0 (npy) or exactly the ignore sentinel, and valid
otherwise. PNG labels are read as 0 = unchanged, >0 = changed.

Exit code 2 if a requested split directory is missing or empty.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DATASETS = ("levir", "sysu", "pcd")
SPLITS = ("train", "test")
MASK_SUFFIXES = {".png", ".npy", ".tif", ".tiff"}


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read {path}")
    return np.asarray(img)


def changed_ratio(mask: np.ndarray) -> tuple[int, int]:
    """Return (changed, valid). Ignore values < 0."""
    flat = np.asarray(mask).reshape(-1).astype(np.float64)
    valid = flat >= 0
    changed = (flat > 0) & valid
    return int(changed.sum()), int(valid.sum())


def collect_masks(root: Path, dataset: str, split: str) -> list[Path]:
    folder = root / dataset / split
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = [p for p in folder.rglob("*") if p.suffix.lower() in MASK_SUFFIXES]
    if not files:
        raise FileNotFoundError(folder)
    return files


def pi_table(root: Path) -> str:
    lines = ["| dataset | split | π | changed | valid |", "|---|---|---|---|---|"]
    for dataset in DATASETS:
        for split in SPLITS:
            files = collect_masks(root, dataset, split)
            changed = valid = 0
            for path in files:
                c, v = changed_ratio(_load_mask(path))
                changed += c
                valid += v
            ratio = changed / valid if valid else float("nan")
            lines.append(f"| {dataset} | {split} | {ratio:.6f} | {changed} | {valid} |")
    lines.append("")
    lines.append("Valid pixels only (ignore label < 0 excluded). Not rounded to the paper's old quotes.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="dataset root with levir/ sysu/ pcd/")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(
            f"dataset root not found: {args.root}\n"
            "Refusing to invent π. Point --root at the directory Member 1 publishes.",
            file=sys.stderr,
        )
        return 2
    try:
        table = pi_table(args.root)
    except FileNotFoundError as exc:
        print(
            f"missing masks under {exc}\n"
            "Expected <root>/<levir|sysu|pcd>/{train,test}/**/*.png. No π was written.",
            file=sys.stderr,
        )
        return 2
    if args.out:
        args.out.write_text(table + "\n")
        print(f"wrote {args.out}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
