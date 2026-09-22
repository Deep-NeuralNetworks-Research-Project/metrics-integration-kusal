"""Paper tables from frozen eval JSON. No p-values.

A cell with one seed is printed without ± and flagged. n is recorded
for every cell.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

HEADLINE = ("precision", "recall", "f1", "iou", "boundary_f1")


def _load(paths: Iterable[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob("*.json"))
        else:
            files = [p]
        for f in files:
            rows.append(json.loads(f.read_text()))
    return rows


def emit_table_i(records: Sequence[Mapping], metrics: Sequence[str] = HEADLINE) -> str:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get("config", "unnamed"))].append(rec)
    header = "| config | " + " | ".join(metrics) + " | n |"
    rule = "|---|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    lines = [
        header,
        rule,
    ]
    flagged = False
    for name in sorted(groups):
        rows = groups[name]
        n = len(rows)
        cells = []
        for key in metrics:
            vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
            if n == 1:
                cells.append(f"{vals[0]:.4f}")
                flagged = True
            else:
                std = float(vals.std(ddof=1)) if n > 1 else 0.0
                cells.append(f"{vals.mean():.4f} ± {std:.4f}")
        flag = " †" if n == 1 else ""
        lines.append(f"| {name}{flag} | " + " | ".join(cells) + f" | {n} |")
    lines.append("")
    if flagged:
        lines.append("† n=1: ± omitted. The caption must not promise an error bar for that cell.")
    lines.append("No significance tests. n is the number of eval JSONs in the cell (seeds or folds).")
    return "\n".join(lines)


def table_i_from_paths(paths: Iterable[str | Path]) -> str:
    return emit_table_i(_load(paths))


def nuisance_markdown(grid: Mapping[str, float]) -> str:
    """Supplementary table: corruption × severity × direction."""
    lines = [
        "| corruption | severity | direction | f1 | retention |",
        "|---|---|---|---|---|",
    ]
    keys = sorted(k for k in grid if k.endswith("/f1") and k.count("/") >= 2)
    for key in keys:
        # name/sN/which/f1
        name, sev, which, _ = key.split("/")
        f1 = grid[key]
        ret = grid.get(f"{name}/{sev}/{which}/retention", float("nan"))
        lines.append(f"| {name} | {sev} | {which} | {f1:.4f} | {ret:.4f} |")
    if len(lines) == 2:
        lines.append("| — | — | — | — | — |")
    return "\n".join(lines)
