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


SIZE_BINS = ("small", "medium", "large")


def emit_change_size_table(records: Sequence[Mapping]) -> str:
    """F1 and SCE_flip per provisional change-size bin.

    Reads the nested ``change_size`` object already stored in each eval JSON.
    """
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get("config", "unnamed"))].append(rec)
    header = "| config | bin | f1 | sce_flip | n |"
    lines = [header, "|---|---|---|---|---|"]
    flagged = False
    for name in sorted(groups):
        rows = groups[name]
        n = len(rows)
        for bin_name in SIZE_BINS:
            f1s = []
            sces = []
            for rec in rows:
                size = rec.get("change_size") or {}
                f1s.append(float(size.get(f"f1_{bin_name}", 0.0)))
                sces.append(float(size.get(f"sce_flip_{bin_name}", 0.0)))
            f1_arr = np.asarray(f1s, dtype=np.float64)
            sce_arr = np.asarray(sces, dtype=np.float64)
            if n == 1:
                f1_cell = f"{f1_arr[0]:.4f}"
                sce_cell = f"{sce_arr[0]:.4f}"
                flagged = True
                flag = " †"
            else:
                f1_cell = f"{f1_arr.mean():.4f} ± {f1_arr.std(ddof=1):.4f}"
                sce_cell = f"{sce_arr.mean():.4f} ± {sce_arr.std(ddof=1):.4f}"
                flag = ""
            lines.append(f"| {name}{flag} | {bin_name} | {f1_cell} | {sce_cell} | {n} |")
    lines.append("")
    lines.append("Bins are provisional (small < 64, medium ≤ 1024, large > 1024) until a LEVIR histogram replaces them.")
    if flagged:
        lines.append("† n=1: ± omitted.")
    lines.append("No significance tests.")
    return "\n".join(lines)


def change_size_table_from_paths(paths: Iterable[str | Path]) -> str:
    return emit_change_size_table(_load(paths))
