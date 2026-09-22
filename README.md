# metrics-integration-kusal

**P5 — Metrics, robustness & paper integration** for *Pair-Order Consistent & Uncertainty-Aware Semantic Change Detection* (University of Moratuwa, CSE). Parent org: [Deep-NeuralNetworks-Research-Project](https://github.com/Deep-NeuralNetworks-Research-Project). P3's matching slice: [modeling-encoders-Weenuka](https://github.com/Deep-NeuralNetworks-Research-Project/modeling-encoders-Weenuka).

This repo holds only the component set owned by P5:

```
src/cdlib/metrics/     src/cdlib/cli/
paper/                 docs/                 configs/corruptions/
scripts/compute_pi.py
```

It follows the frozen metric contract in `CLAUDE.md`: `reset() / update(outputs, batch) / compute() -> dict[str, float]`. Every paper number comes from `python -m cdlib.cli.evaluate`. The schema is documented in [docs/eval_schema.md](docs/eval_schema.md). Add keys; never rename them.

## What is implemented

- **`cli/evaluate.py`** — single eval path. Writes precision, recall, F1, IoU, boundary F1 (2 px), SCE fields, false-alert rates, provisional change-size bins, empty `calibration`, and optional nuisance grid.
- **`metrics/segmentation.py`** — aggregate (primary) and per-image (secondary) P/R/F1/IoU, changed-pixel ratio, ignore-label `-1`. True-negative-only images are excluded from mF1, not nan-dropped.
- **`metrics/boundary_f1.py`** — BF-score at tolerance 2 px (Csurka BMVC 2013). Headline boundary number.
- **`metrics/boundary.py`** — Boundary IoU (secondary), HD95 + ASD, empty-mask guards (no `inf`).
- **`metrics/order.py`** — `sce_flip`, `sce_prob`, `delta_f1_swap`, Spearman ρ, per-ordering PPR, validation-only F1 threshold sweep.
- **`metrics/change_size.py`** — provisional small / medium / large GT component bins (F1 and SCE_flip).
- **`metrics/region.py`** — 8-connected components, min-size sweep `{1,5,20,50}`, image-level false-alert rate, object-level F1 at IoU 0.5.
- **`metrics/calibration.py`** — class-wise + foreground-restricted ECE with equal-mass bins, Brier (+ Murphy reliability), NLL, pair-level AURC/E-AURC (MSR and Soft-Dice-Confidence).
- **`metrics/temperature.py`** — post-hoc $T$; **refuses** `fitted_on="clean_val"` (Ovadia et al.).
- **`metrics/corruptions.py`** + **`configs/corruptions/suite.yaml`** — one-frame-only suite, both directions, seeded by `(pair_id, corruption, severity, direction)`.
- **`metrics/tables.py`** — Table I from eval JSONs (mean ± std, n per cell; n=1 flagged, no p-values).
- **`cli/benchmark.py`** — warm-up excluded, CUDA events, median+IQR, GPU logged, pair-input FLOPs, peak reserved vs allocated memory. **P3 is unblocked on this.**
- **`scripts/compute_pi.py`** — π for LEVIR / SYSU / PCD when mask folders exist; exits clearly if the root is missing.
- **`docs/gpu_queue.md`** — overnight seed / fold order once checkpoints exist.
- **`paper/`** — `acmart` `[sigconf]` skeleton, section files, `references.bib` with `author_year_firstword` keys.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install -e ".[dev]"
pytest -m "not gpu" -q
python -m cdlib.cli.evaluate --dummy --split test --out results/eval.json
python -m cdlib.cli.evaluate --checkpoint PATH --split test --out results/eval.json
python -m cdlib.cli.evaluate --dummy --robustness suite --out results/eval.json
python scripts/compute_pi.py --root /path/to/datasets --out results/pi.md
python -m cdlib.cli.benchmark --model dummy --img-size 64 --device cpu
python -m cdlib.cli.export_masks exp_id=demo --overlays best,median,worst
```

W&B project (team): `moratuwa-cd-p5` — put `WANDB_API_KEY` in Colab/Kaggle secrets, never in git.

## Hand-offs

| To | What |
|---|---|
| Everyone | `cli/evaluate.py` + `docs/eval_schema.md` before any paper number |
| P1 | `docs/nuisance_label.md`, π convention, mask folder layout for `compute_pi.py` |
| P3 | `python -m cdlib.cli.benchmark` — encoder step-time, params, pair FLOPs, memory |
| Member 5 | empty `calibration` object in the eval JSON — add fields, do not fork the script |
| P2, P4 | `docs/dice_calibration_conflict.md` by week 4; confirm SCE field names before quoting |

## Not in this repo

Datasets, trainer, Hydra root, encoders, alignment, pair-order loss — P1/P2/P3/P4. Consume the frozen contracts; stub locally. GPU seed completions wait on checkpoints (see `docs/gpu_queue.md`).
