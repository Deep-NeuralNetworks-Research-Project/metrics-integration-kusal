# metrics-integration-kusal

**P5 — Metrics, robustness & paper integration** for *Pair-Order Consistent & Uncertainty-Aware Semantic Change Detection* (University of Moratuwa, CSE). Parent org: [Deep-NeuralNetworks-Research-Project](https://github.com/Deep-NeuralNetworks-Research-Project). P3's matching slice: [modeling-encoders-Weenuka](https://github.com/Deep-NeuralNetworks-Research-Project/modeling-encoders-Weenuka).

This repo holds only the component set owned by P5:

```
src/cdlib/metrics/     src/cdlib/cli/benchmark.py
paper/                 docs/figure_standards.md  docs/MODEL_CARD.md
```

It follows the frozen metric contract in `CLAUDE.md`: `reset() / update(outputs, batch) / compute() -> dict[str, float]`. `cli/evaluate.py` and `cli/export_masks.py` here are **starter stubs** so the slice is runnable without P2's Hydra CLI; they will be replaced when the team monorepo lands.

## What is implemented

- **`metrics/segmentation.py`** — aggregate (primary) and per-image (secondary) P/R/F1/IoU, changed-pixel ratio, ignore-label `-1`. True-negative-only images are excluded from mF1, not nan-dropped.
- **`tests/test_metrics_handcomputed.py`** — calculator-verified 4×4 and 5×5 fixtures. These are the source of truth.
- **`metrics/boundary.py`** — Boundary IoU (Cheng et al., `dilation_ratio=0.02`), HD95 + ASD, empty-mask guards (no `inf`).
- **`metrics/region.py`** — 8-connected components, min-size sweep `{1,5,20,50}`, image-level false-alert rate, object-level F1 at IoU 0.5.
- **`metrics/calibration.py`** — class-wise + foreground-restricted ECE with equal-mass bins, Brier (+ Murphy reliability), NLL, pair-level AURC/E-AURC (MSR and Soft-Dice-Confidence).
- **`metrics/temperature.py`** — post-hoc $T$; **refuses** `fitted_on="clean_val"` (Ovadia et al.).
- **`metrics/corruptions.py`** — one-frame-only suite, both directions, ffmpeg H.264/HEVC round-trip with cache.
- **`cli/benchmark.py`** — warm-up excluded, CUDA events, median+IQR, GPU logged, pair-input FLOPs (`torch.utils.flop_counter`), peak reserved vs allocated memory. **P3 is unblocked on this.**
- **`metrics/stats.py`** — bootstrap CIs over the test set, ablation tables, **no significance tests at n=3**.
- **`metrics/overlays.py`** — Okabe-Ito TP/FP/FN (no red-green).
- **`paper/`** — `acmart` `[sigconf]` skeleton, section files, `references.bib` with `author_year_firstword` keys.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel
pip install -e ".[dev]"
pytest tests/test_metrics_handcomputed.py -v
pytest -m "not gpu" -q
python -m cdlib.cli.evaluate +metrics=all
python -m cdlib.cli.benchmark --model dummy --img-size 64 --device cpu
python -m cdlib.cli.export_masks exp_id=demo --overlays best,median,worst
```

W&B project (team): `moratuwa-cd-p5` — put `WANDB_API_KEY` in Colab/Kaggle secrets, never in git.

## Hand-offs

| To | What |
|---|---|
| P1 | `docs/nuisance_label.md` — agree ids before either of us ships bins |
| P3 | `python -m cdlib.cli.benchmark` — encoder step-time, params, pair FLOPs, memory |
| P2, P4 | `docs/dice_calibration_conflict.md` by week 4 |

## Not in this repo

Datasets, trainer, Hydra root, encoders, alignment, pair-order loss — P1/P2/P3/P4. Consume the frozen contracts; stub locally.
