# Evaluation JSON schema (version 1)

Every number in the paper comes from `python -m cdlib.cli.evaluate`.
**Add keys. Do not rename them.** Member 5 fills `calibration`; do not fork the script.

```bash
python -m cdlib.cli.evaluate --checkpoint PATH --split test --out results/eval.json
python -m cdlib.cli.evaluate --dummy --split test          # CI / no weights yet
python -m cdlib.cli.evaluate --dummy --robustness suite
```

`--dummy` is the stand-in pair CNN. A checkpoint that is not that module is refused. No score is invented for an unknown architecture.

## Keys

| key | meaning |
|---|---|
| `precision`, `recall`, `f1`, `iou` | aggregate changed-class scores, valid pixels only |
| `boundary_f1` | BF-score, tolerance **2 px** (Csurka et al., BMVC 2013) |
| `boundary_iou` | secondary, dilation_ratio 0.02 |
| `positive_prediction_rate_fwd` / `_rev` | fraction of valid pixels predicted changed, each ordering |
| `sce_flip` | fraction of valid pixels whose hard label flips on swap |
| `sce_prob` | mean absolute probability difference on swap |
| `delta_f1_swap` | aggregate F1(forward) − F1(swapped) |
| `spearman_rho` | Spearman correlation of the two probability maps. Equal constant maps are 1; any other constant map is 0 (rank correlation is undefined) |
| `threshold_f1_optimal_fwd` / `_rev` | chosen on the validation split only |
| `fp_pixels_per_pair` | mean false-positive pixels on no-change pairs |
| `fp_regions_per_pair` | mean 8-connected FP regions on no-change pairs |
| `false_alert_rate` | fraction of no-change pairs with any region ≥ `false_alert_min_area` (20 px) |
| `pi` | changed / valid pixels |
| `n_pixels`, `n_ignored`, `n_pairs`, `seed`, `checkpoint_sha256` | bookkeeping |
| `change_size` | provisional bins: small < 64, medium ≤ 1024, large above. `provisional` is 1.0 until a LEVIR histogram replaces the edges |
| `calibration` | `{}` until Member 5 adds fields |
| `nuisance` | `{}`, or the full severity × corruption × direction grid when `--robustness suite` |

SCE names still need a confirmation from P4 before they are quoted in the paper. The formulas above are the ones this code computes.

Example file: `docs/eval.example.json`. The digest there is `sha256(b"dummy")`, which is what `--dummy` writes.

```json
{
  "schema_version": "1",
  "split": "test",
  "config": "unnamed",
  "seed": 0,
  "checkpoint_sha256": "<hex>",
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "iou": 0.0,
  "boundary_f1": 0.0,
  "boundary_iou": 0.0,
  "positive_prediction_rate_fwd": 0.0,
  "positive_prediction_rate_rev": 0.0,
  "sce_flip": 0.0,
  "sce_prob": 0.0,
  "delta_f1_swap": 0.0,
  "spearman_rho": 0.0,
  "threshold_f1_optimal_fwd": 0.5,
  "threshold_f1_optimal_rev": 0.5,
  "fp_pixels_per_pair": 0.0,
  "fp_regions_per_pair": 0.0,
  "false_alert_rate": 0.0,
  "false_alert_min_area": 20.0,
  "pi": 0.0,
  "n_pixels": 0,
  "n_ignored": 0,
  "n_pairs": 0,
  "change_size": {"provisional": 1.0, "f1_small": 0.0, "sce_flip_small": 0.0},
  "calibration": {},
  "nuisance": {}
}
```
