# P5 architecture — metric contract and evaluation protocol

Locked with the rest of the team in week 1. Changing a contract needs a team-wide PR.

## Metric contract

```
reset() / update(outputs, batch) / compute() -> dict[str, float]
```

`outputs` follows the frozen model forward: `logits [B,1,H,W]` pre-sigmoid, optional `confidence`, `aux`.
`batch` follows the frozen dataset contract: `mask [B,1,H,W]` in `{0,1,-1}`, `nuisance_label`, `img1`/`img2`.

Ignore label `-1` is excluded from every pixel count.

## Pooling protocol (state this verbatim in the paper)

1. **Primary:** aggregate / corpus / micro P/R/F1/IoU — one TP/FP/FN over the whole test set, then the formula once (Corley et al., arXiv:2402.06994).
2. **Secondary:** per-image mF1. True-negative-only images (`TP=FP=FN=0`) are excluded from the average and reported via image-level false-alert rate. They are never silently `nan`-dropped.
3. **Changed-pixel ratio** π is returned next to every result. Overall accuracy is not a headline metric.
4. F1 and IoU at the same pooling are algebraically tied (`F1 = 2 IoU / (1+IoU)`); they are not independent evidence.

## Threshold

One decision threshold, fixed on **validation scenes only**. Record whether it is F1-optimal or precision-constrained. Never retune on test.

## Calibration

Naive all-pixel ECE is invalid at 2–5% change. Report class-wise + foreground-restricted ECE with **equal-mass** bins. Temperature scaling is fitted on **shift-representative** data, not clean val (Ovadia et al., NeurIPS 2019).

## Corruption

Every nuisance transform is applied to **one frame of the pair only**, both directions. See `docs/corruption_severity.md` and `docs/nuisance_label.md`.

## Efficiency

- FLOPs: `torch.utils.flop_counter.FlopCounterMode` (true FLOPs = mul+add). Also report MACs = FLOPs/2.
- Latency: warm-up excluded, CUDA events, `synchronize`, median + IQR. GPU name every run.
- Peak memory headline: `max_memory_reserved`; secondary `max_memory_allocated`.

## Registry

New metric = one file under `src/cdlib/metrics/` + one `METRIC_REGISTRY.register` line in `registry.py`.
