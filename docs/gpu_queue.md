# GPU queue

Re-scoring a saved checkpoint is cheap and can sit between other jobs.
New seeds are not cheap. Do not start them from this slice until the
checkpoint and the dataset loader actually exist.

Overnight order, when they do:

1. The checkpoint Member 1 is blocked on (their first slot).
2. PCD folds 4 and 5, every configuration. This is what turns "three folds" into the 5-fold protocol.
3. SYSU-CD seeds 2 and 3. SYSU is currently a single run, so it has no spread.
4. LEVIR λ=5 seeds 2 and 3. That row has no ± until these exist.

After each run:

```bash
python -m cdlib.cli.evaluate --checkpoint PATH --split test --config NAME --seed N --out results/NAME_seedN.json
```

Then:

```bash
python -c "from cdlib.metrics.tables import table_i_from_paths, change_size_table_from_paths; print(table_i_from_paths(['results'])); print(change_size_table_from_paths(['results']))"
```

A cell with one JSON is printed without ± and marked †. Do not claim an error bar there.

π is a separate job, and only when Member 1's mask folders are on disk:

```bash
python scripts/compute_pi.py --root /path/to/datasets --out results/pi.md
```

The script exits 2 if a split is missing. It will not guess 4.59 or 21.3.
