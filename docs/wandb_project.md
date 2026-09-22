# Weights & Biases project

Documented project name: **`moratuwa-cd-p5`**.

Create it once (any teammate with a W&B account):

```bash
pip install wandb
wandb login   # paste the key; never commit it
wandb project create moratuwa-cd-p5 --entity <your-team-or-username>
```

Or in the browser: https://wandb.ai/ → New project → name `moratuwa-cd-p5` → invite the org.

Put `WANDB_API_KEY` in Colab/Kaggle secrets only. This repo never stores the key.

Until the project exists, local runs can use `wandb offline` or skip logging. Evaluation numbers still come from `python -m cdlib.cli.evaluate`, not from W&B.
