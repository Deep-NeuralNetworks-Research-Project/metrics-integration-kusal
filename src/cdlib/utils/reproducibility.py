"""Seeding utility — coordinate with P2's `utils/seed.py` when the monorepo lands.

Do not fork a second seed API. This helper matches the recipe in
research/05 §7.3: seed python/numpy/torch/cuda, but do **not** force
full cuDNN determinism by default (the speed cost eats Colab quota;
statistical stability across ≥3 seeds is the actual requirement).
"""
from __future__ import annotations


def set_seed(seed: int, deterministic: bool = False) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
