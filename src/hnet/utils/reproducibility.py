"""Reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Covers Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — global legacy seeding is the explicit purpose here
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
