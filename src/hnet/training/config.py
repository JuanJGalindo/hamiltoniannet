"""Training configuration dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration.

    Parameters
    ----------
    epochs : int
    lr : float
        Learning rate for the optimizer.
    optimizer : str
        "adam" (default), "adamw", or "lbfgs".
    scheduler : str, optional
        "cosine", "reduce_on_plateau", or None.
    batch_size : int, optional
        None = full-batch training (default). Set for mini-batch.
    device : str
        "auto", "cuda", "cpu", or "mps".
    seed : int
        Random seed applied before training.
    log_every : int
        Print training progress every N epochs.
    checkpoint_dir : str, optional
        Directory to save best/final checkpoints.
    save_best : bool
        Save checkpoint at best validation loss (requires val_dataset).
    """

    epochs: int = 5000
    lr: float = 1e-3
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    batch_size: Optional[int] = None
    device: str = "auto"
    seed: int = 42
    log_every: int = 500
    checkpoint_dir: Optional[str] = None
    save_best: bool = True

    def __post_init__(self) -> None:
        valid_optimizers = {"adam", "adamw", "lbfgs"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got '{self.optimizer}'"
            )
        valid_schedulers = {"cosine", "reduce_on_plateau", None}
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"scheduler must be one of {valid_schedulers}, got '{self.scheduler}'"
            )
