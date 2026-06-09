"""Unified training loop for all HNN variants."""
from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hnet._protocols import HamiltonianModelProtocol, LossTermProtocol
from hnet.training.config import TrainConfig
from hnet.utils.device import get_device
from hnet.utils.reproducibility import seed_everything


class Trainer:
    """Unified training loop — model-agnostic, dispatches to the loss.

    Works with any model satisfying ``HamiltonianModelProtocol`` and any
    loss satisfying ``LossTermProtocol`` (including ``WeightedLoss``).

    Parameters
    ----------
    model : HamiltonianModelProtocol (any nn.Module with vector_field + energy)
    loss : LossTermProtocol or WeightedLoss
    config : TrainConfig

    Usage
    -----
    trainer = Trainer(model, DerivativeMatchingLoss(), TrainConfig(epochs=5000))
    trainer.fit(dataset)
    print(trainer.history["loss"])
    """

    def __init__(
        self,
        model: HamiltonianModelProtocol,
        loss: LossTermProtocol | Any,
        config: TrainConfig = TrainConfig(),
    ) -> None:
        self.config = config
        self.device = get_device(config.device)
        seed_everything(config.seed)

        if isinstance(model, nn.Module):
            self.model = model.to(self.device)
        else:
            self.model = model

        self.loss_fn = loss
        self._history: dict[str, list[float]] = {"loss": []}
        self._best_loss = float("inf")
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> "Trainer":
        """Train the model.

        Parameters
        ----------
        dataset : torch Dataset with __getitem__ returning dict[str, Tensor]
        val_dataset : optional validation dataset (used for save_best)

        Returns
        -------
        self (for chaining)
        """
        self.model.train()

        for epoch in range(1, self.config.epochs + 1):
            batch = self._get_batch(dataset)
            loss_val, breakdown = self._step(batch)

            self._history["loss"].append(loss_val)
            for k, v in breakdown.items():
                self._history.setdefault(k, []).append(v)

            if self.config.scheduler == "reduce_on_plateau" and self._scheduler is not None:
                self._scheduler.step(loss_val)
            elif self._scheduler is not None:
                self._scheduler.step()

            if epoch % self.config.log_every == 0:
                print(f"Epoch {epoch:>6}/{self.config.epochs}  loss={loss_val:.6f}")

            if self.config.save_best and self.config.checkpoint_dir and loss_val < self._best_loss:
                self._best_loss = loss_val
                self._save(os.path.join(self.config.checkpoint_dir, "best.pt"))

        if self.config.checkpoint_dir:
            self._save(os.path.join(self.config.checkpoint_dir, "final.pt"))

        self.model.eval()
        return self

    def save(self, path: str) -> None:
        """Save model state dict to path."""
        self._save(path)

    def load(self, path: str) -> None:
        """Load model state dict from path."""
        if isinstance(self.model, nn.Module):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            raise TypeError("load() requires an nn.Module model.")

    @property
    def history(self) -> dict[str, list[float]]:
        """Training history: {metric_name: [value_per_epoch]}."""
        return dict(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
        self._optimizer.zero_grad()
        result = self.loss_fn(self.model, batch)
        if isinstance(result, tuple):
            loss, breakdown = result
        else:
            loss, breakdown = result, {}
        loss.backward()
        self._optimizer.step()
        return float(loss.item()), breakdown

    def _get_batch(self, dataset: Dataset) -> dict[str, torch.Tensor]:
        if self.config.batch_size is None:
            # Full-batch: call as_batch() if available, otherwise load all
            if hasattr(dataset, "as_batch"):
                return dataset.as_batch()
            loader = DataLoader(dataset, batch_size=len(dataset))  # type: ignore[arg-type]
            return next(iter(loader))
        else:
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            return next(iter(loader))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        if self.config.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.config.lr)
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.config.lr)
        if self.config.optimizer == "lbfgs":
            return torch.optim.LBFGS(params, lr=self.config.lr)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        if self.config.scheduler is None:
            return None
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=self.config.epochs
            )
        if self.config.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer, patience=200, factor=0.5
            )
        return None

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), path)
