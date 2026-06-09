"""Base classes for composable loss functions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from hnet._protocols import HamiltonianModelProtocol, LossTermProtocol


class LossTerm(ABC):
    """Abstract base for a single loss term.

    Subclass this to add new loss terms without touching any other module.
    The Trainer accepts any object satisfying LossTermProtocol — inheritance
    from this ABC is optional but provides a sensible default weight.
    """

    weight: float = 1.0

    @abstractmethod
    def __call__(
        self,
        model: HamiltonianModelProtocol,
        batch: dict[str, Tensor],
        **kwargs: Any,
    ) -> Tensor:
        """Compute scalar loss tensor."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


class WeightedLoss:
    """Compositor: L_total = Σ_i w_i · L_i(model, batch).

    Parameters
    ----------
    terms : list of (LossTerm, float) tuples
        Each tuple is (loss_term_instance, weight).

    Usage
    -----
    loss = WeightedLoss([
        (DerivativeMatchingLoss(), 1.0),
        (EnergyRegularizationLoss(H_true), 0.1),
    ])
    total, breakdown = loss(model, batch)
    """

    def __init__(self, terms: list[tuple[LossTermProtocol, float]]) -> None:
        if not terms:
            raise ValueError("WeightedLoss requires at least one term.")
        self.terms = terms

    def __call__(
        self,
        model: HamiltonianModelProtocol,
        batch: dict[str, Tensor],
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute total weighted loss and per-term breakdown.

        Returns
        -------
        total : Tensor (scalar)
        breakdown : dict[str, float]  — term name → value for logging
        """
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        breakdown: dict[str, float] = {}
        for term, w in self.terms:
            val = term(model, batch, **kwargs)
            total = total + w * val
            breakdown[type(term).__name__] = float(val.item())
        return total, breakdown

    def __repr__(self) -> str:
        parts = [f"({type(t).__name__}, w={w})" for t, w in self.terms]
        return f"WeightedLoss([{', '.join(parts)}])"
