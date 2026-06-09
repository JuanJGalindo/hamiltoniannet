"""
Structural contracts for the hnet framework.

All public interfaces are defined here as typing.Protocol objects.
Classes satisfy these protocols by duck-typing — no mandatory base class
inheritance is required. Users can implement custom systems and models
in their own codebase without importing any hnet ABC.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor


@runtime_checkable
class PhysicsSystemProtocol(Protocol):
    """Structural contract for any Hamiltonian physical system."""

    @property
    def state_dim(self) -> int:
        """Dimension of the full state vector z."""
        ...

    def hamiltonian(self, z: np.ndarray, **params: Any) -> float:
        """True Hamiltonian H(z). Used only for evaluation, never in training."""
        ...

    def equations_of_motion(self, t: float, z: np.ndarray, **params: Any) -> np.ndarray:
        """RHS of Hamilton's equations: dz/dt = f(t, z)."""
        ...


@runtime_checkable
class HamiltonianModelProtocol(Protocol):
    """Structural contract for any trainable Hamiltonian model."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """PyTorch parameter iterator (satisfied by any nn.Module)."""
        ...

    def vector_field(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Time derivative of the state: dz/dt = f(z). Shape: (..., state_dim)."""
        ...

    def energy(self, z: Tensor) -> Tensor:
        """Scalar energy/Hamiltonian estimate. Shape: (..., 1)."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Switch training/eval mode (satisfied by any nn.Module)."""
        ...

    def eval(self) -> Any:
        """Switch to eval mode (satisfied by any nn.Module)."""
        ...


@runtime_checkable
class IntegratorProtocol(Protocol):
    """Structural contract for any ODE integrator."""

    def integrate(
        self,
        model: HamiltonianModelProtocol,
        z0: np.ndarray,
        t_span: tuple[float, float],
        n_steps: int,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate from z0 over t_span.

        Returns
        -------
        t_eval : np.ndarray, shape (n_steps,)
        Z      : np.ndarray, shape (n_steps, state_dim)
        """
        ...


@runtime_checkable
class LossTermProtocol(Protocol):
    """Structural contract for a composable loss term."""

    def __call__(
        self,
        model: HamiltonianModelProtocol,
        batch: dict[str, Tensor],
        **kwargs: Any,
    ) -> Tensor:
        """Compute scalar loss tensor."""
        ...
