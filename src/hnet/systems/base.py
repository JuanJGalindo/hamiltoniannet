"""Base class and configuration for Hamiltonian physical systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SystemConfig:
    """Immutable configuration for a Hamiltonian system.

    Parameters
    ----------
    state_dim : int
        Dimension of the full state vector z (= 2n for canonical systems).
    phase_dim : int
        Canonical dimension (equals state_dim for canonical systems).
    is_canonical : bool
        True for standard (q,p) systems; False for Lie-Poisson / non-canonical.
    manifold : str, optional
        Constraint manifold name: "sphere", "se3", or None for unconstrained R^{2n}.
    casimir_functions : tuple of callables
        Functions C_i(z) -> float whose level sets define the manifold invariants.
    param_names : tuple of str
        Names of system parameters (e.g., ("mass", "length", "gravity")).
    """

    state_dim: int
    phase_dim: int
    is_canonical: bool = True
    manifold: str | None = None
    casimir_functions: tuple[Callable[[np.ndarray], float], ...] = field(default_factory=tuple)
    param_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {self.state_dim}")
        if self.phase_dim < 1 or self.phase_dim > self.state_dim:
            raise ValueError(
                f"phase_dim must be in [1, state_dim={self.state_dim}], got {self.phase_dim}"
            )


class PhysicsSystem(ABC):
    """Abstract base for Hamiltonian physical systems.

    Owns the oracle dynamics and data factories. Never references any model.

    Subclasses must implement:
        - ``config`` property (returns SystemConfig)
        - ``hamiltonian(z, **params)`` — the true H(z)
        - ``equations_of_motion(t, z, **params)`` — RHS of Hamilton's equations

    All other methods have sensible defaults built on these two.
    """

    @property
    @abstractmethod
    def config(self) -> SystemConfig:
        """System configuration (immutable)."""
        ...

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @abstractmethod
    def hamiltonian(self, z: np.ndarray, **params: Any) -> float:
        """True Hamiltonian H(z). Used only for evaluation, never in training."""
        ...

    @abstractmethod
    def equations_of_motion(self, t: float, z: np.ndarray, **params: Any) -> np.ndarray:
        """RHS: dz/dt = f(t, z). Must return shape (state_dim,)."""
        ...

    # ------------------------------------------------------------------
    # Oracle integration
    # ------------------------------------------------------------------

    def oracle_trajectory(
        self,
        z0: np.ndarray,
        t_span: tuple[float, float],
        n_points: int = 500,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-9,
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the true equations of motion.

        Parameters
        ----------
        z0 : array_like, shape (state_dim,)
        t_span : (t0, tf)
        n_points : number of output time points
        method, rtol, atol : passed to scipy solve_ivp

        Returns
        -------
        t_eval : np.ndarray, shape (n_points,)
        Z      : np.ndarray, shape (n_points, state_dim)
        """
        z0 = np.asarray(z0, dtype=float)
        if z0.shape != (self.state_dim,):
            raise ValueError(f"z0 must have shape ({self.state_dim},), got {z0.shape}")
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def rhs(t: float, z: np.ndarray) -> np.ndarray:
            return self.equations_of_motion(t, z, **params)

        sol = solve_ivp(rhs, t_span, z0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Oracle integration failed: {sol.message}")
        return sol.t, sol.y.T  # (n_points,), (n_points, state_dim)

    # ------------------------------------------------------------------
    # Casimir invariants
    # ------------------------------------------------------------------

    def casimir_errors(self, Z: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate all Casimir invariants along trajectory Z.

        Parameters
        ----------
        Z : np.ndarray, shape (n_points, state_dim)

        Returns
        -------
        dict mapping casimir name to error array of shape (n_points,)
        """
        errors: dict[str, np.ndarray] = {}
        for i, C in enumerate(self.config.casimir_functions):
            values = np.array([C(z) for z in Z])
            errors[f"casimir_{i}"] = np.abs(values - values[0])
        return errors

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.config.state_dim}, "
            f"canonical={self.config.is_canonical})"
        )
