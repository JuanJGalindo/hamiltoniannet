"""Nonlinear pendulum system — the primary test system for all HNN benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np

from hnet.systems.base import PhysicsSystem, SystemConfig
from hnet.systems.registry import register_system


@register_system("nonlinear_pendulum")
class NonlinearPendulum(PhysicsSystem):
    """Nonlinear pendulum with Hamiltonian H(q,p) = p²/2 - cos(q).

    State vector z = [q, p] where q is angle (rad) and p is angular momentum.

    This is the canonical test system used in Greydanus 2019 and
    HNN_Core_Benchmarks.ipynb. The separatrix (H=1) divides libration
    (H<1) from rotation (H>1) regimes.

    Parameters
    ----------
    mass : float
        Pendulum mass (default 1.0, non-dimensionalized).
    length : float
        Pendulum length (default 1.0, non-dimensionalized).
    gravity : float
        Gravitational acceleration (default 1.0, non-dimensionalized).
    """

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        gravity: float = 1.0,
    ) -> None:
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if gravity <= 0:
            raise ValueError(f"gravity must be positive, got {gravity}")
        self._mass = mass
        self._length = length
        self._gravity = gravity
        self._config = SystemConfig(
            state_dim=2,
            phase_dim=2,
            is_canonical=True,
            manifold=None,
            param_names=("mass", "length", "gravity"),
        )

    @property
    def config(self) -> SystemConfig:
        return self._config

    def hamiltonian(self, z: np.ndarray, **params: Any) -> float:
        """H(q,p) = p² / (2·m·l²) - m·g·l·cos(q)."""
        q, p = z[0], z[1]
        m, l, g = self._mass, self._length, self._gravity
        return float(p**2 / (2.0 * m * l**2) - m * g * l * np.cos(q))

    def equations_of_motion(self, t: float, z: np.ndarray, **params: Any) -> np.ndarray:
        """Hamilton's equations: dq/dt = p/(m·l²), dp/dt = -m·g·l·sin(q)."""
        q, p = z[0], z[1]
        m, l, g = self._mass, self._length, self._gravity
        dq = p / (m * l**2)
        dp = -m * g * l * np.sin(q)
        return np.array([dq, dp])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def energy_at(self, q: float, p: float) -> float:
        """Scalar energy evaluation from individual coordinates."""
        return self.hamiltonian(np.array([q, p]))

    def separatrix_energy(self) -> float:
        """Energy of the separatrix (unstable equilibrium at q=π).

        H(π, 0) = -m·g·l·cos(π) = m·g·l divides libration from rotation.
        """
        m, l, g = self._mass, self._length, self._gravity
        return float(m * g * l)

    def sample_phase_space(
        self,
        n_points: int = 1500,
        q_range: tuple[float, float] = (-4.0, 4.0),
        p_range: tuple[float, float] = (-3.5, 3.5),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample random points in phase space.

        Returns
        -------
        Z : np.ndarray, shape (n_points, 2) — random (q,p) pairs
        """
        if rng is None:
            rng = np.random.default_rng()
        q = rng.uniform(*q_range, size=n_points)
        p = rng.uniform(*p_range, size=n_points)
        return np.stack([q, p], axis=1)

    def __repr__(self) -> str:
        return (
            f"NonlinearPendulum(mass={self._mass}, length={self._length}, gravity={self._gravity})"
        )


@register_system("simple_pendulum")
class SimplePendulum(NonlinearPendulum):
    """Small-angle approximation: H(q,p) ≈ p²/2 + q²/2 (harmonic oscillator).

    Valid for |q| << 1. Useful as a sanity-check system where the
    Hamiltonian is exactly quadratic and the HNN should recover it
    with zero approximation error.
    """

    def hamiltonian(self, z: np.ndarray, **params: Any) -> float:
        """H(q,p) = (p² + q²) / 2 (normalized harmonic oscillator)."""
        return float((z[0] ** 2 + z[1] ** 2) / 2.0)

    def equations_of_motion(self, t: float, z: np.ndarray, **params: Any) -> np.ndarray:
        """dq/dt = p, dp/dt = -q."""
        return np.array([z[1], -z[0]])
