"""Derivative dataset with CubicSpline reconstruction.

Migrated from HNN_Core_Benchmarks.ipynb, Cell 2.

Why CubicSpline?
----------------
Finite-difference derivative labels impose a non-symplectic bias during
training (David & Mehats 2023, Prop. 1): the HNN implicitly learns to
minimize the residual of a forward-Euler scheme instead of Hamilton's
equations. CubicSpline reconstruction gives O(h^4) accuracy and avoids
this bias entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import Tensor
from torch.utils.data import Dataset

from hnet._protocols import PhysicsSystemProtocol


@dataclass(frozen=True)
class DataConfig:
    """Configuration for derivative dataset generation."""

    n_trajectories: int = 3
    n_points_per_traj: int = 500
    t_span: tuple[float, float] = (0.0, 20.0)
    noise_std: float = 0.001
    use_spline: bool = True
    seed: int = 42


class DerivativeDataset(Dataset):
    """PyTorch Dataset of (z, z_dot) pairs for supervised HNN training.

    Data generation pipeline:
    1. Sample initial conditions (random or user-specified)
    2. Integrate oracle ODE with high precision (RK45, rtol=1e-9)
    3. Add Gaussian noise to simulate measurement error
    4. Reconstruct derivatives via CubicSpline (or finite differences)

    Parameters
    ----------
    Z : np.ndarray, shape (N, state_dim)
    Z_dot : np.ndarray, shape (N, state_dim)
    device : torch.device
    """

    def __init__(
        self,
        Z: np.ndarray,
        Z_dot: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> None:
        device = torch.device(device) if isinstance(device, str) else device
        self.z = torch.tensor(Z, dtype=torch.float32, device=device)
        self.z_dot = torch.tensor(Z_dot, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"z": self.z[idx], "z_dot": self.z_dot[idx]}

    def as_batch(self) -> dict[str, Tensor]:
        """Return the full dataset as a single batch (full-batch training)."""
        return {"z": self.z, "z_dot": self.z_dot}

    @classmethod
    def from_system(
        cls,
        system: PhysicsSystemProtocol,
        initial_conditions: list[np.ndarray],
        config: DataConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> DerivativeDataset:
        """Generate dataset from a PhysicsSystem.

        Parameters
        ----------
        system : PhysicsSystemProtocol
        initial_conditions : list of z0 arrays, each shape (state_dim,)
        config : DataConfig
        device : torch.device or str

        Returns
        -------
        DerivativeDataset
        """
        if config is None:
            config = DataConfig()
        rng = np.random.default_rng(config.seed)
        all_Z: list[np.ndarray] = []
        all_Z_dot: list[np.ndarray] = []

        for z0 in initial_conditions:
            z0 = np.asarray(z0, dtype=float)
            t_eval, Z_clean = system.oracle_trajectory(
                z0=z0,
                t_span=config.t_span,
                n_points=config.n_points_per_traj,
            )

            # Add measurement noise
            Z_noisy = Z_clean + rng.normal(0, config.noise_std, Z_clean.shape)

            # Reconstruct derivatives
            if config.use_spline:
                Z_dot = _spline_derivatives(t_eval, Z_noisy)
            else:
                Z_dot = _finite_diff_derivatives(t_eval, Z_noisy)

            all_Z.append(Z_noisy)
            all_Z_dot.append(Z_dot)

        Z = np.concatenate(all_Z, axis=0)
        Z_dot = np.concatenate(all_Z_dot, axis=0)
        return cls(Z, Z_dot, device=device)


def _spline_derivatives(t: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Reconstruct time derivatives via CubicSpline (O(h^4) accuracy).

    Parameters
    ----------
    t : np.ndarray, shape (n_points,)
    Z : np.ndarray, shape (n_points, state_dim)

    Returns
    -------
    Z_dot : np.ndarray, shape (n_points, state_dim)
    """
    cs = CubicSpline(t, Z)
    return cs(t, 1)  # First derivative


def _finite_diff_derivatives(t: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Compute derivatives via central finite differences.

    O(h^2) accuracy, but imposes non-symplectic bias — use only for comparison.
    """
    dt = np.diff(t)
    Z_dot = np.zeros_like(Z)
    # Central differences for interior points
    Z_dot[1:-1] = (Z[2:] - Z[:-2]) / (t[2:] - t[:-2])[:, None]
    # Forward/backward for endpoints
    Z_dot[0] = (Z[1] - Z[0]) / dt[0]
    Z_dot[-1] = (Z[-1] - Z[-2]) / dt[-1]
    return Z_dot
