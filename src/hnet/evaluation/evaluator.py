"""Evaluator: rolls out a trained model and computes metrics."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from hnet._protocols import HamiltonianModelProtocol, IntegratorProtocol, PhysicsSystemProtocol
from hnet.evaluation.metrics import (
    casimir_error,
    energy_drift,
    max_energy_error,
    relative_l2_error,
    spin_norm_error,
)
from hnet.utils.device import get_device


class Evaluator:
    """Post-training evaluator: rollout + metrics.

    Decoupled from Trainer — can be used on any loaded checkpoint.

    Parameters
    ----------
    model : HamiltonianModelProtocol
    system : PhysicsSystemProtocol
    integrator : IntegratorProtocol
    device : str or torch.device
    """

    def __init__(
        self,
        model: HamiltonianModelProtocol,
        system: PhysicsSystemProtocol,
        integrator: IntegratorProtocol,
        device: str | torch.device = "auto",
    ) -> None:
        self.model = model
        self.system = system
        self.integrator = integrator
        self.device = get_device(device) if isinstance(device, str) else device

        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def rollout(
        self,
        z0: np.ndarray | list,
        t_span: tuple[float, float],
        n_steps: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the model from z0.

        Returns
        -------
        t_eval : np.ndarray, shape (n_steps,)
        Z_pred : np.ndarray, shape (n_steps, state_dim)
        """
        z0 = np.asarray(z0, dtype=float)
        return self.integrator.integrate(self.model, z0, t_span, n_steps)

    def evaluate(
        self,
        z0: np.ndarray | list,
        t_span: tuple[float, float],
        n_steps: int = 1000,
    ) -> dict[str, float]:
        """Rollout and compute all relevant metrics.

        Parameters
        ----------
        z0 : initial condition
        t_span : (t0, tf)
        n_steps : output time points

        Returns
        -------
        dict with keys: "rel_l2_q", "max_energy_error", "casimir_*", etc.
        """
        z0 = np.asarray(z0, dtype=float)

        # Oracle reference
        t_ref, Z_ref = self.system.oracle_trajectory(z0, t_span, n_points=n_steps)

        # Model prediction
        t_pred, Z_pred = self.rollout(z0, t_span, n_steps)

        results: dict[str, float] = {}

        # Trajectory accuracy (position coordinate only)
        results["rel_l2_q"] = relative_l2_error(Z_pred[:, 0], Z_ref[:, 0])

        # Energy conservation
        results["max_energy_error"] = max_energy_error(
            Z_pred, self.system.hamiltonian, normalize=True
        )

        # Casimir invariants (for non-canonical systems)
        casimir_errs = self.system.casimir_errors(Z_pred)
        for name, arr in casimir_errs.items():
            results[f"max_{name}_error"] = float(np.max(arr))

        # Spin norm (if applicable)
        state_dim = self.system.state_dim
        if state_dim == 3:
            norm_errs = spin_norm_error(Z_pred)
            results["max_spin_norm_error"] = float(np.max(norm_errs))

        return results

    def energy_drift_array(
        self,
        z0: np.ndarray | list,
        t_span: tuple[float, float],
        n_steps: int = 1000,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return time and energy drift array (for plotting).

        Returns
        -------
        t : np.ndarray, shape (n_steps,)
        drift : np.ndarray, shape (n_steps,)
        """
        z0 = np.asarray(z0, dtype=float)
        t, Z = self.rollout(z0, t_span, n_steps)
        drift = energy_drift(Z, self.system.hamiltonian, normalize=normalize)
        return t, drift
