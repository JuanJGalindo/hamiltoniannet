"""Abstract base class for Hamiltonian Neural Networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hnet.utils.autograd import symplectic_gradient


class HamiltonianNet(nn.Module, ABC):
    """Abstract base for all Hamiltonian-structured neural networks.

    Core contract
    -------------
    - ``scalar_field(z)``   Must be implemented by all subclasses.
                            Returns the learned Hamiltonian H_θ(z).
    - ``vector_field(z)``   Default: canonical symplectic gradient [dH/dp, -dH/dq].
                            Override for PoissonHNN, pHNN, sPHNN.
    - ``time_derivative``   Alias for ``vector_field``; used by Trainer and Integrator.

    Design rationale
    ----------------
    ``scalar_field`` and ``vector_field`` are separated so that:
    1. ``scalar_field`` can be evaluated without autograd (energy monitoring).
    2. ``vector_field`` can be overridden cleanly for non-canonical variants.
    3. SHNN changes the LOSS, not the architecture — it inherits both unchanged.

    ``SelfSupervisedHNN`` and ``SNO`` do NOT subclass ``HamiltonianNet``:
    they satisfy ``HamiltonianModelProtocol`` directly. This is honest —
    they are an ODE solver and a flow-map surrogate, not Hamiltonian approximators.
    """

    @abstractmethod
    def scalar_field(self, z: Tensor) -> Tensor:
        """H_θ(z): the learned Hamiltonian scalar field.

        Parameters
        ----------
        z : Tensor, shape (..., state_dim)

        Returns
        -------
        H : Tensor, shape (..., 1)
        """
        ...

    def vector_field(self, z: Tensor, create_graph: bool = True) -> Tensor:
        """Canonical symplectic gradient: dz/dt = [dH/dp, -dH/dq].

        Override this method for non-canonical systems (PoissonHNN, pHNN, sPHNN).

        Parameters
        ----------
        z : Tensor, shape (..., 2n)
        create_graph : bool
            Keep graph for higher-order gradients (True during training).

        Returns
        -------
        dz_dt : Tensor, shape (..., 2n)
        """
        z = z.clone().requires_grad_(True)
        H = self.scalar_field(z)
        return symplectic_gradient(H, z, create_graph=create_graph)

    def time_derivative(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Alias for ``vector_field``. Used by Trainer and Integrator."""
        return self.vector_field(z, **kwargs)

    def get_gradients(self, z: Tensor) -> Tensor:
        """Return ∇_z H_θ(z) as a detached tensor.

        Used by custom integrators that need the gradient without a graph.
        """
        z_req = z.clone().detach().requires_grad_(True)
        H = self.scalar_field(z_req)
        return torch.autograd.grad(H.sum(), z_req)[0].detach()

    def energy(self, z: Tensor) -> Tensor:
        """Evaluate H_θ(z) without gradient tracking (energy monitoring)."""
        with torch.no_grad():
            return self.scalar_field(z)

    def make_scipy_vf(self, device: torch.device | str = "cpu") -> Callable:
        """Return a scipy.integrate.solve_ivp-compatible vector field.

        Usage
        -----
        vf = model.make_scipy_vf(device)
        sol = solve_ivp(vf, t_span, z0, method="RK45")
        """
        if isinstance(device, str):
            device = torch.device(device)

        def vf(t: float, z: np.ndarray) -> np.ndarray:
            z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.enable_grad():
                dz = self.vector_field(z_t, create_graph=False)
            return dz.detach().cpu().numpy().flatten()

        return vf
