"""Baseline unconstrained MLP — the benchmark foil for all HNN comparisons."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hnet.models.backbones.mlp import MLP
from hnet.utils.device import get_device


class BaselineMLP(nn.Module):
    """Direct vector-field regressor: (q,p) → (dq/dt, dp/dt).

    No Hamiltonian structure. Included as the negative control in all
    benchmarks to demonstrate secular energy drift compared to HNN variants.

    This is the model from HNN_Core_Benchmarks.ipynb Cell 2.

    Parameters
    ----------
    input_dim : int
        Dimension of state vector z (= 2 for pendulum).
    hidden_dim : int
    n_layers : int
    activation : str
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,  # outputs dz/dt directly
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Map z → dz/dt directly (no Hamiltonian structure)."""
        return self.net(z)

    def vector_field(self, z: Tensor, **kwargs) -> Tensor:
        """Alias matching HamiltonianModelProtocol — same as forward."""
        return self.forward(z)

    def energy(self, z: Tensor) -> Tensor:
        """BaselineMLP has no meaningful energy. Returns zeros."""
        return torch.zeros(z.shape[:-1] + (1,), device=z.device, dtype=z.dtype)

    def make_scipy_vf(self, device: torch.device | str = "cpu"):
        """Return a scipy.integrate.solve_ivp-compatible vector field."""
        if isinstance(device, str):
            device = torch.device(device)

        def vf(t: float, z: np.ndarray) -> np.ndarray:
            z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dz = self.forward(z_t)
            return dz.detach().cpu().numpy().flatten()

        return vf
