"""HNN — Hamiltonian Neural Network (Greydanus et al., 2019).

Reference
---------
Greydanus, S., Dzamba, M., & Yosinski, J. (2019).
Hamiltonian neural networks. NeurIPS 2019.
"""
from __future__ import annotations

from torch import Tensor

from hnet.models.backbones.mlp import MLP
from hnet.models.base import HamiltonianNet


class HNN(HamiltonianNet):
    """Hamiltonian Neural Network (Greydanus 2019).

    Learns a scalar Hamiltonian H_θ(z) via a plain MLP. The vector field
    is derived from H_θ via the canonical symplectic gradient (inherited
    from HamiltonianNet.vector_field):

        dz/dt = [∂H/∂p, -∂H/∂q]

    This is the simplest and most widely-used HNN variant. It is the
    primary baseline for all conservative-system benchmarks.

    Migrated from HNN_Core_Benchmarks.ipynb, Cell 3 (class HNN).

    Parameters
    ----------
    input_dim : int
        State dimension (= 2 for pendulum, = 4 for double pendulum, etc.).
    hidden_dim : int
        Width of each hidden layer. Default 64 matches the notebook.
    n_layers : int
        Total linear layers. Default 3 (two hidden, one output).
    activation : str
        Nonlinearity. "tanh" (default) per Greydanus 2019.
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
            output_dim=1,
            n_layers=n_layers,
            activation=activation,
        )

    def scalar_field(self, z: Tensor) -> Tensor:
        """H_θ(z): R^{input_dim} → R (scalar Hamiltonian).

        Parameters
        ----------
        z : Tensor, shape (..., input_dim)

        Returns
        -------
        H : Tensor, shape (..., 1)
        """
        return self.net(z)
