"""MLP backbones for Hamiltonian scalar fields."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """Fully-connected MLP with configurable depth and activation.

    Default activation is tanh — the standard choice for HNN scalar fields
    per Greydanus 2019 and all subsequent canonical variants.

    Parameters
    ----------
    input_dim : int
    hidden_dim : int
    output_dim : int
    n_layers : int
        Total number of linear layers (including output). Must be >= 2.
    activation : str
        "tanh" (default), "relu", "silu", "gelu", or "sin".
    """

    _ACTIVATIONS: ClassVar[dict[str, type[nn.Module] | None]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "sin": None,  # handled separately
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 3,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(self._ACTIVATIONS)}, got '{activation}'"
            )

        registered = self._ACTIVATIONS[activation]
        act_cls: type[nn.Module] = Sin if registered is None else registered

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act_cls()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SinMLP(MLP):
    """MLP with sinusoidal activations — bias toward periodic/oscillatory solutions.

    Used by SelfSupervisedHNN (Mattheakis 2022) for the trajectory network.
    Sinusoidal activations encode periodicity as an inductive bias, helping
    the network discover oscillatory trajectories faster than tanh.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        n_layers: int = 3,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            activation="sin",
        )


class Sin(nn.Module):
    """Sinusoidal activation: f(x) = sin(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)
