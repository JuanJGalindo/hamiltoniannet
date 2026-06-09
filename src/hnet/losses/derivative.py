"""Derivative matching loss — the standard HNN training objective."""
from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from hnet._protocols import HamiltonianModelProtocol
from hnet.losses.base import LossTerm


class DerivativeMatchingLoss(LossTerm):
    """MSE loss between predicted and observed time derivatives.

    L = || vector_field(z) - z_dot_obs ||²

    Used by: HNN (Greydanus), SHNN (indirectly), pHNN (Desai),
             AdaptableHNN (Han), PoissonHNN, BaselineMLP.

    Batch keys
    ----------
    "z"     : Tensor, shape (N, state_dim) — state vectors
    "z_dot" : Tensor, shape (N, state_dim) — observed time derivatives
               (CubicSpline-reconstructed to avoid symplectic bias)
    """

    def __call__(
        self,
        model: HamiltonianModelProtocol,
        batch: dict[str, Tensor],
        **kwargs: Any,
    ) -> Tensor:
        z = batch["z"]
        z_dot_obs = batch["z_dot"]
        z_dot_pred = model.vector_field(z)
        return F.mse_loss(z_dot_pred, z_dot_obs)
