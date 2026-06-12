"""Pure autograd utilities for computing symplectic and Poisson gradients."""

from __future__ import annotations

import torch
from torch import Tensor


def symplectic_gradient(
    H: Tensor,
    z: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """Compute the canonical symplectic gradient [dH/dp, -dH/dq].

    Given H(q,p) and z = [q, p] (with q,p each of length n),
    returns the vector field dz/dt = J^{-1} ∇H where J is the
    standard symplectic matrix.

    Parameters
    ----------
    H : Tensor, shape (batch,) or scalar
        Hamiltonian values (must be a graph node with requires_grad).
    z : Tensor, shape (..., 2n)
        State vector with grad enabled.
    create_graph : bool
        Keep computation graph for higher-order gradients (needed during training).

    Returns
    -------
    dz_dt : Tensor, shape (..., 2n)
        Symplectic gradient: [dH/dp, -dH/dq].
    """
    dH = torch.autograd.grad(
        H.sum(),
        z,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    n = z.shape[-1] // 2
    return torch.cat([dH[..., n:], -dH[..., :n]], dim=-1)


def poisson_gradient(
    H: Tensor,
    z: Tensor,
    poisson_tensor_fn,
    create_graph: bool = True,
) -> Tensor:
    """Compute the Poisson bracket gradient: B(z) ∇H.

    Used for non-canonical systems (e.g., Lie-Poisson, rigid body, spin).

    Parameters
    ----------
    H : Tensor
        Hamiltonian values.
    z : Tensor, shape (..., state_dim)
        State vector with grad enabled.
    poisson_tensor_fn : callable
        B(z) -> Tensor, shape (..., state_dim, state_dim).
        The skew-symmetric Poisson tensor.
    create_graph : bool

    Returns
    -------
    dz_dt : Tensor, shape (..., state_dim)
    """
    dH = torch.autograd.grad(
        H.sum(),
        z,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    B = poisson_tensor_fn(z)
    return torch.einsum("...ij,...j->...i", B, dH)


def scalar_gradient(
    scalar: Tensor,
    inputs: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """General-purpose gradient of a scalar w.r.t. inputs."""
    return torch.autograd.grad(
        scalar.sum(),
        inputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
