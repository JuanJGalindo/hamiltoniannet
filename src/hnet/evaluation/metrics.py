"""Pure evaluation metric functions for Hamiltonian systems.

All functions here are pure: no side effects, no class state, no global
variables. This makes them trivially parallelizable and testable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative L2 error: ||pred - true||_2 / ||true||_2.

    Parameters
    ----------
    pred, true : np.ndarray, shape (n_points, ...) or (n_points,)

    Returns
    -------
    float in [0, ∞)
    """
    norm_true = np.linalg.norm(true.flatten())
    if norm_true < 1e-10:
        return float(np.linalg.norm((pred - true).flatten()))
    return float(np.linalg.norm((pred - true).flatten()) / norm_true)


def max_energy_error(
    trajectory: np.ndarray,
    H_true: Callable[[np.ndarray], float],
    normalize: bool = True,
) -> float:
    """Max absolute energy drift along a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_points, state_dim)
    H_true : callable z -> float, the true Hamiltonian
    normalize : bool
        If True, return |ΔH| / |H₀| (scale-invariant).

    Returns
    -------
    float — max energy error
    """
    energies = np.array([H_true(z) for z in trajectory])
    H0 = energies[0]
    errors = np.abs(energies - H0)
    if normalize and abs(H0) > 1e-10:
        return float(np.max(errors) / abs(H0))
    return float(np.max(errors))


def energy_drift(
    trajectory: np.ndarray,
    H_true: Callable[[np.ndarray], float],
    normalize: bool = True,
) -> np.ndarray:
    """Energy drift array along a trajectory (for plotting).

    Returns
    -------
    np.ndarray, shape (n_points,) — |H(t) - H₀| [/ |H₀|]
    """
    energies = np.array([H_true(z) for z in trajectory])
    H0 = energies[0]
    errors = np.abs(energies - H0)
    if normalize and abs(H0) > 1e-10:
        return errors / abs(H0)
    return errors


def casimir_error(
    trajectory: np.ndarray,
    casimir_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Absolute Casimir invariant drift along a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_points, state_dim)
    casimir_fn : callable z -> float

    Returns
    -------
    np.ndarray, shape (n_points,)
    """
    values = np.array([casimir_fn(z) for z in trajectory])
    return np.abs(values - values[0])


def spin_norm_error(trajectory: np.ndarray) -> np.ndarray:
    """Departure from unit spin norm: | ||S(t)|| - 1 |.

    For SpinPrecession systems where S should stay on S².
    """
    norms = np.linalg.norm(trajectory, axis=-1)
    return np.abs(norms - 1.0)


def cosine_similarity_trace(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Cosine similarity between predicted and true trajectory vectors at each time step.

    Decouples magnitude error from phase/direction accuracy.
    A value near 1.0 means the trajectories are aligned regardless of norm.

    Parameters
    ----------
    pred, true : np.ndarray, shape (n_points, state_dim)

    Returns
    -------
    np.ndarray, shape (n_points,)
    """
    dot = np.sum(pred * true, axis=-1)
    norm_pred = np.linalg.norm(pred, axis=-1) + 1e-10
    norm_true = np.linalg.norm(true, axis=-1) + 1e-10
    return dot / (norm_pred * norm_true)
