"""HamiltonianNet (hnet) — A modular framework for Hamiltonian Neural Networks.

Quick start
-----------
>>> import hnet
>>> system = hnet.systems.NonlinearPendulum()
>>> dataset = system.oracle_trajectory(z0=[2.0, 0.9], t_span=(0, 50))
>>> model = hnet.models.HNN(input_dim=2, hidden_dim=64)
>>> config = hnet.TrainConfig(epochs=5000)
>>> trainer = hnet.Trainer(model, hnet.losses.DerivativeMatchingLoss(), config)
"""
from hnet import data, evaluation, integrators, losses, models, systems, utils
from hnet.systems.registry import get_system, register_system
from hnet.training.config import TrainConfig
from hnet.training.trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    # Sub-packages
    "systems",
    "models",
    "losses",
    "data",
    "integrators",
    "evaluation",
    "utils",
    # Top-level shortcuts
    "TrainConfig",
    "Trainer",
    "register_system",
    "get_system",
]
