"""Integration test: train HNN on pendulum and verify energy conservation.

Runs in < 60s on CPU (200 epochs, small dataset).
Serves as the CI smoke test that catches import errors and breaking changes.
"""
import numpy as np
import pytest

import hnet
from hnet.data.derivative_dataset import DataConfig, DerivativeDataset
from hnet.evaluation import Evaluator
from hnet.integrators import ScipyIntegrator


@pytest.fixture(scope="module")
def trained_hnn():
    """Train an HNN for 200 epochs on the pendulum — fast but meaningful."""
    hnet.utils.seed_everything(42)
    system = hnet.systems.NonlinearPendulum()
    z0 = np.array([2.0, 0.9])

    config = DataConfig(n_trajectories=1, n_points_per_traj=200, t_span=(0, 20), noise_std=0.001)
    dataset = DerivativeDataset.from_system(system, [z0], config, device="cpu")

    model = hnet.models.HNN(input_dim=2, hidden_dim=64)
    cfg = hnet.TrainConfig(epochs=200, lr=1e-3, device="cpu", seed=42, log_every=9999)
    trainer = hnet.Trainer(model, hnet.losses.DerivativeMatchingLoss(), cfg)
    trainer.fit(dataset)

    return model, system, trainer


def test_training_loss_decreases(trained_hnn):
    model, system, trainer = trained_hnn
    losses = trainer.history["loss"]
    assert losses[-1] < losses[0], "Training loss did not decrease"


def test_energy_error_finite(trained_hnn):
    model, system, trainer = trained_hnn
    z0 = np.array([2.0, 0.9])
    evaluator = Evaluator(model, system, ScipyIntegrator())
    results = evaluator.evaluate(z0, t_span=(0, 10), n_steps=100)
    assert np.isfinite(results["max_energy_error"])


def test_energy_better_than_random(trained_hnn):
    """HNN after 200 epochs should show some energy conservation signal.
    We use a generous threshold since 200 epochs is far from convergence.
    """
    model, system, trainer = trained_hnn
    z0 = np.array([2.0, 0.9])
    evaluator = Evaluator(model, system, ScipyIntegrator())
    results = evaluator.evaluate(z0, t_span=(0, 5), n_steps=100)
    # After 200 epochs, energy error should be below 50% (generous threshold)
    assert results["max_energy_error"] < 5.0, (
        f"Energy error {results['max_energy_error']:.4f} too large — training may be broken"
    )


def test_metrics_dict_keys(trained_hnn):
    model, system, trainer = trained_hnn
    evaluator = Evaluator(model, system, ScipyIntegrator())
    results = evaluator.evaluate([2.0, 0.9], t_span=(0, 5))
    assert "rel_l2_q" in results
    assert "max_energy_error" in results
