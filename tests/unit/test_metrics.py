"""Unit tests for evaluation metric functions (pure functions)."""
import numpy as np
import pytest

from hnet.evaluation.metrics import (
    casimir_error,
    cosine_similarity_trace,
    energy_drift,
    max_energy_error,
    relative_l2_error,
    spin_norm_error,
)


class TestRelativeL2Error:
    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert relative_l2_error(x, x) == 0.0

    def test_known_error(self):
        pred = np.array([2.0, 0.0])
        true = np.array([1.0, 0.0])
        err = relative_l2_error(pred, true)
        assert abs(err - 1.0) < 1e-10

    def test_multidimensional(self):
        pred = np.zeros((50, 2))
        true = np.ones((50, 2))
        err = relative_l2_error(pred, true)
        assert err > 0

    def test_near_zero_denominator(self):
        pred = np.array([0.001, 0.0])
        true = np.array([0.0, 0.0])
        err = relative_l2_error(pred, true)
        assert np.isfinite(err)


class TestMaxEnergyError:
    def test_zero_energy_drift(self):
        # All points have the same H value (H = constant 5.0)
        traj = np.array([[1.0, 0.0], [2.0, 3.0], [-1.0, 0.5]])
        H = lambda z: 5.0  # truly constant H — zero drift by definition
        assert max_energy_error(traj, H) < 1e-10

    def test_nonzero_drift(self):
        traj = np.array([[1.0, 0.0], [2.0, 0.0]])
        H = lambda z: float(z[0])
        err = max_energy_error(traj, H, normalize=True)
        assert err > 0

    def test_normalized_vs_unnormalized(self):
        # H₀ = 10.0, drift = 2.0, so err_norm = 2/10 = 0.2, err_raw = 2.0
        traj = np.array([[10.0, 0.0], [12.0, 0.0]])
        H = lambda z: float(z[0])
        err_norm = max_energy_error(traj, H, normalize=True)
        err_raw  = max_energy_error(traj, H, normalize=False)
        assert err_norm < err_raw  # 0.2 < 2.0


class TestEnergyDrift:
    def test_shape(self):
        traj = np.random.randn(100, 2)
        H = lambda z: float(z[0]**2 + z[1]**2)
        drift = energy_drift(traj, H)
        assert drift.shape == (100,)

    def test_starts_near_zero(self):
        traj = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        H = lambda z: float(z[0]**2 + z[1]**2)
        drift = energy_drift(traj, H)
        assert float(drift[0]) < 1e-10


class TestCasimirError:
    def test_constant_casimir(self):
        traj = np.ones((50, 3))
        C = lambda z: float(np.dot(z, z))  # always 3.0
        err = casimir_error(traj, C)
        assert np.all(err < 1e-10)

    def test_shape(self):
        traj = np.random.randn(30, 3)
        C = lambda z: float(np.linalg.norm(z))
        err = casimir_error(traj, C)
        assert err.shape == (30,)


class TestSpinNormError:
    def test_unit_sphere(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        traj = np.stack([np.cos(theta), np.sin(theta), np.zeros(100)], axis=1)
        err = spin_norm_error(traj)
        assert np.all(err < 1e-10)

    def test_drift_detected(self):
        traj = np.array([[1.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        err = spin_norm_error(traj)
        assert float(err[1]) > 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        x = np.random.randn(50, 3)
        sim = cosine_similarity_trace(x, x)
        assert np.allclose(sim, 1.0, atol=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        sim = cosine_similarity_trace(a, b)
        assert abs(float(sim[0])) < 1e-10

    def test_shape(self):
        x = np.random.randn(20, 4)
        y = np.random.randn(20, 4)
        sim = cosine_similarity_trace(x, y)
        assert sim.shape == (20,)
