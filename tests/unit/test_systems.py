"""Unit tests for physics systems."""
import numpy as np
import pytest

from hnet.systems import NonlinearPendulum, SimplePendulum, SystemConfig, get_system


class TestSystemConfig:
    def test_valid_config(self):
        cfg = SystemConfig(state_dim=2, phase_dim=2)
        assert cfg.state_dim == 2
        assert cfg.phase_dim == 2
        assert cfg.is_canonical is True

    def test_invalid_state_dim(self):
        with pytest.raises(ValueError, match="state_dim"):
            SystemConfig(state_dim=0, phase_dim=0)

    def test_invalid_phase_dim(self):
        with pytest.raises(ValueError, match="phase_dim"):
            SystemConfig(state_dim=2, phase_dim=5)

    def test_frozen(self):
        cfg = SystemConfig(state_dim=2, phase_dim=2)
        with pytest.raises(Exception):
            cfg.state_dim = 4  # type: ignore[misc]


class TestNonlinearPendulum:
    def setup_method(self):
        self.system = NonlinearPendulum()
        self.z0 = np.array([2.0, 0.9])

    def test_hamiltonian_equilibrium(self):
        # H(0, 0) = -1 (stable equilibrium)
        H = self.system.hamiltonian(np.array([0.0, 0.0]))
        assert abs(H - (-1.0)) < 1e-10

    def test_hamiltonian_symmetry(self):
        # H(q, p) = H(q, -p) (time-reversal symmetry)
        H1 = self.system.hamiltonian(np.array([1.0, 0.5]))
        H2 = self.system.hamiltonian(np.array([1.0, -0.5]))
        assert abs(H1 - H2) < 1e-10

    def test_equations_of_motion_shape(self):
        dz = self.system.equations_of_motion(0.0, self.z0)
        assert dz.shape == (2,)

    def test_energy_conservation_oracle(self):
        # Oracle trajectory should conserve H to machine precision
        t, Z = self.system.oracle_trajectory(self.z0, t_span=(0, 10), n_points=100)
        H0 = self.system.hamiltonian(Z[0])
        energies = [self.system.hamiltonian(z) for z in Z]
        max_drift = max(abs(H - H0) for H in energies)
        assert max_drift < 1e-7, f"Energy drift too large: {max_drift}"

    def test_oracle_trajectory_shape(self):
        t, Z = self.system.oracle_trajectory(self.z0, t_span=(0, 5), n_points=50)
        assert t.shape == (50,)
        assert Z.shape == (50, 2)
        assert np.allclose(Z[0], self.z0, atol=1e-6)

    def test_invalid_z0_shape(self):
        with pytest.raises(ValueError, match="z0 must have shape"):
            self.system.oracle_trajectory(np.array([1.0, 2.0, 3.0]), t_span=(0, 1))

    def test_invalid_mass(self):
        with pytest.raises(ValueError, match="mass"):
            NonlinearPendulum(mass=-1.0)

    def test_casimir_errors_empty_for_canonical(self):
        _, Z = self.system.oracle_trajectory(self.z0, t_span=(0, 5), n_points=20)
        errs = self.system.casimir_errors(Z)
        assert errs == {}

    def test_registry(self):
        system = get_system("nonlinear_pendulum")
        assert isinstance(system, NonlinearPendulum)

    def test_repr(self):
        assert "NonlinearPendulum" in repr(self.system)


class TestSimplePendulum:
    def test_harmonic_oscillator_energy_conservation(self):
        system = SimplePendulum()
        z0 = np.array([0.5, 0.3])
        t, Z = system.oracle_trajectory(z0, t_span=(0, 10), n_points=100)
        H0 = system.hamiltonian(Z[0])
        energies = [system.hamiltonian(z) for z in Z]
        max_drift = max(abs(H - H0) for H in energies)
        assert max_drift < 1e-7

    def test_eom_anti_symmetric(self):
        system = SimplePendulum()
        z = np.array([1.0, 2.0])
        dz = system.equations_of_motion(0.0, z)
        # dq/dt = p, dp/dt = -q
        assert abs(dz[0] - z[1]) < 1e-12
        assert abs(dz[1] + z[0]) < 1e-12
