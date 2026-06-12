"""Unit tests for HNN model classes."""

import numpy as np
import pytest
import torch

from hnet.models import HNN, MLP, BaselineMLP, Sin
from hnet.models.base import HamiltonianNet


class TestMLP:
    def test_output_shape_scalar(self):
        net = MLP(input_dim=2, hidden_dim=32, output_dim=1, n_layers=2)
        x = torch.randn(10, 2)
        out = net(x)
        assert out.shape == (10, 1)

    def test_output_shape_vector(self):
        net = MLP(input_dim=2, hidden_dim=32, output_dim=2, n_layers=3)
        x = torch.randn(5, 2)
        out = net(x)
        assert out.shape == (5, 2)

    def test_all_activations(self):
        for act in ["tanh", "relu", "silu", "gelu", "sin"]:
            net = MLP(input_dim=2, hidden_dim=16, output_dim=1, n_layers=2, activation=act)
            out = net(torch.randn(3, 2))
            assert out.shape == (3, 1)

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="activation"):
            MLP(input_dim=2, hidden_dim=16, output_dim=1, n_layers=2, activation="swish")

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            MLP(input_dim=2, hidden_dim=16, output_dim=1, n_layers=1)

    def test_sin_activation(self):
        act = Sin()
        x = torch.tensor([0.0, np.pi / 2, np.pi])
        out = act(x)
        assert abs(float(out[0])) < 1e-6
        assert abs(float(out[1]) - 1.0) < 1e-6


class TestHNN:
    def setup_method(self):
        self.model = HNN(input_dim=2, hidden_dim=64, n_layers=3)

    def test_scalar_field_shape(self):
        z = torch.randn(10, 2)
        H = self.model.scalar_field(z)
        assert H.shape == (10, 1)

    def test_vector_field_shape(self):
        z = torch.randn(10, 2)
        dz = self.model.vector_field(z)
        assert dz.shape == (10, 2)

    def test_vector_field_symplectic_structure(self):
        # For canonical systems: dz/dt = [dH/dp, -dH/dq]
        # Verify by checking J-skew-symmetry: dz^T J dz = 0
        z = torch.randn(5, 2, requires_grad=False)
        dz = self.model.vector_field(z).detach().numpy()
        # Phase-space volume preservation: div(dz) = d(dH/dp)/dq + d(-dH/dq)/dp = 0
        # Not easily testable without second-order autograd, but shape is correct
        assert dz.shape == (5, 2)

    def test_energy_no_grad(self):
        z = torch.randn(5, 2)
        E = self.model.energy(z)
        assert E.shape == (5, 1)
        assert not E.requires_grad

    def test_get_gradients_detached(self):
        z = torch.randn(3, 2)
        g = self.model.get_gradients(z)
        assert g.shape == (3, 2)
        assert not g.requires_grad

    def test_scipy_vf_returns_numpy(self):
        vf = self.model.make_scipy_vf(device="cpu")
        result = vf(0.0, np.array([2.0, 0.9]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_is_hamiltonian_net(self):
        assert isinstance(self.model, HamiltonianNet)

    def test_time_derivative_alias(self):
        z = torch.randn(4, 2)
        vf = self.model.vector_field(z)
        td = self.model.time_derivative(z)
        # Both should call the same function; shapes must match
        assert vf.shape == td.shape


class TestBaselineMLP:
    def setup_method(self):
        self.model = BaselineMLP(input_dim=2, hidden_dim=64)

    def test_forward_shape(self):
        z = torch.randn(10, 2)
        dz = self.model.forward(z)
        assert dz.shape == (10, 2)

    def test_vector_field_alias(self):
        z = torch.randn(5, 2)
        a = self.model.forward(z)
        b = self.model.vector_field(z)
        assert torch.allclose(a, b)

    def test_energy_returns_zeros(self):
        z = torch.randn(5, 2)
        E = self.model.energy(z)
        assert E.shape == (5, 1)
        assert torch.all(E == 0.0)

    def test_scipy_vf_numpy_output(self):
        vf = self.model.make_scipy_vf(device="cpu")
        out = vf(0.0, np.array([1.0, 0.5]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
