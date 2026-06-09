"""Unit tests for loss functions."""
import pytest
import torch

from hnet.losses import DerivativeMatchingLoss, LossTerm, WeightedLoss
from hnet.models import HNN, BaselineMLP


class TestDerivativeMatchingLoss:
    def setup_method(self):
        self.loss = DerivativeMatchingLoss()
        self.model = HNN(input_dim=2, hidden_dim=32)

    def test_zero_loss_for_exact_prediction(self):
        # If predicted == observed, loss should be ~0
        z = torch.randn(20, 2)
        z_dot_true = self.model.vector_field(z).detach()
        batch = {"z": z, "z_dot": z_dot_true}
        val = self.loss(self.model, batch)
        assert float(val.item()) < 1e-8

    def test_positive_loss(self):
        z = torch.randn(20, 2)
        z_dot_wrong = torch.zeros(20, 2)
        batch = {"z": z, "z_dot": z_dot_wrong}
        val = self.loss(self.model, batch)
        assert float(val.item()) > 0

    def test_scalar_output(self):
        z = torch.randn(10, 2)
        z_dot = torch.randn(10, 2)
        batch = {"z": z, "z_dot": z_dot}
        val = self.loss(self.model, batch)
        assert val.shape == torch.Size([])

    def test_baseline_mlp_compatible(self):
        model = BaselineMLP(input_dim=2)
        z = torch.randn(10, 2)
        z_dot = torch.randn(10, 2)
        batch = {"z": z, "z_dot": z_dot}
        val = self.loss(model, batch)
        assert float(val.item()) >= 0


class TestWeightedLoss:
    def test_single_term_identity(self):
        term = DerivativeMatchingLoss()
        model = HNN(input_dim=2, hidden_dim=32)
        z = torch.randn(10, 2)
        z_dot = torch.randn(10, 2)
        batch = {"z": z, "z_dot": z_dot}

        weighted = WeightedLoss([(term, 1.0)])
        total, breakdown = weighted(model, batch)
        direct = term(model, batch)
        assert abs(float(total.item()) - float(direct.item())) < 1e-6

    def test_weight_scaling(self):
        term = DerivativeMatchingLoss()
        model = HNN(input_dim=2, hidden_dim=32)
        z = torch.randn(10, 2)
        z_dot = torch.randn(10, 2)
        batch = {"z": z, "z_dot": z_dot}

        w1 = WeightedLoss([(term, 1.0)])
        w2 = WeightedLoss([(term, 2.0)])
        v1, _ = w1(model, batch)
        v2, _ = w2(model, batch)
        assert abs(float(v2.item()) / float(v1.item()) - 2.0) < 1e-5

    def test_breakdown_keys(self):
        term = DerivativeMatchingLoss()
        model = HNN(input_dim=2, hidden_dim=32)
        z = torch.randn(5, 2)
        z_dot = torch.randn(5, 2)
        batch = {"z": z, "z_dot": z_dot}

        weighted = WeightedLoss([(term, 1.0)])
        _, breakdown = weighted(model, batch)
        assert "DerivativeMatchingLoss" in breakdown

    def test_empty_terms_raises(self):
        with pytest.raises(ValueError, match="at least one term"):
            WeightedLoss([])
