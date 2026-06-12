"""Protocol-conformance tests: lock the public structural boundary.

Every concrete class shipped by hnet must structurally satisfy the
corresponding ``typing.Protocol`` in ``hnet._protocols``. These assertions
fail CI when a protocol and its implementations drift apart — e.g., when
internal code starts calling a method that the published protocol omits.

Design policy (architectural audit, Task 4):
- ``typing.Protocol`` defines every public-facing boundary; third-party
  extensions interface purely via structural subtyping.
- ABCs (``PhysicsSystem``, ``HamiltonianNet``, ``LossTerm``) exist only for
  internal code reuse and default implementations; inheritance from them is
  optional. Note: ``LossTerm.weight`` is a documented default not consumed by
  ``WeightedLoss`` (weights are passed explicitly as tuples) — retained
  unchanged for interface stability.
"""

import numpy as np

from hnet._protocols import (
    HamiltonianModelProtocol,
    IntegratorProtocol,
    LossTermProtocol,
    PhysicsSystemProtocol,
)
from hnet.integrators import ScipyIntegrator
from hnet.losses import DerivativeMatchingLoss, WeightedLoss
from hnet.models import HNN, BaselineMLP
from hnet.systems import NonlinearPendulum, SimplePendulum


class TestPhysicsSystemProtocol:
    def test_nonlinear_pendulum_conforms(self):
        assert isinstance(NonlinearPendulum(), PhysicsSystemProtocol)

    def test_simple_pendulum_conforms(self):
        assert isinstance(SimplePendulum(), PhysicsSystemProtocol)

    def test_protocol_covers_evaluator_usage(self):
        """Evaluator calls oracle_trajectory and casimir_errors — both must be
        protocol members so that protocol-conformant third-party systems work."""
        system = NonlinearPendulum()
        t, Z = system.oracle_trajectory(np.array([2.0, 0.9]), t_span=(0, 1), n_points=10)
        assert t.shape == (10,)
        assert Z.shape == (10, 2)
        assert system.casimir_errors(Z) == {}

    def test_minimal_duck_typed_system_conforms(self):
        """A third-party system needs no hnet ABC import to satisfy the protocol."""

        class ExternalSystem:
            @property
            def state_dim(self) -> int:
                return 2

            def hamiltonian(self, z, **params):
                return float(z[0] ** 2 + z[1] ** 2) / 2.0

            def equations_of_motion(self, t, z, **params):
                return np.array([z[1], -z[0]])

            def oracle_trajectory(self, z0, t_span, n_points=500, **kwargs):
                t = np.linspace(t_span[0], t_span[1], n_points)
                return t, np.tile(z0, (n_points, 1))

            def casimir_errors(self, Z):
                return {}

        assert isinstance(ExternalSystem(), PhysicsSystemProtocol)


class TestHamiltonianModelProtocol:
    def test_hnn_conforms(self):
        assert isinstance(HNN(input_dim=2, hidden_dim=16), HamiltonianModelProtocol)

    def test_baseline_mlp_conforms(self):
        assert isinstance(BaselineMLP(input_dim=2, hidden_dim=16), HamiltonianModelProtocol)


class TestIntegratorProtocol:
    def test_scipy_integrator_conforms(self):
        assert isinstance(ScipyIntegrator(), IntegratorProtocol)


class TestLossTermProtocol:
    def test_derivative_matching_loss_conforms(self):
        assert isinstance(DerivativeMatchingLoss(), LossTermProtocol)

    def test_weighted_loss_conforms(self):
        loss = WeightedLoss([(DerivativeMatchingLoss(), 1.0)])
        assert isinstance(loss, LossTermProtocol)
