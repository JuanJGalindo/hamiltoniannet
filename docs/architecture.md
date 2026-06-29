# Architecture

> Stub — expand as the framework grows. Summarizes the design enforced in code and CI.

## Principles

1. **Protocol-first.** Public interfaces are `typing.Protocol` objects in `src/hnet/_protocols.py`. Classes satisfy them by structural subtyping — no mandatory base class. Users extend the framework without importing any hnet ABC.
2. **Strict layering.** Imports flow downward only; `import-linter` enforces it on every push.
3. **Frozen configuration.** All config objects are `@dataclass(frozen=True)` — immutable, hashable, self-documenting.
4. **Pure functions** for metrics and geometry.

## Layering

```
visualization
  → evaluation
    → training
      → systems | models | integrators | losses | data   (peer layer)
        → utils
          → _protocols
```

A module may import from layers below it, never above. The contract lives in `pyproject.toml` under `[tool.importlinter]`.

## Protocols

| Protocol | Key surface | Implemented by |
|---|---|---|
| `PhysicsSystemProtocol` | `hamiltonian(z)`, `equations_of_motion(t, z)`, `state_dim` | `PhysicsSystem` subclasses (`NonlinearPendulum`, `SimplePendulum`) |
| `HamiltonianModelProtocol` | `vector_field(z)`, `energy(z)`, `parameters()` | `HamiltonianNet` subclasses, `BaselineMLP`, future `SelfSupervisedHNN`/`SNO` |
| `IntegratorProtocol` | `integrate(model, z0, t_span, n_steps)` | `ScipyIntegrator` |
| `LossTermProtocol` | `__call__(model, batch)` | `DerivativeMatchingLoss`, `WeightedLoss` |

All are `@runtime_checkable`.

## Subpackages

| Package | Responsibility | Status |
|---|---|---|
| `systems/` | Ground-truth physics + registry | pendulum live |
| `models/` | `HamiltonianNet` ABC, `HNN`, `BaselineMLP`, MLP backbones | live |
| `losses/` | `LossTerm`, `WeightedLoss`, `DerivativeMatchingLoss` | live |
| `data/` | `DerivativeDataset` + CubicSpline reconstruction | live |
| `training/` | `Trainer`, `TrainConfig` | live |
| `integrators/` | `ScipyIntegrator` (symplectic variants planned) | partial |
| `evaluation/` | pure `metrics`, `Evaluator` | live |
| `geometry/` | Lie-Poisson / Darboux / projections | placeholder (Phase 5) |
| `visualization/` | phase portraits, energy drift, reports | placeholder (Phase 1 wk4) |
| `utils/` | autograd, device, reproducibility | live |

## Extension guide

See [CONTRIBUTING.md](../CONTRIBUTING.md) → "Adding a new variant". The short version: subclass `HamiltonianNet` (canonical) or satisfy `HamiltonianModelProtocol` (everything else), register systems with `@register_system`, and respect the layer your module belongs to.

## References

- Build log: [`../Handoff.md`](../Handoff.md).
- Full design rationale and phases: [`../../SpinesAI_Direct/HNNs_WorkPlan.md`](../../SpinesAI_Direct/HNNs_WorkPlan.md).
