# HamiltonianNet (`hnet`)

A modular framework for **Hamiltonian Neural Networks** — physics-informed models that learn a scalar Hamiltonian `H_θ` and derive dynamics from its symplectic gradient, so energy is conserved *by construction* rather than by penalty.

> **Status:** alpha (v0.1.0). Phase 1 (canonical / conservative systems) complete — `HNN` and `BaselineMLP` on the nonlinear pendulum, with a full training/evaluation/benchmark stack. Variants SHNN, Self-Supervised HNN, pHNN, sPHNN, PoissonHNN, HGN, SNO are on the [roadmap](../SpinesAI_Direct/HNNs_WorkPlan.md).

This library productizes the research in **[SpinesAI_Direct](../SpinesAI_Direct)** (notebooks + literature review).

## Install

```bash
# From source (this repo) — recommended while in development:
pip install -e ".[dev]"

# Target once published:
pip install hamiltoniannet
```

Core dependencies: `torch>=2.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.6`. `pyproject.toml` is the source of truth; `requirements*.txt` are convenience mirrors (see [below](#dependency-files)). Optional extras: `[kaggle]`, `[hgn]`, `[sno]`, `[docs]`, `[dev]`.

## Quickstart

```python
import numpy as np
import hnet
from hnet.data.derivative_dataset import DataConfig, DerivativeDataset
from hnet.evaluation import Evaluator
from hnet.integrators import ScipyIntegrator

hnet.utils.seed_everything(42)
system = hnet.systems.NonlinearPendulum()        # H(q,p) = p²/2 − cos(q)
z0 = np.array([2.0, 0.9])                         # near-separatrix initial condition

# CubicSpline-reconstructed derivative labels (avoids non-symplectic bias)
dataset = DerivativeDataset.from_system(
    system, [z0], DataConfig(n_trajectories=1, n_points_per_traj=1000, t_span=(0.0, 50.0))
)

model = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)
trainer = hnet.Trainer(model, hnet.losses.DerivativeMatchingLoss(), hnet.TrainConfig(epochs=5000))
trainer.fit(dataset)

evaluator = Evaluator(model, system, ScipyIntegrator())
print(evaluator.evaluate(z0, t_span=(0.0, 100.0)))   # {'rel_l2_q': ..., 'max_energy_error': ...}
```

A complete, plotted version is in [`examples/01_canonical_hnn.py`](examples/01_canonical_hnn.py).

## Design in two sentences

Every public interface is a `typing.Protocol` (`PhysicsSystemProtocol`, `HamiltonianModelProtocol`, `IntegratorProtocol`, `LossTermProtocol`), so you extend the framework by *satisfying* a protocol — no mandatory base class. Modules form a strict dependency layering enforced in CI by `import-linter`:

```
visualization → evaluation → training → {systems | models | integrators | losses | data} → utils → _protocols
```

See [`docs/architecture.md`](docs/architecture.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for the rules, and [CLAUDE.md](CLAUDE.md) for agent/contributor context.

## Dependency files

| File | Purpose |
|---|---|
| `pyproject.toml` | **Source of truth** (PEP 621) — core deps + extras. |
| `requirements.txt` | Flexible core mirror (`>=` bounds). |
| `requirements-dev.txt` | Flexible dev mirror (`-r requirements.txt` + tooling). |
| `requirements-lock.txt` | Pinned snapshot of the validated environment (reproducible). |

## References

- Greydanus, Dzamba & Yosinski (2019), *Hamiltonian Neural Networks*.
- David & Mehats (2023), *Symplectic learning for HNNs* — derivative-label bias.
- Mattheakis et al. (2022), *Hamiltonian neural networks for solving equations of motion*.

Full corpus (16 papers): [`../SpinesAI_Direct/bib/`](../SpinesAI_Direct/bib).

## License

MIT © 2026 Juan José Galindo Márquez.
