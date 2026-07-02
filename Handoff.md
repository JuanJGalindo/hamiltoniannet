# HamiltonianNet (hnet) — Implementation Handoff

**Date:** 2026-06-09  
**Framework version:** 0.1.0  
**Phase completed:** Week 1 of the 6-month roadmap (Phase 1 — Canonical/Conservative Systems, Core Infrastructure)

---

## What Was Built

This document describes every file created during the first implementation session. The goal was to lay the robust, extensible foundation that all future HNN variants (Phases 2–6) will build on.

---

## Project Location

```
C:\Users\juanj\Desktop\WorkSpace\PhysicsSurrogates\hamiltoniannet\
```

Install the package in editable mode from that directory:

```bash
pip install -e .
```

Import as:

```python
import hnet
```

---

## Package Structure (current state)

```
hamiltoniannet/
├── pyproject.toml                    ← build config, dependencies, ruff/mypy/pytest settings
├── README.md
├── Handoff.md                        ← this file
│
├── src/hnet/
│   ├── __init__.py                   ← public API surface (re-exports all user-facing symbols)
│   ├── _protocols.py                 ← typing.Protocol definitions (structural contracts)
│   │
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── base.py                   ← PhysicsSystem ABC + SystemConfig dataclass
│   │   ├── pendulum.py               ← NonlinearPendulum + SimplePendulum
│   │   └── registry.py              ← @register_system decorator + get_system()
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   ← HamiltonianNet ABC
│   │   ├── hnn.py                    ← HNN (Greydanus 2019)
│   │   ├── baseline.py               ← BaselineMLP (benchmark foil)
│   │   └── backbones/
│   │       ├── __init__.py
│   │       └── mlp.py                ← MLP, SinMLP, Sin (custom sinusoidal activation)
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── base.py                   ← LossTerm ABC + WeightedLoss compositor
│   │   └── derivative.py             ← DerivativeMatchingLoss
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── derivative_dataset.py    ← DerivativeDataset + DataConfig + CubicSpline reconstruction
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py                 ← TrainConfig (frozen dataclass)
│   │   └── trainer.py               ← Trainer (unified training loop)
│   │
│   ├── integrators/
│   │   ├── __init__.py
│   │   └── scipy_wrapper.py         ← ScipyIntegrator (wraps solve_ivp)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               ← 6 pure metric functions
│   │   └── evaluator.py             ← Evaluator (rollout + metrics)
│   │
│   ├── geometry/
│   │   └── __init__.py              ← placeholder (Week 16: non-canonical geometry)
│   │
│   └── visualization/
│       └── __init__.py              ← placeholder (Week 4: plotting utilities)
│
├── tests/
│   ├── unit/
│   │   ├── test_systems.py          ← 16 tests for PhysicsSystem, pendulum, registry
│   │   ├── test_models.py           ← 18 tests for HNN, BaselineMLP, MLP, Sin
│   │   ├── test_losses.py           ← 8 tests for DerivativeMatchingLoss, WeightedLoss
│   │   └── test_metrics.py          ← 16 tests for all 6 metric functions
│   └── integration/
│       └── test_pendulum_hnn.py     ← 4 tests: full train + evaluate pipeline (< 5s on CPU)
│
├── examples/
│   └── 01_canonical_hnn.py          ← full HNN vs Baseline training + energy plot
│
├── benchmarks/                       ← Kaggle local benchmark tasks (empty, scaffolded)
│
└── .github/
    └── workflows/
        └── ci.yml                   ← GitHub Actions (pytest + ruff + mypy, Python 3.10-3.12)
```

---

## File-by-File Descriptions

### `src/hnet/_protocols.py`

Defines all public interfaces as `typing.Protocol` objects. This is the single most important architectural decision: no class needs to inherit from any hnet base class — it only needs to structurally satisfy the protocol.

| Protocol | Key methods |
|---|---|
| `PhysicsSystemProtocol` | `hamiltonian(z)`, `equations_of_motion(t, z)`, `state_dim` |
| `HamiltonianModelProtocol` | `vector_field(z)`, `energy(z)`, `parameters()` |
| `IntegratorProtocol` | `integrate(model, z0, t_span, n_steps)` |
| `LossTermProtocol` | `__call__(model, batch)` |

All are decorated with `@runtime_checkable` so `isinstance()` checks work at runtime.

---

### `src/hnet/systems/base.py`

**`SystemConfig`** — frozen dataclass holding system metadata:
- `state_dim`, `phase_dim`, `is_canonical`, `manifold`, `casimir_functions`, `param_names`
- Frozen (`@dataclass(frozen=True)`) — cannot be mutated after construction

**`PhysicsSystem`** — abstract base class. Subclasses must implement:
- `config` property → `SystemConfig`
- `hamiltonian(z, **params)` → float
- `equations_of_motion(t, z, **params)` → np.ndarray

Provides for free:
- `oracle_trajectory(z0, t_span, ...)` — integrates via scipy RK45 at rtol=1e-9
- `casimir_errors(Z)` — evaluates all Casimir invariants along a trajectory

---

### `src/hnet/systems/pendulum.py`

**`NonlinearPendulum`** — the primary test system for all benchmarks.

- Hamiltonian: `H(q,p) = p²/2 - cos(q)`
- Equations of motion: `dq/dt = p`, `dp/dt = -sin(q)` (non-dimensionalized)
- The near-separatrix IC `[q₀=2.0, p₀=0.9]` from `HNN_Core_Benchmarks.ipynb` is used throughout
- Registered as `"nonlinear_pendulum"` in the system registry

**`SimplePendulum`** — small-angle (harmonic) approximation: `H(q,p) = (p²+q²)/2`. Useful sanity check where the Hamiltonian is exactly quadratic.

---

### `src/hnet/systems/registry.py`

```python
@register_system("pendulum")
class NonlinearPendulum(PhysicsSystem): ...

# User extension — zero changes to library code:
@hnet.register_system("my_system")
class MySystem(hnet.PhysicsSystem): ...

system = hnet.get_system("my_system")
```

`SYSTEM_REGISTRY` is a plain dict. The `@register_system` decorator adds to it. `get_system(name)` raises a descriptive `KeyError` listing all available names if the key is not found.

---

### `src/hnet/models/base.py`

**`HamiltonianNet`** — abstract base for all canonical HNN variants. Subclasses must implement only `scalar_field(z)`.

Provides for free:
- `vector_field(z)` — canonical symplectic gradient `[dH/dp, -dH/dq]` via `torch.autograd.grad` with `create_graph=True`
- `time_derivative(z)` — alias for `vector_field`, used by Trainer and Integrator
- `get_gradients(z)` — returns `∇_z H_θ(z)` as a detached tensor (for custom integrators)
- `energy(z)` — evaluates `scalar_field` inside `torch.no_grad()` (for monitoring)
- `make_scipy_vf(device)` — wraps the model as a `scipy.integrate.solve_ivp`-compatible function

**Design:** `SelfSupervisedHNN` and `SNO` do NOT subclass `HamiltonianNet`. They satisfy `HamiltonianModelProtocol` directly. Forcing them to implement `scalar_field()` would be a structural lie.

---

### `src/hnet/models/hnn.py`

**`HNN`** — direct port of the `HNN` class from `HNN_Core_Benchmarks.ipynb`, Cell 3.

```python
model = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3, activation="tanh")
```

`scalar_field(z)` delegates to an internal `MLP`. `vector_field(z)` is inherited from `HamiltonianNet` and computes the canonical symplectic gradient automatically via autograd.

---

### `src/hnet/models/baseline.py`

**`BaselineMLP`** — unconstrained direct regressor: `(q,p) → (dq/dt, dp/dt)`. No Hamiltonian structure. Always included in benchmarks as the negative control demonstrating secular energy drift.

---

### `src/hnet/models/backbones/mlp.py`

**`MLP`** — configurable fully-connected network. Supports activations: `tanh` (default), `relu`, `silu`, `gelu`, `sin`.

**`SinMLP`** — MLP with sinusoidal activations. Used by `SelfSupervisedHNN` (Mattheakis 2022) for oscillatory trajectory learning.

**`Sin`** — `nn.Module` wrapping `torch.sin(x)`. Used as a drop-in activation layer.

---

### `src/hnet/losses/base.py`

**`LossTerm`** — abstract callable. Subclass to add a new loss without changing any other file.

**`WeightedLoss`** — compositor: `L = Σ w_i · L_i(model, batch)`. Returns both the total loss and a per-term breakdown dict for logging.

```python
loss = WeightedLoss([
    (DerivativeMatchingLoss(), 1.0),
    (EnergyRegularizationLoss(H_true), 0.1),
])
total, breakdown = loss(model, batch)
# breakdown = {"DerivativeMatchingLoss": 0.03, "EnergyRegularizationLoss": 0.001}
```

---

### `src/hnet/losses/derivative.py`

**`DerivativeMatchingLoss`** — `MSE(vector_field(z), z_dot_obs)`.

The standard training loss for HNN, pHNN, AdaptableHNN, PoissonHNN, and BaselineMLP. Expects batch keys `"z"` and `"z_dot"`.

---

### `src/hnet/data/derivative_dataset.py`

**`DataConfig`** — frozen dataclass: `n_trajectories`, `n_points_per_traj`, `t_span`, `noise_std`, `use_spline`, `seed`.

**`DerivativeDataset`** — PyTorch `Dataset` of `(z, z_dot)` pairs. Built via `DerivativeDataset.from_system(system, initial_conditions, config)`.

**Why CubicSpline?** Finite-difference derivative labels impose a non-symplectic bias: the trained HNN implicitly learns to minimize a forward-Euler residual instead of Hamilton's equations (David & Mehats 2023, Proposition 1). `CubicSpline.derivative()` gives O(h⁴) accuracy and eliminates this bias entirely. This is why `use_spline=True` is the default.

---

### `src/hnet/training/config.py`

**`TrainConfig`** — frozen dataclass:

| Field | Default | Notes |
|---|---|---|
| `epochs` | 5000 | |
| `lr` | 1e-3 | |
| `optimizer` | `"adam"` | also `"adamw"`, `"lbfgs"` |
| `scheduler` | `None` | also `"cosine"`, `"reduce_on_plateau"` |
| `batch_size` | `None` | None = full-batch |
| `device` | `"auto"` | auto-selects CUDA/MPS/CPU |
| `seed` | 42 | applied before training |
| `log_every` | 500 | print every N epochs |
| `checkpoint_dir` | `None` | saves `best.pt` and `final.pt` |

---

### `src/hnet/training/trainer.py`

**`Trainer`** — unified training loop, model-agnostic. Works with any model satisfying `HamiltonianModelProtocol` and any loss satisfying `LossTermProtocol`.

```python
trainer = hnet.Trainer(model, loss, config)
trainer.fit(dataset)
print(trainer.history["loss"])  # per-epoch loss values
trainer.save("weights.pt")
```

- Full-batch by default (calls `dataset.as_batch()` when available)
- Calls `seed_everything(config.seed)` before training
- Moves model to the resolved device automatically
- Saves checkpoints to `checkpoint_dir` if specified

---

### `src/hnet/integrators/scipy_wrapper.py`

**`ScipyIntegrator`** — wraps any `HamiltonianModelProtocol` model as a `solve_ivp` vector field.

Default tolerances: `rtol=1e-6`, `atol=1e-9`. Tighter than the oracle (rtol=1e-9) but looser than reference integration — chosen to reveal true long-time energy behavior rather than integrator error.

```python
integrator = ScipyIntegrator(method="RK45", rtol=1e-6, atol=1e-9)
t, Z = integrator.integrate(model, z0=[2.0, 0.9], t_span=(0, 100), n_steps=1200)
```

---

### `src/hnet/evaluation/metrics.py`

All pure functions — no side effects, no class state:

| Function | Returns | Use |
|---|---|---|
| `relative_l2_error(pred, true)` | float | trajectory accuracy |
| `max_energy_error(traj, H_true, normalize)` | float | energy conservation |
| `energy_drift(traj, H_true, normalize)` | np.ndarray | plotting |
| `casimir_error(traj, casimir_fn)` | np.ndarray | non-canonical invariants |
| `spin_norm_error(traj)` | np.ndarray | S² manifold constraint |
| `cosine_similarity_trace(pred, true)` | np.ndarray | phase alignment |

---

### `src/hnet/evaluation/evaluator.py`

**`Evaluator`** — post-training evaluation. Decoupled from Trainer; works on loaded checkpoints.

```python
evaluator = hnet.evaluation.Evaluator(model, system, ScipyIntegrator())
results = evaluator.evaluate(z0=[2.0, 0.9], t_span=(0, 100))
# results = {"rel_l2_q": 0.003, "max_energy_error": 0.0018}

t, Z_pred = evaluator.rollout(z0=[2.0, 0.9], t_span=(0, 100))
t, drift   = evaluator.energy_drift_array(z0=[2.0, 0.9], t_span=(0, 100))
```

---

### `utils/` subpackage

| Module | Key exports |
|---|---|
| `autograd.py` | `symplectic_gradient(H, z)`, `poisson_gradient(H, z, B_fn)`, `scalar_gradient(f, x)` |
| `device.py` | `get_device("auto")`, `to_device(tensor, device)` |
| `reproducibility.py` | `seed_everything(seed)` — covers Python, NumPy, PyTorch CPU+CUDA |

---

## Test Suite

### Running tests

```bash
cd C:\Users\juanj\Desktop\WorkSpace\PhysicsSurrogates\hamiltoniannet

# All tests
python -m pytest tests/ -v

# Unit tests only (fast, ~2s)
python -m pytest tests/unit/ -v

# Integration tests only (~3.5s)
python -m pytest tests/integration/ -v

# Save results to a file
python -m pytest tests/ -v > test_results.txt 2>&1

# With coverage report
pip install pytest-cov
python -m pytest tests/ --cov=hnet --cov-report=term-missing
```

### Test results (session on 2026-06-09)

```
Platform: win32, Python 3.13.11, pytest 9.0.3

tests/unit/test_losses.py        8 passed
tests/unit/test_metrics.py      16 passed
tests/unit/test_models.py       18 passed
tests/unit/test_systems.py      16 passed
tests/integration/test_pendulum_hnn.py  4 passed

Total: 62 passed in ~5.5s (CPU)
```

Test results are **not saved to a file automatically** — they are printed to the terminal. To persist them, use:

```bash
python -m pytest tests/ -v > test_results.txt 2>&1
```

### What each test file covers

**`tests/unit/test_systems.py`** (16 tests)
- `SystemConfig` validation (invalid dims, frozen enforcement)
- `NonlinearPendulum`: Hamiltonian at equilibrium, time-reversal symmetry, EOM shape, energy conservation on oracle trajectory, IC shape validation, invalid constructor args
- `SimplePendulum`: harmonic EOM correctness
- Registry: `get_system("nonlinear_pendulum")` returns correct type

**`tests/unit/test_models.py`** (18 tests)
- `MLP`: output shapes, all 5 activations, invalid activation/n_layers error handling
- `Sin`: sinusoidal output at known values (0, π/2, π)
- `HNN`: `scalar_field` shape, `vector_field` shape, `energy` no-grad, `get_gradients` detached, `make_scipy_vf` numpy output, `isinstance(HamiltonianNet)`, `time_derivative` alias
- `BaselineMLP`: forward shape, `vector_field` alias, energy returns zeros, scipy vf

**`tests/unit/test_losses.py`** (8 tests)
- `DerivativeMatchingLoss`: zero loss for exact prediction, positive for wrong prediction, scalar output, compatible with BaselineMLP
- `WeightedLoss`: single-term identity, weight scaling (2× weight → 2× loss), breakdown keys, empty terms raises ValueError

**`tests/unit/test_metrics.py`** (16 tests)
- `relative_l2_error`: zero error, known error, multidimensional, near-zero denominator
- `max_energy_error`: truly constant H, nonzero drift, normalized < unnormalized when |H₀| > 1
- `energy_drift`: shape, starts at zero
- `casimir_error`: constant invariant, shape
- `spin_norm_error`: unit sphere, drift detected
- `cosine_similarity_trace`: identical vectors = 1.0, orthogonal = 0.0, shape

**`tests/integration/test_pendulum_hnn.py`** (4 tests)
- Trains HNN for 200 epochs on the pendulum (fast version of the full 5000-epoch experiment)
- Verifies: loss decreases, energy error is finite, energy better than a random threshold, result dict has expected keys

---

## Architecture Rules (do not break these)

### 1. Dependency direction

Imports must only flow downward:

```
_protocols  →  (nothing)
utils       →  _protocols
systems     →  utils, _protocols
models      →  utils, _protocols
data        →  systems, utils
losses      →  models, utils
integrators →  models, utils
training    →  losses, data, models, utils
evaluation  →  systems, integrators, models, utils
visualization → evaluation
```

Never import `training` or `evaluation` from inside `models` or `systems`. CI enforces this with `import-linter`.

### 2. Frozen dataclasses for all config

`SystemConfig`, `DataConfig`, `TrainConfig` are all `@dataclass(frozen=True)`. Do not replace them with plain dicts.

### 3. Pure functions for metrics and geometry

All functions in `evaluation/metrics.py` and `geometry/` must remain pure (no side effects, no class state). If you need stateful evaluation, put it in `Evaluator`, not in `metrics.py`.

### 4. Protocol first, ABC second

If adding a new variant that does not fit `HamiltonianNet` (e.g., it has no `scalar_field`), do NOT force it to inherit. Make it satisfy `HamiltonianModelProtocol` directly. `SelfSupervisedHNN` and `SNO` follow this pattern.

---

## What Comes Next (Weeks 5–6 per the Roadmap)

Following the plan at [`../SpinesAI_Direct/HNNs_WorkPlan.md`](../SpinesAI_Direct/HNNs_WorkPlan.md) (6-month roadmap):

**Week 5 (SHNN):**
- `src/hnet/losses/symplectic.py` — `SymplecticSchemeLoss` (symplectic Euler + implicit midpoint)
- `src/hnet/models/shnn.py` — `SHNN` class (same architecture as HNN, different loss)
- `src/hnet/data/state_pair_dataset.py` — `StatePairDataset` for (y₀, y₁) pairs
- `src/hnet/integrators/symplectic.py` — `SymplecticEuler`, `StormerVerlet`

**Week 6 (SelfSupervisedHNN):**
- `src/hnet/models/selfsup_hnn.py` — `SelfSupervisedHNN` with `trial_solution(t, z0)`
- `src/hnet/losses/collocation.py` — `CollocationLoss` (ODE residual + energy term, λ=0.1)
- `src/hnet/training/curriculum.py` — `CausalCurriculum` callback
- `src/hnet/data/collocation_dataset.py` — stochastic time-point sampler

**Source to read before implementing:**
- `C:\Users\juanj\Desktop\WorkSpace\PhysicsSurrogates\SpinesAI_Direct\Core\HNN_Core_Benchmarks.ipynb` Cell 5 — the SelfSupervisedHNN training loop is the canonical reference
- `C:\Users\juanj\Desktop\WorkSpace\PhysicsSurrogates\SpinesAI_Direct\bib\BasicHNNs\SHNNs_DavidMehats2023.pdf` — required for correct `SymplecticSchemeLoss` scheme-point interpolation

---

## Quick-Start Verification

Run this to confirm everything is working after a fresh install:

```bash
cd C:\Users\juanj\Desktop\WorkSpace\PhysicsSurrogates\hamiltoniannet
pip install -e .
python -m pytest tests/ -v
```

Expected output: `62 passed in ~5s`.

To run the full HNN training example (5000 epochs, ~2-3 min on CPU):

```bash
python examples/01_canonical_hnn.py
```

This reproduces the Baseline vs HNN comparison from `HNN_Core_Benchmarks.ipynb` and saves `01_canonical_hnn_results.png`.
