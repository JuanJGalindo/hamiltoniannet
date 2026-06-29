# CLAUDE.md — hamiltoniannet (`hnet`)

Context for AI assistants and contributors. Read this before changing structure.

## What this is

A modular, **protocol-first** framework for Hamiltonian Neural Networks. Package name `hamiltoniannet`, import name `hnet`, source under `src/hnet/`. It productizes the research in the sibling repo **[SpinesAI_Direct](../SpinesAI_Direct)**; the canonical reference implementations being ported live in that repo's `Core/HNN_Core_Benchmarks.ipynb`. The roadmap is [`../SpinesAI_Direct/HNNs_WorkPlan.md`](../SpinesAI_Direct/HNNs_WorkPlan.md); the Week-1 build log is [`Handoff.md`](Handoff.md).

**Current status:** v0.1.0 alpha, Phase 1 complete (`HNN`, `BaselineMLP`, training/eval/benchmark infra). 62 tests pass; CI on Python 3.10–3.12.

## Architecture rules (do not break)

1. **Dependency direction** — imports flow downward only, enforced by `import-linter`:
   ```
   visualization → evaluation → training → {systems | models | integrators | losses | data} → utils → _protocols
   ```
   Never import `training`/`evaluation` from inside `models`/`systems`.
2. **Frozen configs** — `SystemConfig`, `DataConfig`, `TrainConfig` are `@dataclass(frozen=True)`. Don't replace with dicts.
3. **Pure metrics** — everything in `evaluation/metrics.py` and `geometry/` is a pure function (no state, no side effects). Stateful evaluation goes in `Evaluator`.
4. **Protocol first, ABC second** — a variant that doesn't fit `HamiltonianNet` (no `scalar_field`) must satisfy `HamiltonianModelProtocol` directly rather than inherit. `SelfSupervisedHNN` and `SNO` follow this.

## Conventions

- **Physics notation is intentional** and is whitelisted in ruff: `N806`, `N803`, `E741`, `N812` are ignored so `q, p, z, H, l`, and `import torch.nn.functional as F` are allowed.
- **Line length 100**, double-quote formatting, spaces — via `ruff format` (config in `pyproject.toml`, mirrored in `.editorconfig`).
- **Canonical vector field** is `[∂H/∂p, −∂H/∂q]` via autograd with `create_graph=True` (`HamiltonianNet.vector_field`, override for non-canonical variants).
- **Derivative labels** default to **CubicSpline** (O(h⁴)); finite differences impose a non-symplectic bias (David & Mehats 2023, Prop. 1) — this is why `DataConfig.use_spline=True`.
- **mypy** is non-strict with `ignore_missing_imports` (PyTorch stubs are incomplete).

## Local CI (run before pushing)

```bash
ruff format --check .
ruff check .
lint-imports
mypy src/hnet/ --ignore-missing-imports
pytest tests/ -q          # expect 62 passed
```

## Adding a variant (recipe)

1. New canonical model → subclass `HamiltonianNet`, implement `scalar_field(z)`; `vector_field` is inherited. Put it in `src/hnet/models/`.
2. Non-canonical / non-Hamiltonian surrogate → satisfy `HamiltonianModelProtocol` directly.
3. New system → subclass `PhysicsSystem` and register with `@register_system("name")`.
4. New loss → subclass `LossTerm` (compose via `WeightedLoss`); no other module changes.
5. Add unit tests under `tests/unit/`; keep integration tests < 60 s on CPU.
6. Respect the layer the new module belongs to (rule 1).

## Dependency files

`pyproject.toml` is the source of truth. `requirements.txt` (core) and `requirements-dev.txt` (tooling) are flexible mirrors; `requirements-lock.txt` pins the validated environment. Update all of them when you change a dependency.

## Related

[SpinesAI_Direct](../SpinesAI_Direct) (research) · [`docs/architecture.md`](docs/architecture.md) · [`docs/literature_review.md`](docs/literature_review.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · root [`WORKSPACE_OVERVIEW.md`](../WORKSPACE_OVERVIEW.md).
