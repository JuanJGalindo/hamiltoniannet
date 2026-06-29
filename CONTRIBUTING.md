# Contributing to hamiltoniannet

Thanks for contributing. This library favors **robustness and extensibility first**: protocol-based interfaces, a strictly layered architecture, and frozen configuration. Please keep changes aligned with these.

## Setup

```bash
pip install -e ".[dev]"
# or: pip install -r requirements-dev.txt
pre-commit install        # optional, runs ruff on commit
```

PyTorch CPU build: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Local CI (must pass before a PR)

Run the same checks CI runs (`.github/workflows/ci.yml`), in order:

```bash
ruff format --check .                       # formatting
ruff check .                                # linting
lint-imports                                # layered-architecture contract
mypy src/hnet/ --ignore-missing-imports     # types
pytest tests/ -q                            # 62 tests, < ~6 s on CPU
```

Coverage: `pytest tests/ --cov=hnet --cov-report=term-missing`.

## Architecture rules (enforced)

1. **Dependency direction** — downward only; `import-linter` checks it:
   ```
   visualization → evaluation → training → {systems | models | integrators | losses | data} → utils → _protocols
   ```
2. **Frozen dataclasses** for all config (`SystemConfig`, `DataConfig`, `TrainConfig`).
3. **Pure functions** for metrics (`evaluation/metrics.py`) and geometry — stateful logic goes in `Evaluator`.
4. **Protocol first, ABC second** — don't force a model to inherit `HamiltonianNet` if it has no `scalar_field`; satisfy `HamiltonianModelProtocol` instead.

## Conventions

- **Line length 100**, double quotes, spaces (`ruff format`). Config in `pyproject.toml`, mirrored in `.editorconfig`.
- **Physics single-letter names are allowed** (`q, p, z, H, l`) — ruff ignores `N806/N803/E741/N812` deliberately. Keep using them for phase-space coordinates.
- **NumPy-style docstrings**; full type hints (`from __future__ import annotations`, `X | None`, not `Optional`).
- **Derivative labels** use CubicSpline by default (David & Mehats 2023).

## Adding a new variant

1. **Model** — canonical: subclass `HamiltonianNet`, implement `scalar_field(z)` (vector field inherited). Non-canonical/operator: satisfy `HamiltonianModelProtocol`. File under `src/hnet/models/`.
2. **System** — subclass `PhysicsSystem`, decorate with `@register_system("name")`.
3. **Loss** — subclass `LossTerm`; compose via `WeightedLoss`. No edits to other modules.
4. **Tests** — add unit tests in `tests/unit/`; keep `tests/integration/` < 60 s.
5. **Docs** — note the variant in `docs/literature_review.md` with its reference.

Reference notebooks for ports live in [`../SpinesAI_Direct/Core/HNN_Core_Benchmarks.ipynb`](../SpinesAI_Direct/Core/HNN_Core_Benchmarks.ipynb). The phase plan is [`../SpinesAI_Direct/HNNs_WorkPlan.md`](../SpinesAI_Direct/HNNs_WorkPlan.md).

## Dependencies

`pyproject.toml` is the source of truth. If you add a dependency, update `requirements.txt` / `requirements-dev.txt` (bounds) **and** `requirements-lock.txt` (pin).

## Commits & PRs

- Keep PRs scoped to one layer/feature where possible.
- Ensure the full local CI sequence passes.
- Reference the roadmap week / research notebook the change derives from.
