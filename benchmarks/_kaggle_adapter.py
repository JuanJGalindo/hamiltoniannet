"""Optional Kaggle Benchmarks SDK adapter.

The Kaggle Benchmarks platform decorates benchmark entry points with
``@kbench.task`` and injects an ``llm`` model handle at evaluation time.
hnet benchmarks measure physical invariants (energy conservation, Casimir
drift, trajectory accuracy), not language-model output, so every entry
point accepts the ``llm`` parameter and ignores it — it exists solely to
satisfy the platform calling convention.

When the SDK is not installed (the default local setup), ``task`` degrades
to a no-op registration decorator, so every script in this directory runs
with plain ``python benchmarks/<script>.py``. Install the SDK with
``pip install -e ".[kaggle]"`` to enable push/run/download against the
remote service (see docs/kaggle_setup.md).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import kaggle_benchmarks as kbench

    HAS_KAGGLE_BENCHMARKS = True

    def task(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register the entry point with the Kaggle Benchmarks SDK."""
        return kbench.task(name=name)

except ImportError:
    HAS_KAGGLE_BENCHMARKS = False

    def task(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """No-op fallback: tag the function with its task name and return it."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            fn.benchmark_task_name = name  # type: ignore[attr-defined]
            return fn

        return decorator
