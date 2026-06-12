"""Benchmark 02: trajectory accuracy — HNN vs unconstrained Baseline MLP.

Trains both models on identical pendulum data and compares the relative L2
error of the position coordinate against the oracle trajectory. The
Baseline MLP is the negative control: it matches derivatives pointwise but
accumulates secular drift without the symplectic structure.

Local run:  python benchmarks/02_trajectory_accuracy.py [--smoke]
"""

from __future__ import annotations

import numpy as np
from _common import export_artifacts, parse_args, rollout_run, smoke_overrides, train_model
from _kaggle_adapter import task

import hnet

TASK_NAME = "hnn-trajectory-accuracy"
Z0 = np.array([2.0, 0.9])


@task(name=TASK_NAME)
def trajectory_accuracy_task(
    llm=None, *, epochs: int = 3000, n_points: int = 1000, n_steps: int = 1000, device: str = "auto"
) -> float:
    """Returns the HNN relative L2 error on q (lower is better)."""
    hnet.utils.seed_everything(42)
    system = hnet.systems.NonlinearPendulum()
    t_span = (0.0, 100.0) if n_steps >= 1000 else (0.0, 20.0)

    runs = []
    for label, model in [
        ("HNN", hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)),
        ("BaselineMLP", hnet.models.BaselineMLP(input_dim=2, hidden_dim=64, n_layers=3)),
    ]:
        train_model(model, system, [Z0], epochs=epochs, n_points=n_points, device=device)
        runs.append(rollout_run(label, model, system, Z0, t_span, n_steps=n_steps, device=device))

    export_artifacts(TASK_NAME, runs, system, headline_metric="relative_l2_q")
    return runs[0].metrics["relative_l2_q"]


if __name__ == "__main__":
    args = parse_args(__doc__)
    params = smoke_overrides(args)
    value = trajectory_accuracy_task(device=args.device, **params)
    print(f"\n[{TASK_NAME}] HNN Rel-L2(q) = {value:.6f}")
