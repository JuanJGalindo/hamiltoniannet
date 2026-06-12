"""Benchmark 05: HNN data efficiency.

Sweeps the number of training samples per trajectory
(n ∈ {100, 300, 1000}) at fixed noise and epochs, tracking how trajectory
accuracy and energy conservation scale with data volume. Hamiltonian
structure acts as an inductive bias, so the HNN is expected to degrade
gracefully relative to unconstrained regressors.

Local run:  python benchmarks/05_data_efficiency.py [--smoke]
"""

from __future__ import annotations

import numpy as np
from _common import export_artifacts, parse_args, rollout_run, smoke_overrides, train_model
from _kaggle_adapter import task

import hnet

TASK_NAME = "hnn-data-efficiency"
Z0 = np.array([2.0, 0.9])
DATASET_SIZES = (100, 300, 1000)


@task(name=TASK_NAME)
def data_efficiency_task(
    llm=None, *, epochs: int = 3000, n_points: int = 1000, n_steps: int = 1000, device: str = "auto"
) -> float:
    """Returns the mean Rel-L2(q) across dataset sizes (lower is better)."""
    system = hnet.systems.NonlinearPendulum()
    t_span = (0.0, 100.0) if n_steps >= 1000 else (0.0, 20.0)
    # In smoke mode scale the sweep down proportionally to the reduced n_points.
    sizes = DATASET_SIZES if n_points >= 1000 else tuple(max(s // 5, 40) for s in DATASET_SIZES)

    runs = []
    for size in sizes:
        hnet.utils.seed_everything(42)
        hnn = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)
        train_model(hnn, system, [Z0], epochs=epochs, n_points=size, device=device)
        runs.append(
            rollout_run(f"HNN n={size}", hnn, system, Z0, t_span, n_steps=n_steps, device=device)
        )

    export_artifacts(TASK_NAME, runs, system, headline_metric="relative_l2_q")
    return float(np.mean([run.metrics["relative_l2_q"] for run in runs]))


if __name__ == "__main__":
    args = parse_args(__doc__)
    params = smoke_overrides(args)
    value = data_efficiency_task(device=args.device, **params)
    print(f"\n[{TASK_NAME}] mean Rel-L2(q) over size sweep = {value:.6f}")
