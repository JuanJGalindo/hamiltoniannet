"""Benchmark 04: HNN robustness to measurement noise.

Sweeps the Gaussian noise level injected into the training trajectories
(σ ∈ {1e-4, 1e-3, 1e-2}) and tracks the degradation of energy conservation
and trajectory accuracy. CubicSpline derivative reconstruction filters
part of the noise; this benchmark quantifies the remaining sensitivity.

Local run:  python benchmarks/04_noise_robustness.py [--smoke]
"""

from __future__ import annotations

import numpy as np
from _common import export_artifacts, parse_args, rollout_run, smoke_overrides, train_model
from _kaggle_adapter import task

import hnet

TASK_NAME = "hnn-noise-robustness"
Z0 = np.array([2.0, 0.9])
NOISE_LEVELS = (1e-4, 1e-3, 1e-2)


@task(name=TASK_NAME)
def noise_robustness_task(
    llm=None, *, epochs: int = 3000, n_points: int = 1000, n_steps: int = 1000, device: str = "auto"
) -> float:
    """Returns the mean max |ΔH|/|H₀| across noise levels (lower is better)."""
    system = hnet.systems.NonlinearPendulum()
    t_span = (0.0, 100.0) if n_steps >= 1000 else (0.0, 20.0)

    runs = []
    for noise_std in NOISE_LEVELS:
        hnet.utils.seed_everything(42)
        hnn = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)
        train_model(
            hnn, system, [Z0], epochs=epochs, noise_std=noise_std, n_points=n_points, device=device
        )
        runs.append(
            rollout_run(
                f"HNN σ={noise_std:g}", hnn, system, Z0, t_span, n_steps=n_steps, device=device
            )
        )

    export_artifacts(TASK_NAME, runs, system, headline_metric="max_energy_deviation")
    return float(np.mean([run.metrics["max_energy_deviation"] for run in runs]))


if __name__ == "__main__":
    args = parse_args(__doc__)
    params = smoke_overrides(args)
    value = noise_robustness_task(device=args.device, **params)
    print(f"\n[{TASK_NAME}] mean max_energy_deviation over noise sweep = {value:.6f}")
