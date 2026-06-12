"""Benchmark 01: HNN energy conservation on the nonlinear pendulum.

Trains a canonical HNN (Greydanus 2019) on near-separatrix pendulum data
and measures the maximum normalized Hamiltonian deviation |ΔH|/|H₀| over a
long-horizon rollout. This is the headline physical-invariant benchmark of
the conservative-systems track.

Local run:  python benchmarks/01_conservative_energy_conservation.py [--smoke]
"""

from __future__ import annotations

import numpy as np
from _common import export_artifacts, parse_args, rollout_run, smoke_overrides, train_model
from _kaggle_adapter import task

import hnet

TASK_NAME = "hnn-conservative-energy-conservation"
Z0 = np.array([2.0, 0.9])  # near-separatrix initial condition
T_EVAL = (0.0, 100.0)


@task(name=TASK_NAME)
def conservative_energy_task(
    llm=None, *, epochs: int = 3000, n_points: int = 1000, n_steps: int = 1000, device: str = "auto"
) -> float:
    """Returns max |ΔH|/|H₀| of the trained HNN (lower is better)."""
    hnet.utils.seed_everything(42)
    system = hnet.systems.NonlinearPendulum()

    hnn = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)
    train_model(hnn, system, [Z0], epochs=epochs, n_points=n_points, device=device)

    t_span = T_EVAL if n_steps >= 1000 else (0.0, 20.0)
    run = rollout_run("HNN", hnn, system, Z0, t_span, n_steps=n_steps, device=device)
    export_artifacts(TASK_NAME, [run], system, headline_metric="max_energy_deviation")
    return run.metrics["max_energy_deviation"]


if __name__ == "__main__":
    args = parse_args(__doc__)
    params = smoke_overrides(args)
    value = conservative_energy_task(device=args.device, **params)
    print(f"\n[{TASK_NAME}] max_energy_deviation = {value:.6f}")
