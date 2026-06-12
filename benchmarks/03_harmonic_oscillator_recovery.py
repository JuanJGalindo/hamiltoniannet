"""Benchmark 03: exact Hamiltonian recovery on the harmonic oscillator.

The SimplePendulum system has the exactly quadratic Hamiltonian
H(q,p) = (q² + p²)/2. An HNN with sufficient capacity recovers it up to a
gauge constant, so both the energy deviation and the trajectory error are
expected near the noise floor. This benchmark is the sanity gate of the
conservative track: regressions here indicate broken training rather than
model-capacity limits.

Local run:  python benchmarks/03_harmonic_oscillator_recovery.py [--smoke]
"""

from __future__ import annotations

import numpy as np
from _common import export_artifacts, parse_args, rollout_run, smoke_overrides, train_model
from _kaggle_adapter import task

import hnet

TASK_NAME = "hnn-harmonic-oscillator-recovery"
Z0 = np.array([1.0, 0.0])  # unit-energy circle in phase space


@task(name=TASK_NAME)
def harmonic_recovery_task(
    llm=None, *, epochs: int = 3000, n_points: int = 1000, n_steps: int = 1000, device: str = "auto"
) -> float:
    """Returns max |ΔH|/|H₀| of the trained HNN on the oscillator."""
    hnet.utils.seed_everything(42)
    system = hnet.systems.SimplePendulum()

    hnn = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)
    train_model(
        hnn, system, [Z0], epochs=epochs, n_points=n_points, t_span=(0.0, 20.0), device=device
    )

    run = rollout_run("HNN", hnn, system, Z0, (0.0, 50.0), n_steps=n_steps, device=device)
    export_artifacts(TASK_NAME, [run], system, headline_metric="max_energy_deviation")
    return run.metrics["max_energy_deviation"]


if __name__ == "__main__":
    args = parse_args(__doc__)
    params = smoke_overrides(args)
    value = harmonic_recovery_task(device=args.device, **params)
    print(f"\n[{TASK_NAME}] max_energy_deviation = {value:.6f}")
