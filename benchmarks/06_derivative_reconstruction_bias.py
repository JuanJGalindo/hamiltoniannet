"""Benchmark 06: derivative-label reconstruction bias — CubicSpline vs FD.

Finite-difference derivative labels impose a non-symplectic bias during
HNN training (David & Méhats 2023, Prop. 1): the network implicitly fits
the residual of a forward-integration scheme instead of Hamilton's
equations. CubicSpline reconstruction (O(h⁴)) avoids the bias. This
benchmark trains identical HNNs on both label types and reports the
energy-conservation degradation ratio FD / spline (≥ 1 expected; larger
values confirm the bias).

Local run:  python benchmarks/06_derivative_reconstruction_bias.py [--smoke]
"""

from __future__ import annotations

import numpy as np
from _common import export_artifacts, parse_args, rollout_run, smoke_overrides, train_model
from _kaggle_adapter import task

import hnet

TASK_NAME = "hnn-derivative-reconstruction-bias"
Z0 = np.array([2.0, 0.9])


@task(name=TASK_NAME)
def derivative_bias_task(
    llm=None, *, epochs: int = 3000, n_points: int = 1000, n_steps: int = 1000, device: str = "auto"
) -> float:
    """Returns max-energy-deviation ratio (finite-difference / spline)."""
    system = hnet.systems.NonlinearPendulum()
    t_span = (0.0, 100.0) if n_steps >= 1000 else (0.0, 20.0)

    runs = []
    for label, use_spline in [("HNN spline labels", True), ("HNN FD labels", False)]:
        hnet.utils.seed_everything(42)
        hnn = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3)
        train_model(
            hnn,
            system,
            [Z0],
            epochs=epochs,
            n_points=n_points,
            use_spline=use_spline,
            device=device,
        )
        runs.append(rollout_run(label, hnn, system, Z0, t_span, n_steps=n_steps, device=device))

    export_artifacts(TASK_NAME, runs, system, headline_metric="max_energy_deviation")
    spline_err = runs[0].metrics["max_energy_deviation"]
    fd_err = runs[1].metrics["max_energy_deviation"]
    return fd_err / max(spline_err, 1e-12)


if __name__ == "__main__":
    args = parse_args(__doc__)
    params = smoke_overrides(args)
    value = derivative_bias_task(device=args.device, **params)
    print(f"\n[{TASK_NAME}] energy-deviation ratio FD/spline = {value:.4f}")
