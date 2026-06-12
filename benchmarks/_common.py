"""Shared infrastructure for hnet benchmark tracking scripts.

Provides the training helper, rollout-metric computation, CLI argument
parsing (``--epochs``, ``--smoke``), and the artifact exporter required by
every benchmark: each run writes high-resolution validation plots and
metric summary tables to ``benchmarks/.artifacts/<benchmark_task_name>/``:

    phase_portrait.png
    energy_drift_profile.png
    metrics.csv
    metrics.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering for CI and remote runs
import matplotlib.pyplot as plt
import numpy as np

import hnet
from hnet.data.derivative_dataset import DataConfig, DerivativeDataset
from hnet.evaluation.metrics import (
    energy_drift,
    max_energy_error,
    relative_l2_error,
)
from hnet.integrators import ScipyIntegrator

ARTIFACTS_ROOT = Path(__file__).resolve().parent / ".artifacts"

PLOT_COLORS = ("crimson", "darkorange", "steelblue", "seagreen", "purple", "sienna")


@dataclass
class BenchmarkRun:
    """One labeled rollout: prediction, oracle reference, and its metrics."""

    label: str
    t: np.ndarray
    Z_pred: np.ndarray
    Z_ref: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)


def parse_args(description: str) -> argparse.Namespace:
    """Standard CLI for every benchmark script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--epochs", type=int, default=3000, help="training epochs per model (default 3000)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fast validation run: 150 epochs, reduced dataset and rollout",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help='torch device: "auto", "cpu", or "cuda"'
    )
    return parser.parse_args()


def train_model(
    model,
    system,
    initial_conditions: list[np.ndarray],
    *,
    epochs: int,
    noise_std: float = 0.001,
    n_points: int = 1000,
    t_span: tuple[float, float] = (0.0, 50.0),
    use_spline: bool = True,
    device: str = "auto",
    seed: int = 42,
):
    """Train any HamiltonianModelProtocol model on system-generated data.

    Returns the fitted ``hnet.Trainer`` (model is trained in place).
    """
    resolved_device = hnet.utils.get_device(device)
    data_config = DataConfig(
        n_trajectories=len(initial_conditions),
        n_points_per_traj=n_points,
        t_span=t_span,
        noise_std=noise_std,
        use_spline=use_spline,
        seed=seed,
    )
    dataset = DerivativeDataset.from_system(
        system, initial_conditions, data_config, device=resolved_device
    )
    train_config = hnet.TrainConfig(
        epochs=epochs,
        lr=1e-3,
        device=str(resolved_device),
        seed=seed,
        log_every=max(epochs // 4, 1),
    )
    trainer = hnet.Trainer(model, hnet.losses.DerivativeMatchingLoss(), train_config)
    trainer.fit(dataset)
    return trainer


def rollout_run(
    label: str,
    model,
    system,
    z0: np.ndarray,
    t_span: tuple[float, float],
    n_steps: int = 1000,
    device: str = "auto",
) -> BenchmarkRun:
    """Roll out a trained model, compute the oracle reference, attach metrics.

    Metrics: Relative L2 Error (q), Mean Squared Error (full state),
    Maximum Energy Invariant Deviation |ΔH|/|H₀|, and Casimir Symmetry
    Conservation Error for systems that define Casimir invariants.
    """
    resolved_device = hnet.utils.get_device(device)
    integrator = ScipyIntegrator(rtol=1e-6, atol=1e-9, device=resolved_device)
    _t_ref, Z_ref = system.oracle_trajectory(z0, t_span, n_points=n_steps)
    t_pred, Z_pred = integrator.integrate(model, z0, t_span, n_steps)

    metrics: dict[str, float] = {
        "relative_l2_q": relative_l2_error(Z_pred[:, 0], Z_ref[:, 0]),
        "mse_state": float(np.mean((Z_pred - Z_ref) ** 2)),
        "max_energy_deviation": max_energy_error(Z_pred, system.hamiltonian, normalize=True),
    }
    for name, arr in system.casimir_errors(Z_pred).items():
        metrics[f"max_{name}_error"] = float(np.max(arr))

    return BenchmarkRun(label=label, t=t_pred, Z_pred=Z_pred, Z_ref=Z_ref, metrics=metrics)


def _metric_annotation(run: BenchmarkRun) -> str:
    return (
        f"{run.label}: RelL2(q)={run.metrics['relative_l2_q']:.3g}, "
        f"MSE={run.metrics['mse_state']:.3g}, "
        f"max|ΔH|/|H₀|={run.metrics['max_energy_deviation']:.3g}"
    )


def export_artifacts(
    task_name: str,
    runs: list[BenchmarkRun],
    system,
    headline_metric: str = "max_energy_deviation",
) -> Path:
    """Write all required artifacts for one benchmark task.

    Layout: ``benchmarks/.artifacts/<task_name>/{phase_portrait,
    energy_drift_profile}.png`` plus ``metrics.{csv,md}``. Charts are
    annotated with the physical invariant tracking metrics of every run.
    """
    out_dir = ARTIFACTS_ROOT / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation = "\n".join(_metric_annotation(run) for run in runs)

    # Phase portrait -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(runs[0].Z_ref[:, 0], runs[0].Z_ref[:, 1], "k-", lw=1.6, alpha=0.85, label="Oracle")
    for run, color in zip(runs, PLOT_COLORS, strict=False):
        ax.plot(run.Z_pred[:, 0], run.Z_pred[:, 1], "--", color=color, lw=1.2, label=run.label)
    ax.set_xlabel("q (rad)")
    ax.set_ylabel("p")
    ax.set_title(f"Phase Portrait — {task_name}")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, annotation, fontsize=7, family="monospace", va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.02 * len(runs), 1, 1))
    fig.savefig(out_dir / "phase_portrait.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Energy drift profile -----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for run, color in zip(runs, PLOT_COLORS, strict=False):
        drift = energy_drift(run.Z_pred, system.hamiltonian, normalize=True)
        ax.semilogy(
            run.t,
            drift + 1e-12,
            color=color,
            lw=1.2,
            label=f"{run.label} (max={run.metrics['max_energy_deviation']:.3g})",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|ΔH| / |H₀|")
    ax.set_title(f"Energy Conservation Degradation — {task_name}")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, annotation, fontsize=7, family="monospace", va="bottom")
    fig.tight_layout(rect=(0, 0.04 + 0.02 * len(runs), 1, 1))
    fig.savefig(out_dir / "energy_drift_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Metric summary tables ----------------------------------------------
    metric_keys = sorted({key for run in runs for key in run.metrics})
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", *metric_keys])
        for run in runs:
            writer.writerow(
                [run.label, *[f"{run.metrics.get(k, float('nan')):.6g}" for k in metric_keys]]
            )

    md_lines = [
        f"# Metrics — {task_name}",
        "",
        f"Headline metric: `{headline_metric}`",
        "",
        "| run | " + " | ".join(metric_keys) + " |",
        "|" + "---|" * (len(metric_keys) + 1),
    ]
    for run in runs:
        cells = " | ".join(f"{run.metrics.get(k, float('nan')):.6g}" for k in metric_keys)
        md_lines.append(f"| {run.label} | {cells} |")
    (out_dir / "metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return out_dir


def smoke_overrides(args: argparse.Namespace) -> dict:
    """Resolve effective run parameters honoring ``--smoke``."""
    if args.smoke:
        return {"epochs": 150, "n_points": 200, "n_steps": 200}
    return {"epochs": args.epochs, "n_points": 1000, "n_steps": 1000}
