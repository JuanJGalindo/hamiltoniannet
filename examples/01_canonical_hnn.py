"""Example 01: Canonical HNN on the Nonlinear Pendulum.

Migrated from HNN_Demo.ipynb and HNN_Core_Benchmarks.ipynb.

Demonstrates:
- Data generation via oracle trajectory + CubicSpline derivative reconstruction
- HNN training (Greydanus 2019)
- BaselineMLP training (unconstrained, for comparison)
- Energy conservation comparison over long-time rollout
- Phase portrait and energy drift plots

Physics system: Nonlinear pendulum H(q,p) = p²/2 - cos(q)
Initial condition: q₀=2.0 rad, p₀=0.9 (near separatrix, libration regime)
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

import hnet
from hnet.data.derivative_dataset import DataConfig, DerivativeDataset
from hnet.integrators import ScipyIntegrator
from hnet.evaluation import Evaluator

# ── Configuration ─────────────────────────────────────────────────────────────

SEED = 42
hnet.utils.seed_everything(SEED)
DEVICE = hnet.utils.get_device("auto")

Z0 = np.array([2.0, 0.9])    # near-separatrix IC (from HNN_Core_Benchmarks.ipynb)
T_TRAIN = (0.0, 50.0)
T_EVAL  = (0.0, 100.0)
N_POINTS = 1000
NOISE_STD = 0.001
EPOCHS = 5000
HIDDEN_DIM = 64
N_LAYERS = 3

# ── Physics System ─────────────────────────────────────────────────────────────

system = hnet.systems.NonlinearPendulum()
print(f"System: {system}")
print(f"H(q₀, p₀) = {system.hamiltonian(Z0):.4f}  (separatrix at H=1.0)")

# ── Generate Training Data ─────────────────────────────────────────────────────

data_config = DataConfig(
    n_trajectories=1,
    n_points_per_traj=N_POINTS,
    t_span=T_TRAIN,
    noise_std=NOISE_STD,
    use_spline=True,   # CubicSpline reconstruction (avoids non-symplectic bias)
    seed=SEED,
)
dataset = DerivativeDataset.from_system(system, [Z0], data_config, device=DEVICE)
print(f"Training dataset: {len(dataset)} (z, z_dot) pairs")

# ── Train Baseline MLP ────────────────────────────────────────────────────────

print("\n--- Training Baseline MLP ---")
baseline = hnet.models.BaselineMLP(input_dim=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
baseline_cfg = hnet.TrainConfig(
    epochs=EPOCHS, lr=1e-3, device=str(DEVICE), seed=SEED, log_every=1000
)
baseline_trainer = hnet.Trainer(baseline, hnet.losses.DerivativeMatchingLoss(), baseline_cfg)
baseline_trainer.fit(dataset)

# ── Train HNN ─────────────────────────────────────────────────────────────────

print("\n--- Training HNN (Greydanus 2019) ---")
hnn = hnet.models.HNN(input_dim=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
hnn_cfg = hnet.TrainConfig(
    epochs=EPOCHS, lr=1e-3, device=str(DEVICE), seed=SEED, log_every=1000
)
hnn_trainer = hnet.Trainer(hnn, hnet.losses.DerivativeMatchingLoss(), hnn_cfg)
hnn_trainer.fit(dataset)

# ── Evaluate ──────────────────────────────────────────────────────────────────

integrator = ScipyIntegrator(rtol=1e-6, atol=1e-9, device=DEVICE)

baseline_eval = Evaluator(baseline, system, integrator, device=DEVICE)
hnn_eval = Evaluator(hnn, system, integrator, device=DEVICE)

print("\n--- Evaluation over 100s rollout ---")
baseline_metrics = baseline_eval.evaluate(Z0, T_EVAL, n_steps=1200)
hnn_metrics = hnn_eval.evaluate(Z0, T_EVAL, n_steps=1200)

print(f"{'Model':<12} {'Rel-L2 q':<14} {'Max ΔH/|H₀|':<14}")
print("-" * 42)
print(f"{'Baseline':<12} {baseline_metrics['rel_l2_q']:<14.4f} {baseline_metrics['max_energy_error']:<14.4f}")
print(f"{'HNN':<12} {hnn_metrics['rel_l2_q']:<14.4f} {hnn_metrics['max_energy_error']:<14.4f}")

# ── Reference Trajectory ──────────────────────────────────────────────────────

t_ref, Z_ref = system.oracle_trajectory(Z0, T_EVAL, n_points=1200, rtol=1e-9, atol=1e-9)

# Predict trajectories
t_base, Z_base = baseline_eval.rollout(Z0, T_EVAL, n_steps=1200)
t_hnn, Z_hnn   = hnn_eval.rollout(Z0, T_EVAL, n_steps=1200)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Canonical HNN vs Baseline — Nonlinear Pendulum (near separatrix)", fontsize=13)

# Phase portrait
ax = axes[0]
ax.plot(Z_ref[:, 0], Z_ref[:, 1], "k-",  lw=1.5, label="Oracle", alpha=0.8)
ax.plot(Z_base[:, 0], Z_base[:, 1], "--", color="darkorange", lw=1.2, label="Baseline MLP")
ax.plot(Z_hnn[:, 0],  Z_hnn[:, 1],  "-",  color="crimson",   lw=1.2, label="HNN")
ax.set_xlabel("q (rad)")
ax.set_ylabel("p")
ax.set_title("Phase Portrait")
ax.legend(fontsize=9)

# Energy drift
ax = axes[1]
from hnet.evaluation.metrics import energy_drift
drift_base = energy_drift(Z_base, system.hamiltonian, normalize=True)
drift_hnn  = energy_drift(Z_hnn,  system.hamiltonian, normalize=True)
ax.semilogy(t_base, drift_base + 1e-12, "--", color="darkorange", label="Baseline MLP")
ax.semilogy(t_hnn,  drift_hnn  + 1e-12, "-",  color="crimson",   label="HNN")
ax.set_xlabel("Time (s)")
ax.set_ylabel("|ΔH| / |H₀|")
ax.set_title("Energy Conservation (log scale)")
ax.legend(fontsize=9)
ax.set_ylim([1e-6, 10])

# Training loss
ax = axes[2]
ax.semilogy(baseline_trainer.history["loss"], "--", color="darkorange", label="Baseline MLP", alpha=0.8)
ax.semilogy(hnn_trainer.history["loss"],      "-",  color="crimson",   label="HNN", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Training Convergence")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("01_canonical_hnn_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to 01_canonical_hnn_results.png")
plt.show()
