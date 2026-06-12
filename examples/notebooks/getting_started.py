"""Getting started with hnet — reactive marimo notebook.

Authored as the marimo successor to examples/01_canonical_hnn.py (no legacy
.ipynb existed). marimo eliminates hidden-state and out-of-order execution
errors: every variable is defined in exactly one cell and the dependency
graph re-executes reactively.

Run interactively:   marimo edit examples/notebooks/getting_started.py
Run as an app:       marimo run  examples/notebooks/getting_started.py
Run as a script:     python examples/notebooks/getting_started.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Hamiltonian Neural Networks — Getting Started

        Trains a canonical **HNN** (Greydanus 2019) and an unconstrained
        **Baseline MLP** on the nonlinear pendulum
        $H(q, p) = p^2/2 - \cos(q)$, then compares long-time energy
        conservation and phase-portrait topology — the correctness criteria
        for Hamiltonian surrogates (output-vector matching is not).
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    import hnet
    from hnet.data.derivative_dataset import DataConfig, DerivativeDataset
    from hnet.evaluation import Evaluator
    from hnet.evaluation.metrics import energy_drift, max_energy_error, relative_l2_error
    from hnet.integrators import ScipyIntegrator

    return (
        DataConfig,
        DerivativeDataset,
        Evaluator,
        ScipyIntegrator,
        energy_drift,
        hnet,
        max_energy_error,
        np,
        plt,
        relative_l2_error,
    )


@app.cell
def _(hnet):
    SEED = 42
    hnet.utils.seed_everything(SEED)
    DEVICE = hnet.utils.get_device("auto")
    pendulum = hnet.systems.NonlinearPendulum()
    return DEVICE, SEED, pendulum


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(
        start=0.0, stop=0.01, step=0.001, value=0.001, label="Measurement noise σ"
    )
    epochs_slider = mo.ui.slider(
        start=500, stop=5000, step=500, value=2000, label="Training epochs"
    )
    train_button = mo.ui.run_button(label="Train models")
    mo.vstack([noise_slider, epochs_slider, train_button])
    return epochs_slider, noise_slider, train_button


@app.cell
def _(
    DEVICE,
    DataConfig,
    DerivativeDataset,
    SEED,
    epochs_slider,
    hnet,
    mo,
    noise_slider,
    np,
    pendulum,
    train_button,
):
    mo.stop(
        mo.app_meta().mode != "script" and not train_button.value,
        mo.md("*Click **Train models** to generate data and fit both models.*"),
    )

    z0_train = np.array([2.0, 0.9])  # near-separatrix initial condition
    train_data_config = DataConfig(
        n_trajectories=1,
        n_points_per_traj=1000,
        t_span=(0.0, 50.0),
        noise_std=float(noise_slider.value),
        use_spline=True,  # CubicSpline labels avoid non-symplectic bias
        seed=SEED,
    )
    train_dataset = DerivativeDataset.from_system(
        pendulum, [z0_train], train_data_config, device=DEVICE
    )

    def train_model(model_to_train):
        train_config = hnet.TrainConfig(
            epochs=int(epochs_slider.value),
            lr=1e-3,
            device=str(DEVICE),
            seed=SEED,
            log_every=max(int(epochs_slider.value) // 4, 1),
        )
        model_trainer = hnet.Trainer(
            model_to_train, hnet.losses.DerivativeMatchingLoss(), train_config
        )
        model_trainer.fit(train_dataset)
        return model_trainer

    baseline_model = hnet.models.BaselineMLP(input_dim=2, hidden_dim=64, n_layers=3).to(DEVICE)
    hnn_model = hnet.models.HNN(input_dim=2, hidden_dim=64, n_layers=3).to(DEVICE)

    baseline_trainer = train_model(baseline_model)
    hnn_trainer = train_model(hnn_model)
    return baseline_model, baseline_trainer, hnn_model, hnn_trainer


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Reactive diagnostics

        The sliders below drive **evaluation only** — drag them to probe the
        trained surrogates at new initial conditions and horizons without
        retraining. Near the separatrix ($H = 1$) the baseline drifts across
        the homoclinic orbit while the HNN stays on the level set of
        $H_\theta$.
        """
    )
    return


@app.cell
def _(mo):
    q0_slider = mo.ui.slider(start=-3.0, stop=3.0, step=0.1, value=2.0, label="q₀ (rad)")
    p0_slider = mo.ui.slider(start=-2.5, stop=2.5, step=0.1, value=0.9, label="p₀")
    horizon_slider = mo.ui.slider(start=10, stop=100, step=10, value=50, label="Horizon (s)")
    mo.vstack([q0_slider, p0_slider, horizon_slider])
    return horizon_slider, p0_slider, q0_slider


@app.cell
def _(
    DEVICE,
    Evaluator,
    ScipyIntegrator,
    baseline_model,
    hnn_model,
    horizon_slider,
    np,
    p0_slider,
    pendulum,
    q0_slider,
):
    z0_eval = np.array([float(q0_slider.value), float(p0_slider.value)])
    t_span_eval = (0.0, float(horizon_slider.value))
    n_eval_steps = 600

    rollout_integrator = ScipyIntegrator(rtol=1e-6, atol=1e-9, device=DEVICE)
    baseline_evaluator = Evaluator(baseline_model, pendulum, rollout_integrator, device=DEVICE)
    hnn_evaluator = Evaluator(hnn_model, pendulum, rollout_integrator, device=DEVICE)

    _t_oracle, Z_oracle = pendulum.oracle_trajectory(z0_eval, t_span_eval, n_points=n_eval_steps)
    t_baseline, Z_baseline = baseline_evaluator.rollout(z0_eval, t_span_eval, n_steps=n_eval_steps)
    t_hnn, Z_hnn = hnn_evaluator.rollout(z0_eval, t_span_eval, n_steps=n_eval_steps)
    return Z_baseline, Z_hnn, Z_oracle, t_baseline, t_hnn, z0_eval


@app.cell
def _(
    Z_baseline,
    Z_hnn,
    Z_oracle,
    baseline_trainer,
    energy_drift,
    hnn_trainer,
    pendulum,
    plt,
    t_baseline,
    t_hnn,
):
    diag_fig, diag_axes = plt.subplots(1, 3, figsize=(16, 4.5))
    diag_fig.suptitle("Canonical HNN vs Baseline — Nonlinear Pendulum", fontsize=13)

    ax_phase = diag_axes[0]
    ax_phase.plot(Z_oracle[:, 0], Z_oracle[:, 1], "k-", lw=1.5, label="Oracle", alpha=0.8)
    ax_phase.plot(Z_baseline[:, 0], Z_baseline[:, 1], "--", color="darkorange", label="Baseline")
    ax_phase.plot(Z_hnn[:, 0], Z_hnn[:, 1], "-", color="crimson", label="HNN")
    ax_phase.set_xlabel("q (rad)")
    ax_phase.set_ylabel("p")
    ax_phase.set_title("Phase Portrait")
    ax_phase.legend(fontsize=9)

    ax_energy = diag_axes[1]
    drift_baseline = energy_drift(Z_baseline, pendulum.hamiltonian, normalize=True)
    drift_hnn = energy_drift(Z_hnn, pendulum.hamiltonian, normalize=True)
    ax_energy.semilogy(t_baseline, drift_baseline + 1e-12, "--", color="darkorange")
    ax_energy.semilogy(t_hnn, drift_hnn + 1e-12, "-", color="crimson")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("|ΔH| / |H₀|")
    ax_energy.set_title("Energy Conservation (log scale)")
    ax_energy.set_ylim([1e-6, 10])

    ax_loss = diag_axes[2]
    ax_loss.semilogy(baseline_trainer.history["loss"], "--", color="darkorange", alpha=0.8)
    ax_loss.semilogy(hnn_trainer.history["loss"], "-", color="crimson", alpha=0.8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Convergence")

    diag_fig.tight_layout()
    diag_fig
    return


@app.cell
def _(
    Z_baseline,
    Z_hnn,
    Z_oracle,
    max_energy_error,
    mo,
    pendulum,
    relative_l2_error,
    z0_eval,
):
    h0_eval = pendulum.hamiltonian(z0_eval)
    metrics_table = mo.md(
        f"""
        | Model | Rel-L2 (q) | Max \\|ΔH\\| / \\|H₀\\| |
        |---|---|---|
        | Baseline MLP | {relative_l2_error(Z_baseline[:, 0], Z_oracle[:, 0]):.4f} \
| {max_energy_error(Z_baseline, pendulum.hamiltonian):.4f} |
        | HNN | {relative_l2_error(Z_hnn[:, 0], Z_oracle[:, 0]):.4f} \
| {max_energy_error(Z_hnn, pendulum.hamiltonian):.4f} |

        Initial condition H(q₀, p₀) = {h0_eval:.4f} (separatrix at H = 1.0).
        """
    )
    metrics_table
    return


if __name__ == "__main__":
    app.run()
