# Kaggle Benchmarks — Local Development Setup

The `benchmarks/` suite integrates with the Kaggle Benchmarks platform through an
optional adapter ([benchmarks/_kaggle_adapter.py](../benchmarks/_kaggle_adapter.py)).
Every script runs locally with plain Python; the Kaggle SDK is required only for
push/run/download cycles against the remote service.

## 1. Installation

```bash
pip install -e ".[kaggle]"   # installs kaggle-benchmarks + kaggle-cli extras
```

Without this extra, `_kaggle_adapter.task` degrades to a no-op registration
decorator and the suite remains fully functional offline.

## 2. Environment initialization

```bash
kaggle benchmarks auth
kaggle benchmarks init
```

`auth` stores the API credential (from the Kaggle account settings page);
`init` scaffolds the task manifest in the working directory.

## 3. Execution pipeline lifecycle

The operational path is strictly local-first:

| Stage | Command |
|---|---|
| 1. Local execution verification | `python benchmarks/01_conservative_energy_conservation.py --smoke` |
| 2. Task push | `kaggle b t push` |
| 3. Model evaluation run | `kaggle b t run -m <model>` |
| 4. Pull metrics | `kaggle b t download` |

A task must pass stage 1 (exit code 0, artifacts written) before any push.
Use `--smoke` for fast validation (150 epochs) and the default configuration
(3000 epochs) for reportable numbers.

## 4. Task signature contract

The platform injects an `llm` model handle into every `@kbench.task` entry
point. hnet benchmarks evaluate physical invariants, not language-model
output, so the parameter is accepted and ignored:

```python
from _kaggle_adapter import task

@task(name="hnn-conservative-energy-conservation")
def conservative_energy_task(llm=None, *, epochs: int = 3000, ...) -> float:
    # returns the headline metric mapped to hnet core evaluation metrics
    ...
```

## 5. Registered tasks

| Script | Task name | Headline metric |
|---|---|---|
| `01_conservative_energy_conservation.py` | `hnn-conservative-energy-conservation` | max \|ΔH\|/\|H₀\| |
| `02_trajectory_accuracy.py` | `hnn-trajectory-accuracy` | Rel-L2(q), HNN |
| `03_harmonic_oscillator_recovery.py` | `hnn-harmonic-oscillator-recovery` | max \|ΔH\|/\|H₀\| |
| `04_noise_robustness.py` | `hnn-noise-robustness` | mean max \|ΔH\|/\|H₀\| over σ sweep |
| `05_data_efficiency.py` | `hnn-data-efficiency` | mean Rel-L2(q) over size sweep |
| `06_derivative_reconstruction_bias.py` | `hnn-derivative-reconstruction-bias` | energy-deviation ratio FD/spline |

## 6. Artifacts

Every execution exports validation plots and metric tables to
`benchmarks/.artifacts/<task_name>/`:

```
benchmarks/.artifacts/<task_name>/phase_portrait.png
benchmarks/.artifacts/<task_name>/energy_drift_profile.png
benchmarks/.artifacts/<task_name>/metrics.csv
benchmarks/.artifacts/<task_name>/metrics.md
```

Charts are rendered at 300 dpi and annotated with Relative L2 Error, Mean
Squared Error, Maximum Energy Invariant Deviation, and (where the system
defines them) Casimir Symmetry Conservation Error. The directory is
git-ignored; artifacts are regenerated on every run.
