# Kaggle Setup — Remote GPU Execution and Benchmarks Integration

The `benchmarks/` suite runs locally with plain Python. Remote execution on
Kaggle servers uses two distinct mechanisms:

1. **Kaggle Kernels (primary)** — executes the full benchmark suite on a
   Kaggle-provided GPU with explicit accelerator selection. Managed by
   [benchmarks/kaggle/deploy.py](../benchmarks/kaggle/deploy.py).
2. **Kaggle Benchmarks platform (optional)** — task registration through the
   `kaggle-benchmarks` SDK adapter
   ([benchmarks/_kaggle_adapter.py](../benchmarks/_kaggle_adapter.py)). This
   platform targets LLM evaluation and exposes **no GPU/accelerator control**;
   the adapter degrades to a no-op decorator when the SDK is absent.

## 1. Installation and credentials

```bash
pip install -e ".[kaggle]"   # installs kaggle>=2.0 (official CLI) + kaggle-benchmarks
```

Authenticate with an API token from the Kaggle account settings page, via
either mechanism:

- `~/.kaggle/kaggle.json` (downloaded token file), or
- `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.

## 2. GPU execution via Kaggle Kernels (primary path)

[benchmarks/kaggle/run_all_benchmarks.py](../benchmarks/kaggle/run_all_benchmarks.py)
is a self-contained kernel script: it clones the public repository, installs
`hnet`, asserts CUDA availability (the run aborts on CPU-only allocation), and
executes all six benchmark scripts with `--device cuda`. Artifacts are copied
to the kernel output directory together with an aggregated `summary.md`.

| Stage | Command |
|---|---|
| 1. Push + launch on GPU | `python benchmarks/kaggle/deploy.py push --accelerator NvidiaTeslaT4` |
| 2. Monitor run state | `python benchmarks/kaggle/deploy.py status` |
| 3. Pull artifacts | `python benchmarks/kaggle/deploy.py download` |

`push` renders `benchmarks/kaggle/kernel-metadata.json` (git-ignored; the
kernel `id` is namespaced by the authenticated username) with
`enable_gpu: true` and `enable_internet: true`, then invokes
`kaggle kernels push -p benchmarks/kaggle --accelerator <ID>`. Use
`push --dry-run` to inspect the rendered manifest and command without
contacting the service.

### Accelerator selection

| Accelerator ID | Tier |
|---|---|
| `NvidiaTeslaT4`, `NvidiaTeslaT4Highmem`, `NvidiaTeslaP100` | Free quota |
| `NvidiaL4`, `NvidiaL4X1`, `NvidiaTeslaA100`, `NvidiaH100`, `NvidiaRtxPro6000` | Paid tiers |

The default is `NvidiaTeslaT4`. The HNN workloads (3-layer MLPs with
second-order autograd) saturate no GPU class; the free tier is sufficient.

### Smoke versus production runs

Kernels accept no CLI arguments. Flip the `SMOKE` constant at the top of
`run_all_benchmarks.py` to `True` for a validation push (150 epochs per
model), confirm the artifacts arrive, then restore `SMOKE = False` for the
production run (3000 epochs per model).

### Quick start

```bash
pip install "kaggle>=2.0"
# place API token from kaggle.com/settings at ~/.kaggle/kaggle.json
python benchmarks/kaggle/deploy.py push      # launches on NvidiaTeslaT4
python benchmarks/kaggle/deploy.py status    # poll until "complete"
python benchmarks/kaggle/deploy.py download  # artifacts -> benchmarks/.artifacts/kaggle/
```

### Downloaded artifact layout

```
benchmarks/.artifacts/kaggle/
├── artifacts/
│   ├── summary.md
│   └── <task_name>/
│       ├── phase_portrait.png
│       ├── energy_drift_profile.png
│       ├── metrics.csv
│       └── metrics.md
└── <kernel log files>
```

## 3. Kaggle Benchmarks platform (optional adapter)

The `kaggle-benchmarks` SDK decorates entry points with `@kbench.task` and
injects an `llm` model-proxy handle at evaluation time. hnet benchmarks
measure physical invariants, not language-model output, so every entry point
accepts the `llm` parameter and ignores it — it exists solely to satisfy the
platform calling convention:

```python
from _kaggle_adapter import task

@task(name="hnn-conservative-energy-conservation")
def conservative_energy_task(llm=None, *, epochs: int = 3000, ...) -> float:
    # returns the headline metric mapped to hnet core evaluation metrics
    ...
```

The platform CLI lifecycle, for reference (commands operate on single
self-contained task files and provide no hardware selection):

```bash
kaggle benchmarks init                     # credentials + dev environment
kaggle b t push <task-slug> -f <file.py>   # upload task (converted to notebook)
kaggle b t run  <task-slug> -m <model>     # repeat -m for multiple models
kaggle b t download <task-slug> -o <dir>
```

GPU-bound benchmark execution uses the Kernels path in section 2.

## 4. Registered tasks

| Script | Task name | Headline metric |
|---|---|---|
| `01_conservative_energy_conservation.py` | `hnn-conservative-energy-conservation` | max \|ΔH\|/\|H₀\| |
| `02_trajectory_accuracy.py` | `hnn-trajectory-accuracy` | Rel-L2(q), HNN |
| `03_harmonic_oscillator_recovery.py` | `hnn-harmonic-oscillator-recovery` | max \|ΔH\|/\|H₀\| |
| `04_noise_robustness.py` | `hnn-noise-robustness` | mean max \|ΔH\|/\|H₀\| over σ sweep |
| `05_data_efficiency.py` | `hnn-data-efficiency` | mean Rel-L2(q) over size sweep |
| `06_derivative_reconstruction_bias.py` | `hnn-derivative-reconstruction-bias` | energy-deviation ratio FD/spline |

## 5. Local execution and artifacts

Every benchmark runs offline with plain Python; `--smoke` selects the fast
validation configuration (150 epochs) and the default configuration (3000
epochs) produces reportable numbers:

```bash
python benchmarks/01_conservative_energy_conservation.py --smoke
```

Each execution exports validation plots and metric tables to
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
