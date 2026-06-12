"""Kaggle GPU kernel entry point: run the full hnet benchmark suite.

This file is the single ``code_file`` pushed by ``deploy.py`` via
``kaggle kernels push`` (kernel_type=script, enable_gpu=true,
enable_internet=true). Kernels upload exactly one code file, so this
script is self-contained: it clones the public repository, installs the
package, and executes every benchmark script with ``--device cuda``.

Artifacts (phase portraits, energy-drift profiles, metric tables) are
copied into ``artifacts/`` under the kernel working directory so that
``kaggle kernels output`` retrieves them, together with a ``summary.md``
aggregating the per-task metric tables.

Kernels accept no CLI arguments: flip ``SMOKE`` to ``True`` before a
validation push (150 epochs per model) and back to ``False`` for the
full 3000-epoch production run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SMOKE = False

REPO_URL = "https://github.com/JuanJGalindo/hamiltoniannet"
CLONE_DIR = Path.cwd() / "hamiltoniannet"
OUTPUT_DIR = Path.cwd() / "artifacts"

BENCHMARK_SCRIPTS = (
    "01_conservative_energy_conservation.py",
    "02_trajectory_accuracy.py",
    "03_harmonic_oscillator_recovery.py",
    "04_noise_robustness.py",
    "05_data_efficiency.py",
    "06_derivative_reconstruction_bias.py",
)


def run(cmd: list[str], **kwargs) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def bootstrap() -> None:
    """Clone the public repository and install the hnet package."""
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)])
    run([sys.executable, "-m", "pip", "install", "--quiet", str(CLONE_DIR)])


def assert_gpu() -> None:
    """Fail fast when no CUDA device is allocated: this run exists to use it."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. This kernel must be pushed with "
            "enable_gpu=true and a GPU accelerator (see benchmarks/kaggle/deploy.py)."
        )
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)


def run_benchmarks() -> None:
    extra = ["--smoke"] if SMOKE else []
    for script in BENCHMARK_SCRIPTS:
        print(
            f"\n=== benchmarks/{script} (device=cuda{', smoke' if SMOKE else ''}) ===", flush=True
        )
        run(
            [sys.executable, f"benchmarks/{script}", "--device", "cuda", *extra],
            cwd=CLONE_DIR,
        )


def collect_artifacts() -> None:
    """Copy benchmark artifacts into the kernel output directory and summarize."""
    source = CLONE_DIR / "benchmarks" / ".artifacts"
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(source, OUTPUT_DIR)

    summary_lines = [f"# hnet benchmark summary ({'smoke' if SMOKE else 'full'} run)", ""]
    for metrics_md in sorted(OUTPUT_DIR.glob("*/metrics.md")):
        summary_lines.append(metrics_md.read_text(encoding="utf-8").rstrip())
        summary_lines.append("")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    n_files = sum(1 for p in OUTPUT_DIR.rglob("*") if p.is_file())
    print(f"\nExported {n_files} artifact files to {OUTPUT_DIR}", flush=True)


def main() -> None:
    bootstrap()
    assert_gpu()
    run_benchmarks()
    collect_artifacts()


if __name__ == "__main__":
    main()
