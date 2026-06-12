"""Local deployment wrapper for the Kaggle GPU benchmark kernel.

Wraps the official Kaggle CLI (``pip install "kaggle>=2.0"``, installed by
the ``[kaggle]`` extra) to push, monitor, and download the benchmark
kernel defined by ``run_all_benchmarks.py``:

    python benchmarks/kaggle/deploy.py push [--accelerator NvidiaTeslaT4]
    python benchmarks/kaggle/deploy.py status
    python benchmarks/kaggle/deploy.py download

``push`` renders ``kernel-metadata.json`` (git-ignored; the kernel ``id``
is namespaced by the authenticated Kaggle username) and launches a run on
the selected GPU accelerator. ``download`` pulls the run output into
``benchmarks/.artifacts/kaggle/``.

Credentials: ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables (Kaggle account settings -> API).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = KERNEL_DIR.parent / ".artifacts" / "kaggle"
KERNEL_SLUG = "hnet-benchmarks-gpu"

GPU_ACCELERATORS = (
    "NvidiaTeslaT4",
    "NvidiaTeslaT4Highmem",
    "NvidiaTeslaP100",
    "NvidiaL4",
    "NvidiaL4X1",
    "NvidiaTeslaA100",
    "NvidiaH100",
    "NvidiaRtxPro6000",
)


def resolve_username() -> str:
    """Resolve the Kaggle username from the environment or kaggle.json."""
    username = os.environ.get("KAGGLE_USERNAME")
    if username:
        return username
    credentials = Path.home() / ".kaggle" / "kaggle.json"
    if credentials.exists():
        username = json.loads(credentials.read_text(encoding="utf-8")).get("username")
        if username:
            return username
    raise SystemExit(
        "Kaggle username not found. Set KAGGLE_USERNAME/KAGGLE_KEY or place the API "
        "token from the Kaggle account settings page at ~/.kaggle/kaggle.json."
    )


def kernel_id(username: str) -> str:
    return f"{username}/{KERNEL_SLUG}"


def render_metadata(username: str) -> dict:
    """Kernel manifest consumed by ``kaggle kernels push``."""
    return {
        "id": kernel_id(username),
        "title": "hnet HNN Benchmarks (GPU)",
        "code_file": "run_all_benchmarks.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
    }


def write_metadata(username: str) -> Path:
    path = KERNEL_DIR / "kernel-metadata.json"
    path.write_text(json.dumps(render_metadata(username), indent=2) + "\n", encoding="utf-8")
    return path


def run_kaggle(args: list[str]) -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            'Kaggle CLI not found on PATH. Install it with: pip install "kaggle>=2.0" '
            '(or pip install -e ".[kaggle]").'
        )
    cmd = ["kaggle", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def cmd_push(args: argparse.Namespace) -> None:
    username = resolve_username()
    metadata_path = write_metadata(username)
    print(f"Rendered {metadata_path}:")
    print(metadata_path.read_text(encoding="utf-8"))
    push_args = ["kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", args.accelerator]
    if args.dry_run:
        print(f"[dry-run] kaggle {' '.join(push_args)}")
        return
    run_kaggle(push_args)
    print(
        f"\nPushed {kernel_id(username)} (accelerator={args.accelerator}). "
        f"Monitor with: python {Path(sys.argv[0]).name} status"
    )


def cmd_status(_args: argparse.Namespace) -> None:
    run_kaggle(["kernels", "status", kernel_id(resolve_username())])


def cmd_download(_args: argparse.Namespace) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run_kaggle(["kernels", "output", kernel_id(resolve_username()), "-p", str(DOWNLOAD_DIR)])
    print(f"\nArtifacts downloaded to {DOWNLOAD_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="render kernel-metadata.json and launch a GPU run")
    push.add_argument(
        "--accelerator",
        default="NvidiaTeslaT4",
        choices=GPU_ACCELERATORS,
        help="Kaggle GPU accelerator ID (default: NvidiaTeslaT4, free tier)",
    )
    push.add_argument(
        "--dry-run",
        action="store_true",
        help="render metadata and print the push command without executing it",
    )
    push.set_defaults(func=cmd_push)

    status = sub.add_parser("status", help="report the latest kernel run state")
    status.set_defaults(func=cmd_status)

    download = sub.add_parser("download", help="pull run output into benchmarks/.artifacts/kaggle/")
    download.set_defaults(func=cmd_download)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
