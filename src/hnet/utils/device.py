"""Device management utilities."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device from a preference string.

    Parameters
    ----------
    preference : str
        "auto"  — use CUDA if available, else CPU
        "cuda"  — force CUDA (raises if unavailable)
        "cpu"   — force CPU
        "mps"   — Apple Silicon GPU (raises if unavailable)

    Returns
    -------
    torch.device
    """
    preference = preference.lower().strip()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(preference)


def to_device(tensor_or_module, device: str | torch.device = "auto"):
    """Move a tensor or nn.Module to the resolved device."""
    dev = get_device(device) if isinstance(device, str) else device
    return tensor_or_module.to(dev)
