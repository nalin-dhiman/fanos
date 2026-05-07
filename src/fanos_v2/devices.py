"""Device-selection helpers."""

from __future__ import annotations

import warnings

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cuda``, ``mps``, or ``cpu`` into a usable device.

    Unsupported accelerator requests fall back to CPU with a warning instead of
    crashing after datasets have been downloaded.
    """

    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested, but this PyTorch build has no CUDA support. Falling back to CPU.")
        return torch.device("cpu")

    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested, but it is not available in this environment. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(name)


def device_summary() -> dict[str, str]:
    """Return a compact device summary for reports."""

    summary = {
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()),
        "mps_built": str(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
        "mps_available": str(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        summary["cuda_device"] = torch.cuda.get_device_name(0)
    return summary
