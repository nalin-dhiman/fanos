"""Compatibility namespace for the PyPI package named ``fanos``.

The v0.2/v0.4 codebase exposes the new implementation as ``fanos_v2`` while
the PyPI project remains ``fanos`` for continuity with earlier releases.
"""

from fanos_v2 import (
    FANoSV2,
    FANoSV2Diagnostics,
    FANoSV2Fast,
    Quantized4BitTensor,
    SparseTopKTensor,
    average_gradients,
    ddp_safe_optimizer_step,
    densify_topk,
    dequantize_4bit,
    device_summary,
    dynamic_variance_clip,
    low_rank_approximation,
    quantize_4bit,
    quantized_gradient_residual,
    resolve_device,
    sparsify_topk,
)

FANoS = FANoSV2
FANoSFast = FANoSV2Fast

__all__ = [
    "FANoS",
    "FANoSFast",
    "FANoSV2",
    "FANoSV2Fast",
    "FANoSV2Diagnostics",
    "Quantized4BitTensor",
    "SparseTopKTensor",
    "average_gradients",
    "ddp_safe_optimizer_step",
    "densify_topk",
    "dequantize_4bit",
    "device_summary",
    "dynamic_variance_clip",
    "low_rank_approximation",
    "quantize_4bit",
    "quantized_gradient_residual",
    "resolve_device",
    "sparsify_topk",
]
