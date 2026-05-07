"""FANoS-v2 PyTorch optimizer.

Public API:
    from fanos_v2 import FANoSV2, FANoSV2Fast
"""

from .memory import (
    Quantized4BitTensor,
    SparseTopKTensor,
    dequantize_4bit,
    densify_topk,
    dynamic_variance_clip,
    low_rank_approximation,
    quantize_4bit,
    sparsify_topk,
)
from .distributed import average_gradients, ddp_safe_optimizer_step, quantized_gradient_residual
from .devices import device_summary, resolve_device
from .optimizer import FANoSV2, FANoSV2Diagnostics, FANoSV2Fast

__all__ = [
    "FANoSV2",
    "FANoSV2Fast",
    "FANoSV2Diagnostics",
    "Quantized4BitTensor",
    "SparseTopKTensor",
    "dequantize_4bit",
    "densify_topk",
    "dynamic_variance_clip",
    "low_rank_approximation",
    "quantize_4bit",
    "sparsify_topk",
    "average_gradients",
    "ddp_safe_optimizer_step",
    "quantized_gradient_residual",
    "device_summary",
    "resolve_device",
]
__version__ = "0.2.0a0"
