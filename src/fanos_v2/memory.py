"""Memory and communication helpers for FANoS-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Quantized4BitTensor:
    """Packed signed 4-bit tensor with one symmetric scale."""

    packed: torch.Tensor
    scale: torch.Tensor
    shape: torch.Size
    dtype: torch.dtype


@dataclass(frozen=True)
class SparseTopKTensor:
    """Flat top-k sparse representation for gradient communication tests."""

    indices: torch.Tensor
    values: torch.Tensor
    shape: torch.Size
    nnz: int


def low_rank_approximation(tensor: torch.Tensor, rank: int = 8) -> torch.Tensor:
    """Return a truncated-SVD approximation for a matrix-like tensor.

    Higher-dimensional tensors are flattened to ``(prod(shape[:-1]), shape[-1])``
    and reshaped back. Vectors are returned unchanged.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    if tensor.ndim < 2:
        return tensor.clone()

    original_shape = tensor.shape
    matrix = tensor.reshape(-1, original_shape[-1]).to(dtype=torch.float32)
    max_rank = min(rank, matrix.shape[0], matrix.shape[1])
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    approx = (u[:, :max_rank] * s[:max_rank]) @ vh[:max_rank, :]
    return approx.reshape(original_shape).to(dtype=tensor.dtype)


def quantize_4bit(tensor: torch.Tensor, eps: float = 1e-12) -> Quantized4BitTensor:
    """Symmetrically quantize a tensor into packed signed 4-bit values.

    Values are clipped to [-8, 7]. Two signed nibbles are packed into one uint8.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    flat = tensor.detach().reshape(-1).to(dtype=torch.float32)
    scale = flat.abs().max().clamp_min(eps) / 7.0
    q = torch.round(flat / scale).clamp(-8, 7).to(torch.int16)
    encoded = (q + 8).to(torch.uint8)
    if encoded.numel() % 2 == 1:
        encoded = torch.cat([encoded, torch.zeros(1, dtype=torch.uint8, device=encoded.device)])
    low = encoded[0::2]
    high = encoded[1::2] << 4
    packed = low | high
    return Quantized4BitTensor(packed=packed, scale=scale, shape=tensor.shape, dtype=tensor.dtype)


def dequantize_4bit(qtensor: Quantized4BitTensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Dequantize a tensor produced by :func:`quantize_4bit`."""

    packed = qtensor.packed if device is None else qtensor.packed.to(device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    encoded = torch.stack([low, high], dim=1).reshape(-1)[: int(torch.tensor(qtensor.shape).prod().item())]
    signed = encoded.to(torch.int16) - 8
    values = signed.to(dtype=torch.float32) * qtensor.scale.to(device=packed.device)
    return values.reshape(qtensor.shape).to(dtype=qtensor.dtype)


def sparsify_topk(tensor: torch.Tensor, density: float = 0.01) -> SparseTopKTensor:
    """Keep the largest-magnitude entries of a tensor."""

    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    flat = tensor.detach().reshape(-1)
    nnz = max(1, int(round(flat.numel() * density)))
    if nnz >= flat.numel():
        indices = torch.arange(flat.numel(), device=flat.device)
        values = flat.clone()
    else:
        _, indices = torch.topk(flat.abs(), nnz, sorted=False)
        values = flat[indices].clone()
    return SparseTopKTensor(indices=indices, values=values, shape=tensor.shape, nnz=int(nnz))


def densify_topk(stensor: SparseTopKTensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Reconstruct a dense tensor from :func:`sparsify_topk` output."""

    target_device = device or stensor.values.device
    flat = torch.zeros(int(torch.tensor(stensor.shape).prod().item()), dtype=stensor.values.dtype, device=target_device)
    flat[stensor.indices.to(device=target_device)] = stensor.values.to(device=target_device)
    return flat.reshape(stensor.shape)


def dynamic_variance_clip(grad: torch.Tensor, variance_ema: torch.Tensor, clip_factor: float = 3.0, eps: float = 1e-8) -> torch.Tensor:
    """Clip gradients elementwise using a running variance estimate."""

    if clip_factor <= 0:
        raise ValueError("clip_factor must be positive")
    bound = clip_factor * variance_ema.to(dtype=grad.dtype).sqrt().add(eps)
    return grad.clamp(min=-bound, max=bound)
