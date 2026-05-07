"""Distributed-training helpers for FANoS-v2 experiments."""

from __future__ import annotations

from typing import Iterable

import torch

from .memory import dequantize_4bit, quantize_4bit


def average_gradients(parameters: Iterable[torch.nn.Parameter], group=None) -> None:
    """Average dense gradients across a torch.distributed process group."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size(group=group)
    if world_size <= 1:
        return

    for param in parameters:
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError("average_gradients does not support sparse gradients")
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, group=group)
        param.grad.div_(world_size)


def quantized_gradient_residual(grad: torch.Tensor) -> torch.Tensor:
    """Return quantization residual for error-feedback experiments."""

    qgrad = quantize_4bit(grad)
    restored = dequantize_4bit(qgrad, device=grad.device)
    return grad - restored


def ddp_safe_optimizer_step(optimizer: torch.optim.Optimizer, parameters: Iterable[torch.nn.Parameter], group=None) -> None:
    """Average gradients if distributed is active, then step the optimizer."""

    average_gradients(parameters, group=group)
    optimizer.step()
