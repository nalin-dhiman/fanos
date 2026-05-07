"""Small deterministic benchmark against common PyTorch optimizers.

This script is intentionally lightweight. It is a sanity check for optimizer
behavior, not a claim of superiority.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import torch

from fanos_v2 import FANoSV2


def make_problem(dim: int, condition: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x0 = torch.randn(dim, generator=generator)
    spectrum = torch.logspace(0.0, torch.log10(torch.tensor(condition)).item(), dim)
    return x0, spectrum


def loss_fn(x: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    return 0.5 * (spectrum.to(x.device) * x.square()).mean()


def run_one(name: str, x0: torch.Tensor, spectrum: torch.Tensor, steps: int, lr: float) -> dict[str, float]:
    x = torch.nn.Parameter(x0.clone())
    if name == "fanosv2":
        opt = FANoSV2([x], lr=lr, grad_clip=1.0, target_scale=0.2)
    elif name == "adamw":
        opt = torch.optim.AdamW([x], lr=lr, weight_decay=0.0)
    elif name == "sgd":
        opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
    elif name == "rmsprop":
        opt = torch.optim.RMSprop([x], lr=lr, momentum=0.9)
    else:
        raise ValueError(f"unknown optimizer: {name}")

    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(x, spectrum)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    final_loss = float(loss_fn(x, spectrum).detach())

    state_bytes = 0
    if hasattr(opt, "state_size_bytes"):
        state_bytes = int(opt.state_size_bytes())
    else:
        for state in opt.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    state_bytes += value.numel() * value.element_size()

    return {"loss": final_loss, "seconds": elapsed, "state_mb": state_bytes / 1024**2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--condition", type=float, default=1_000.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    x0, spectrum = make_problem(args.dim, args.condition, args.seed)
    rows = []
    for name in ["fanosv2", "adamw", "sgd", "rmsprop"]:
        result = run_one(name, x0, spectrum, args.steps, args.lr)
        rows.append({"optimizer": name, **result})
        print(
            f"{name:8s} loss={result['loss']:.6e} "
            f"time={result['seconds']:.3f}s state={result['state_mb']:.3f}MiB"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
