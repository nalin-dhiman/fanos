"""Small image-classification benchmark with external dataset storage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fanos_v2 import FANoSV2, FANoSV2Fast, resolve_device


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: int = 28, classes: int = 10) -> None:
        super().__init__()
        pooled = image_size // 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * pooled * pooled, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_optimizer(name: str, params, args):
    lr = args.lr
    if name in {"fanosv2", "fanosv2fast"}:
        opt_cls = FANoSV2Fast if name == "fanosv2fast" else FANoSV2
        preset = "auto" if name == "fanosv2fast" and args.fanos_preset == "default" else args.fanos_preset
        kwargs = dict(
            preset=preset,
            lr=lr,
            momentum=args.fanos_momentum,
            thermostat_lr=args.fanos_thermostat_lr,
            target_scale=args.fanos_target_scale,
            lr_bounds=(lr * args.fanos_lr_min_mult, lr * args.fanos_lr_max_mult),
            preconditioner_power=args.fanos_preconditioner_power,
            adaptive_preconditioner_power=args.fanos_adaptive_preconditioner_power,
            warmup_steps=args.fanos_warmup_steps,
            warmup_start_momentum=args.fanos_warmup_start_momentum,
            warmup_start_lr_scale=args.fanos_warmup_start_lr_scale,
            thermostat_warmup_steps=args.fanos_thermostat_warmup_steps,
            grad_norm_interval=args.fanos_grad_norm_interval,
            sanitize_gradients=args.fanos_sanitize_gradients,
        )
        if name == "fanosv2fast" and args.fanos_thermostat_interval == 1:
            kwargs["thermostat_interval"] = 4
        else:
            kwargs["thermostat_interval"] = args.fanos_thermostat_interval
        if name == "fanosv2fast" and args.fanos_grad_clip == 1.0:
            kwargs["grad_clip"] = None
        else:
            kwargs["grad_clip"] = None if args.fanos_grad_clip <= 0.0 else args.fanos_grad_clip
        if args.fanos_adaptive_lr is not None:
            kwargs["adaptive_lr"] = args.fanos_adaptive_lr
        if args.fanos_record_diagnostics is not None:
            kwargs["record_diagnostics"] = args.fanos_record_diagnostics
        if args.fanos_diagnostics_interval != 1:
            kwargs["diagnostics_interval"] = args.fanos_diagnostics_interval
        return opt_cls(params, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer: {name}")


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == target).float().mean().item())


def state_mb(optimizer: torch.optim.Optimizer) -> float:
    if hasattr(optimizer, "state_size_bytes"):
        return optimizer.state_size_bytes() / 1024**2
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total / 1024**2


def run_optimizer(name: str, args) -> dict[str, float | str]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset_specs = {
        "mnist": (datasets.MNIST, 1, 28),
        "fashionmnist": (datasets.FashionMNIST, 1, 28),
        "cifar10": (datasets.CIFAR10, 3, 32),
    }
    dataset_cls, in_channels, image_size = dataset_specs[args.dataset]
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = dataset_cls(str(args.data_root / "vision"), train=True, transform=transform, download=args.download)
    test_ds = dataset_cls(str(args.data_root / "vision"), train=False, transform=transform, download=args.download)
    train_ds = Subset(train_ds, list(range(min(args.train_samples, len(train_ds)))))
    test_ds = Subset(test_ds, list(range(min(args.test_samples, len(test_ds)))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = SmallCNN(in_channels=in_channels, image_size=image_size).to(device)
    optimizer = make_optimizer(name, model.parameters(), args)
    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    elapsed = time.perf_counter() - start

    model.eval()
    acc_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in test_loader:
            acc_sum += accuracy(model(x.to(device)), y.to(device))
            n_batches += 1

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    return {
        "optimizer": name,
        "loss": last_loss,
        "top1": acc_sum / max(1, n_batches),
        "seconds": elapsed,
        "state_mb": state_mb(optimizer),
        "peak_gpu_mb": peak_mb,
        "device": str(device),
        "dataset": args.dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dataset", choices=["mnist", "fashionmnist", "cifar10"], default="mnist")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fanos-target-scale", type=float, default=0.10)
    parser.add_argument("--fanos-momentum", type=float, default=0.85)
    parser.add_argument("--fanos-thermostat-lr", type=float, default=3e-3)
    parser.add_argument("--fanos-grad-clip", type=float, default=1.0)
    parser.add_argument("--fanos-preset", choices=["default", "auto", "pinn"], default="default")
    parser.add_argument("--fanos-preconditioner-power", type=float, default=1.0)
    parser.add_argument("--fanos-adaptive-preconditioner-power", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fanos-warmup-steps", type=int, default=0)
    parser.add_argument("--fanos-warmup-start-momentum", type=float, default=0.0)
    parser.add_argument("--fanos-warmup-start-lr-scale", type=float, default=1.0)
    parser.add_argument("--fanos-thermostat-warmup-steps", type=int, default=0)
    parser.add_argument("--fanos-thermostat-interval", type=int, default=1)
    parser.add_argument("--fanos-grad-norm-interval", type=int, default=1)
    parser.add_argument("--fanos-sanitize-gradients", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fanos-record-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fanos-diagnostics-interval", type=int, default=1)
    parser.add_argument("--fanos-adaptive-lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fanos-lr-min-mult", type=float, default=0.1)
    parser.add_argument("--fanos-lr-max-mult", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--optimizers", nargs="+", default=["fanosv2", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()

    rows = [run_optimizer(name, args) for name in args.optimizers]
    for row in rows:
        print(
            f"{row['optimizer']:8s} loss={row['loss']:.4f} top1={row['top1']:.3f} "
            f"time={row['seconds']:.2f}s state={row['state_mb']:.3f}MiB gpu={row['peak_gpu_mb']:.1f}MiB"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
