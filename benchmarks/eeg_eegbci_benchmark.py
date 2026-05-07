"""EEGBCI cross-subject benchmark for FANoS-v2.

This is a small, honest EEG harness. It uses PhysioNet EEG Motor Movement /
Imagery via MNE and stores files under ../datasets by default.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fanos_v2 import FANoSV2, FANoSV2Fast, resolve_device


class EEGNetLite(nn.Module):
    def __init__(self, channels: int, samples: int, classes: int) -> None:
        super().__init__()
        pool_kernel = max(1, samples // 32)
        pooled_samples = ((samples - pool_kernel) // pool_kernel) + 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, 24, kernel_size=15, padding=7),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel),
            nn.Flatten(),
            nn.Linear(24 * pooled_samples, 64),
            nn.ELU(),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_subject(subject: int, runs: list[int], data_root: Path, tmin: float, tmax: float):
    os.environ.setdefault("MPLCONFIGDIR", str(data_root / ".mplconfig"))
    import mne
    from mne.datasets import eegbci

    files = eegbci.load_data(subjects=subject, runs=runs, path=str(data_root / "eeg"), update_path=False)
    raws = [mne.io.read_raw_edf(path, preload=True, verbose=False) for path in files]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.filter(7.0, 30.0, fir_design="firwin", verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    selected = {key: value for key, value in event_id.items() if key in {"T1", "T2"}}
    if len(selected) < 2:
        raise RuntimeError(f"Subject {subject} did not expose T1/T2 events for runs {runs}")
    epochs = mne.Epochs(raw, events, selected, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    x = epochs.get_data(copy=True).astype("float32")
    labels = epochs.events[:, -1]
    values = sorted(selected.values())
    y = np.array([values.index(label) for label in labels], dtype="int64")
    x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-6)
    return torch.from_numpy(x), torch.from_numpy(y)


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


def state_mb(optimizer: torch.optim.Optimizer) -> float:
    if hasattr(optimizer, "state_size_bytes"):
        return optimizer.state_size_bytes() / 1024**2
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total / 1024**2


def run_fold(name: str, train_data, test_data, args) -> dict[str, float | str]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    x_train, y_train = train_data
    x_test, y_test = test_data
    model = EEGNetLite(channels=x_train.shape[1], samples=x_train.shape[2], classes=int(y_train.max().item() + 1)).to(device)
    optimizer = make_optimizer(name, model.parameters(), args)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device))
        top1 = float((logits.argmax(dim=1).cpu() == y_test).float().mean().item())

    return {"optimizer": name, "loss": final_loss, "top1": top1, "seconds": seconds, "state_mb": state_mb(optimizer), "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "datasets")
    parser.add_argument("--train-subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--test-subject", type=int, default=2)
    parser.add_argument("--runs", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fanos-target-scale", type=float, default=0.10)
    parser.add_argument("--fanos-momentum", type=float, default=0.85)
    parser.add_argument("--fanos-thermostat-lr", type=float, default=3e-3)
    parser.add_argument("--fanos-grad-clip", type=float, default=1.0)
    parser.add_argument("--fanos-preset", choices=["default", "auto", "pinn"], default="default")
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
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--optimizers", nargs="+", default=["fanosv2", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()

    train_xs, train_ys = [], []
    for subject in args.train_subjects:
        x, y = load_subject(subject, args.runs, args.data_root, args.tmin, args.tmax)
        train_xs.append(x)
        train_ys.append(y)
    train_data = (torch.cat(train_xs), torch.cat(train_ys))
    test_data = load_subject(args.test_subject, args.runs, args.data_root, args.tmin, args.tmax)

    rows = [run_fold(name, train_data, test_data, args) for name in args.optimizers]
    for row in rows:
        print(
            f"{row['optimizer']:8s} loss={row['loss']:.4f} top1={row['top1']:.3f} "
            f"time={row['seconds']:.2f}s state={row['state_mb']:.3f}MiB"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
