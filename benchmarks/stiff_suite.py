"""Stiff-objective benchmark suite for FANoS-v2.

Tasks are intentionally lightweight and reproducible:
- Rosenbrock
- ill-conditioned quadratic
- noisy small-data regression
- ODE parameter fitting
- 1D Poisson PINN
- synthetic sequence memory
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import nn

from fanos_v2 import FANoSV2, resolve_device


@dataclass
class Result:
    task: str
    optimizer: str
    seed: int
    final_loss: float
    metric: float
    seconds: float
    state_mb: float
    status: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def optimizer_state_mb(opt: torch.optim.Optimizer) -> float:
    if hasattr(opt, "state_size_bytes"):
        return opt.state_size_bytes() / 1024**2
    total = 0
    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total / 1024**2


FANOS_PRESETS = {
    "default": {},
    "auto": {
        "optimizer_preset": "auto",
    },
    "pinn": {
        "lr": 5e-4,
        "target_scale": 0.05,
        "momentum": 0.75,
        "thermostat_lr": 0.001,
        "adaptive_lr": True,
        "preconditioner_power": 0.5,
        "warmup_steps": 200,
        "warmup_start_momentum": 0.0,
        "thermostat_warmup_steps": 200,
    },
    "pinn_raw": {
        "lr": 1e-3,
        "target_scale": 0.10,
        "momentum": 0.75,
        "thermostat_lr": 0.003,
        "adaptive_lr": True,
        "preconditioner_power": 0.0,
        "warmup_steps": 200,
        "warmup_start_momentum": 0.0,
        "thermostat_warmup_steps": 200,
    },
    "sequence": {
        "lr": 1e-3,
        "target_scale": 0.05,
        "momentum": 0.70,
        "thermostat_lr": 0.001,
        "adaptive_lr": True,
        "preconditioner_power": 1.0,
        "warmup_steps": 300,
        "warmup_start_momentum": 0.0,
        "thermostat_warmup_steps": 100,
    },
    "no_precond": {
        "preconditioner": "none",
        "lr": 1e-3,
        "target_scale": 0.10,
        "momentum": 0.85,
        "thermostat_lr": 0.003,
        "adaptive_lr": True,
        "warmup_steps": 200,
        "warmup_start_momentum": 0.0,
    },
}


def make_optimizer(name: str, params, lr: float, grad_clip: float | None = 1.0, fanos_preset: str = "default"):
    if name == "fanosv2":
        preset = FANOS_PRESETS[fanos_preset]
        lr = float(preset.get("lr", lr))
        return FANoSV2(
            params,
            preset=str(preset.get("optimizer_preset", "default")),
            lr=lr,
            grad_clip=grad_clip,
            momentum=float(preset.get("momentum", 0.85)),
            thermostat_lr=float(preset.get("thermostat_lr", 0.003)),
            target_scale=float(preset.get("target_scale", 0.10)),
            adaptive_lr=bool(preset.get("adaptive_lr", True)),
            lr_bounds=(lr * 0.1, lr * 2.0),
            preconditioner=str(preset.get("preconditioner", "diag")),
            preconditioner_power=float(preset.get("preconditioner_power", 1.0)),
            warmup_steps=int(preset.get("warmup_steps", 0)),
            warmup_start_momentum=float(preset.get("warmup_start_momentum", 0.0)),
            thermostat_warmup_steps=int(preset.get("thermostat_warmup_steps", 0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer: {name}")


def safe_result(task: str, name: str, seed: int, start: float, opt: torch.optim.Optimizer, loss: torch.Tensor, metric: float) -> Result:
    status = "ok" if torch.isfinite(loss).all() else "nonfinite"
    return Result(task, name, seed, float(loss.detach().cpu()), float(metric), time.perf_counter() - start, optimizer_state_mb(opt), status)


def rosenbrock_loss(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(100.0 * (x[1:] - x[:-1].square()).square() + (1.0 - x[:-1]).square())


def run_rosenbrock(name: str, seed: int, device: torch.device, steps: int, fanos_preset: str) -> Result:
    set_seed(seed)
    x = nn.Parameter(torch.empty(100, device=device).uniform_(-2.0, 2.0))
    opt = make_optimizer(name, [x], lr=0.001 if name != "sgd" else 0.0005, grad_clip=1.0, fanos_preset=fanos_preset)
    start = time.perf_counter()
    loss = torch.tensor(float("inf"), device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = rosenbrock_loss(x)
        loss.backward()
        opt.step()
    metric = float(torch.linalg.norm(x.detach() - 1.0).cpu())
    return safe_result("rosenbrock100", name, seed, start, opt, loss, metric)


def run_ill_conditioned_quadratic(name: str, seed: int, device: torch.device, steps: int, fanos_preset: str) -> Result:
    set_seed(seed)
    dim = 512
    x = nn.Parameter(torch.randn(dim, device=device))
    spectrum = torch.logspace(0.0, 6.0, dim, device=device)
    opt = make_optimizer(name, [x], lr=0.01 if name != "sgd" else 0.001, grad_clip=1.0, fanos_preset=fanos_preset)
    start = time.perf_counter()
    loss = torch.tensor(float("inf"), device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * (spectrum * x.square()).mean()
        loss.backward()
        opt.step()
    metric = float(torch.linalg.norm(x.detach()).cpu())
    return safe_result("ill_conditioned_quadratic", name, seed, start, opt, loss, metric)


class TinyMLP(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_noisy_regression(name: str, seed: int, device: torch.device, steps: int, fanos_preset: str) -> Result:
    set_seed(seed)
    x = torch.linspace(-3, 3, 32, device=device).unsqueeze(1)
    y_clean = torch.sin(3 * x) + 0.2 * x
    y = y_clean + 0.25 * torch.randn_like(y_clean)
    model = TinyMLP().to(device)
    opt = make_optimizer(name, model.parameters(), lr=0.003 if name != "sgd" else 0.001, grad_clip=1.0, fanos_preset=fanos_preset)
    start = time.perf_counter()
    loss = torch.tensor(float("inf"), device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        metric = float(torch.nn.functional.mse_loss(model(x), y_clean).cpu())
    return safe_result("noisy_small_regression", name, seed, start, opt, loss, metric)


def run_ode_fit(name: str, seed: int, device: torch.device, steps: int, fanos_preset: str) -> Result:
    set_seed(seed)
    t = torch.linspace(0, 2, 50, device=device)
    true_k = torch.tensor(1.7, device=device)
    y = torch.exp(-true_k * t) + 0.01 * torch.randn_like(t)
    raw_k = nn.Parameter(torch.tensor(0.0, device=device))
    opt = make_optimizer(name, [raw_k], lr=0.03 if name != "sgd" else 0.01, grad_clip=1.0, fanos_preset=fanos_preset)
    start = time.perf_counter()
    loss = torch.tensor(float("inf"), device=device)
    for _ in range(steps):
        opt.zero_grad()
        k = torch.nn.functional.softplus(raw_k)
        pred = torch.exp(-k * t)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
    metric = float(torch.abs(torch.nn.functional.softplus(raw_k).detach() - true_k).cpu())
    return safe_result("ode_exp_fit", name, seed, start, opt, loss, metric)


class PINN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_poisson_pinn(name: str, seed: int, device: torch.device, steps: int, fanos_preset: str) -> Result:
    set_seed(seed)
    model = PINN().to(device)
    x_col = torch.linspace(0, 1, 64, device=device).unsqueeze(1)
    x_col.requires_grad_(True)
    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    y_bc = torch.zeros_like(x_bc)
    preset = fanos_preset if name == "fanosv2" and fanos_preset != "default" else "default"
    opt = make_optimizer(name, model.parameters(), lr=0.001 if name != "sgd" else 0.0005, grad_clip=1.0, fanos_preset=preset)
    start = time.perf_counter()
    loss = torch.tensor(float("inf"), device=device)
    for _ in range(steps):
        opt.zero_grad()
        u = model(x_col)
        du = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u = torch.autograd.grad(du, x_col, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        forcing = -(torch.pi**2) * torch.sin(torch.pi * x_col)
        residual = d2u - forcing
        loss_pde = residual.square().mean()
        loss_bc = torch.nn.functional.mse_loss(model(x_bc), y_bc)
        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_eval = torch.linspace(0, 1, 128, device=device).unsqueeze(1)
        u_true = torch.sin(torch.pi * x_eval)
        metric = float(torch.nn.functional.mse_loss(model(x_eval), u_true).cpu())
    return safe_result("poisson_pinn_1d", name, seed, start, opt, loss, metric)


class TinyRNN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden, nonlinearity="tanh", batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y[:, -1])


def run_sequence_memory(name: str, seed: int, device: torch.device, steps: int, fanos_preset: str) -> Result:
    set_seed(seed)
    n = 128
    length = 40
    bits = torch.randint(0, 2, (n,), device=device)
    x = torch.zeros(n, length, 2, device=device)
    x[:, 0, 0] = bits.float()
    x[:, :, 1] = torch.randn(n, length, device=device) * 0.05
    y = bits.long()
    model = TinyRNN().to(device)
    preset = fanos_preset if name == "fanosv2" and fanos_preset != "default" else "default"
    opt = make_optimizer(name, model.parameters(), lr=0.003 if name != "sgd" else 0.001, grad_clip=1.0, fanos_preset=preset)
    start = time.perf_counter()
    loss = torch.tensor(float("inf"), device=device)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        metric = float((model(x).argmax(dim=1) == y).float().mean().cpu())
    return safe_result("sequence_memory", name, seed, start, opt, loss, metric)


TASKS = {
    "rosenbrock100": run_rosenbrock,
    "ill_conditioned_quadratic": run_ill_conditioned_quadratic,
    "noisy_small_regression": run_noisy_regression,
    "ode_exp_fit": run_ode_fit,
    "poisson_pinn_1d": run_poisson_pinn,
    "sequence_memory": run_sequence_memory,
}


def write_csv(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Result.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[Result]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, str], list[Result]] = {}
    for row in rows:
        groups.setdefault((row.task, row.optimizer), []).append(row)
    out = []
    for (task, optimizer), group in groups.items():
        losses = [row.final_loss for row in group]
        metrics = [row.metric for row in group]
        out.append(
            {
                "task": task,
                "optimizer": optimizer,
                "n": len(group),
                "loss_mean": sum(losses) / len(losses),
                "metric_mean": sum(metrics) / len(metrics),
                "seconds_mean": sum(row.seconds for row in group) / len(group),
                "state_mb_mean": sum(row.state_mb for row in group) / len(group),
                "ok": sum(row.status == "ok" for row in group),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument("--optimizers", nargs="+", default=["fanosv2", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fanos-preset", choices=list(FANOS_PRESETS.keys()), default="default")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "stiff_suite.csv")
    parser.add_argument("--summary-out", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "stiff_suite_summary.csv")
    args = parser.parse_args()

    device = resolve_device(args.device)
    rows: list[Result] = []
    for task in args.tasks:
        for seed in args.seeds:
            for optimizer in args.optimizers:
                try:
                    result = TASKS[task](optimizer, seed, device, args.steps, args.fanos_preset)
                except Exception as exc:
                    result = Result(task, optimizer, seed, float("nan"), float("nan"), 0.0, 0.0, f"error:{type(exc).__name__}:{exc}")
                rows.append(result)
                print(
                    f"{task:26s} {optimizer:8s} seed={seed} "
                    f"loss={result.final_loss:.6g} metric={result.metric:.6g} "
                    f"time={result.seconds:.2f}s status={result.status}"
                )

    write_csv(args.out, rows)
    summary_rows = summarize(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="") as f:
        fieldnames = ["task", "optimizer", "n", "loss_mean", "metric_mean", "seconds_mean", "state_mb_mean", "ok"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Raw results: {args.out}")
    print(f"Summary: {args.summary_out}")


if __name__ == "__main__":
    main()
