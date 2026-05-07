"""FANoS-v2 hyperparameter sweep for available local benchmarks."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT.parent
DEFAULT_DATA_ROOT = MAIN_DIR / "datasets"
DEFAULT_RESULTS_ROOT = MAIN_DIR / "results"
DEFAULT_REPORT_ROOT = MAIN_DIR / "reports"


def parse_floats(values: str) -> list[float]:
    return [float(value) for value in values.split(",") if value]


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows[0]


def run(cmd: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    output = proc.stdout + f"\n[exit={proc.returncode} elapsed={elapsed:.2f}s]\n"
    log_path.write_text(output)
    print(output)
    return int(proc.returncode), output


def rank_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def key(row: dict[str, str]):
        top1 = float(row.get("top1", "nan"))
        loss = float(row.get("loss", "inf"))
        seconds = float(row.get("seconds", "inf"))
        return (-top1, loss, seconds)

    return sorted(rows, key=key)


def write_report(task: str, rows: list[dict[str, str]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ranked = rank_rows(rows)
    headers = list(ranked[0].keys()) if ranked else []
    lines = [
        f"# FANoS-v2 {task} Sweep",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if ranked:
        lines.extend(
            [
                "## Best Config",
                "",
                "```json",
                json.dumps(ranked[0], indent=2),
                "```",
                "",
                "## All Results",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        for row in ranked:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    else:
        lines.append("_No successful rows._")
    report_path.write_text("\n".join(lines) + "\n")


def run_vision_sweep(args) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    out_dir = args.results_root / "sweeps" / "vision"
    combos = list(itertools.product(args.lrs, args.target_scales, args.momentums, args.thermostat_lrs))
    for idx, (lr, target_scale, momentum, thermostat_lr) in enumerate(combos, start=1):
        out_csv = out_dir / f"vision_{idx:03d}.csv"
        code, _ = run(
            [
                sys.executable,
                "benchmarks/vision_benchmark.py",
                "--epochs",
                str(args.vision_epochs),
                "--train-samples",
                str(args.vision_train_samples),
                "--test-samples",
                str(args.vision_test_samples),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--data-root",
                str(args.data_root),
                "--optimizers",
                "fanosv2",
                "--lr",
                str(lr),
                "--fanos-target-scale",
                str(target_scale),
                "--fanos-momentum",
                str(momentum),
                "--fanos-thermostat-lr",
                str(thermostat_lr),
                "--fanos-grad-clip",
                str(args.grad_clip),
                "--out",
                str(out_csv),
            ],
            args.results_root / "logs" / "sweeps" / f"vision_{idx:03d}.log",
        )
        if code == 0:
            row = read_first_row(out_csv)
            row.update({"lr": lr, "target_scale": target_scale, "momentum": momentum, "thermostat_lr": thermostat_lr})
            rows.append(row)
    return rows


def run_eeg_sweep(args) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    out_dir = args.results_root / "sweeps" / "eeg"
    combos = list(itertools.product(args.lrs, args.target_scales, args.momentums, args.thermostat_lrs))
    for idx, (lr, target_scale, momentum, thermostat_lr) in enumerate(combos, start=1):
        out_csv = out_dir / f"eeg_{idx:03d}.csv"
        code, _ = run(
            [
                sys.executable,
                "benchmarks/eeg_eegbci_benchmark.py",
                "--train-subjects",
                *[str(subject) for subject in args.eeg_train_subjects],
                "--test-subject",
                str(args.eeg_test_subject),
                "--runs",
                *[str(run_id) for run_id in args.eeg_runs],
                "--epochs",
                str(args.eeg_epochs),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--data-root",
                str(args.data_root),
                "--optimizers",
                "fanosv2",
                "--lr",
                str(lr),
                "--fanos-target-scale",
                str(target_scale),
                "--fanos-momentum",
                str(momentum),
                "--fanos-thermostat-lr",
                str(thermostat_lr),
                "--fanos-grad-clip",
                str(args.grad_clip),
                "--out",
                str(out_csv),
            ],
            args.results_root / "logs" / "sweeps" / f"eeg_{idx:03d}.log",
        )
        if code == 0:
            row = read_first_row(out_csv)
            row.update({"lr": lr, "target_scale": target_scale, "momentum": momentum, "thermostat_lr": thermostat_lr})
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["vision", "eeg", "both"], default="both")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lrs", type=parse_floats, default=parse_floats("0.0005,0.001,0.002"))
    parser.add_argument("--target-scales", type=parse_floats, default=parse_floats("0.05,0.1,0.2"))
    parser.add_argument("--momentums", type=parse_floats, default=parse_floats("0.85,0.9"))
    parser.add_argument("--thermostat-lrs", type=parse_floats, default=parse_floats("0.003,0.01"))
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--vision-epochs", type=int, default=2)
    parser.add_argument("--vision-train-samples", type=int, default=10000)
    parser.add_argument("--vision-test-samples", type=int, default=2000)

    parser.add_argument("--eeg-epochs", type=int, default=10)
    parser.add_argument("--eeg-train-subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--eeg-test-subject", type=int, default=5)
    parser.add_argument("--eeg-runs", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.report_root = args.report_root.resolve()

    if args.task in {"vision", "both"}:
        rows = run_vision_sweep(args)
        write_csv(rows, args.results_root / "fanos_vision_sweep.csv")
        write_report("Vision", rows, args.report_root / "fanos_vision_sweep.md")

    if args.task in {"eeg", "both"}:
        rows = run_eeg_sweep(args)
        write_csv(rows, args.results_root / "fanos_eeg_sweep.csv")
        write_report("EEG", rows, args.report_root / "fanos_eeg_sweep.md")

    print(f"Reports written under {args.report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
