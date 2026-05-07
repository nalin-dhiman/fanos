"""Overnight FANoS-v2 study with repeated seeds and aggregate report."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT.parent
DEFAULT_DATA_ROOT = MAIN_DIR / "datasets"
DEFAULT_RESULTS_ROOT = MAIN_DIR / "results" / "night_study"
DEFAULT_REPORT_ROOT = MAIN_DIR / "reports"


FANOS_CONFIGS = {
    "stable": {"lr": 0.001, "target_scale": 0.10, "momentum": 0.85, "thermostat_lr": 0.003},
    "vision_sweep_best": {"lr": 0.001, "target_scale": 0.05, "momentum": 0.85, "thermostat_lr": 0.003},
    "eeg_sweep_best": {"lr": 0.001, "target_scale": 0.10, "momentum": 0.90, "thermostat_lr": 0.003},
    "low_lr": {"lr": 0.0005, "target_scale": 0.10, "momentum": 0.85, "thermostat_lr": 0.003},
    "auto": {"lr": 0.001, "target_scale": 0.10, "momentum": 0.85, "thermostat_lr": 0.003, "preset": "auto"},
    "auto_fast_sync": {
        "lr": 0.001,
        "target_scale": 0.10,
        "momentum": 0.85,
        "thermostat_lr": 0.003,
        "preset": "auto",
        "thermostat_interval": 8,
        "grad_norm_interval": 8,
        "sanitize_gradients": False,
    },
}


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    log_path.write_text(proc.stdout + f"\n[exit={proc.returncode} elapsed={elapsed:.2f}s]\n")
    print(proc.stdout)
    print(f"[exit={proc.returncode} elapsed={elapsed:.2f}s]")
    return int(proc.returncode)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["task"], row["optimizer"], row.get("config", "baseline"))
        groups.setdefault(key, []).append(row)

    summary = []
    for (task, optimizer, config), group_rows in groups.items():
        top1 = [float(row["top1"]) for row in group_rows]
        loss = [float(row["loss"]) for row in group_rows]
        seconds = [float(row["seconds"]) for row in group_rows]
        state = [float(row["state_mb"]) for row in group_rows]
        summary.append(
            {
                "task": task,
                "optimizer": optimizer,
                "config": config,
                "n": len(group_rows),
                "top1_mean": statistics.mean(top1),
                "top1_std": statistics.stdev(top1) if len(top1) > 1 else 0.0,
                "top1_best": max(top1),
                "top1_worst": min(top1),
                "loss_mean": statistics.mean(loss),
                "loss_std": statistics.stdev(loss) if len(loss) > 1 else 0.0,
                "loss_best": min(loss),
                "loss_worst": max(loss),
                "seconds_mean": statistics.mean(seconds),
                "state_mb_mean": statistics.mean(state),
            }
        )

    adamw_by_task = {
        str(row["task"]): row
        for row in summary
        if row["optimizer"] == "adamw" and row["config"] == "baseline"
    }
    for row in summary:
        adamw = adamw_by_task.get(str(row["task"]))
        if adamw is None:
            row["delta_top1_vs_adamw"] = ""
            row["delta_seconds_pct_vs_adamw"] = ""
            continue
        row["delta_top1_vs_adamw"] = float(row["top1_mean"]) - float(adamw["top1_mean"])
        adamw_seconds = float(adamw["seconds_mean"])
        row["delta_seconds_pct_vs_adamw"] = (
            100.0 * (float(row["seconds_mean"]) - adamw_seconds) / adamw_seconds if adamw_seconds > 0 else ""
        )

    return sorted(summary, key=lambda row: (row["task"], -float(row["top1_mean"]), float(row["loss_mean"])))


def markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def run_task_baselines(task: str, seed: int, args) -> list[dict[str, str]]:
    out = args.results_root / f"{task}_baselines_seed{seed}.csv"
    if task == "vision":
        cmd = [
            sys.executable,
            "benchmarks/vision_benchmark.py",
            "--epochs",
            str(args.vision_epochs),
            "--train-samples",
            str(args.vision_train_samples),
            "--test-samples",
            str(args.vision_test_samples),
            "--dataset",
            args.vision_dataset,
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--data-root",
            str(args.data_root),
            "--optimizers",
            "adamw",
            "sgd",
            "rmsprop",
            "--out",
            str(out),
        ]
    else:
        cmd = [
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
            str(seed),
            "--data-root",
            str(args.data_root),
            "--optimizers",
            "adamw",
            "sgd",
            "rmsprop",
            "--out",
            str(out),
        ]
    code = run(cmd, args.results_root / "logs" / f"{task}_baselines_seed{seed}.log")
    rows = read_rows(out) if code == 0 else []
    for row in rows:
        row.update({"task": task, "seed": seed, "config": "baseline"})
    return rows


def run_task_fanos(task: str, seed: int, config_name: str, config: dict[str, float], args) -> list[dict[str, str]]:
    out = args.results_root / f"{task}_fanos_{config_name}_seed{seed}.csv"
    common = [
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--data-root",
        str(args.data_root),
        "--optimizers",
        "fanosv2",
        "--lr",
        str(config["lr"]),
        "--fanos-target-scale",
        str(config["target_scale"]),
        "--fanos-momentum",
        str(config["momentum"]),
        "--fanos-thermostat-lr",
        str(config["thermostat_lr"]),
        "--out",
        str(out),
    ]
    if "preset" in config:
        common.extend(["--fanos-preset", str(config["preset"])])
    if "thermostat_interval" in config:
        common.extend(["--fanos-thermostat-interval", str(config["thermostat_interval"])])
    if "grad_norm_interval" in config:
        common.extend(["--fanos-grad-norm-interval", str(config["grad_norm_interval"])])
    if config.get("sanitize_gradients") is False:
        common.append("--no-fanos-sanitize-gradients")
    if task == "vision":
        cmd = [
            sys.executable,
            "benchmarks/vision_benchmark.py",
            "--epochs",
            str(args.vision_epochs),
            "--train-samples",
            str(args.vision_train_samples),
            "--test-samples",
            str(args.vision_test_samples),
            "--dataset",
            args.vision_dataset,
            *common,
        ]
    else:
        cmd = [
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
            *common,
        ]
    code = run(cmd, args.results_root / "logs" / f"{task}_fanos_{config_name}_seed{seed}.log")
    rows = read_rows(out) if code == 0 else []
    for row in rows:
        row.update(
            {
                "task": task,
                "seed": seed,
                "config": config_name,
                "lr": config["lr"],
                "target_scale": config["target_scale"],
                "momentum": config["momentum"],
                "thermostat_lr": config["thermostat_lr"],
                "preset": config.get("preset", "default"),
            }
        )
    return rows


def write_report(args, all_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> Path:
    args.report_root.mkdir(parents=True, exist_ok=True)
    default_results = DEFAULT_RESULTS_ROOT.resolve()
    report_name = "fanos_night_study_report.md"
    if args.results_root.resolve() != default_results:
        report_name = f"fanos_night_study_{args.results_root.name}_report.md"
    report_path = args.report_root / report_name
    report_path.write_text(
        f"""# FANoS-v2 Night Study Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Aggregate Summary

{markdown_table(summary_rows)}

## Raw Rows

{markdown_table(all_rows)}

## Notes

- Baselines are AdamW, SGD, and RMSProp at the benchmark default learning rate.
- FANoS configurations are fixed presets chosen from the previous sweep plus one lower-LR guardrail.
- `delta_top1_vs_adamw` and `delta_seconds_pct_vs_adamw` are computed within each task against the AdamW baseline.
- Treat this as stronger evidence than a single seed, but not a final paper result unless the selected seeds/folds match the target protocol.
"""
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=["vision", "eeg"], default=["vision", "eeg"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--configs", nargs="+", default=list(FANOS_CONFIGS.keys()))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--vision-epochs", type=int, default=5)
    parser.add_argument("--vision-train-samples", type=int, default=60000)
    parser.add_argument("--vision-test-samples", type=int, default=10000)
    parser.add_argument("--vision-dataset", choices=["mnist", "fashionmnist", "cifar10"], default="mnist")
    parser.add_argument("--eeg-epochs", type=int, default=10)
    parser.add_argument("--eeg-train-subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--eeg-test-subject", type=int, default=5)
    parser.add_argument("--eeg-runs", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.report_root = args.report_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)

    configs = {name: FANOS_CONFIGS[name] for name in args.configs}
    all_rows: list[dict[str, str]] = []
    for task in args.tasks:
        for seed in args.seeds:
            all_rows.extend(run_task_baselines(task, seed, args))
            for config_name, config in configs.items():
                all_rows.extend(run_task_fanos(task, seed, config_name, config, args))

    raw_path = args.results_root / "night_study_raw.csv"
    summary_path = args.results_root / "night_study_summary.csv"
    summary_rows = summarize(all_rows)
    write_rows(raw_path, all_rows)
    write_rows(summary_path, summary_rows)
    report_path = write_report(args, all_rows, summary_rows)
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
