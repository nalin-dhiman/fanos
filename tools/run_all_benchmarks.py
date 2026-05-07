"""Run FANoS-v2 benchmark suite and generate a Markdown report.

Datasets, results, and reports are written outside the project directory by
default:

    ../datasets
    ../results
    ../reports
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT.parent
DEFAULT_DATA_ROOT = MAIN_DIR / "datasets"
DEFAULT_RESULTS_ROOT = MAIN_DIR / "results"
DEFAULT_REPORT_ROOT = MAIN_DIR / "reports"


def run_command(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    if dry_run:
        log_path.write_text(f"DRY RUN: {printable}\n")
        return 0

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
    log_path.write_text(proc.stdout + f"\n[exit={proc.returncode} elapsed={elapsed:.2f}s]\n")
    print(proc.stdout)
    print(f"[exit={proc.returncode} elapsed={elapsed:.2f}s]")
    return int(proc.returncode)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows generated._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def collect_environment() -> dict[str, str]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        from fanos_v2 import device_summary, resolve_device

        env.update(device_summary())
        env["requested_device"] = str(getattr(collect_environment, "requested_device", "unknown"))
        env["resolved_device"] = str(resolve_device(str(getattr(collect_environment, "requested_device", "auto"))))
    except Exception as exc:  # pragma: no cover - report helper
        env["torch_error"] = repr(exc)
    return env


def write_report(args, status: dict[str, int], started: float, finished: float) -> Path:
    args.report_root.mkdir(parents=True, exist_ok=True)
    report_path = args.report_root / "fanos_v2_benchmark_report.md"
    env = collect_environment()

    quadratic_rows = read_csv(args.results_root / "quadratic_full.csv")
    vision_rows = read_csv(args.results_root / "vision_full.csv")
    eeg_rows = read_csv(args.results_root / "eeg_full.csv")

    content = f"""# FANoS-v2 Benchmark Report

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

Duration: {(finished - started) / 60.0:.2f} minutes

## Configuration

```json
{json.dumps(vars(args), indent=2, default=str)}
```

## Environment

```json
{json.dumps(env, indent=2)}
```

## Command Status

```json
{json.dumps(status, indent=2)}
```

## Quadratic

{markdown_table(quadratic_rows)}

## Vision

{markdown_table(vision_rows)}

## EEGBCI Cross-Subject

{markdown_table(eeg_rows)}

## Notes

- Data root: `{args.data_root}`
- Results root: `{args.results_root}`
- This report records benchmark outputs; it does not imply FANoS-v2 wins unless the tables show it across meaningful seeds and settings.
- For paper-quality claims, rerun with multiple seeds and hardware profiling.
"""
    report_path.write_text(content)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizers", nargs="+", default=["fanosv2", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--quadratic-steps", type=int, default=None)
    parser.add_argument("--quadratic-dim", type=int, default=None)
    parser.add_argument("--vision-epochs", type=int, default=None)
    parser.add_argument("--vision-train-samples", type=int, default=None)
    parser.add_argument("--vision-test-samples", type=int, default=None)
    parser.add_argument("--eeg-epochs", type=int, default=None)
    parser.add_argument("--eeg-train-subjects", type=int, nargs="+", default=None)
    parser.add_argument("--eeg-test-subject", type=int, default=None)
    parser.add_argument("--eeg-runs", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()
    collect_environment.requested_device = args.device

    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.report_root = args.report_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    log_root = args.results_root / "logs"

    if args.profile == "smoke":
        args.quadratic_steps = args.quadratic_steps or 50
        args.quadratic_dim = args.quadratic_dim or 64
        args.vision_epochs = args.vision_epochs or 1
        args.vision_train_samples = args.vision_train_samples or 512
        args.vision_test_samples = args.vision_test_samples or 256
        args.eeg_epochs = args.eeg_epochs or 1
        args.eeg_train_subjects = args.eeg_train_subjects or [1]
        args.eeg_test_subject = args.eeg_test_subject or 2
    else:
        args.quadratic_steps = args.quadratic_steps or 2000
        args.quadratic_dim = args.quadratic_dim or 2048
        args.vision_epochs = args.vision_epochs or 5
        args.vision_train_samples = args.vision_train_samples or 60000
        args.vision_test_samples = args.vision_test_samples or 10000
        args.eeg_epochs = args.eeg_epochs or 10
        args.eeg_train_subjects = args.eeg_train_subjects or [1, 2, 3, 4]
        args.eeg_test_subject = args.eeg_test_subject or 5

    started = time.perf_counter()
    status: dict[str, int] = {}
    py = sys.executable

    if not args.skip_download:
        status["fetch_mnist"] = run_command(
            [py, "tools/fetch_datasets.py", "--dataset", "mnist", "--data-root", str(args.data_root)],
            log_root / "fetch_mnist.log",
            args.dry_run,
        )
        for subject in sorted(set(args.eeg_train_subjects + [args.eeg_test_subject])):
            status[f"fetch_eegbci_s{subject}"] = run_command(
                [
                    py,
                    "tools/fetch_datasets.py",
                    "--dataset",
                    "eegbci",
                    "--subject",
                    str(subject),
                    "--runs",
                    *[str(run) for run in args.eeg_runs],
                    "--data-root",
                    str(args.data_root),
                ],
                log_root / f"fetch_eegbci_s{subject}.log",
                args.dry_run,
            )

    status["quadratic"] = run_command(
        [
            py,
            "benchmarks/quadratic_compare.py",
            "--steps",
            str(args.quadratic_steps),
            "--dim",
            str(args.quadratic_dim),
            "--out",
            str(args.results_root / "quadratic_full.csv"),
        ],
        log_root / "quadratic.log",
        args.dry_run,
    )

    status["vision"] = run_command(
        [
            py,
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
            *args.optimizers,
            "--out",
            str(args.results_root / "vision_full.csv"),
        ],
        log_root / "vision.log",
        args.dry_run,
    )

    status["eeg"] = run_command(
        [
            py,
            "benchmarks/eeg_eegbci_benchmark.py",
            "--train-subjects",
            *[str(subject) for subject in args.eeg_train_subjects],
            "--test-subject",
            str(args.eeg_test_subject),
            "--runs",
            *[str(run) for run in args.eeg_runs],
            "--epochs",
            str(args.eeg_epochs),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--data-root",
            str(args.data_root),
            "--optimizers",
            *args.optimizers,
            "--out",
            str(args.results_root / "eeg_full.csv"),
        ],
        log_root / "eeg.log",
        args.dry_run,
    )

    finished = time.perf_counter()
    report = write_report(args, status, started, finished)
    print(f"\nReport written to: {report}")
    return 0 if all(code == 0 for code in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
