"""One-command FANoS-v2 research runner.

This orchestrates dataset fetches, repeated-seed vision studies, stiff suites,
PINN-specific checks, optional EEG, and final report generation. It is intended
for overnight/manual runs where the user should not need to wake up and launch
the next benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT.parent
DEFAULT_DATA_ROOT = MAIN_DIR / "datasets"
DEFAULT_RESULTS_ROOT = MAIN_DIR / "results" / "full_research"
DEFAULT_REPORT_ROOT = MAIN_DIR / "reports"


def run(cmd: list[str], log_path: Path, dry_run: bool, stop_on_failure: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "$ " + " ".join(cmd)
    print(rendered)
    if dry_run:
        log_path.write_text(rendered + "\n[dry-run]\n")
        return 0

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    output = proc.stdout + f"\n[exit={proc.returncode} elapsed={elapsed:.2f}s]\n"
    log_path.write_text(output)
    print(proc.stdout)
    print(f"[exit={proc.returncode} elapsed={elapsed:.2f}s]")
    if stop_on_failure and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def run_fetches(args) -> None:
    if args.skip_download:
        return
    datasets = []
    if "vision" in args.blocks:
        datasets.extend(args.vision_datasets)
    for dataset in dict.fromkeys(datasets):
        run(
            [
                sys.executable,
                "tools/fetch_datasets.py",
                "--dataset",
                dataset,
                "--data-root",
                str(args.data_root),
            ],
            args.results_root / "logs" / f"fetch_{dataset}.log",
            args.dry_run,
            args.stop_on_failure,
        )

    if "eeg" in args.blocks:
        for subject in args.eeg_subjects:
            run(
                [
                    sys.executable,
                    "tools/fetch_datasets.py",
                    "--dataset",
                    "eegbci",
                    "--subject",
                    str(subject),
                    "--runs",
                    *[str(run_id) for run_id in args.eeg_runs],
                    "--data-root",
                    str(args.data_root),
                ],
                args.results_root / "logs" / f"fetch_eeg_subject{subject}.log",
                args.dry_run,
                args.stop_on_failure,
            )


def run_vision(args) -> None:
    if "vision" not in args.blocks:
        return
    for dataset in args.vision_datasets:
        train_samples = min(args.vision_train_samples, 50000) if dataset == "cifar10" else args.vision_train_samples
        run(
            [
                sys.executable,
                "tools/run_night_study.py",
                "--tasks",
                "vision",
                "--vision-dataset",
                dataset,
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--configs",
                *args.configs,
                "--device",
                args.device,
                "--vision-epochs",
                str(args.vision_epochs),
                "--vision-train-samples",
                str(train_samples),
                "--vision-test-samples",
                str(args.vision_test_samples),
                "--data-root",
                str(args.data_root),
                "--results-root",
                str(args.results_root / f"{dataset}_{len(args.seeds)}seed"),
                "--report-root",
                str(args.report_root),
            ],
            args.results_root / "logs" / f"vision_{dataset}.log",
            args.dry_run,
            args.stop_on_failure,
        )


def run_stiff(args) -> None:
    if "stiff" not in args.blocks:
        return
    run(
        [
            sys.executable,
            "benchmarks/stiff_suite.py",
            "--optimizers",
            "fanosv2",
            "adamw",
            "sgd",
            "rmsprop",
            "--fanos-preset",
            "auto",
            "--seeds",
            *[str(seed) for seed in args.stiff_seeds],
            "--steps",
            str(args.stiff_steps),
            "--device",
            args.device,
            "--out",
            str(args.results_root / "stiff_auto.csv"),
            "--summary-out",
            str(args.results_root / "stiff_auto_summary.csv"),
        ],
        args.results_root / "logs" / "stiff_auto.log",
        args.dry_run,
        args.stop_on_failure,
    )


def run_pinn(args) -> None:
    if "pinn" not in args.blocks:
        return
    run(
        [
            sys.executable,
            "benchmarks/stiff_suite.py",
            "--tasks",
            "poisson_pinn_1d",
            "--optimizers",
            "fanosv2",
            "adamw",
            "sgd",
            "rmsprop",
            "--fanos-preset",
            "pinn",
            "--seeds",
            *[str(seed) for seed in args.stiff_seeds],
            "--steps",
            str(args.stiff_steps),
            "--device",
            args.device,
            "--out",
            str(args.results_root / "pinn_preset.csv"),
            "--summary-out",
            str(args.results_root / "pinn_preset_summary.csv"),
        ],
        args.results_root / "logs" / "pinn_preset.log",
        args.dry_run,
        args.stop_on_failure,
    )


def run_eeg(args) -> None:
    if "eeg" not in args.blocks:
        return
    test_subject = args.eeg_test_subject
    train_subjects = [subject for subject in args.eeg_subjects if subject != test_subject]
    run(
        [
            sys.executable,
            "tools/run_night_study.py",
            "--tasks",
            "eeg",
            "--seeds",
            *[str(seed) for seed in args.eeg_seeds],
            "--configs",
            "low_lr",
            "auto",
            "stable",
            "eeg_sweep_best",
            "--device",
            args.device,
            "--eeg-epochs",
            str(args.eeg_epochs),
            "--eeg-train-subjects",
            *[str(subject) for subject in train_subjects],
            "--eeg-test-subject",
            str(test_subject),
            "--eeg-runs",
            *[str(run_id) for run_id in args.eeg_runs],
            "--data-root",
            str(args.data_root),
            "--results-root",
            str(args.results_root / f"eeg_{len(args.eeg_seeds)}seed"),
            "--report-root",
            str(args.report_root),
        ],
        args.results_root / "logs" / "eeg.log",
        args.dry_run,
        args.stop_on_failure,
    )


def write_manifest(args) -> Path:
    args.results_root.mkdir(parents=True, exist_ok=True)
    manifest = args.results_root / "RUN_MANIFEST.txt"
    manifest.write_text(
        "\n".join(
            [
                f"generated={time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"device={args.device}",
                f"blocks={' '.join(args.blocks)}",
                f"vision_datasets={' '.join(args.vision_datasets)}",
                f"seeds={' '.join(str(seed) for seed in args.seeds)}",
                f"configs={' '.join(args.configs)}",
                f"results_root={args.results_root}",
                f"report_root={args.report_root}",
                "",
            ]
        )
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", nargs="+", choices=["vision", "stiff", "pinn", "eeg"], default=["vision", "stiff", "pinn"])
    parser.add_argument("--vision-datasets", nargs="+", choices=["mnist", "fashionmnist", "cifar10"], default=["mnist", "fashionmnist", "cifar10"])
    parser.add_argument("--configs", nargs="+", default=["low_lr", "auto", "stable", "vision_sweep_best"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--vision-epochs", type=int, default=5)
    parser.add_argument("--vision-train-samples", type=int, default=60000)
    parser.add_argument("--vision-test-samples", type=int, default=10000)
    parser.add_argument("--stiff-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stiff-steps", type=int, default=2000)
    parser.add_argument("--eeg-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eeg-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--eeg-test-subject", type=int, default=5)
    parser.add_argument("--eeg-runs", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--eeg-epochs", type=int, default=10)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.report_root = args.report_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    manifest = write_manifest(args)
    print(f"Manifest: {manifest}")
    run_fetches(args)
    run_vision(args)
    run_stiff(args)
    run_pinn(args)
    run_eeg(args)

    run(
        [
            sys.executable,
            "tools/build_decision_report.py",
            "--results-root",
            str(args.results_root),
            "--report-root",
            str(args.report_root),
            "--report-name",
            f"fanos_v2_decision_report_{args.results_root.name}.md",
        ],
        args.results_root / "logs" / "decision_report.log",
        args.dry_run,
        False,
    )
    print(f"Done. Results root: {args.results_root}")
    print(f"Reports root: {args.report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
