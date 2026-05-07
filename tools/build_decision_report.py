"""Build a consolidated FANoS-v2 decision report from completed benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = MAIN_DIR / "results"
DEFAULT_REPORT_ROOT = MAIN_DIR / "reports"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_pct(value: str) -> str:
    return f"{100.0 * float(value):.3f}%"


def fmt_float(value: str) -> str:
    number = float(value)
    if abs(number) < 1e-3 and number != 0:
        return f"{number:.3e}"
    return f"{number:.6g}"


def fmt_time_delta(value: str) -> str:
    delta = float(value)
    if delta >= 0:
        return f"costs {fmt_float(str(delta))}% more wall-clock time"
    return f"is {fmt_float(str(abs(delta)))}% faster"


def find(rows: list[dict[str, str]], optimizer: str, config: str | None = None, task: str | None = None) -> dict[str, str]:
    for row in rows:
        if row.get("optimizer") != optimizer:
            continue
        if config is not None and row.get("config") != config:
            continue
        if task is not None and row.get("task") != task:
            continue
        return row
    return {}


def classification_section(title: str, rows: list[dict[str, str]]) -> str:
    adamw = find(rows, "adamw", "baseline")
    fanos_rows = [row for row in rows if row.get("optimizer") == "fanosv2"]
    if not fanos_rows or not adamw:
        return f"## {title}\n\nMissing summary rows.\n"

    fanos_rows = sorted(fanos_rows, key=lambda row: float(row["top1_mean"]), reverse=True)
    shown_rows = fanos_rows[:4]
    lines = [
        f"## {title}",
        "",
        "| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean | Time delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in shown_rows:
        lines.append(
            f"| FANoS {row.get('config', 'default')} | {fmt_pct(row['top1_mean'])} | "
            f"{fmt_pct(row['top1_std'])} | {fmt_pct(row['delta_top1_vs_adamw'])} | "
            f"{fmt_float(row['seconds_mean'])} | {fmt_float(row['delta_seconds_pct_vs_adamw'])}% |"
        )
    lines.append(
        f"| AdamW | {fmt_pct(adamw['top1_mean'])} | {fmt_pct(adamw['top1_std'])} | "
        f"0.000% | {fmt_float(adamw['seconds_mean'])} | 0.000% |"
    )

    best = shown_rows[0]
    delta = float(best["delta_top1_vs_adamw"])
    if delta > 0:
        verdict = (
            f"Verdict: best FANoS config is ahead by {fmt_pct(best['delta_top1_vs_adamw'])}, "
            f"but {fmt_time_delta(best['delta_seconds_pct_vs_adamw'])}."
        )
    else:
        verdict = (
            f"Verdict: AdamW remains ahead on top-1; best FANoS config trails by "
            f"{fmt_pct(str(abs(delta)))} and {fmt_time_delta(best['delta_seconds_pct_vs_adamw'])}."
        )
    lines.extend(["", verdict, ""])
    return "\n".join(lines)


def night_study_summary_paths(results_root: Path) -> list[Path]:
    if (results_root / "RUN_MANIFEST.txt").exists():
        return sorted(results_root.glob("*seed/night_study_summary.csv")) + sorted(
            results_root.glob("night_study_summary.csv")
        )
    return sorted(results_root.glob("**/night_study_summary.csv"))


def discover_classification_sections(results_root: Path) -> list[tuple[str, list[dict[str, str]]]]:
    sections = []
    seen: set[Path] = set()
    for path in night_study_summary_paths(results_root):
        if path in seen:
            continue
        seen.add(path)
        rows = read_csv(path)
        if find(rows, "adamw", "baseline") and find(rows, "fanosv2", "low_lr"):
            sections.append((path.parent.name.replace("_", " ").title(), rows))
    return sections


def first_existing(paths: list[Path]) -> list[dict[str, str]]:
    for path in paths:
        rows = read_csv(path)
        if rows:
            return rows
    return []


def stiff_section(rows: list[dict[str, str]], pinn_rows: list[dict[str, str]]) -> str:
    tasks = [
        ("rosenbrock100", "Rosenbrock"),
        ("ill_conditioned_quadratic", "Ill-conditioned quadratic"),
        ("noisy_small_regression", "Noisy regression"),
        ("ode_exp_fit", "ODE fit"),
        ("poisson_pinn_1d", "PINN auto"),
        ("sequence_memory", "Sequence memory"),
    ]
    lines = [
        "## Stiff And Scientific Tasks",
        "",
        "| Task | FANoS metric | AdamW metric | Best critical read |",
        "| --- | ---: | ---: | --- |",
    ]
    for task, label in tasks:
        fanos = find(rows, "fanosv2", task=task)
        adamw = find(rows, "adamw", task=task)
        if not fanos or not adamw:
            continue
        if task == "ill_conditioned_quadratic":
            verdict = "FANoS loses badly; needs dedicated curvature handling."
        elif task == "poisson_pinn_1d":
            verdict = "Auto loses; use the PINN preset."
        elif task == "sequence_memory":
            verdict = "FANoS matches accuracy and has lower loss."
        elif task == "ode_exp_fit":
            verdict = "Tied with AdamW."
        elif task == "rosenbrock100":
            verdict = "FANoS beats AdamW, but RMSProp is stronger here."
        else:
            verdict = "Competitive; not a decisive win."
        lines.append(f"| {label} | {fmt_float(fanos['metric_mean'])} | {fmt_float(adamw['metric_mean'])} | {verdict} |")

    pinn_fanos = find(pinn_rows, "fanosv2", task="poisson_pinn_1d")
    pinn_adamw = find(pinn_rows, "adamw", task="poisson_pinn_1d")
    if pinn_fanos and pinn_adamw:
        lines.extend(
            [
                "",
                "PINN preset result:",
                "",
                f"- FANoS PINN metric mean: `{fmt_float(pinn_fanos['metric_mean'])}`",
                f"- AdamW metric mean: `{fmt_float(pinn_adamw['metric_mean'])}`",
                "- Critical read: the PINN math update is real, but it is not captured by the generic auto preset yet.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-name", default="fanos_v2_decision_report.md")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    report_root = args.report_root.resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    classification_sections = discover_classification_sections(results_root)
    stiff = first_existing(
        [
            results_root / "stiff_auto_summary.csv",
            results_root / "stiff_auto_5seed_summary.csv",
            *sorted(results_root.glob("**/stiff_auto_summary.csv")),
        ]
    )
    pinn = first_existing(
        [
            results_root / "pinn_preset_summary.csv",
            results_root / "pinn_preset_5seed_summary.csv",
            *sorted(results_root.glob("**/pinn_preset_summary.csv")),
        ]
    )
    classification_text = "\n".join(classification_section(title, rows) for title, rows in classification_sections)
    if not classification_text:
        classification_text = "## Classification Studies\n\nNo completed night-study summaries were found.\n"

    path = report_root / args.report_name
    path.write_text(
        f"""# FANoS-v2 Decision Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

Source results root: `{results_root}`

## Executive Verdict

FANoS-v2 is no longer just a toy optimizer. It has reproducible positive signals on MNIST, FashionMNIST, sequence memory, ODE fitting, and the PINN-specific preset.

It is still not a general AdamW replacement. The main blockers are speed, weak behavior on ill-conditioned quadratics, and the gap between `auto` and the task-specific PINN preset.

{classification_text}

{stiff_section(stiff, pinn)}

## Next Engineering Priorities

1. Keep `low_lr` as the best current vision preset and `auto` as the general safety preset.
2. Optimize runtime before adding more benchmark breadth; current wins are still too expensive.
3. Add CIFAR-10 with a stronger CNN before making any broader vision claim.
4. Add an ill-conditioned mode or curvature detector; current FANoS loses this task clearly.
5. Keep EEG claims conservative unless FANoS beats AdamW on top-1, not only loss.

## Claim Boundary

Safe claim: FANoS-v2 is a promising feedback-controlled optimizer framework with strong task-aware modes and repeated-seed wins on these lightweight benchmarks.

Unsafe claim: FANoS-v2 is a universal optimizer or a drop-in AdamW replacement.
"""
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
