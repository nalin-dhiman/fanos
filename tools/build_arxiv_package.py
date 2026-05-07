"""Build the FANoS-v2 v0.2 paper package from local experiment CSVs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results"
PACKAGE = ROOT / "paper" / "fanos_v2_v02"
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"
DATA = PACKAGE / "data"

FINAL_FAST_ROOTS = {
    "MNIST": RESULTS / "fanosv2fast_default_mnist_mps_20260507_222936",
    "Fashion-MNIST": RESULTS / "fanosv2fast_default_fashionmnist_mps_20260507_223017",
    "CIFAR-10": RESULTS / "fanosv2fast_default_cifar10_mps_20260507_223058",
}

GRADNORM_ROOTS = {
    "MNIST k=1": RESULTS / "gradnorm_sweep_k1_mps_20260506_223104",
    "MNIST k=2": RESULTS / "gradnorm_sweep_k2_mps_20260506_223156",
    "MNIST k=4": RESULTS / "gradnorm_sweep_k4_mps_20260506_223238",
    "MNIST k=8": RESULTS / "gradnorm_sweep_k8_mps_20260506_223316",
    "Fashion-MNIST k=2": RESULTS / "gradnorm_fashionmnist_k2_mps_20260507_213549",
    "Fashion-MNIST k=4": RESULTS / "gradnorm_fashionmnist_k4_mps_20260507_213633",
    "CIFAR-10 k=2": RESULTS / "gradnorm_cifar10_k2_mps_20260507_213715",
    "CIFAR-10 k=4": RESULTS / "gradnorm_cifar10_k4_mps_20260507_213814",
}

FULL_RESEARCH = RESULTS / "full_research_mps_fixed_20260505_110606"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def read_seed_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("seed_*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["optimizer"]].append(row)

    summary: dict[str, dict[str, float]] = {}
    for optimizer, opt_rows in grouped.items():
        top1 = [float(row["top1"]) for row in opt_rows]
        seconds = [float(row["seconds"]) for row in opt_rows]
        loss = [float(row["loss"]) for row in opt_rows]
        summary[optimizer] = {
            "n": float(len(opt_rows)),
            "top1_mean": mean(top1),
            "top1_std": std(top1),
            "seconds_mean": mean(seconds),
            "seconds_std": std(seconds),
            "loss_mean": mean(loss),
            "loss_std": std(loss),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def tex_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_final_fast_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, root in FINAL_FAST_ROOTS.items():
        summary = summarize_rows(read_seed_rows(root))
        adamw = summary["adamw"]
        fanos = summary["fanosv2fast"]
        rows.append(
            {
                "dataset": dataset,
                "adamw_top1_mean": adamw["top1_mean"],
                "adamw_top1_std": adamw["top1_std"],
                "adamw_seconds_mean": adamw["seconds_mean"],
                "adamw_seconds_std": adamw["seconds_std"],
                "fanos_top1_mean": fanos["top1_mean"],
                "fanos_top1_std": fanos["top1_std"],
                "fanos_seconds_mean": fanos["seconds_mean"],
                "fanos_seconds_std": fanos["seconds_std"],
                "delta_top1_pp": 100.0 * (fanos["top1_mean"] - adamw["top1_mean"]),
                "time_delta_pct": 100.0 * (fanos["seconds_mean"] / adamw["seconds_mean"] - 1.0),
                "source": str(root.relative_to(ROOT.parent)),
            }
        )
    return rows


def build_gradnorm_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, root in GRADNORM_ROOTS.items():
        summary = summarize_rows(read_seed_rows(root))
        adamw = summary["adamw"]
        fanos = summary["fanosv2fast"]
        dataset, k_label = label.rsplit(" ", 1)
        rows.append(
            {
                "dataset": dataset,
                "k": k_label.replace("k=", ""),
                "fanos_top1_mean": fanos["top1_mean"],
                "fanos_top1_std": fanos["top1_std"],
                "fanos_seconds_mean": fanos["seconds_mean"],
                "fanos_seconds_std": fanos["seconds_std"],
                "delta_top1_pp": 100.0 * (fanos["top1_mean"] - adamw["top1_mean"]),
                "time_delta_pct": 100.0 * (fanos["seconds_mean"] / adamw["seconds_mean"] - 1.0),
                "source": str(root.relative_to(ROOT.parent)),
            }
        )
    return rows


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_final_table(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & AdamW top-1 & FANoSFast top-1 & $\Delta$ top-1 & AdamW s & FANoSFast s & $\Delta$ time \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & "
            f"{pct(float(row['adamw_top1_mean']))} $\\pm$ {pct(float(row['adamw_top1_std']))} & "
            f"{pct(float(row['fanos_top1_mean']))} $\\pm$ {pct(float(row['fanos_top1_std']))} & "
            f"{tex_num(float(row['delta_top1_pp']), 2)} pp & "
            f"{tex_num(float(row['adamw_seconds_mean']), 2)} & "
            f"{tex_num(float(row['fanos_seconds_mean']), 2)} & "
            f"{tex_num(float(row['time_delta_pct']), 1)}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "vision_fast_results.tex").write_text("\n".join(lines) + "\n")


def write_gradnorm_table(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Interval & FANoSFast top-1 & $\Delta$ top-1 & FANoSFast s & $\Delta$ time \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['k']} & "
            f"{pct(float(row['fanos_top1_mean']))} $\\pm$ {pct(float(row['fanos_top1_std']))} & "
            f"{tex_num(float(row['delta_top1_pp']), 2)} pp & "
            f"{tex_num(float(row['fanos_seconds_mean']), 2)} & "
            f"{tex_num(float(row['time_delta_pct']), 1)}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "gradnorm_ablation.tex").write_text("\n".join(lines) + "\n")


def write_scientific_tables() -> None:
    pinn = read_summary_csv(FULL_RESEARCH / "pinn_preset_summary.csv")
    stiff = read_summary_csv(FULL_RESEARCH / "stiff_auto_summary.csv")
    eeg = read_summary_csv(FULL_RESEARCH / "eeg_5seed" / "night_study_summary.csv")

    pinn_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Optimizer & residual metric & loss & seconds \\",
        r"\midrule",
    ]
    for row in pinn:
        pinn_lines.append(
            f"{row['optimizer']} & {float(row['metric_mean']):.3e} & "
            f"{float(row['loss_mean']):.3e} & {float(row['seconds_mean']):.2f} \\\\"
        )
    pinn_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "pinn_preliminary.tex").write_text("\n".join(pinn_lines) + "\n")

    selected_tasks = {
        "rosenbrock100",
        "noisy_small_regression",
        "ode_exp_fit",
        "poisson_pinn_1d",
        "sequence_memory",
    }
    stiff_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Task & Optimizer & metric & seconds \\",
        r"\midrule",
    ]
    for row in stiff:
        if row["task"] in selected_tasks and row["optimizer"] in {"fanosv2", "adamw", "rmsprop"}:
            stiff_lines.append(
                f"{row['task'].replace('_', '\\_')} & {row['optimizer']} & "
                f"{float(row['metric_mean']):.3e} & {float(row['seconds_mean']):.2f} \\\\"
            )
    stiff_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "scientific_preliminary.tex").write_text("\n".join(stiff_lines) + "\n")

    eeg_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & top-1 & $\Delta$ top-1 & $\Delta$ time \\",
        r"\midrule",
    ]
    for row in eeg:
        if row["optimizer"] == "fanosv2" and row["config"] != "auto":
            continue
        if row["optimizer"] not in {"fanosv2", "adamw"}:
            continue
        name = "FANoS auto" if row["optimizer"] == "fanosv2" else "AdamW"
        eeg_lines.append(
            f"{name} & {pct(float(row['top1_mean']))} $\\pm$ {pct(float(row['top1_std']))} & "
            f"{100.0 * float(row['delta_top1_vs_adamw']):.2f} pp & "
            f"{float(row['delta_seconds_pct_vs_adamw']):.1f}\\% \\\\"
        )
    eeg_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "eeg_preliminary.tex").write_text("\n".join(eeg_lines) + "\n")

    write_csv(DATA / "pinn_preliminary.csv", pinn, list(pinn[0].keys()))
    write_csv(DATA / "stiff_preliminary.csv", stiff, list(stiff[0].keys()))
    write_csv(DATA / "eeg_preliminary.csv", eeg, list(eeg[0].keys()))


def plot_final_results(rows: list[dict[str, object]]) -> None:
    datasets = [str(row["dataset"]) for row in rows]
    delta_top1 = [float(row["delta_top1_pp"]) for row in rows]
    time_delta = [float(row["time_delta_pct"]) for row in rows]

    plt.figure(figsize=(7.0, 4.0))
    bars = plt.bar(datasets, delta_top1, color=["#2a9d8f", "#457b9d", "#6a4c93"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Top-1 delta vs AdamW (percentage points)")
    plt.title("FANoSV2Fast accuracy delta on final 5-seed vision runs")
    for bar, value in zip(bars, delta_top1):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:+.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES / "vision_accuracy_delta.pdf")
    plt.savefig(FIGURES / "vision_accuracy_delta.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.0))
    bars = plt.bar(datasets, time_delta, color=["#e76f51", "#f4a261", "#8d99ae"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Runtime delta vs AdamW (%)")
    plt.title("FANoSV2Fast runtime overhead on final 5-seed vision runs")
    for bar, value in zip(bars, time_delta):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"+{value:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES / "vision_runtime_delta.pdf")
    plt.savefig(FIGURES / "vision_runtime_delta.png", dpi=180)
    plt.close()


def plot_gradnorm(rows: list[dict[str, object]]) -> None:
    mnist = [row for row in rows if row["dataset"] == "MNIST"]
    ks = [int(row["k"]) for row in mnist]
    acc = [float(row["delta_top1_pp"]) for row in mnist]
    runtime = [float(row["time_delta_pct"]) for row in mnist]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax1.plot(ks, acc, marker="o", color="#2a9d8f", label="Top-1 delta")
    ax1.set_xlabel("Gradient-norm refresh interval")
    ax1.set_ylabel("Top-1 delta vs AdamW (pp)", color="#2a9d8f")
    ax1.tick_params(axis="y", labelcolor="#2a9d8f")
    ax2 = ax1.twinx()
    ax2.plot(ks, runtime, marker="s", color="#e76f51", label="Runtime delta")
    ax2.set_ylabel("Runtime delta vs AdamW (%)", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")
    plt.title("MNIST scalar-sync ablation")
    fig.tight_layout()
    plt.savefig(FIGURES / "gradnorm_interval_ablation.pdf")
    plt.savefig(FIGURES / "gradnorm_interval_ablation.png", dpi=180)
    plt.close()


def write_manifest() -> None:
    manifest = [
        "# FANoS-v2 v0.2 Paper Package Manifest",
        "",
        "Generated artifacts are derived from local CSV result folders under `../results`.",
        "",
        "## Final vision sources",
    ]
    for dataset, root in FINAL_FAST_ROOTS.items():
        manifest.append(f"- {dataset}: `{root.relative_to(ROOT.parent)}`")
    manifest.extend(["", "## Ablation sources"])
    for label, root in GRADNORM_ROOTS.items():
        manifest.append(f"- {label}: `{root.relative_to(ROOT.parent)}`")
    manifest.extend(
        [
            "",
            "## Preliminary non-vision sources",
            f"- Full fixed research root: `{FULL_RESEARCH.relative_to(ROOT.parent)}`",
            "",
            "## Files",
            "- `fanos_v2_v02.tex`: arXiv-style manuscript.",
            "- `figures/*.pdf`: paper figures.",
            "- `tables/*.tex`: generated LaTeX tables.",
            "- `data/*.csv`: aggregated CSV inputs for audit.",
            "- `REPRODUCIBILITY.md`: commands and claim boundaries.",
            "- `ARXIV_SUBMISSION_CHECKLIST.md`: package/submission checklist.",
        ]
    )
    (PACKAGE / "MANIFEST.md").write_text("\n".join(manifest) + "\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    final_rows = build_final_fast_summary()
    gradnorm_rows = build_gradnorm_summary()

    write_csv(
        DATA / "vision_fast_summary.csv",
        final_rows,
        [
            "dataset",
            "adamw_top1_mean",
            "adamw_top1_std",
            "adamw_seconds_mean",
            "adamw_seconds_std",
            "fanos_top1_mean",
            "fanos_top1_std",
            "fanos_seconds_mean",
            "fanos_seconds_std",
            "delta_top1_pp",
            "time_delta_pct",
            "source",
        ],
    )
    write_csv(
        DATA / "gradnorm_ablation_summary.csv",
        gradnorm_rows,
        [
            "dataset",
            "k",
            "fanos_top1_mean",
            "fanos_top1_std",
            "fanos_seconds_mean",
            "fanos_seconds_std",
            "delta_top1_pp",
            "time_delta_pct",
            "source",
        ],
    )
    write_final_table(final_rows)
    write_gradnorm_table(gradnorm_rows)
    write_scientific_tables()
    plot_final_results(final_rows)
    plot_gradnorm(gradnorm_rows)
    write_manifest()
    print(f"paper package data written to {PACKAGE}")


if __name__ == "__main__":
    main()
