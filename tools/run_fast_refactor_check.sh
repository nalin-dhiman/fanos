#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./fanos_virtualenv/bin/python"
DEVICE="${1:-mps}"
DATASET="${2:-mnist}"
ROOT="../results/fast_refactor_${DATASET}_${DEVICE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/profiles"

"$PY" -m pytest

"$PY" -m cProfile -o "$ROOT/profiles/fanosv2.prof" \
  benchmarks/vision_benchmark.py \
  --dataset "$DATASET" \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device "$DEVICE" \
  --optimizers fanosv2 \
  --fanos-preset auto \
  --out "$ROOT/profiles/fanosv2.csv"

"$PY" -m cProfile -o "$ROOT/profiles/fanosv2fast.prof" \
  benchmarks/vision_benchmark.py \
  --dataset "$DATASET" \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device "$DEVICE" \
  --optimizers fanosv2fast \
  --out "$ROOT/profiles/fanosv2fast.csv"

"$PY" -m cProfile -o "$ROOT/profiles/adamw.prof" \
  benchmarks/vision_benchmark.py \
  --dataset "$DATASET" \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device "$DEVICE" \
  --optimizers adamw \
  --out "$ROOT/profiles/adamw.csv"

for seed in 0 1 2 3 4; do
  "$PY" benchmarks/vision_benchmark.py \
    --dataset "$DATASET" \
    --epochs 5 \
    --train-samples 10000 \
    --test-samples 2000 \
    --device "$DEVICE" \
    --seed "$seed" \
    --optimizers adamw fanosv2 fanosv2fast \
    --fanos-preset auto \
    --out "$ROOT/seed_${seed}.csv"
done

"$PY" - "$ROOT" <<'PY'
import csv
import math
import pstats
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])

print("\n### Single-seed profile metrics")
for name in ["fanosv2", "fanosv2fast", "adamw"]:
    with (root / "profiles" / f"{name}.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    print(
        f"{name:12s} top1={float(row['top1']):.4f} "
        f"seconds={float(row['seconds']):.4f} state_mb={float(row['state_mb']):.4f}"
    )

print("\n### Repeated-seed metrics")
rows_by_opt = defaultdict(list)
for path in sorted(root.glob("seed_*.csv")):
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows_by_opt[row["optimizer"]].append(row)

def mean(values):
    return sum(values) / max(1, len(values))

def std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))

summary = {}
for opt, rows in sorted(rows_by_opt.items()):
    top1 = [float(r["top1"]) for r in rows]
    seconds = [float(r["seconds"]) for r in rows]
    summary[opt] = {"top1": mean(top1), "seconds": mean(seconds)}
    print(
        f"{opt:12s} top1={mean(top1):.4f} +/- {std(top1):.4f} "
        f"seconds={mean(seconds):.2f} +/- {std(seconds):.2f}"
    )

if "adamw" in summary:
    adamw = summary["adamw"]
    print("\n### Deltas vs AdamW")
    for opt in ["fanosv2", "fanosv2fast"]:
        if opt not in summary:
            continue
        acc_delta = 100.0 * (summary[opt]["top1"] - adamw["top1"])
        time_delta = 100.0 * (summary[opt]["seconds"] / adamw["seconds"] - 1.0)
        print(f"{opt:12s} top1_delta={acc_delta:+.3f}pp time_delta={time_delta:+.1f}%")

for name in ["fanosv2", "fanosv2fast", "adamw"]:
    print(f"\n### {name} profile")
    pstats.Stats(str(root / "profiles" / f"{name}.prof")).strip_dirs().sort_stats("cumtime").print_stats(20)
PY

echo
echo "Fast refactor check written to: $ROOT"
