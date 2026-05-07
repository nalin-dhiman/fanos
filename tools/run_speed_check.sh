#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./fanos_virtualenv/bin/python"
DEVICE="${1:-mps}"
PROFILE_DIR="../results/profiles/speed_check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DIR"

"$PY" -m pytest

"$PY" -m cProfile -o "$PROFILE_DIR/fanos_exact.prof" \
  benchmarks/vision_benchmark.py \
  --dataset mnist \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device "$DEVICE" \
  --optimizers fanosv2 \
  --fanos-preset auto \
  --out "$PROFILE_DIR/fanos_exact.csv"

"$PY" -m cProfile -o "$PROFILE_DIR/fanos_fast_sync.prof" \
  benchmarks/vision_benchmark.py \
  --dataset mnist \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device "$DEVICE" \
  --optimizers fanosv2 \
  --fanos-preset auto \
  --fanos-thermostat-interval 8 \
  --fanos-grad-norm-interval 8 \
  --no-fanos-sanitize-gradients \
  --out "$PROFILE_DIR/fanos_fast_sync.csv"

"$PY" -m cProfile -o "$PROFILE_DIR/adamw.prof" \
  benchmarks/vision_benchmark.py \
  --dataset mnist \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device "$DEVICE" \
  --optimizers adamw \
  --out "$PROFILE_DIR/adamw.csv"

"$PY" - <<PY
import csv
import pstats
from pathlib import Path

root = Path("$PROFILE_DIR")
print("\\n### Run metrics")
for path in ["fanos_exact.csv", "fanos_fast_sync.csv", "adamw.csv"]:
    with (root / path).open(newline="") as f:
        row = next(csv.DictReader(f))
    print(
        f"{path:22s} top1={float(row['top1']):.4f} "
        f"seconds={float(row['seconds']):.4f} state_mb={float(row['state_mb']):.4f}"
    )

for name in ["fanos_exact", "fanos_fast_sync", "adamw"]:
    print(f"\\n### {name} profile")
    pstats.Stats(str(root / f"{name}.prof")).strip_dirs().sort_stats("cumtime").print_stats(25)
PY

echo
echo "Profiles written to: $PROFILE_DIR"
