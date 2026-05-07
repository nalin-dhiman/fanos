#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./fanos_virtualenv/bin/python"
RESULTS_DIR="../results"
REPORT_DIR="../reports"

if [[ $# -gt 0 ]]; then
  RUN_ROOT="$1"
else
  RUN_ROOT="$(ls -dt "$RESULTS_DIR"/full_research_mps_fixed_* 2>/dev/null | head -n 1)"
fi

if [[ -z "${RUN_ROOT:-}" || ! -d "$RUN_ROOT" ]]; then
  echo "No fixed full-research result folder found. Pass one explicitly, e.g.:"
  echo "bash tools/run_v02_cleanup.sh ../results/full_research_mps_fixed_YYYYMMDD_HHMMSS"
  exit 1
fi

echo "Using run root: $RUN_ROOT"

"$PY" -m pytest

"$PY" tools/run_night_study.py \
  --tasks eeg \
  --seeds 0 1 2 3 4 \
  --configs low_lr auto stable eeg_sweep_best \
  --device mps \
  --eeg-epochs 10 \
  --eeg-train-subjects 1 2 3 4 \
  --eeg-test-subject 5 \
  --eeg-runs 3 4 \
  --data-root ../datasets \
  --results-root "$RUN_ROOT/eeg_5seed" \
  --report-root "$REPORT_DIR"

"$PY" tools/build_decision_report.py \
  --results-root "$RUN_ROOT" \
  --report-root "$REPORT_DIR" \
  --report-name fanos_v2_v0.2_evidence_report.md

PROFILE_DIR="$RESULTS_DIR/profiles/$(basename "$RUN_ROOT")"
mkdir -p "$PROFILE_DIR"

"$PY" -m cProfile -o "$PROFILE_DIR/fanos_mnist.prof" \
  benchmarks/vision_benchmark.py \
  --dataset mnist \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device mps \
  --optimizers fanosv2 \
  --fanos-preset auto \
  --out "$PROFILE_DIR/fanos_mnist.csv"

"$PY" -m cProfile -o "$PROFILE_DIR/adamw_mnist.prof" \
  benchmarks/vision_benchmark.py \
  --dataset mnist \
  --epochs 1 \
  --train-samples 10000 \
  --test-samples 1000 \
  --device mps \
  --optimizers adamw \
  --out "$PROFILE_DIR/adamw_mnist.csv"

"$PY" - <<PY
import pstats
from pathlib import Path

profile_dir = Path("$PROFILE_DIR")
for name in ["fanos_mnist", "adamw_mnist"]:
    print(f"\\n### {name}")
    pstats.Stats(str(profile_dir / f"{name}.prof")).strip_dirs().sort_stats("cumtime").print_stats(40)
PY

echo
echo "Clean report: $REPORT_DIR/fanos_v2_v0.2_evidence_report.md"
echo "Profiles: $PROFILE_DIR"
