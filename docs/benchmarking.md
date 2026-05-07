# Benchmarking Guide

The benchmark scripts are designed to be honest harnesses, not proof of superiority. They write data and results outside `fanos_v2_project` by default.

## Data Root

Default:

```text
../datasets
```

From this project, that resolves to:

```text
/Users/nalin/Downloads/fanos_v2/datasets
```

## Fetch Datasets

```bash
python tools/fetch_datasets.py --dataset mnist
python tools/fetch_datasets.py --dataset fashionmnist
python tools/fetch_datasets.py --dataset cifar10
python tools/fetch_datasets.py --dataset eegbci --subject 1 --runs 3 4
python tools/fetch_datasets.py --dataset eegbci --subject 2 --runs 3 4
```

The EEG starter uses PhysioNet EEGBCI through MNE. It is a public motor movement / imagery dataset suitable for validating EEG preprocessing, cross-subject evaluation code, and optimizer stability. It is not the same as HMC, `motor_mv_img`, or workload datasets.

## Vision Smoke Benchmark

```bash
python benchmarks/vision_benchmark.py \
  --epochs 1 \
  --train-samples 512 \
  --test-samples 256 \
  --optimizers fanosv2 adamw sgd rmsprop \
  --out ../results/vision_smoke.csv
```

FashionMNIST uses the same small CNN shape and is the first out-of-MNIST image check:

```bash
python benchmarks/vision_benchmark.py \
  --dataset fashionmnist \
  --epochs 1 \
  --train-samples 1024 \
  --test-samples 512 \
  --optimizers fanosv2 adamw \
  --lr 0.001 \
  --fanos-preset auto \
  --out ../results/fashion_auto_smoke.csv
```

CIFAR-10 uses a 3-channel version of the same small CNN:

```bash
python benchmarks/vision_benchmark.py \
  --dataset cifar10 \
  --epochs 1 \
  --train-samples 2048 \
  --test-samples 512 \
  --optimizers fanosv2 adamw \
  --lr 0.001 \
  --fanos-preset auto \
  --out ../results/cifar10_auto_smoke.csv
```

## Full Research Runner

Use this when you want the whole queue to run without launching the next command manually:

```bash
python tools/run_full_research_study.py \
  --blocks vision stiff pinn \
  --vision-datasets mnist fashionmnist cifar10 \
  --seeds 0 1 2 3 4 \
  --configs low_lr auto stable vision_sweep_best \
  --device mps \
  --vision-epochs 5 \
  --stiff-steps 2000 \
  --results-root ../results/full_research_mps \
  --report-root ../reports
```

Add `eeg` to `--blocks` for EEGBCI. Use `--dry-run` to print the exact queue and `--skip-download` once datasets are already present.

Metrics:

- final training loss
- top-1 accuracy
- wall-clock time
- optimizer-state memory
- peak GPU memory when CUDA is available

## EEG Cross-Subject Smoke Benchmark

```bash
python benchmarks/eeg_eegbci_benchmark.py \
  --train-subjects 1 \
  --test-subject 2 \
  --runs 3 4 \
  --epochs 1 \
  --optimizers fanosv2 adamw \
  --out ../results/eeg_smoke.csv
```

The script performs:

- MNE EDF loading
- EEG channel selection
- 7-30 Hz filtering
- T1/T2 event extraction
- epoch normalization
- train-subject to held-out-subject evaluation

## Larger Targets

ResNet-50, ViT-S, Llama-60M, HMC, workload, and other EEG datasets should be added as separate experiment configs with exact dataset links, preprocessing, seeds, batch sizes, hardware, and profiler settings. Do not infer those results from smoke tests.

## Repeated-Seed Comparison

Use the night-study runner for formal mean/std/best/worst and AdamW deltas:

```bash
python tools/run_night_study.py \
  --tasks vision \
  --vision-dataset mnist \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --configs low_lr auto stable vision_sweep_best \
  --device cpu \
  --vision-epochs 5 \
  --vision-train-samples 60000 \
  --vision-test-samples 10000 \
  --results-root ../results/vision_10seed_auto \
  --report-root ../reports
```

Then repeat with `--vision-dataset fashionmnist` before making any broader vision claim.

## Hardware Profiling

On CUDA systems, use:

```bash
python benchmarks/vision_benchmark.py --device cuda
```

On Apple Silicon, use:

```bash
python benchmarks/vision_benchmark.py --device mps
```

If PyTorch reports MPS unavailable, the helper falls back to CPU. Check with:

```bash
python - <<'PY'
import torch
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
PY
```

For publication-quality profiling, add PyTorch Profiler or Nsight traces and report:

- step time
- peak memory
- optimizer-state memory
- energy-to-target if power counters are available
