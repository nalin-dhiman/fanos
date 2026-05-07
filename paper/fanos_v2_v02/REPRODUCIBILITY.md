# FANoS-v2 v0.2 Reproducibility Notes

This package is a draft research release package, not a claim of production
readiness.

## Final Vision Evidence

The final v0.2 fast-preset validation uses these local result folders:

- `../results/fanosv2fast_default_mnist_mps_20260507_222936`
- `../results/fanosv2fast_default_fashionmnist_mps_20260507_223017`
- `../results/fanosv2fast_default_cifar10_mps_20260507_223058`

Regenerate paper tables and plots:

```bash
cd /Users/nalin/Downloads/fanos_v2/fanos_v2_project
MPLCONFIGDIR=/private/tmp/matplotlib ./fanos_virtualenv/bin/python tools/build_arxiv_package.py
```

Rerun the final validation:

```bash
cd /Users/nalin/Downloads/fanos_v2/fanos_v2_project

for dataset in mnist fashionmnist cifar10; do
  ROOT="../results/fanosv2fast_default_${dataset}_mps_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ROOT"
  for seed in 0 1 2 3 4; do
    ./fanos_virtualenv/bin/python benchmarks/vision_benchmark.py \
      --dataset "$dataset" \
      --epochs 5 \
      --train-samples 10000 \
      --test-samples 2000 \
      --device mps \
      --seed "$seed" \
      --optimizers adamw fanosv2fast \
      --fanos-preset auto \
      --out "$ROOT/seed_${seed}.csv"
  done
done
```

## Final Claim

Safe:

```text
FANoS-v2 shows repeated-seed accuracy gains on lightweight vision benchmarks,
while remaining materially slower than AdamW.
```

Unsafe:

```text
FANoS-v2 is a universal optimizer.
FANoS-v2 is a drop-in AdamW replacement.
FANoS-v2 is broadly proven for PINNs or physics problems.
```

## Current Blocking Issue

The current implementation still has optimizer-step overhead. The next v0.3
engineering task is a dense diagonal fast path using foreach-style tensor
operations and fewer per-parameter Python conversions.

