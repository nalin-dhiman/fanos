# FANoS-v2

FANoS-v2 is a PyTorch optimizer for experiments with feedback-controlled momentum on stiff objectives. It is not a default replacement for AdamW. The goal of this implementation is consistency, stability instrumentation, and a clear path toward lower-memory variants.

## Install

```bash
pip install fanos
```

For editable development:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv fanos_env
source fanos_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

The checked local environment uses Python 3.13.5 and PyTorch 2.8.0. TensorFlow is not required for the PyTorch optimizer core. Add it separately only for TensorFlow-specific experiments.

## Quickstart

```python
import torch
from fanos import FANoS
from fanos_v2 import FANoSV2, FANoSV2Fast

model = torch.nn.Linear(10, 1)
opt = FANoS(model.parameters(), lr=1e-3, grad_clip=1.0)

x = torch.randn(64, 10)
y = torch.randn(64, 1)

loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
opt.step()
opt.zero_grad()

print(opt.diagnostics()[0])
```

For the current best general guardrails, use:

```python
opt = FANoSV2(model.parameters(), lr=1e-3, grad_clip=1.0, preset="auto")
```

`preset="auto"` keeps the standard parameter-unit update, starts with low momentum, delays thermostat damping, and lets the RMS preconditioner soften only when the feedback controller sees unstable update energy. It is meant as a safer general preset, not a replacement for task-specific tuning.

## Core Update

FANoS-v2 defaults to an update buffer `u` in parameter units:

```text
pre_g = g / (sqrt(s) + eps)
rho = momentum * exp(-lr * zeta)
u = rho * u - lr * pre_g
theta = theta + u
```

The thermostat compares update energy with a target proposed-step energy and adjusts non-negative friction `zeta` using a clipped log-ratio controller.
The RMS preconditioner uses bias correction by default, which makes early steps much less brittle when `beta2` is close to one.
For residual-heavy scientific objectives such as PINNs, use `preconditioner_power < 1` or `preconditioner_power=0` to avoid over-normalizing PDE and boundary-loss gradients.
The `preset="auto"` path keeps `preconditioner_power=1.0` for ordinary training, but enables adaptive softening when the previous thermostat error is large. This avoided the sequence-memory stall in smoke tests while preserving normal image-classification startup.

For paper-equation audits, use:

```python
opt = FANoSV2(model.parameters(), lr=1e-3, update_mode="physical")
```

That mode stores a descent velocity `v` and applies:

```text
v = rho * v + pre_g
theta = theta - lr * v
```

The default `update_mode="parameter"` is recommended for public training because it removes the old `theta += v` versus `theta += lr*v` ambiguity.

See [docs/math.md](docs/math.md) for the mathematical notes.

## Efficiency Options

- `preconditioner="diag"`: full diagonal RMS state plus update buffer.
- `preconditioner="factored"`: row/column second-moment factors for matrix-like tensors.
- `preconditioner="none"`: feedback momentum without RMS preconditioning.
- `state_dtype=torch.bfloat16`: optional lower-precision optimizer state.
- `adaptive_lr=True`: optional gradient-stability learning-rate modulation with `lr_bounds`.

Use `optimizer.state_size_bytes()` to estimate tensor-state memory.

The package also exports experimental memory/communication helpers:

```python
from fanos_v2 import (
    low_rank_approximation,
    quantize_4bit,
    dequantize_4bit,
    sparsify_topk,
    densify_topk,
    dynamic_variance_clip,
)
```

These are intended for benchmark and distributed-training experiments. They are deliberately separate from the optimizer step so convergence behavior stays auditable.

## Examples

```bash
python examples/rosenbrock_demo.py
python tools/fetch_datasets.py --dataset mnist
python tools/fetch_datasets.py --dataset fashionmnist
python tools/fetch_datasets.py --dataset cifar10
python tools/fetch_datasets.py --dataset eegbci --subject 1 --runs 3 4
python tools/fetch_datasets.py --dataset eegbci --subject 2 --runs 3 4
python benchmarks/quadratic_compare.py --steps 500
python benchmarks/vision_benchmark.py --epochs 1 --train-samples 512 --test-samples 256
python benchmarks/vision_benchmark.py --dataset fashionmnist --epochs 1 --train-samples 1024 --test-samples 512 --optimizers fanosv2 adamw --fanos-preset auto
python benchmarks/eeg_eegbci_benchmark.py --train-subjects 1 --test-subject 2 --runs 3 4 --epochs 1
pytest
```

## One-Command Benchmark Sweep

From `fanos_v2_project`:

```bash
./fanos_virtualenv/bin/python tools/run_all_benchmarks.py --profile full --device auto
```

This will fetch missing datasets into `../datasets`, write CSVs/logs into `../results`, and generate:

```text
../reports/fanos_v2_benchmark_report.md
```

The default full profile runs:

- quadratic benchmark: 2048 dimensions, 2000 steps
- MNIST benchmark: 60,000 train samples, 10,000 test samples, 5 epochs
- EEGBCI benchmark: train subjects 1-4, test subject 5, runs 3 and 4, 10 epochs

For a faster check:

```bash
./fanos_virtualenv/bin/python tools/run_all_benchmarks.py --profile smoke
```

## Full Research Run

This is the one-command runner for leaving the machine overnight. It can fetch datasets, run MNIST, FashionMNIST, CIFAR-10, stiff objectives, the PINN preset, optional EEG, and build reports.

```bash
./fanos_virtualenv/bin/python tools/run_full_research_study.py \
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

Use `--skip-download` after the datasets are already present. Add `eeg` to `--blocks` if you also want the EEGBCI study in the same run.

For a quick command preview without running:

```bash
./fanos_virtualenv/bin/python tools/run_full_research_study.py --dry-run
```

## Overnight Study

This is the better command for serious tuning evidence. It repeats seeds, compares baselines against several fixed FANoS presets, and writes aggregate mean/std tables:

```bash
./fanos_virtualenv/bin/python tools/run_night_study.py \
  --tasks vision eeg \
  --seeds 0 1 2 3 4 \
  --configs low_lr auto stable vision_sweep_best eeg_sweep_best \
  --device cpu \
  --vision-dataset mnist \
  --vision-epochs 5 \
  --vision-train-samples 60000 \
  --vision-test-samples 10000 \
  --eeg-epochs 10
```

It writes:

```text
../results/night_study/night_study_raw.csv
../results/night_study/night_study_summary.csv
../reports/fanos_night_study_report.md
```

For GPU or accelerator auto-detection:

```bash
./fanos_virtualenv/bin/python tools/run_all_benchmarks.py --profile full --device auto
```

For Apple Silicon, use `--device mps` or `--device auto`. In this checked Mac environment, PyTorch reports `mps_built=True` but `mps_available=False`, so the runners currently fall back to CPU. Verify with:

```bash
./fanos_virtualenv/bin/python - <<'PY'
import torch
print(torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
PY
```

Use `--skip-download` to resume after datasets are already present.

For the current speed bottleneck, compare exact FANoS, fast-sync FANoS, and AdamW on the same small run:

```bash
bash tools/run_speed_check.sh mps
```

The fast-sync path uses `--fanos-thermostat-interval 8`, `--fanos-grad-norm-interval 8`, and `--no-fanos-sanitize-gradients`. Treat it as a performance candidate until its accuracy has been revalidated on repeated seeds.

For the real optimizer refactor path, compare the exact reference optimizer against the opt-in `fanosv2fast` class:

```bash
bash tools/run_fast_refactor_check.sh mps mnist
```

`fanosv2fast` keeps `FANoSV2` untouched and uses faster training defaults: `preset="auto"`, no adaptive LR, no gradient clipping, thermostat updates every 4 steps, and diagnostics off by default. Treat it as an experimental speed preset until it is validated outside the lightweight vision suite.

For optimizer experiments that intentionally remove gradient-norm scalar synchronization, pass `--fanos-grad-clip 0 --no-fanos-adaptive-lr`. This is an accuracy-risky speed test, not a recommended default.

Large benchmark targets such as ResNet-50, ViT-S, Llama-60m, HMC, and ADFTD should live in separate reproducible experiment configs with fixed seeds, exact datasets, hardware notes, and baseline sweeps. The current repository includes the optimizer core and lightweight sanity benchmarks only.

See [docs/benchmarking.md](docs/benchmarking.md) for dataset and benchmark details.

## Current Smoke Results

These are tiny CPU smoke runs, not claims of superiority.

MNIST subset, one epoch, 512 train samples, 256 test samples:

```text
fanosv2  loss=2.6516 top1=0.129 time=0.13s state=0.808MiB
adamw    loss=2.2398 top1=0.168 time=0.11s state=0.808MiB
sgd      loss=2.2936 top1=0.137 time=0.11s state=0.404MiB
rmsprop  loss=1.2470 top1=0.598 time=0.11s state=0.808MiB
```

EEGBCI train subject 1, test subject 2, one epoch:

```text
fanosv2  loss=2.1143 top1=0.500 time=0.03s state=0.553MiB
adamw    loss=0.7926 top1=0.500 time=0.01s state=0.553MiB
```

The 10-seed MNIST CPU study now shows `low_lr` FANoS ahead of AdamW on mean top-1, but slower per run:

```text
FANoS low_lr      top1_mean=0.9899  seconds_mean=70.9
AdamW baseline    top1_mean=0.9879  seconds_mean=65.0
RMSProp baseline  top1_mean=0.9817  seconds_mean=65.0
SGD baseline      top1_mean=0.9675  seconds_mean=63.2
```

Critical interpretation: this is a real positive signal on MNIST, not proof of a universal optimizer. FANoS-v2 is strongest today on Rosenbrock/stiff nonconvex tests, competitive on MNIST after tuning, repaired on the sequence-memory smoke with warmup, and promising for PINNs only with the softer `pinn` preset. EEGBCI and ill-conditioned quadratics remain weak or inconclusive.

## Reproducibility Checklist

- Set random seeds in each experiment.
- Report learning-rate sweeps, not only the best run.
- Log `zeta`, `rho`, update energy, target energy, gradient norm, and clip scale.
- Compare against AdamW with gradient clipping for serious claims.
- Report wall-clock time, peak memory, and energy-to-target when hardware counters are available.
- For EEG tasks such as HMC or ADFTD, report dataset split protocol, preprocessing, model architecture, and seed-level confidence intervals.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
