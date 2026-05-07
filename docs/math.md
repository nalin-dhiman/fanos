# FANoS-v2 Mathematical Notes

FANoS-v2 is a feedback-controlled momentum optimizer. The recommended implementation state is an update buffer `u` in parameter units:

```text
s_t = beta2 s_{t-1} + (1 - beta2) g_t^2
hat{s}_t = s_t / (1 - beta2^t)
pre_g_t = g_t / (sqrt(hat{s}_t) + eps)^alpha
rho_t = rho_max exp(-eta zeta_t)
u_{t+1} = rho_t u_t - eta pre_g_t
theta_{t+1} = theta_t + u_{t+1}
```

This makes `eta` the ordinary learning rate. The first step is comparable to AdamW or SGD rather than being implicitly scaled by `eta^2`. The second-moment estimate is bias-corrected by default; without this correction, early FANoS steps can be much too large when `beta2` is close to one.

The exponent `alpha = preconditioner_power` controls how strongly the RMS state shapes the update:

- `alpha=1`: full RMS preconditioning.
- `alpha=0.5`: softer RMS preconditioning; this is the best current PINN setting in the Poisson-1D test.
- `alpha=0`: raw-gradient feedback momentum.

This is important for PINNs and residual-based scientific losses, where full RMS normalization can distort the balance between PDE, boundary, and data terms. In the current Poisson-1D test, `alpha=0.5` is much better than full RMS, while an overly conservative raw-gradient PINN setting underfits.

## Auto Preset

The experimental `preset="auto"` is the current bridge from hand-tuned modes toward a general optimizer. It does not try to identify the task by name. Instead it applies startup guardrails and adapts preconditioning from recent feedback:

```text
rho_max,t ramps from low momentum to the requested momentum
zeta_t is held fixed during thermostat warmup
alpha_t = alpha_target - c max(0, |previous_log_error| - 1)
alpha_t is clipped to [alpha_min, alpha_max]
```

In code, `alpha_t` is `preconditioner_power_effective` in the diagnostics. The current default auto bounds are `[0.5, 1.0]`, with target `1.0`. This means ordinary image and sequence tasks start with full RMS-style preconditioning, while unstable energy feedback can soften toward `0.5`.

This is deliberately conservative. The PINN result shows why: Poisson-1D prefers a consistently softer `alpha=0.5`, while short classification and sequence tasks are harmed if auto starts too soft. The honest current rule is:

- Use `preset="auto"` for general experiments and smoke testing.
- Use the explicit PINN configuration, or set `preconditioner_power=0.5`, for residual-heavy scientific objectives.
- Treat adaptive alpha as a first version of generalization logic, not as a solved task detector.

For direct comparison to the paper-style semi-implicit physical velocity, the implementation also supports:

```text
rho_t = rho_max exp(-eta zeta_t)
v_{t+1} = rho_t v_t + pre_g_t
theta_{t+1} = theta_t - eta v_{t+1}
```

This mode is selected with `update_mode="physical"`. It is useful for equation audits and ablations. The parameter-unit mode is the default because it makes the code path unambiguous for deep-learning users.

The thermostat observes mean update energy:

```text
K_t = mean(u_t^2)
K*_t = alpha mean((eta pre_g_t)^2) + eps
```

It updates friction using a clipped log-ratio controller:

```text
e_t = clip(log((EMA(K_t) + eps) / (EMA(K*_t) + eps)), -e_max, e_max)
zeta_{t+1} = clip((1 - lambda) zeta_t + gamma e_t, zeta_min, zeta_max)
```

The default bounds are non-negative, so the thermostat damps excess update energy instead of injecting energy through negative friction. Signed friction can be explored by passing a negative lower bound, but it is not the recommended public default.

## Practical Defaults

- `lr`: tune like AdamW, usually start in the `1e-4` to `3e-3` range for neural nets.
- `momentum`: base momentum before adaptive damping, default `0.85`.
- `target_scale`: thermostat energy target, default `0.10`.
- `thermostat_lr`: thermostat controller gain, default `0.003`.
- `preconditioner_power`: RMS preconditioner strength, default `1.0`.
- `adaptive_preconditioner_power`: when enabled, softens the effective RMS exponent after large thermostat energy errors.
- `thermostat_warmup_steps`: delay friction updates while energy EMAs initialize, useful for PINNs and sequence tasks.
- `thermostat_interval`: update thermostat diagnostics every N steps. Values above one reduce scalar synchronization overhead on accelerators, but should be validated because damping reacts more slowly.
- `grad_norm_interval`: reuse the last scalar gradient norm for several steps. This reduces accelerator-to-host synchronization in clipping/adaptive-LR paths, but makes clipping and adaptive LR slightly stale between refreshes.
- `sanitize_gradients=False`: skips per-step `isfinite` replacement for speed. Only use this when upstream training already guards against nonfinite gradients or when running controlled benchmarks.
- `record_diagnostics=False`: disables per-step diagnostic snapshot creation for training-only runs. This reduces Python overhead without changing the tensor update.
- `FANoSV2Fast`: experimental wrapper that keeps exact `FANoSV2` intact and opts into faster runtime defaults for benchmark validation.
- `grad_clip`: recommended for stiff or noisy objectives.
- `preconditioner="diag"`: research default, Adam-like memory.
- `preconditioner="factored"`: lower-memory matrix state using row and column second-moment factors.
- `state_dtype=torch.bfloat16`: optional low-memory state for large models; validate before serious runs.
- `adaptive_lr=True`: optional learning-rate damping when the current gradient norm rises above its EMA.

## Memory and Communication Tools

The package includes helpers for research experiments:

- `low_rank_approximation`: truncated SVD for matrix-like tensors.
- `quantize_4bit` and `dequantize_4bit`: symmetric packed signed 4-bit tensor storage.
- `sparsify_topk` and `densify_topk`: top-k gradient communication simulation.
- `dynamic_variance_clip`: elementwise clipping from a variance estimate.

These tools are not silently applied inside the default optimizer. Use them explicitly in benchmark or distributed-training wrappers so the mathematical behavior of the core step stays visible.

## Stability Guardrails

- Gradients are clipped before updating second-moment or thermostat statistics.
- Friction is bounded by projection.
- The controller uses log energy ratios for scale invariance.
- Diagnostics expose `zeta`, effective momentum `rho`, update energy, target energy, gradient norm, and clip scale.
- Auto diagnostics also expose the effective `preconditioner_power` used for the step.

## Critical Notes

The paper-style helper

```text
thermostat_control(momentum, temp, target_temp)
```

implements the raw proportional temperature correction requested in the prompt. The optimizer itself uses the bounded log-energy controller by default because the raw ratio can flip sign or over-damp when `temp / target_temp` is large. Use the helper for ablations; use the default controller for public training runs unless an experiment explicitly studies signed or raw damping.
