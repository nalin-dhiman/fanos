"""FANoS-v2 optimizer for PyTorch.

FANoS-v2 is a feedback-controlled momentum optimizer. It stores the momentum
state as a parameter-update buffer ``u`` and applies ``theta += u``. This makes
the learning rate a normal first-order step size and removes the velocity-unit
ambiguity in earlier FANoS sketches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Tuple
import math

import torch

Preset = Literal["default", "auto", "pinn"]
Preconditioner = Literal["diag", "factored", "none"]
UpdateMode = Literal["parameter", "physical"]


@dataclass(frozen=True)
class FANoSV2Diagnostics:
    """Snapshot of group-level thermostat and clipping diagnostics."""

    step: int
    group: int
    zeta: float
    rho: float
    update_energy: float
    target_energy: float
    log_error: float
    grad_norm: float
    clip_scale: float
    preconditioner: str
    preconditioner_power: float
    update_mode: str
    lr_effective: float


class FANoSV2(torch.optim.Optimizer):
    """Friction-adaptive, feedback-controlled momentum optimizer.

    Default update for one parameter group:

    ``s <- beta2*s + (1-beta2)*g^2``
    ``pre_g <- g / (sqrt(s) + eps)``
    ``rho <- momentum * exp(-lr*zeta)``
    ``u <- rho*u - lr*pre_g``
    ``theta <- theta + u``

    For paper-equation audits, pass ``update_mode="physical"``. That stores a
    descent velocity ``v`` and applies ``theta <- theta - lr*v``.

    The thermostat observes mean update energy ``mean(u^2)`` and compares it to
    a target based on the proposed preconditioned step energy
    ``target_scale * mean((lr*pre_g)^2)``. A clipped log-ratio controller updates
    non-negative friction ``zeta`` by default.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        preset: Preset = "default",
        lr: float = 1e-3,
        beta2: float = 0.999,
        eps: float = 1e-8,
        momentum: float = 0.85,
        thermostat_lr: float = 3e-3,
        thermostat_decay: float = 0.0,
        temp_beta: float = 0.95,
        target_scale: float = 0.10,
        zeta_bounds: Tuple[float, float] = (0.0, 50.0),
        max_log_error: float = 5.0,
        grad_clip: Optional[float] = None,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        preconditioner: Preconditioner = "diag",
        preconditioner_power: float = 1.0,
        adaptive_preconditioner_power: bool = False,
        preconditioner_power_bounds: Tuple[float, float] = (0.5, 1.0),
        preconditioner_power_warmup_steps: int = 0,
        preconditioner_power_instability_gain: float = 0.15,
        factored_min_dim: int = 2,
        state_dtype: Optional[torch.dtype] = None,
        bias_correction: bool = True,
        update_mode: UpdateMode = "parameter",
        adaptive_lr: Optional[bool] = None,
        lr_bounds: Optional[Tuple[float, float]] = None,
        lr_adapt_beta: float = 0.95,
        lr_adapt_gain: float = 0.25,
        warmup_steps: int = 0,
        warmup_start_momentum: float = 0.0,
        warmup_start_lr_scale: float = 1.0,
        thermostat_warmup_steps: int = 0,
        thermostat_interval: int = 1,
        grad_norm_interval: int = 1,
        sanitize_gradients: bool = True,
        record_diagnostics: bool = True,
        diagnostics_interval: int = 1,
    ):
        if preset not in {"default", "auto", "pinn"}:
            raise ValueError("preset must be 'default', 'auto', or 'pinn'")
        if preset == "auto":
            momentum = 0.85 if momentum == 0.85 else momentum
            target_scale = 0.10 if target_scale == 0.10 else target_scale
            thermostat_lr = 0.003 if thermostat_lr == 3e-3 else thermostat_lr
            adaptive_lr = True if adaptive_lr is None else adaptive_lr
            adaptive_preconditioner_power = True
            preconditioner_power_bounds = (0.5, 1.0)
            warmup_steps = 200 if warmup_steps == 0 else warmup_steps
            warmup_start_momentum = 0.0 if warmup_start_momentum == 0.0 else warmup_start_momentum
            thermostat_warmup_steps = 100 if thermostat_warmup_steps == 0 else thermostat_warmup_steps
        if preset == "pinn":
            lr = 5e-4 if lr == 1e-3 else lr
            momentum = 0.75 if momentum == 0.85 else momentum
            target_scale = 0.05 if target_scale == 0.10 else target_scale
            thermostat_lr = 0.001 if thermostat_lr == 3e-3 else thermostat_lr
            adaptive_lr = True if adaptive_lr is None else adaptive_lr
            preconditioner_power = 0.5 if preconditioner_power == 1.0 else preconditioner_power
            warmup_steps = 200 if warmup_steps == 0 else warmup_steps
            warmup_start_momentum = 0.0 if warmup_start_momentum == 0.0 else warmup_start_momentum
            thermostat_warmup_steps = 200 if thermostat_warmup_steps == 0 else thermostat_warmup_steps
        if adaptive_lr is None:
            adaptive_lr = False

        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if thermostat_lr < 0:
            raise ValueError("thermostat_lr must be non-negative")
        if not 0.0 <= thermostat_decay < 1.0:
            raise ValueError("thermostat_decay must be in [0, 1)")
        if not 0.0 <= temp_beta < 1.0:
            raise ValueError("temp_beta must be in [0, 1)")
        if target_scale <= 0:
            raise ValueError("target_scale must be positive")
        zmin, zmax = zeta_bounds
        if zmax < zmin:
            raise ValueError("require zeta_min <= zeta_max")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be positive when provided")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if preconditioner not in {"diag", "factored", "none"}:
            raise ValueError("preconditioner must be 'diag', 'factored', or 'none'")
        if preconditioner_power < 0:
            raise ValueError("preconditioner_power must be non-negative")
        amin, amax = preconditioner_power_bounds
        if amin < 0 or amax < amin:
            raise ValueError("preconditioner_power_bounds must satisfy 0 <= min <= max")
        if preconditioner_power_warmup_steps < 0:
            raise ValueError("preconditioner_power_warmup_steps must be non-negative")
        if preconditioner_power_instability_gain < 0:
            raise ValueError("preconditioner_power_instability_gain must be non-negative")
        if factored_min_dim < 2:
            raise ValueError("factored_min_dim must be >= 2")
        if update_mode not in {"parameter", "physical"}:
            raise ValueError("update_mode must be 'parameter' or 'physical'")
        if not 0.0 <= lr_adapt_beta < 1.0:
            raise ValueError("lr_adapt_beta must be in [0, 1)")
        if lr_adapt_gain < 0:
            raise ValueError("lr_adapt_gain must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= warmup_start_momentum < 1.0:
            raise ValueError("warmup_start_momentum must be in [0, 1)")
        if warmup_start_lr_scale <= 0:
            raise ValueError("warmup_start_lr_scale must be positive")
        if thermostat_warmup_steps < 0:
            raise ValueError("thermostat_warmup_steps must be non-negative")
        if thermostat_interval < 1:
            raise ValueError("thermostat_interval must be >= 1")
        if grad_norm_interval < 1:
            raise ValueError("grad_norm_interval must be >= 1")
        if diagnostics_interval < 1:
            raise ValueError("diagnostics_interval must be >= 1")
        if lr_bounds is not None:
            lr_min, lr_max = lr_bounds
            if lr_min <= 0 or lr_max < lr_min:
                raise ValueError("lr_bounds must satisfy 0 < min <= max")

        defaults = dict(
            preset=preset,
            lr=float(lr),
            beta2=float(beta2),
            eps=float(eps),
            momentum=float(momentum),
            thermostat_lr=float(thermostat_lr),
            thermostat_decay=float(thermostat_decay),
            temp_beta=float(temp_beta),
            target_scale=float(target_scale),
            zeta_bounds=(float(zmin), float(zmax)),
            max_log_error=float(max_log_error),
            grad_clip=grad_clip,
            weight_decay=float(weight_decay),
            decoupled_weight_decay=bool(decoupled_weight_decay),
            preconditioner=preconditioner,
            preconditioner_power=float(preconditioner_power),
            adaptive_preconditioner_power=bool(adaptive_preconditioner_power),
            preconditioner_power_bounds=(float(amin), float(amax)),
            preconditioner_power_warmup_steps=int(preconditioner_power_warmup_steps),
            preconditioner_power_instability_gain=float(preconditioner_power_instability_gain),
            factored_min_dim=int(factored_min_dim),
            state_dtype=state_dtype,
            bias_correction=bool(bias_correction),
            update_mode=update_mode,
            adaptive_lr=bool(adaptive_lr),
            lr_bounds=lr_bounds,
            lr_adapt_beta=float(lr_adapt_beta),
            lr_adapt_gain=float(lr_adapt_gain),
            warmup_steps=int(warmup_steps),
            warmup_start_momentum=float(warmup_start_momentum),
            warmup_start_lr_scale=float(warmup_start_lr_scale),
            thermostat_warmup_steps=int(thermostat_warmup_steps),
            thermostat_interval=int(thermostat_interval),
            grad_norm_interval=int(grad_norm_interval),
            sanitize_gradients=bool(sanitize_gradients),
            record_diagnostics=bool(record_diagnostics),
            diagnostics_interval=int(diagnostics_interval),
        )
        super().__init__(params, defaults)

        self._step_count = 0
        self._last_diagnostics: list[FANoSV2Diagnostics] = []
        for group in self.param_groups:
            group.setdefault("zeta", 0.0)
            group.setdefault("temp_ema", 0.0)
            group.setdefault("target_ema", 0.0)
            group.setdefault("grad_norm_ema", 0.0)
            group.setdefault("last_log_error", 0.0)
            group.setdefault("last_grad_norm", 0.0)
            group.setdefault("_grad_norm_initialized", False)
            group.setdefault("preconditioner_power_effective", float(group["preconditioner_power"]))
            group.setdefault("_thermostat_initialized", False)
            group.setdefault("_lr_initialized", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        diagnostics: list[FANoSV2Diagnostics] = []

        for group_idx, group in enumerate(self.param_groups):
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            lr = float(group["lr"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            momentum = float(group["momentum"])
            thermostat_lr = float(group["thermostat_lr"])
            thermostat_decay = float(group["thermostat_decay"])
            temp_beta = float(group["temp_beta"])
            target_scale = float(group["target_scale"])
            zmin, zmax = group["zeta_bounds"]
            max_log_error = float(group["max_log_error"])
            grad_clip = group["grad_clip"]
            weight_decay = float(group["weight_decay"])
            decoupled_wd = bool(group["decoupled_weight_decay"])
            preconditioner = group["preconditioner"]
            preconditioner_power = self._effective_preconditioner_power(group)
            factored_min_dim = int(group["factored_min_dim"])
            state_dtype = group["state_dtype"]
            update_mode = group["update_mode"]
            adaptive_lr = bool(group["adaptive_lr"])
            lr_bounds = group["lr_bounds"]
            lr_adapt_beta = float(group["lr_adapt_beta"])
            lr_adapt_gain = float(group["lr_adapt_gain"])
            warmup_steps = int(group["warmup_steps"])
            warmup_start_momentum = float(group["warmup_start_momentum"])
            warmup_start_lr_scale = float(group["warmup_start_lr_scale"])
            thermostat_warmup_steps = int(group["thermostat_warmup_steps"])
            thermostat_interval = int(group["thermostat_interval"])
            grad_norm_interval = int(group["grad_norm_interval"])
            sanitize_gradients = bool(group["sanitize_gradients"])
            record_diagnostics = bool(group["record_diagnostics"])
            diagnostics_interval = int(group["diagnostics_interval"])
            control_step = self._step_count % thermostat_interval == 0
            diagnostics_step = record_diagnostics and self._step_count % diagnostics_interval == 0

            grad_norm = 0.0
            needs_grad_norm = grad_clip is not None or adaptive_lr
            if needs_grad_norm:
                grad_norm = self._maybe_grad_norm(params, group, grad_norm_interval, sanitize_gradients)
            lr_effective = self._effective_lr(group, lr, grad_norm, adaptive_lr, lr_bounds, lr_adapt_beta, lr_adapt_gain)
            warmup = self._warmup_factor(self._step_count, warmup_steps)
            lr_effective *= warmup_start_lr_scale + (1.0 - warmup_start_lr_scale) * warmup
            momentum_effective = warmup_start_momentum + (momentum - warmup_start_momentum) * warmup
            clip_scale = 1.0
            if grad_clip is not None and grad_norm > 0.0:
                clip_scale = min(1.0, float(grad_clip) / (grad_norm + 1e-12))

            zeta = float(group.get("zeta", 0.0))
            rho = momentum_effective * math.exp(-lr_effective * zeta)
            rho = max(0.0, min(0.999, rho))

            update_energy_sum: Optional[torch.Tensor] = None
            target_energy_sum: Optional[torch.Tensor] = None
            count = 0

            for p in params:
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("FANoSV2 does not support sparse gradients")

                if sanitize_gradients:
                    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
                if clip_scale != 1.0:
                    grad = grad.mul(clip_scale)

                if weight_decay != 0.0:
                    if decoupled_wd:
                        p.mul_(1.0 - lr * weight_decay)
                    else:
                        grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state, preconditioner, factored_min_dim, state_dtype, update_mode)
                state["step"] += 1

                pre_grad = self._precondition(grad, state, group, preconditioner_power)

                if update_mode == "physical":
                    velocity = state["v"]
                    velocity.mul_(rho).add_(pre_grad.to(dtype=velocity.dtype), alpha=1.0)
                    p.add_(velocity.to(dtype=p.dtype), alpha=-lr_effective)
                    step_delta = velocity.to(dtype=torch.float32).mul(-lr_effective)
                else:
                    update = state["u"]
                    update.mul_(rho).add_(pre_grad.to(dtype=update.dtype), alpha=-lr_effective)
                    p.add_(update.to(dtype=p.dtype))
                    step_delta = update.to(dtype=torch.float32)

                if control_step:
                    update_energy = step_delta.pow(2).sum()
                    proposed_energy = (lr_effective * pre_grad).to(dtype=torch.float32).pow(2).sum()
                    update_energy_sum = update_energy if update_energy_sum is None else update_energy_sum + update_energy
                    target_energy_sum = proposed_energy if target_energy_sum is None else target_energy_sum + proposed_energy
                    count += p.numel()

            log_error = 0.0
            update_energy_value = 0.0
            target_energy_value = 0.0
            if count > 0 and update_energy_sum is not None and target_energy_sum is not None:
                update_energy_value = float(update_energy_sum.item()) / float(count)
                target_energy_value = target_scale * float(target_energy_sum.item()) / float(count) + eps

                initialized = bool(group.get("_thermostat_initialized", False))
                if initialized:
                    temp_ema = temp_beta * float(group["temp_ema"]) + (1.0 - temp_beta) * update_energy_value
                    target_ema = temp_beta * float(group["target_ema"]) + (1.0 - temp_beta) * target_energy_value
                else:
                    temp_ema = update_energy_value
                    target_ema = target_energy_value
                    group["_thermostat_initialized"] = True

                log_error = math.log((temp_ema + eps) / (target_ema + eps))
                log_error = max(-max_log_error, min(max_log_error, log_error))
                if self._step_count > thermostat_warmup_steps:
                    zeta = (1.0 - thermostat_decay) * zeta + thermostat_lr * log_error
                    zeta = max(float(zmin), min(float(zmax), zeta))

                group["zeta"] = float(zeta)
                group["temp_ema"] = float(temp_ema)
                group["target_ema"] = float(target_ema)
                group["last_log_error"] = float(log_error)

            if diagnostics_step:
                diagnostics.append(
                    FANoSV2Diagnostics(
                        step=self._step_count,
                        group=group_idx,
                        zeta=float(group.get("zeta", zeta)),
                        rho=float(rho),
                        update_energy=float(update_energy_value),
                        target_energy=float(target_energy_value),
                        log_error=float(log_error),
                        grad_norm=float(grad_norm),
                        clip_scale=float(clip_scale),
                        preconditioner=str(preconditioner),
                        preconditioner_power=float(preconditioner_power),
                        update_mode=str(update_mode),
                        lr_effective=float(lr_effective),
                    )
                )

        if diagnostics or any(bool(group["record_diagnostics"]) for group in self.param_groups):
            self._last_diagnostics = diagnostics
        return loss

    def diagnostics(self) -> list[FANoSV2Diagnostics]:
        """Return the most recent per-group diagnostic snapshots."""

        return list(self._last_diagnostics)

    def state_size_bytes(self) -> int:
        """Estimate optimizer tensor-state memory in bytes."""

        total = 0
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return int(total)

    @staticmethod
    def thermostat_control(momentum: torch.Tensor, temperature: float, target_temperature: float) -> torch.Tensor:
        """Paper-style scalar thermostat helper.

        This is provided for tests and educational use. The optimizer's default
        controller uses a bounded log-energy variant, which is more scale stable.
        """

        if target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        return momentum * (1.0 - float(temperature) / float(target_temperature))

    @staticmethod
    def fanos_update(theta: torch.Tensor, velocity: torch.Tensor, lr: float) -> torch.Tensor:
        """Functional semi-implicit paper update ``theta <- theta - lr*v``."""

        if lr <= 0:
            raise ValueError("lr must be positive")
        return theta - lr * velocity

    @staticmethod
    def _effective_lr(
        group: dict[str, Any],
        lr: float,
        grad_norm: float,
        adaptive_lr: bool,
        lr_bounds: Optional[Tuple[float, float]],
        beta: float,
        gain: float,
    ) -> float:
        if not adaptive_lr or grad_norm <= 0.0:
            return lr

        initialized = bool(group.get("_lr_initialized", False))
        if initialized:
            ema = beta * float(group["grad_norm_ema"]) + (1.0 - beta) * grad_norm
        else:
            ema = grad_norm
            group["_lr_initialized"] = True
        group["grad_norm_ema"] = float(ema)

        log_ratio = math.log((grad_norm + 1e-12) / (ema + 1e-12))
        lr_eff = lr * math.exp(-gain * log_ratio)
        if lr_bounds is not None:
            lr_min, lr_max = lr_bounds
            lr_eff = max(float(lr_min), min(float(lr_max), lr_eff))
        return float(lr_eff)

    @staticmethod
    def _warmup_factor(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return max(0.0, min(1.0, float(step) / float(warmup_steps)))

    def _effective_preconditioner_power(self, group: dict[str, Any]) -> float:
        target_power = float(group["preconditioner_power"])
        if not bool(group["adaptive_preconditioner_power"]):
            group["preconditioner_power_effective"] = target_power
            return target_power

        alpha_min, alpha_max = group["preconditioner_power_bounds"]
        alpha_min = float(alpha_min)
        alpha_max = float(alpha_max)
        warmup_steps = int(group["preconditioner_power_warmup_steps"])
        gain = float(group["preconditioner_power_instability_gain"])
        progress = self._warmup_factor(self._step_count, warmup_steps)
        scheduled = alpha_min + (min(target_power, alpha_max) - alpha_min) * progress
        instability = max(0.0, abs(float(group.get("last_log_error", 0.0))) - 1.0)
        alpha = scheduled - gain * instability
        alpha = max(alpha_min, min(alpha_max, alpha))
        group["preconditioner_power_effective"] = float(alpha)
        return float(alpha)

    @staticmethod
    def _maybe_grad_norm(
        params: list[torch.nn.Parameter],
        group: dict[str, Any],
        grad_norm_interval: int,
        sanitize_gradients: bool,
    ) -> float:
        initialized = bool(group.get("_grad_norm_initialized", False))
        should_refresh = not initialized or int(group.get("_grad_norm_step", 0)) % grad_norm_interval == 0
        group["_grad_norm_step"] = int(group.get("_grad_norm_step", 0)) + 1
        if should_refresh:
            grad_norm = FANoSV2._grad_norm(params, sanitize_gradients=sanitize_gradients)
            group["last_grad_norm"] = float(grad_norm)
            group["_grad_norm_initialized"] = True
            return float(grad_norm)
        return float(group.get("last_grad_norm", 0.0))

    @staticmethod
    def _grad_norm(params: list[torch.nn.Parameter], sanitize_gradients: bool = True) -> float:
        total: Optional[torch.Tensor] = None
        for p in params:
            grad = p.grad.detach()
            if grad.is_sparse:
                raise RuntimeError("FANoSV2 does not support sparse gradients")
            if sanitize_gradients:
                grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
            sq = grad.to(dtype=torch.float32).pow(2).sum()
            total = sq if total is None else total + sq
        if total is None:
            return 0.0
        return math.sqrt(float(total.item()))

    @staticmethod
    def _state_dtype_for(param: torch.Tensor, state_dtype: Optional[torch.dtype]) -> torch.dtype:
        if state_dtype is not None:
            return state_dtype
        return param.dtype if param.is_floating_point() else torch.float32

    def _init_state(
        self,
        param: torch.Tensor,
        state: dict[str, Any],
        preconditioner: Preconditioner,
        factored_min_dim: int,
        state_dtype: Optional[torch.dtype],
        update_mode: UpdateMode,
    ) -> None:
        dtype = self._state_dtype_for(param, state_dtype)
        if update_mode == "physical":
            state["v"] = torch.zeros_like(param, dtype=dtype, memory_format=torch.preserve_format)
        else:
            state["u"] = torch.zeros_like(param, dtype=dtype, memory_format=torch.preserve_format)
        state["step"] = 0
        if preconditioner == "diag":
            state["sq"] = torch.zeros_like(param, dtype=dtype, memory_format=torch.preserve_format)
            state["preconditioner_kind"] = "diag"
        elif preconditioner == "factored" and param.ndim >= factored_min_dim:
            state["row_sq"] = torch.zeros(param.shape[:-1], dtype=dtype, device=param.device)
            state["col_sq"] = torch.zeros(param.shape[-1], dtype=dtype, device=param.device)
            state["preconditioner_kind"] = "factored"
        elif preconditioner == "factored":
            state["sq"] = torch.zeros_like(param, dtype=dtype, memory_format=torch.preserve_format)
            state["preconditioner_kind"] = "diag"
        else:
            state["preconditioner_kind"] = "none"

    def _precondition(
        self,
        grad: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        preconditioner_power: float,
    ) -> torch.Tensor:
        beta2 = float(group["beta2"])
        eps = float(group["eps"])
        bias_correction = bool(group["bias_correction"])
        kind = state["preconditioner_kind"]
        step = int(state.get("step", 1))
        correction = 1.0
        if bias_correction and beta2 > 0.0:
            correction = max(1.0 - beta2**step, 1e-16)

        if kind == "none":
            return grad

        if kind == "diag":
            sq = state["sq"]
            grad_stats = grad.to(dtype=sq.dtype)
            sq.mul_(beta2).addcmul_(grad_stats, grad_stats, value=1.0 - beta2)
            sq_hat = sq.to(dtype=grad.dtype) / correction
            denom = sq_hat.sqrt().add_(eps)
            if preconditioner_power != 1.0:
                denom = denom.pow(preconditioner_power)
            return grad / denom

        row_sq = state["row_sq"]
        col_sq = state["col_sq"]
        grad_stats = grad.to(dtype=row_sq.dtype)
        grad2 = grad_stats.pow(2)
        row_sq.mul_(beta2).add_(grad2.mean(dim=-1), alpha=1.0 - beta2)
        reduce_dims = tuple(range(grad2.ndim - 1))
        col_sq.mul_(beta2).add_(grad2.mean(dim=reduce_dims), alpha=1.0 - beta2)

        row = (row_sq.to(dtype=grad.dtype) / correction).clamp_min(eps)
        col = (col_sq.to(dtype=grad.dtype) / correction).clamp_min(eps)
        normalizer = row.mean().clamp_min(eps)
        denom_sq = row.unsqueeze(-1) * col / normalizer
        denom = denom_sq.sqrt().add_(eps)
        if preconditioner_power != 1.0:
            denom = denom.pow(preconditioner_power)
        return grad / denom


class FANoSV2Fast(FANoSV2):
    """Experimental fast-training profile for FANoS-v2.

    This keeps :class:`FANoSV2` as the exact reference implementation and only
    changes opt-in runtime defaults that reduce scalar synchronization and
    Python diagnostics overhead. Use it for speed experiments, not as the
    evidence-report baseline until repeated-seed accuracy is revalidated.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs: Any):
        kwargs.setdefault("preset", "auto")
        kwargs.setdefault("adaptive_lr", False)
        kwargs.setdefault("grad_clip", None)
        kwargs.setdefault("thermostat_interval", 4)
        kwargs.setdefault("grad_norm_interval", 1)
        kwargs.setdefault("sanitize_gradients", True)
        kwargs.setdefault("record_diagnostics", False)
        super().__init__(params, **kwargs)
