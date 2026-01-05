
"""
FANoS-NHC: FANoS with a Nose-Hoover chain (NHC) thermostat.

This is an *upgrade* for robustness: multiple friction variables per group can reduce "ringing"
and improve multi-timescale stability.

This file implements a simple, one-gradient-per-step chain update (explicit in zetas).
For production-grade molecular dynamics, more sophisticated reversible integrators exist; for
optimization, this simple scheme is a good starting point when combined with:
- EMA smoothing of T_inst
- friction clipping
- small chain length (K=2 or 3)

Usage:
    opt = FANoSChain(model.parameters(), chain_length=2,
                     Q=(1.0, 10.0),  # Q1 fast, Q2 slow
                     T0_max=1e-3, ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import math
import torch


@dataclass
class TemperatureSchedule:
    Tmax: float = 1e-3
    Tmin: float = 0.0
    tau: float = 20000.0

    def __call__(self, step: int) -> float:
        if self.tau <= 0:
            return float(self.Tmin)
        return float(self.Tmin + (self.Tmax - self.Tmin) * math.exp(-step / self.tau))


class FANoSChain(torch.optim.Optimizer):
    """
    FANoS + Nose-Hoover chain thermostat (per parameter group).

    Dynamics:
        v <- v - lr*(g/m + zeta1*v)
        theta <- theta + lr*v

    Thermostat chain (explicit Euler-like):
        zeta1 <- zeta1 + lr*((T_ema - T0)/Q1 - zeta2*zeta1)
        zetaj <- zetaj + lr*((Q_{j-1}*zeta_{j-1}^2 - T0)/Qj - zeta_{j+1}*zetaj)
        zetaK <- zetaK + lr*((Q_{K-1}*zeta_{K-1}^2 - T0)/QK)

    Practical notes:
    - Choose K small (2 or 3).
    - Q1 small-ish, Q2.. larger (slower).
    - Always clip zetas.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.999,
        eps: float = 1e-8,
        chain_length: int = 2,
        Q: Sequence[float] = (1.0, 10.0),
        T0_max: float = 1e-3,
        T0_min: float = 0.0,
        tau: float = 20000.0,
        rho_T: float = 0.9,
        zeta_clip: float = 10.0,
        grad_clip: Optional[float] = 1.0,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
    ):
        if chain_length < 1:
            raise ValueError("chain_length must be >= 1")
        if len(Q) != chain_length:
            raise ValueError("Q must have length == chain_length")
        if any(q <= 0 for q in Q):
            raise ValueError("All Q values must be positive")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            chain_length=chain_length,
            Q=tuple(float(q) for q in Q),
            rho_T=rho_T,
            zeta_clip=zeta_clip,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

        self.schedule = TemperatureSchedule(Tmax=T0_max, Tmin=T0_min, tau=tau)
        self._step_count: int = 0

        for group in self.param_groups:
            group.setdefault("zetas", [0.0 for _ in range(chain_length)])
            group.setdefault("T_ema", float(T0_max))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        step = self._step_count
        T0 = self.schedule(step)

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["beta"]
            eps: float = group["eps"]
            K: int = group["chain_length"]
            Qs = group["Q"]
            rho_T: float = group["rho_T"]
            zeta_clip: float = group["zeta_clip"]
            grad_clip: Optional[float] = group["grad_clip"]
            weight_decay: float = group["weight_decay"]
            decoupled_wd: bool = group["decoupled_weight_decay"]

            zetas = [float(z) for z in group.get("zetas", [0.0] * K)]
            T_ema: float = float(group.get("T_ema", T0))

            # Per-group grad norm clipping
            if grad_clip is not None:
                sq = 0.0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("FANoSChain does not support sparse gradients")
                    sq += float(g.detach().pow(2).sum().item())
                gnorm = math.sqrt(sq) if sq > 0 else 0.0
                clip_scale = min(1.0, grad_clip / (gnorm + 1e-12))
            else:
                clip_scale = 1.0

            temp_sum = 0.0
            temp_count = 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("FANoSChain does not support sparse gradients")
                if not torch.isfinite(g).all():
                    continue

                # Weight decay
                if weight_decay != 0.0:
                    if decoupled_wd:
                        p.data.mul_(1.0 - lr * weight_decay)
                    else:
                        g = g.add(p.data, alpha=weight_decay)

                if clip_scale != 1.0:
                    g = g.mul(clip_scale)

                state = self.state[p]
                if len(state) == 0:
                    state["v"] = torch.zeros_like(p.data)
                    state["s"] = torch.zeros_like(p.data)

                v: torch.Tensor = state["v"]
                s: torch.Tensor = state["s"]

                s.mul_(beta).addcmul_(g, g, value=1.0 - beta)
                m = s.sqrt().add_(eps)

                # Velocity update uses only zeta1 as friction on v
                zeta1 = zetas[0]
                v.mul_(1.0 - lr * zeta1)
                v.addcdiv_(g, m, value=-lr)

                p.data.add_(v, alpha=lr)

                temp_sum += float((m * v * v).sum().item())
                temp_count += v.numel()

            if temp_count > 0:
                T_inst = temp_sum / float(temp_count)
                T_ema = rho_T * T_ema + (1.0 - rho_T) * T_inst

                # Chain thermostat updates (explicit Euler-like)
                # zeta1
                if K == 1:
                    zetas[0] = zetas[0] + lr * ((T_ema - T0) / Qs[0])
                else:
                    zetas[0] = zetas[0] + lr * (((T_ema - T0) / Qs[0]) - zetas[1] * zetas[0])
                    # middle zetas
                    for j in range(1, K - 1):
                        zetas[j] = zetas[j] + lr * (((Qs[j - 1] * zetas[j - 1] ** 2 - T0) / Qs[j]) - zetas[j + 1] * zetas[j])
                    # last zeta
                    zetas[K - 1] = zetas[K - 1] + lr * ((Qs[K - 2] * zetas[K - 2] ** 2 - T0) / Qs[K - 1])

                # Clip all zetas
                zetas = [max(-zeta_clip, min(zeta_clip, z)) for z in zetas]

            group["zetas"] = [float(z) for z in zetas]
            group["T_ema"] = float(T_ema)

        return loss
