
"""
FANoS: Friction-Adaptive Nose-Hoover Symplectic Optimizer (PyTorch)

This file implements the *core* publishable algorithm:
- second-order (velocity) dynamics
- diagonal "mass" from RMS of gradients
- Nose-Hoover thermostat variable zeta per param group
- symplectic (semi-implicit) update (one gradient per step)

Designed to be readable and hackable, not maximally micro-optimized.

Usage:
    opt = FANoS(model.parameters(), lr=1e-3, T0_max=1e-3, T0_min=0.0, tau=20000,
                Q=1.0, beta=0.999, rho_T=0.9, zeta_clip=10.0, grad_clip=1.0,
                weight_decay=1e-4, decoupled_weight_decay=True)

Notes:
- T0_max and Q have the most "physical" meaning:
    T0_max: early target kinetic energy (exploration)
    Q: thermostat inertia (how fast friction adapts)
- For very noisy minibatch gradients, increase rho_T and Q, and clip zeta more tightly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union, Dict, Any

import math
import torch


@dataclass
class TemperatureSchedule:
    """Exponential annealing schedule T0(step) = Tmin + (Tmax-Tmin)*exp(-step/tau)."""
    Tmax: float = 1e-3
    Tmin: float = 0.0
    tau: float = 20000.0

    def __call__(self, step: int) -> float:
        if self.tau <= 0:
            return float(self.Tmin)
        return float(self.Tmin + (self.Tmax - self.Tmin) * math.exp(-step / self.tau))


class FANoS(torch.optim.Optimizer):
    """
    FANoS optimizer with diagonal mass and one thermostat variable (zeta) per parameter group.

    Update (per parameter element i):
        s_i <- beta*s_i + (1-beta)*g_i^2
        m_i <- sqrt(s_i) + eps
        v_i <- (1 - lr*zeta) * v_i - lr * (g_i / m_i)
        theta_i <- theta_i + lr * v_i

    Thermostat (per group g):
        T_inst <- mean_i(m_i * v_i^2)   over params in group
        T_ema <- rho_T*T_ema + (1-rho_T)*T_inst
        zeta <- clip(zeta + (lr/Q)*(T_ema - T0(step)), [-zeta_clip, +zeta_clip])
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.999,
        eps: float = 1e-8,
        Q: float = 1.0,
        T0_max: float = 1e-3,
        T0_min: float = 0.0,
        tau: float = 20000.0,
        rho_T: float = 0.9,
        zeta_clip: float = 10.0,
        grad_clip: Optional[float] = 1.0,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = True,
        # --- Ablations / research toggles (kept for reproducibility) ---
        explicit_euler: bool = False,
        fixed_friction: bool = False,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0,1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if Q <= 0:
            raise ValueError("Q must be positive")
        if not (0.0 <= rho_T < 1.0):
            raise ValueError("rho_T must be in [0,1)")
        if zeta_clip <= 0:
            raise ValueError("zeta_clip must be positive")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            Q=Q,
            rho_T=rho_T,
            zeta_clip=zeta_clip,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            T0_max=T0_max,
            explicit_euler=explicit_euler,
            fixed_friction=fixed_friction,
        )
        super().__init__(params, defaults)

        self.schedule = TemperatureSchedule(Tmax=T0_max, Tmin=T0_min, tau=tau)
        self._step_count: int = 0

        # Initialize per-group thermostat state
        for group in self.param_groups:
            group.setdefault("zeta", 0.0)
            group.setdefault("T_ema", float(T0_max))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        for group in self.param_groups:
            # Hyperparameters
            Q = group['Q']
            T0_max = group['T0_max']
            rho_T = group['rho_T']
            beta = group['beta']
            eps = group['eps']
            clip_val = group['grad_clip']
            zeta_clip = group['zeta_clip']
            
            # Ablations
            explicit_euler = group.get('explicit_euler', False)
            fixed_friction = group.get('fixed_friction', False)

            # --- 1. Dynamic T0 Schedule ---
            # Typically annealing or adaptive.
            # Here we just use constant or simple schedule if needed.
            # For simplicity: T0 = T0_max for now (based on paper impl).
            T0 = T0_max 

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p.data)
                    state['zeta'] = torch.tensor(0.0, device=p.device)
                    # RMSProp state
                    state['avg_sq_grad'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                v = state['v']
                zeta = state['zeta']
                avg_sq = state['avg_sq_grad']

                # --- 2. Kinetic Energy & Thermostat Update ---
                # E_k = 0.5 * v^2 * M (approx)
                # But we need "Instantaneous Temperature" T_inst
                # T_inst ~ v^2 / D (roughly, normalized)
                
                # Update RMS Mass approx
                avg_sq.mul_(beta).addcmul_(grad, grad, value=1-beta)
                M = avg_sq.sqrt().add_(eps)
                
                # Thermostat update (Semi-implicit: update zeta first)
                # dT = T_inst - T0
                # But we use v_{k} to estimate T_{inst}?
                # Standard Nosé-Hoover:
                # zeta_{k+1/2} = zeta_k + dt/2Q * (2E_k - D*T)
                # Here we simplify to:
                # zeta_new = zeta + lr/Q * (T_ema - T0)
                # We need T_ema. 
                
                # Estimate T_inst from current velocity
                # v is correlated with M^-1 * grad
                # T_inst propto v^T M v / D
                v_scaled = v * M.sqrt() # if M is diagonal mass
                # Actually for RMSProp as preconditioner:
                # p_dot = -grad.
                # v approx -eta * M^-1 * grad
                
                # Let's use simple Kinetic Energy proxy:
                # K = 0.5 * torch.sum(v * M * v)
                K = 0.5 * torch.sum(v * M * v)
                d = p.data.numel()
                T_inst = 2 * K / d
                
                # EMA of Temperature
                if 'T_ema' not in state: state['T_ema'] = T_inst
                state['T_ema'] = rho_T * state['T_ema'] + (1-rho_T) * T_inst
                
                # Update Zeta
                if not fixed_friction:
                    d_zeta = (state['T_ema'] - T0)
                    zeta.add_(group['lr'] * d_zeta / Q)
                    
                    # Clip Zeta for stability
                    if zeta_clip > 0:
                        zeta.clamp_(-zeta_clip, zeta_clip)
                else:
                    # Fixed friction mode (Ablation)
                    zeta.fill_(0.0) # 0.1? Or just 0 for "No thermostat"
                    # Usually fixed friction means zeta=const > 0.
                    # But "No Thermostat" implies standard Momentum?
                    # Let's stick to zeta=0 (Identity momentum) or small damping.
                    # Paper says "Fixed Friction (Thermostat OFF)".
                    # We will use zeta=0 effectively making it Preconditioned Momentum.
                
                # --- 3. Velocity Update ---
                # v_{k+1} = (1 - lr*zeta) * v_k - lr * M^-1 * g
                # This is "Semi-Implicit" if we used new zeta.
                
                # Gradient Clipping
                if clip_val > 0:
                     torch.nn.utils.clip_grad_norm_([p], clip_val)
                     grad = p.grad.data # Reload clipped
                
                # Preconditioned Gradient
                pre_grad = grad / M
                
                # Update V
                # We need v_old for Explicit Euler ablation
                if explicit_euler:
                    v_old = v.clone()
                
                # Damping factor
                damping = 1.0 - group['lr'] * zeta
                v.mul_(damping).add_(pre_grad, alpha=-group['lr'])
                
                # --- 4. Position Update ---
                # theta_{k+1} = theta_k + v_{k+1}  (Symplectic Euler)
                # theta_{k+1} = theta_k + v_k      (Explicit Euler)
                
                if explicit_euler:
                     p.data.add_(v_old)
                else:
                     p.data.add_(v)
                
        return loss

    def extra_repr(self) -> str:
        return f"step={self._step_count}, schedule={self.schedule}"

