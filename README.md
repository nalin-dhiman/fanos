# FANoS Optimizer (PyTorch)

**FANoS** = *Friction-Adaptive Nosé–Hoover Symplectic momentum*.

This package provides a PyTorch `torch.optim.Optimizer` implementation of FANoS:
- semi-implicit (symplectic-Euler) momentum update
- a Nosé–Hoover-inspired thermostat variable that adapts friction using kinetic-energy feedback
- optional diagonal RMS “mass” (preconditioner) and optional global gradient clipping

The accompanying paper is included in the release repo; this library is just the clean optimizer code.

## Install

```bash
pip install fanos
```

## Quickstart

```python
import torch
from fanos import FANoS

model = torch.nn.Linear(10, 1)
opt = FANoS(model.parameters(), lr=1e-3, grad_clip=1.0)

x = torch.randn(64, 10)
y = torch.randn(64, 1)

loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
opt.step()
opt.zero_grad()
```

## Notes (read this before hype happens)

FANoS is a research optimizer. In the paper's reported protocols it:
- helps vs *unclipped* AdamW/RMSProp on Rosenbrock-100D,
- but is not a general replacement for strong baselines like AdamW + clipping,
- and can be unstable or high-variance on some problems without tuning.

So: treat it as a tool for experiments, not a default choice for production.

## Citation

Add the paper citation from `CITATION.cff` in the repo.


[![PyPI version](https://badge.fury.io/py/fanos.svg)](https://pypi.org/project/fanos/)
