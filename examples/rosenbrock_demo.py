"""Minimal demo: optimize a scalar Rosenbrock-like function in PyTorch."""
import torch
from fanos import FANoS

def rosenbrock(x):
    # x shape (d,)
    return torch.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

def main():
    torch.manual_seed(0)
    d = 20
    x = torch.nn.Parameter(torch.empty(d).uniform_(-2, 2))
    opt = FANoS([x], lr=3e-2, grad_clip=1.0, T0_max=1e-3)

    for k in range(2000):
        opt.zero_grad()
        loss = rosenbrock(x)
        loss.backward()
        opt.step()
        if (k+1) % 200 == 0:
            print(k+1, float(loss.detach().cpu()))

if __name__ == "__main__":
    main()
