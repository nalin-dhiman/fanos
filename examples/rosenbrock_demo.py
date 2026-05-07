"""Optimize Rosenbrock with FANoS-v2."""

import torch

from fanos_v2 import FANoSV2


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(100.0 * (x[1:] - x[:-1].square()).square() + (1.0 - x[:-1]).square())


def main() -> None:
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.empty(20).uniform_(-2.0, 2.0))
    opt = FANoSV2([x], lr=3e-3, grad_clip=1.0, target_scale=0.2)

    for step in range(1, 2001):
        opt.zero_grad()
        loss = rosenbrock(x)
        loss.backward()
        opt.step()

        if step % 200 == 0:
            diag = opt.diagnostics()[0]
            print(
                f"{step:4d} loss={float(loss.detach()):.6f} "
                f"zeta={diag.zeta:.4f} rho={diag.rho:.4f} "
                f"clip={diag.clip_scale:.3f}"
            )


if __name__ == "__main__":
    main()
