import torch
from fanos import FANoS

def test_fanos_smoke_step():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    opt = FANoS(model.parameters(), lr=1e-3, grad_clip=1.0)

    x = torch.randn(16, 5)
    y = torch.randn(16, 1)

    loss0 = torch.nn.functional.mse_loss(model(x), y)
    loss0.backward()
    opt.step()
    opt.zero_grad()

    loss1 = torch.nn.functional.mse_loss(model(x), y)
    assert torch.isfinite(loss1).all()
