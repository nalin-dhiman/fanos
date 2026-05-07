import math

import torch

from fanos_v2 import (
    FANoSV2,
    FANoSV2Fast,
    dequantize_4bit,
    densify_topk,
    dynamic_variance_clip,
    low_rank_approximation,
    quantize_4bit,
    quantized_gradient_residual,
    sparsify_topk,
)


def test_smoke_step_keeps_loss_finite():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )
    opt = FANoSV2(model.parameters(), lr=1e-3, grad_clip=1.0)

    x = torch.randn(32, 5)
    y = torch.randn(32, 1)

    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()

    next_loss = torch.nn.functional.mse_loss(model(x), y)
    assert torch.isfinite(next_loss)
    assert opt.diagnostics()


def test_quadratic_converges_from_large_initial_point():
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor([8.0, -4.0]))
    opt = FANoSV2([x], lr=0.04, grad_clip=10.0, target_scale=0.2)

    initial = None
    for step in range(300):
        opt.zero_grad()
        loss = 0.5 * (x * x).sum()
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        opt.step()

    final = float((0.5 * (x * x).sum()).detach())
    assert initial is not None
    assert final < 0.05 * initial


def test_zeta_is_non_negative_and_bounded():
    x = torch.nn.Parameter(torch.tensor([10.0]))
    opt = FANoSV2([x], lr=0.1, thermostat_lr=1.0, zeta_bounds=(0.0, 0.25), target_scale=1e-6)

    for _ in range(20):
        opt.zero_grad()
        (x.square().sum()).backward()
        opt.step()

    zeta = opt.param_groups[0]["zeta"]
    assert 0.0 <= zeta <= 0.25


def test_clipping_happens_before_second_moment_update():
    x = torch.nn.Parameter(torch.tensor([0.0]))
    opt = FANoSV2([x], lr=0.01, beta2=0.0, grad_clip=1.0)

    x.grad = torch.tensor([1000.0])
    opt.step()

    sq = opt.state[x]["sq"]
    assert torch.allclose(sq, torch.ones_like(sq), atol=1e-6)
    assert math.isclose(opt.diagnostics()[0].clip_scale, 0.001, rel_tol=1e-6)


def test_second_moment_bias_correction_prevents_large_first_step():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2([x], lr=0.1, momentum=0.0, preconditioner="diag", thermostat_lr=0.0)

    x.grad = torch.tensor([2.0])
    opt.step()

    assert torch.allclose(x.detach(), torch.tensor([0.9]), atol=1e-5)


def test_preconditioner_power_zero_matches_raw_gradient_step():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        momentum=0.0,
        preconditioner="diag",
        preconditioner_power=0.0,
        thermostat_lr=0.0,
    )

    x.grad = torch.tensor([2.0])
    opt.step()

    assert torch.allclose(x.detach(), torch.tensor([0.8]), atol=1e-6)


def test_auto_preset_enables_guarded_startup_and_adaptive_alpha():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2([x], preset="auto", lr=0.1, thermostat_lr=0.0)

    group = opt.param_groups[0]
    assert group["adaptive_lr"] is True
    assert group["adaptive_preconditioner_power"] is True
    assert group["warmup_steps"] == 200
    assert group["thermostat_warmup_steps"] == 100

    x.grad = torch.tensor([2.0])
    opt.step()

    diag = opt.diagnostics()[0]
    assert 0.5 <= diag.preconditioner_power <= 1.0
    assert diag.rho < 0.01


def test_auto_preset_respects_explicit_adaptive_lr_override():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2([x], preset="auto", adaptive_lr=False)

    assert opt.param_groups[0]["adaptive_lr"] is False


def test_pinn_preset_uses_softer_task_specific_defaults():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2([x], preset="pinn")

    group = opt.param_groups[0]
    assert group["lr"] == 5e-4
    assert group["momentum"] == 0.75
    assert group["target_scale"] == 0.05
    assert group["thermostat_lr"] == 0.001
    assert group["adaptive_lr"] is True
    assert group["preconditioner_power"] == 0.5
    assert group["warmup_steps"] == 200
    assert group["thermostat_warmup_steps"] == 200


def test_thermostat_interval_skips_energy_control_between_updates():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        momentum=0.0,
        preconditioner="none",
        thermostat_lr=1.0,
        thermostat_interval=3,
    )

    for _ in range(2):
        x.grad = torch.tensor([2.0])
        opt.step()
        x.grad = None
        diag = opt.diagnostics()[0]
        assert diag.update_energy == 0.0
        assert opt.param_groups[0]["zeta"] == 0.0

    x.grad = torch.tensor([2.0])
    opt.step()
    assert opt.diagnostics()[0].update_energy > 0.0


def test_grad_norm_interval_reuses_last_scalar_sync():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        momentum=0.0,
        preconditioner="none",
        thermostat_lr=0.0,
        grad_clip=10.0,
        grad_norm_interval=3,
    )

    x.grad = torch.tensor([2.0])
    opt.step()
    first_norm = opt.diagnostics()[0].grad_norm

    x.grad = torch.tensor([5.0])
    opt.step()
    second_norm = opt.diagnostics()[0].grad_norm

    assert first_norm == 2.0
    assert second_norm == first_norm


def test_diagnostics_can_be_recorded_less_often():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2([x], lr=0.1, momentum=0.0, preconditioner="none", diagnostics_interval=2)

    x.grad = torch.tensor([2.0])
    opt.step()
    assert opt.diagnostics() == []

    x.grad = torch.tensor([2.0])
    opt.step()
    assert opt.diagnostics()[0].step == 2


def test_fast_optimizer_is_opt_in_and_keeps_reference_defaults_unchanged():
    x_ref = torch.nn.Parameter(torch.tensor([1.0]))
    x_fast = torch.nn.Parameter(torch.tensor([1.0]))

    ref = FANoSV2([x_ref])
    fast = FANoSV2Fast([x_fast])

    assert ref.param_groups[0]["thermostat_interval"] == 1
    assert ref.param_groups[0]["record_diagnostics"] is True
    assert fast.param_groups[0]["preset"] == "auto"
    assert fast.param_groups[0]["adaptive_lr"] is False
    assert fast.param_groups[0]["grad_clip"] is None
    assert fast.param_groups[0]["thermostat_interval"] == 4
    assert fast.param_groups[0]["record_diagnostics"] is False

    x_fast.grad = torch.tensor([2.0])
    fast.step()
    assert fast.diagnostics() == []


def test_adaptive_preconditioner_power_softens_after_instability():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        preconditioner_power=1.0,
        adaptive_preconditioner_power=True,
        preconditioner_power_bounds=(0.5, 1.0),
        preconditioner_power_warmup_steps=0,
        preconditioner_power_instability_gain=0.25,
    )
    opt.param_groups[0]["last_log_error"] = 3.0

    x.grad = torch.tensor([2.0])
    opt.step()

    assert math.isclose(opt.diagnostics()[0].preconditioner_power, 0.5, rel_tol=1e-6)


def test_thermostat_warmup_delays_zeta_update():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        momentum=0.0,
        preconditioner="none",
        thermostat_lr=1.0,
        thermostat_warmup_steps=2,
        zeta_bounds=(0.0, 10.0),
    )

    for _ in range(2):
        x.grad = torch.tensor([10.0])
        opt.step()
        x.grad = None
        assert opt.param_groups[0]["zeta"] == 0.0

    x.grad = torch.tensor([10.0])
    opt.step()
    assert opt.param_groups[0]["zeta"] > 0.0


def test_state_dict_round_trip_preserves_group_state():
    torch.manual_seed(0)
    x1 = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    opt1 = FANoSV2([x1], lr=0.01, grad_clip=1.0)

    for _ in range(5):
        opt1.zero_grad()
        (x1.square().sum()).backward()
        opt1.step()

    state = opt1.state_dict()

    x2 = torch.nn.Parameter(x1.detach().clone())
    opt2 = FANoSV2([x2], lr=0.01, grad_clip=1.0)
    opt2.load_state_dict(state)

    assert opt2.param_groups[0]["zeta"] == opt1.param_groups[0]["zeta"]
    assert opt2.param_groups[0]["temp_ema"] == opt1.param_groups[0]["temp_ema"]


def test_factored_preconditioner_uses_row_and_column_state():
    w = torch.nn.Parameter(torch.randn(4, 6))
    opt = FANoSV2([w], lr=0.01, preconditioner="factored")

    loss = w.square().sum()
    loss.backward()
    opt.step()

    state = opt.state[w]
    assert state["preconditioner_kind"] == "factored"
    assert state["row_sq"].shape == (4,)
    assert state["col_sq"].shape == (6,)
    assert "sq" not in state


def test_none_preconditioner_is_plain_feedback_momentum():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2([x], lr=0.1, momentum=0.0, preconditioner="none", thermostat_lr=0.0)

    x.grad = torch.tensor([2.0])
    opt.step()

    assert torch.allclose(x.detach(), torch.tensor([0.8]), atol=1e-6)


def test_physical_update_mode_matches_paper_step_sign():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        momentum=0.0,
        preconditioner="none",
        thermostat_lr=0.0,
        update_mode="physical",
    )

    x.grad = torch.tensor([2.0])
    opt.step()

    assert torch.allclose(x.detach(), torch.tensor([0.8]), atol=1e-6)
    assert "v" in opt.state[x]
    assert opt.diagnostics()[0].update_mode == "physical"


def test_functional_thermostat_and_update_helpers():
    momentum = torch.tensor([2.0])
    controlled = FANoSV2.thermostat_control(momentum, temperature=0.25, target_temperature=1.0)
    theta_next = FANoSV2.fanos_update(torch.tensor([1.0]), torch.tensor([2.0]), lr=0.1)

    assert torch.allclose(controlled, torch.tensor([1.5]))
    assert torch.allclose(theta_next, torch.tensor([0.8]))


def test_adaptive_lr_is_reported_and_bounded():
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = FANoSV2(
        [x],
        lr=0.1,
        preconditioner="none",
        adaptive_lr=True,
        lr_bounds=(0.01, 0.2),
    )

    x.grad = torch.tensor([2.0])
    opt.step()

    diag = opt.diagnostics()[0]
    assert 0.01 <= diag.lr_effective <= 0.2


def test_low_rank_approximation_preserves_shape():
    matrix = torch.randn(6, 4)
    approx = low_rank_approximation(matrix, rank=2)

    assert approx.shape == matrix.shape
    assert torch.isfinite(approx).all()


def test_quantize_4bit_round_trip_is_reasonable():
    tensor = torch.linspace(-1.0, 1.0, steps=17)
    qtensor = quantize_4bit(tensor)
    restored = dequantize_4bit(qtensor)

    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert torch.max((restored - tensor).abs()) < 0.20
    assert qtensor.packed.numel() <= math.ceil(tensor.numel() / 2)


def test_sparse_topk_round_trip_keeps_largest_entries():
    tensor = torch.tensor([0.1, -4.0, 0.2, 3.0])
    sparse = sparsify_topk(tensor, density=0.5)
    dense = densify_topk(sparse)

    assert sparse.nnz == 2
    assert torch.allclose(dense, torch.tensor([0.0, -4.0, 0.0, 3.0]))


def test_dynamic_variance_clip_limits_large_entries():
    grad = torch.tensor([-10.0, 0.5, 10.0])
    variance = torch.ones_like(grad)
    clipped = dynamic_variance_clip(grad, variance, clip_factor=2.0)

    assert torch.allclose(clipped, torch.tensor([-2.0, 0.5, 2.0]), atol=1e-6)


def test_quantized_gradient_residual_reconstructs_error_shape():
    grad = torch.randn(9)
    residual = quantized_gradient_residual(grad)

    assert residual.shape == grad.shape
    assert torch.isfinite(residual).all()
