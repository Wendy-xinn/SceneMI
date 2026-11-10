import math
import torch
from scenemi.generalization_losses import geodesic_squared, focused_objective, shape_objective


def test_so3_has_finite_identity_gradient_and_correct_large_angle():
    r = torch.eye(3).requires_grad_()
    loss = geodesic_squared(r, torch.eye(3))
    loss.backward()
    assert loss < 1e-10 and torch.isfinite(r.grad).all()
    half_turn = torch.diag(torch.tensor([-1., -1., 1.]))
    assert torch.allclose(geodesic_squared(half_turn, torch.eye(3)), torch.tensor(math.pi**2), atol=1e-5)


def test_focused_loss_gates_high_noise_and_matches_motion_not_smoothness():
    # Nondegenerate 22-joint rest and identity 6D-residual motion.
    rest = torch.randn(2, 22, 3)*.1
    rest[:, 0] = 0
    motion = torch.zeros(2, 128, 201)
    motion[:, :, 0] = torch.linspace(0, .3, 128)
    prediction = motion.clone().requires_grad_()
    batch = dict(motion=motion, target_rest=rest)
    policy = dict(max_timestep=600, root_strength=1.25, position_scale_m=.5,
        velocity_scale_m_s=1., weights=dict(torso_chain_rad2=.05, lower_position_normalized=.1,
        lower_local_rad2=.05, lower_velocity_normalized=.025))
    result = focused_objective(prediction, batch, dict(rest=rest), torch.ones(2), torch.tensor([100, 900]), policy, .25)
    assert result['focused_increment'].abs() < 1e-9
    result['focused_increment'].backward()
    assert torch.isfinite(prediction.grad).all()
    wrong = motion.clone()
    wrong[:, :, 3] = .2
    result = focused_objective(wrong, batch, dict(rest=rest), torch.ones(2), torch.tensor([999, 999]), policy, .25)
    assert result['focused_increment'] == 0


def test_shape_loss_uses_format_specific_coefficients_and_finite_gradient():
    rest = torch.randn(4, 22, 3, requires_grad=True)
    native = torch.zeros(4, 10, requires_grad=True)
    smplh = torch.zeros(4, 16, requires_grad=True)
    details = dict(rest=rest, predicted_native_betas=native, predicted_betas=smplh,
                   template_logits=torch.zeros(4, 3, requires_grad=True))
    batch = dict(target_rest=rest.detach()+.01, body_format=torch.tensor([0, 1, 2, 3]),
                 target_native_betas=torch.ones(4, 10), target_betas=torch.ones(4, 16), target_template=torch.zeros(4, dtype=torch.long))
    c = dict(bone_scale_m=.1, height_scale_m=.25, weights=dict(shape_rest=.5,
        native_shape_betas=.05, native_template=.01, amass_shape_betas=.05, bone_length=.05, skeleton_height=.05))
    shape_objective(details, batch, c)['total'].backward()
    assert native.grad[1:3].abs().sum() > 0
    assert native.grad[[0, 3]].abs().sum() == 0
    assert smplh.grad[0].abs().sum() > 0
    assert smplh.grad[1:].abs().sum() == 0
    assert torch.isfinite(rest.grad).all()
