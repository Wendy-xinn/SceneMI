"""Numerical readout/gradient checks, not end-to-end camera robustness claims."""
import torch
import pytest

from tests.model import test_ego_alignment as alignment_fixture
from scenemi.geometry.head_anchor import camera_residual, translate_motion
from scenemi.geometry.temporal_head_anchor import TemporalHeadAnchor
from scenemi.anchored_scene_model import SCENE_INPUTS, mask_scene_inputs
from scenemi.anchored_training import scene_keep_mask


def policy(**changes):
    return dict(data_loss='l2', data_weight=1., acceleration_weight=.03**4,
        jerk_weight=.03**6, huber_delta_m=.02, irls_iterations=20,
        missing_ridge=1e-8, **changes)


def fixture():
    example = alignment_fixture.EgoAlignmentTests(); example.setUp()
    return example


def test_differentiable_readout_matches_validated_B_and_preserves_rotations():
    f = fixture(); anchor = TemporalHeadAnchor(policy(), frames=8)
    output = anchor(f.motion, f.rest, f.mount, f.camera)
    reference = translate_motion(f.motion, f.rest,
        anchor.operator @ camera_residual(f.motion, f.rest, f.mount, f.camera))
    torch.testing.assert_close(output, reference, rtol=1e-12, atol=1e-12)
    assert torch.equal(output[..., 3:135], f.motion[..., 3:135])


def test_full_readout_has_correct_finite_gradients_through_camera_fk_and_root():
    f = fixture(); anchor = TemporalHeadAnchor(policy(), frames=8)
    motion = f.motion.clone().requires_grad_()
    torch.manual_seed(9); projection = torch.randn_like(motion)
    def loss(value):
        return (anchor(value, f.rest, f.mount, f.camera)*projection).sum()
    gradient = torch.autograd.grad(loss(motion), motion)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[..., :3].norm()>0 and gradient[..., 3:135].norm()>0
    for index in ((0, 3, 0), (0, 4, 8), (0, 3, 95)):
        plus, minus = motion.detach().clone(), motion.detach().clone()
        plus[index] += 1e-6; minus[index] -= 1e-6
        numerical = (loss(plus)-loss(minus))/(2e-6)
        torch.testing.assert_close(gradient[index], numerical, rtol=2e-5, atol=1e-7)


def test_missing_stream_does_not_invent_translation_and_masked_frames_are_ignored():
    anchor = TemporalHeadAnchor(policy(), frames=8)
    e = torch.randn(2, 8, 3, dtype=torch.float64)
    assert torch.count_nonzero(anchor.correction(e, torch.zeros(2, 8)))==0
    availability = torch.ones(2, 8); availability[:, 3]=0
    contaminated = e.clone(); contaminated[:, 3]=1e6
    torch.testing.assert_close(anchor.correction(e, availability), anchor.correction(contaminated, availability))


def test_optional_huber_downweights_an_isolated_synthetic_camera_spike():
    l2 = TemporalHeadAnchor(policy(), frames=128)
    robust_policy = policy(); robust_policy['data_loss']='huber'
    robust = TemporalHeadAnchor(robust_policy, frames=128)
    e = torch.zeros(1, 128, 3, dtype=torch.float64); e[:, 64, 0]=.5
    assert robust.correction(e).norm()<l2.correction(e).norm()*.5


def test_scene_dropout_removes_every_scene_channel_and_preserves_camera_payload():
    public = {key:torch.ones(2, 3, 4) for key in SCENE_INPUTS}
    public.update(camera=torch.randn(2, 8, 9), camera_available=torch.ones(2, 8, 2))
    dropped = mask_scene_inputs(public, torch.tensor([True, False]))
    for key in SCENE_INPUTS:
        assert torch.equal(dropped[key][0], public[key][0])
        assert (dropped[key][1]==(-1 if key=='frame_scene_owner' else 0)).all()
        assert public[key].eq(1).all()
    assert dropped['camera'] is public['camera']
    assert dropped['camera_available'] is public['camera_available']


def test_batch128_uses_48_native_scene_and_80_camera_only_examples():
    sources = ('amass', 'egobody', 'trumans', 'grail')
    pairs = [(None, dict(dataset=source)) for source in sources for _ in range(32)]
    mask = scene_keep_mask(pairs, .5, 'cpu')
    assert int(mask.sum())==48 and int((~mask).sum())==80
    assert not mask[:32].any()
    assert [int(mask[i:i+32].sum()) for i in (32, 64, 96)]==[16, 16, 16]


def test_invalid_availability_is_rejected():
    anchor = TemporalHeadAnchor(policy(), frames=8)
    with pytest.raises(ValueError): anchor.correction(torch.zeros(1, 8, 3), torch.full((1, 8), 1.1))
