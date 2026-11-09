"""Unknown geometry, bounded ground, physical gradients and residual dynamics."""
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from scenemi.config import ROOT
from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView
from scenemi.geometry.surface_query import plane_gap
from scenemi.refinement.model import ANCHOR_JOINTS
from scenemi.refinement.spatial_losses import observed_patches, geometry_supervision, residual_penalty
from scenemi.refinement.spatial_training import load_config


def fixture(index=None):
    c = load_config(ROOT/'configs/ego_scene_spatial_refinement.yaml')
    policy = yaml.safe_load((ROOT/c['spatial']['policy']).read_text())
    x, z = np.meshgrid(np.arange(-.5, .51, .04), np.arange(-.5, .51, .04))
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel())).astype(np.float32)
    index = index if index is not None else ObservedSceneIndex(points,
        np.tile([0., 1., 0.], (len(points), 1)), np.ones(len(points), bool))
    model = SimpleNamespace(spatial_policy=policy, geometry_policy=c['geometry'])
    labels = torch.as_tensor(ANCHOR_JOINTS)[None]
    model.skin_points = lambda motion, details, formats: (motion.reshape(1, -1, 67, 3), labels)
    shape = (1, 8, 67, 3);target = torch.zeros(shape)
    batch = dict(scene_views=(SceneView(index, np.zeros(3), np.eye(3)),), body_format=torch.ones(1, dtype=torch.long),
        frame_scene_points=torch.zeros(1, 8, 1, 3), frame_scene_owner=torch.zeros(1, 8, 1, dtype=torch.long),
        frame_normals=torch.zeros(1, 8, 1, 3), frame_normal_valid=torch.zeros(1, 8, 1, dtype=torch.bool),
        contact_weight=torch.ones(1, 8, 2), motion=target.flatten(2), target_rest=None,
        target_native_betas=None, target_template=None)
    return model, batch, target, c


def test_far_context_does_not_become_a_measured_contact_or_collision_patch():
    index = ObservedSceneIndex(np.array([[.4, 0, 0]], np.float32), np.array([[0., 1., 0.]]), np.ones(1, bool))
    model, batch, points, _ = fixture(index)
    result = observed_patches(model, points, batch, ground=True)
    assert not result['valid'].any() and not result['confidence'].any()


def test_bounded_ground_can_fill_a_small_hole_for_feet_without_inventing_hand_support():
    model, batch, points, _ = fixture()
    original = batch['scene_views'][0].index
    # A hole just outside measured near radius but inside the independently
    # declared 25 cm support interpolation bound.
    keep = np.linalg.norm(original.points[:, [0, 2]], axis=-1) > .21
    index = ObservedSceneIndex(original.points[keep], original.normals[keep], np.ones(keep.sum(), bool))
    batch['scene_views'] = (SceneView(index, np.zeros(3), np.eye(3)),)
    near = observed_patches(model, points, batch, ground=False)
    support = observed_patches(model, points, batch, ground=True)
    assert not near['valid'].any()
    expected = torch.tensor([j in (7, 8, 10, 11) for j in ANCHOR_JOINTS])
    assert torch.equal(support['valid'][0, 0], expected)
    assert (support['confidence'][..., expected] > 0).all()


def test_patch_selection_is_detached_but_plane_residual_has_finite_body_gradient():
    model, batch, points, _ = fixture();points[..., 1] = -.1;points.requires_grad_()
    patch = observed_patches(model, points, batch)
    loss = (-plane_gap(points, patch)).clamp_min(0).mean();loss.backward()
    assert not patch['centre'].requires_grad
    assert torch.isfinite(points.grad).all() and (points.grad[..., 1] < 0).all()


def test_source_body_penetration_masks_physical_supervision():
    model, batch, points, _ = fixture();points[..., 1] = -.1
    batch['motion'] = points.flatten(2).clone()
    prediction = points.flatten(2).clone().requires_grad_()
    terms, stats, _ = geometry_supervision(model, prediction, dict(scene_active=torch.ones(1)), batch)
    assert terms['penetration'] == 0 and stats['observed_surface_coverage'] == 0


def test_unknown_scene_produces_no_geometry_force_or_fabricated_support():
    model, batch, points, _ = fixture(ObservedSceneIndex(np.empty((0, 3))))
    prediction = points.flatten(2).clone().requires_grad_()
    terms, stats, _ = geometry_supervision(model, prediction, dict(scene_active=torch.ones(1)), batch)
    loss = sum(terms.values());loss.backward()
    assert loss == 0 and prediction.grad.abs().max() == 0
    assert stats['observed_support_coverage'] == 0 and stats['observed_surface_coverage'] == 0


def test_contact_reliability_mask_removes_untrusted_support_labels():
    model, batch, points, _ = fixture();batch['contact_weight'].zero_()
    _, _, labels = geometry_supervision(model, points.flatten(2), dict(scene_active=torch.ones(1)), batch)
    assert labels['reliability'].sum() == 0


def test_temporal_penalty_targets_only_new_correction_and_detects_high_frequency():
    policy = dict(magnitude=0., first_difference=.1, second_difference=.1)
    active = torch.ones(1);limits = torch.ones(22)
    constant = torch.full((1, 8, 22, 3), .1, requires_grad=True)
    smooth, _ = residual_penalty(constant, limits, active, policy)
    noisy = constant.detach().clone();noisy[:, ::2] *= -1;noisy.requires_grad_()
    high, _ = residual_penalty(noisy, limits, active, policy);high.backward()
    assert smooth == 0 and high > 0 and torch.isfinite(noisy.grad).all()


def test_new_configuration_has_four_groups_frozen_prior_and_equal_source_quota():
    c = load_config(ROOT/'configs/ego_scene_spatial_refinement.yaml')
    assert set(c['spatial_loss']['weights']) == {'motion', 'contact', 'geometry', 'residual'}
    assert 'refinement_loss' not in c and c['model']['scene_modules_trainable'] is False
    assert c['training']['effective_batch_size'] == 128 and c['data']['source_quota_per_update'] == 32
    assert 'phase_b' in c['spatial']['prior'] and c['data']['excluded_datasets'] == ['rich']
