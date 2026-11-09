"""Read-only FK/head and bounded-correction diagnostics for scene refiners.

These are measurements, not extra training losses. Counterfactual head
orientations use the same residual once with selected joints; they never run
another refiner or infer additive attribution through the kinematic chain.
"""
import math

import torch

from scenemi.refinement.model import LOWER, TORSO, apply_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def rotation_angle(first, second):
    """SO(3) angle in radians, including exact zero and pi for diagnostics."""
    relative = first.transpose(-1, -2) @ second
    skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                        relative[..., 0, 2]-relative[..., 2, 0],
                        relative[..., 1, 0]-relative[..., 0, 1]), -1)
    sine = .5*skew.norm(dim=-1)
    cosine = .5*(relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)
    return torch.atan2(sine, cosine.clamp(-1, 1))


def _mean(value, valid):
    valid = valid.to(value)
    return (value*valid).sum(-1)/valid.sum(-1).clamp_min(1)


@torch.no_grad()
def residual_statistics(omega, limits, gate=None, *, saturation_fraction=.9):
    if not 0 < saturation_fraction <= 1:
        raise ValueError('Saturation threshold must be a fraction in (0,1]')
    angles = omega.norm(dim=-1)
    ratio = angles/limits.clamp_min(1e-12)[None, None]
    # Torso/lower output uses per-axis tanh*limit/sqrt(3), whereas root
    # yaw uses tanh*limit. Norm occupancy alone misses single-axis saturation.
    component_ratio = omega.abs().amax(-1)*math.sqrt(3)/limits.clamp_min(1e-12)[None,None]
    component_ratio[:,:,0] = omega[:,:,0,1].abs()/limits[0].clamp_min(1e-12)
    active = ((limits > 0)[None, None].expand_as(angles) if gate is None
              else gate.bool() & (limits > 0)[None, None])
    result = {}
    for name, joints in (('root', (0,)), ('spine', TORSO), ('lower', LOWER)):
        value = angles[:, :, joints].flatten(1)*180/math.pi
        mask = active[:, :, joints].flatten(1)
        saturated = (ratio[:, :, joints] >= saturation_fraction).flatten(1) & mask
        component_saturated = (component_ratio[:,:,joints] >= saturation_fraction).flatten(1) & mask
        prefix = 'residual_'+name
        result.update({prefix+'_mean_deg': value.mean(-1),
            prefix+'_active_mean_deg': _mean(value, mask),
            prefix+'_max_deg': value.amax(-1),
            prefix+'_p95_deg': torch.quantile(value, .95, dim=-1),
            prefix+'_enabled_fraction': mask.float().mean(-1),
            prefix+'_norm_budget_fraction': _mean(ratio[:,:,joints].flatten(1),mask),
            prefix+'_saturation_fraction': _mean(saturated.float(), mask),
            prefix+'_component_saturation_fraction': _mean(component_saturated.float(),mask),
            prefix+'_component_saturated_count': component_saturated.sum(-1).to(value),
            prefix+'_saturated_count': saturated.sum(-1).to(value)})
    return result


@torch.no_grad()
def motion_diagnostics(model, final, baseline, details, batch, *, saturation_fraction=.9):
    obs = details['observations']
    rotation_mask = obs['camera_available'][..., 1] > 0
    position_mask = obs['camera_available'][..., 0] > 0
    observed = rotation_from_6d(obs['camera'][..., 3:]-final.new_tensor([1.,0,0,0,1,0]))
    before = global_rotations(baseline)[:, :, 15]
    after = global_rotations(final)[:, :, 15]
    error = lambda rotation: _mean(rotation_angle(rotation, observed)*180/math.pi, rotation_mask)
    positions = forward_kinematics(final, details['rest'])
    camera = positions[:, :, 15]+(after @ details['head_camera_translation'][:, None, :, None])[..., 0]
    omega = details['rotation_residual']
    result = dict(head_observation_position_mm=_mean(
        (camera-obs['camera'][..., :3]*2).norm(dim=-1)*1000, position_mask),
        head_observation_orientation_deg=error(after),
        head_observation_orientation_before_deg=error(before),
        head_observation_rotation_frames=rotation_mask.sum(-1).to(final),
        head_observation_position_frames=position_mask.sum(-1).to(final))
    result['head_observation_orientation_change_deg'] = error(after)-error(before)
    for name, joints in (('root_only', (0,)), ('spine_only', TORSO)):
        selected = torch.zeros_like(omega)
        selected[:, :, joints] = omega[:, :, joints]
        counterfactual = apply_rotations(baseline, selected, details['rest'])
        result['head_orientation_'+name+'_deg'] = error(global_rotations(counterfactual)[:, :, 15])
    result.update(residual_statistics(omega, model.refiner.limits,
        details.get('correction_gate'), saturation_fraction=saturation_fraction))
    if 'motion' in batch and 'target_rest' in batch:
        target = forward_kinematics(batch['motion'], batch['target_rest'])
        relative = positions[:, :, LOWER]-positions[:, :, :1]
        truth = target[:, :, LOWER]-target[:, :, :1]
        error_mm = (relative-truth).norm(dim=-1)*1000
        ratio = omega[:, :, LOWER].abs().amax(-1)*math.sqrt(3)/model.refiner.limits[list(LOWER)].clamp_min(1e-12)
        saturated = ratio >= saturation_fraction
        result['residual_lower_saturated_fk_error_mm'] = _mean(error_mm.flatten(1), saturated.flatten(1))
    return result


@torch.no_grad()
def query_change_diagnostics(model, final, baseline, details, public, *, stride=4):
    """Compare detached neighbourhoods on one immutable map after each B.

    Only validation invokes this extra query. The training geometry objective
    already reselects patches at final output positions and differentiates
    distance to those fixed patches. No third refiner is evaluated here.
    """
    import numpy as np
    from scenemi.geometry.support_evidence import execute

    formats = public['body_format']
    before, _ = model.skin_points(baseline[:, ::stride], details, formats)
    after, _ = model.skin_points(final[:, ::stride], details, formats)
    pending = []
    for i, view in enumerate(public.get('scene_views', (None,)*len(final))):
        if view is None or formats[i] <= 0:
            pending.append(None)
            continue
        options = dict(final_anchors=after[i].cpu().numpy(), dynamic=tuple(
            public[key][i, ::stride].cpu().numpy() for key in
            ('frame_scene_points','frame_scene_owner','frame_normals','frame_normal_valid')))
        executor = getattr(model, 'query_executor', None)
        pending.append(executor.submit('compare', view, before[i].cpu().numpy(), model.spatial_policy, **options)
            if executor is not None else execute('compare', view, before[i].cpu().numpy(), model.spatial_policy, **options))
    result = []
    for i, future in enumerate(pending):
        row = dict(query_observed_map_points=0., query_anchor_displacement_mm=float(
            (after[i]-before[i]).norm(dim=-1).mean()*1000))
        defaults = ('near_neighbour_change','context_neighbour_change','nearest_surface_change',
            'nearest_before_mm','nearest_after_mm','nearest_distance_change_mm',
            'nearest_before_coverage','nearest_after_coverage','nearest_common_coverage',
            'near_comparable_fraction','context_comparable_fraction',
            'scale_overlap_before','scale_overlap_after','scale_jaccard_before','scale_jaccard_after',
            'context_novel_fraction_before','context_novel_fraction_after')
        row.update({'query_'+key:0. for key in defaults})
        if future is not None:
            item = future.result() if hasattr(future, 'result') else future
            mean = lambda value, valid: float(np.asarray(value)[valid].mean()) if np.asarray(valid).any() else 0.
            before_finite = np.isfinite(item['nearest_before'])
            after_finite = np.isfinite(item['nearest_after'])
            common = before_finite & after_finite
            difference = np.zeros_like(item['nearest_before'])
            np.subtract(item['nearest_after'],item['nearest_before'],out=difference,where=common)
            row.update(query_observed_map_points=float(len(public['scene_views'][i].index.points)),
                query_near_neighbour_change=mean(item['near_change'], item['near_union']),
                query_context_neighbour_change=mean(item['context_change'], item['context_union']),
                query_nearest_surface_change=mean(item['nearest_surface_changed'], item['nearest_surface_union']),
                query_nearest_before_mm=mean(item['nearest_before'], before_finite)*1000,
                query_nearest_after_mm=mean(item['nearest_after'], after_finite)*1000,
                query_nearest_distance_change_mm=mean(difference, common)*1000,
                query_nearest_before_coverage=float(before_finite.mean()),
                query_nearest_after_coverage=float(after_finite.mean()),
                query_nearest_common_coverage=float(common.mean()),
                query_near_comparable_fraction=float(item['near_union'].mean()),
                query_context_comparable_fraction=float(item['context_union'].mean()),
                query_scale_overlap_before=mean(item['scale_overlap_before'], item['scale_overlap_valid_before']),
                query_scale_overlap_after=mean(item['scale_overlap_after'], item['scale_overlap_valid_after']),
                query_scale_jaccard_before=mean(item['scale_jaccard_before'], item['scale_overlap_valid_before']),
                query_scale_jaccard_after=mean(item['scale_jaccard_after'], item['scale_overlap_valid_after']),
                query_context_novel_fraction_before=mean(item['context_novel_fraction_before'],item['context_present_before']),
                query_context_novel_fraction_after=mean(item['context_novel_fraction_after'],item['context_present_after']))
        result.append(row)
    return result
