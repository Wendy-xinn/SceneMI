"""Four loss groups on final FK+B output, using complete observed scene maps.

Context neighbours are input features only. Contact and physical constraints
require a measured near surface or a bounded, independently observed support
candidate. GT supplies labels/reliability checks only, never model features.
"""
import numpy as np
import torch
from torch.nn import functional as F

from scenemi.generalization_losses import geodesic_squared
from scenemi.geometry.surface_query import plane_gap
from scenemi.refinement.losses import masked_mean
from scenemi.refinement.model import ANCHOR_JOINTS, CONTACT_PARTS, LOWER
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


FEET = tuple(j in (7, 8, 10, 11) for j in ANCHOR_JOINTS)


@torch.no_grad()
def observed_patches(model, points, batch, *, dynamic=True, ground=False):
    """Nearest measured patch, optionally bounded static support interpolation.

    CPU discrete search is detached; plane_gap(predicted, patch) remains fully
    differentiable. Dynamic owners 1..99 use their current frame only. Normal
    validity and the 20 cm measured-surface radius are explicit masks.
    """
    from scenemi.geometry.spatial_execution import nearest_patches
    executor = getattr(model, 'query_executor', None)
    fields, pending = [], []
    for i, view in enumerate(batch['scene_views']):
        anchors = points[i].detach().cpu().numpy()
        if view is None or batch['body_format'][i] <= 0:
            view = None
        moving = None
        if dynamic and view is not None:
            moving = tuple(batch[k][i].detach().cpu().numpy() for k in
                ('frame_scene_points', 'frame_scene_owner', 'frame_normals', 'frame_normal_valid'))
        options = dict(dynamic=moving, ground_mask=np.asarray(FEET) if ground else None)
        if executor is not None and view is not None:
            pending.append(executor.submit('patches', view, anchors, model.spatial_policy, **options))
        else:
            pending.append(nearest_patches(view, anchors, model.spatial_policy, **options))
    for item in pending:
        fields.append(item.result() if hasattr(item, 'result') else item)
    return {k: torch.as_tensor(np.stack([f[k] for f in fields]), device=points.device,
        dtype=torch.bool if k == 'valid' else points.dtype) for k in fields[0]}


def residual_penalty(omega, limits, active, policy):
    """Per-frame differences of bounded rotation corrections, never body pose."""
    normalized = omega / limits.clamp_min(1e-6)[None, None, :, None]
    mean = lambda x: masked_mean(x.flatten(1).mean(1), active)
    magnitude = mean(normalized.square())
    first = mean(torch.diff(normalized, dim=1).abs()) if omega.shape[1] > 1 else magnitude*0
    second = mean(torch.diff(normalized, n=2, dim=1).abs()) if omega.shape[1] > 2 else magnitude*0
    return (policy['magnitude']*magnitude + policy['first_difference']*first
        + policy['second_difference']*second), dict(residual_magnitude=magnitude,
            residual_first_difference=first, residual_second_difference=second)


def geometry_supervision(model, prediction, details, batch, *, labels=True):
    """Shared geometry metrics and training labels on the complete CPU index."""
    g = model.geometry_policy; stride = g['loss_frame_stride']; ts = slice(None, None, stride)
    formats = batch['body_format']; active = details['scene_active'].to(prediction)
    skin, parts = model.skin_points(prediction[:, ts], details, formats)
    observed = dict(batch, **{k: batch[k][:, ts] for k in
        ('frame_scene_points', 'frame_scene_owner', 'frame_normals', 'frame_normal_valid')})
    with torch.no_grad():
        teacher = dict(rest=batch['target_rest'], predicted_native_betas=batch['target_native_betas'],
            predicted_template=batch['target_template'])
        target, _ = model.skin_points(batch['motion'][:, ts], teacher, formats)
        measured = observed_patches(model, skin, observed)
        support = observed_patches(model, target, observed, dynamic=False, ground=True)
        native = (formats > 0)[:, None, None].expand(skin.shape[:-1])
        valid = measured['valid'] & native & (plane_gap(target, measured) >= -g['source_penetration_tolerance_m'])
        feet = torch.as_tensor(FEET, device=skin.device)[None, None].expand_as(valid)
        velocity = torch.diff(target, dim=1)*(30/stride)
        velocity = torch.cat((velocity[:, :1], velocity), 1)
        speed = velocity.norm(dim=-1)
        side = ((parts == 8) | (parts == 11)).long()[:, None].expand_as(valid)
        source_weight = batch['contact_weight'][:, ts].gather(2, side).clamp(0, 1)
        reliability = torch.where(feet, source_weight, batch['contact_weight'][:, ts].amax(-1, keepdim=True))
        target_gap = plane_gap(target, support)
        support_valid = (support['valid'] & native & (support['normal'][..., 1] >= g['support_normal_y'])
            & (target_gap.abs() <= g['source_contact_tolerance_m']))
        stance = feet & native & (speed <= g['stance_speed_mps'])
        weight = reliability*stance*support_valid*support['confidence']
    gap = plane_gap(skin, measured)
    depth = (-gap-g['penetration_tolerance_m']).clamp_min(0)
    crossing = F.smooth_l1_loss(depth/g['distance_scale_m'], torch.zeros_like(depth), reduction='none')
    support_gap = plane_gap(skin, support)
    excess = (support_gap.abs()-g['contact_tolerance_m']).clamp_min(0)/g['distance_scale_m']
    contact = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')
    tangent_velocity = torch.diff(skin, dim=1)*(30/stride)
    normal = support['normal'][:, 1:]
    tangent_velocity = tangent_velocity-(tangent_velocity*normal).sum(-1, keepdim=True)*normal
    interval = stance[:, 1:] & stance[:, :-1]
    interval_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    interval_weight *= ((support['normal'][:, 1:]*support['normal'][:, :-1]).sum(-1) >= .95)
    slip = ((tangent_velocity.norm(dim=-1)-g['sliding_tolerance_mps']).clamp_min(0)/g['velocity_scale_mps']).square()
    nominal = lambda x, w, n: (x*w).flatten(1).sum(1)/n.flatten(1).sum(1).clamp_min(1)
    terms = dict(penetration=(nominal(crossing, valid, native)*active).mean(),
        support=(nominal(contact, weight, stance)*active).mean(),
        sliding=(nominal(slip, interval_weight, interval)*active).mean())
    with torch.no_grad():
        average = lambda x, w: (x*w).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(1)
        stats = dict(observed_surface_penetration_mm=average(depth, valid)*1000,
            observed_surface_penetrating_fraction=average((depth > 0).to(depth), valid),
            observed_surface_coverage=valid.float().flatten(1).mean(1),
            observed_support_gap_mm=average(support_gap.abs(), weight)*1000,
            observed_support_coverage=(weight > 0).float().flatten(1).mean(1),
            observed_support_sliding_mps=average(tangent_velocity.norm(dim=-1), interval_weight))
    if not labels:
        return terms, stats, None
    return terms, stats, dict(target_gap=target_gap, speed=speed, support=support,
        native=native, reliability=reliability, stride=stride)


def objective(model, prediction, details, batch, policy):
    target = batch['motion']; active = details['scene_active'].to(prediction)
    position = forward_kinematics(prediction, details['rest'])
    truth = forward_kinematics(target, batch['target_rest']).detach()
    local = rotation_from_6d(prediction[..., 3:135].reshape(*prediction.shape[:2], 22, 6))
    gt_local = rotation_from_6d(target[..., 3:135].reshape(*target.shape[:2], 22, 6)).detach()
    mean = lambda x: masked_mean(x.flatten(1).mean(1), active)
    obs = details['observations']; head_rotation = global_rotations(prediction)[:, :, 15]
    observed_rotation = rotation_from_6d(obs['camera'][..., 3:]-prediction.new_tensor([1., 0, 0, 0, 1, 0]))
    camera = position[:, :, 15]+(head_rotation @ details['head_camera_translation'][:, None, :, None])[..., 0]
    lower, gt_lower = position[:, :, LOWER]-position[:, :, :1], truth[:, :, LOWER]-truth[:, :, :1]
    components = dict(rotation=mean(geodesic_squared(local, gt_local)),
        position=mean(((position-truth)/policy['position_scale_m']).square()),
        lower_velocity=mean(((torch.diff(lower, dim=1)-torch.diff(gt_lower, dim=1))*30/policy['velocity_scale_mps']).square()),
        head_position=masked_mean(((camera-obs['camera'][..., :3]*2)/policy['head_position_scale_m']).square().mean(-1),
            obs['camera_available'][..., 0]*active[:, None]),
        head_orientation=masked_mean(geodesic_squared(head_rotation, observed_rotation), obs['camera_available'][..., 1]*active[:, None]))
    geometry, stats, supervision = geometry_supervision(model, prediction, details, batch)
    cp = policy['contact']; gap = supervision['target_gap']; speed = supervision['speed']; support = supervision['support']
    positive = (gap.abs() <= cp['positive_gap_m']) & (speed <= cp['positive_speed_mps'])
    negative = (gap >= cp['negative_gap_m']) | (speed >= cp['negative_speed_mps'])
    selected = torch.as_tensor([j in CONTACT_PARTS for j in ANCHOR_JOINTS], device=prediction.device)[None, None]
    valid = support['valid'] & supervision['native'] & selected & (positive | negative)
    valid &= (support['normal'][..., 1] >= cp['support_normal_y']) & (gap >= -model.geometry_policy['source_penetration_tolerance_m'])
    ts = slice(None, None, supervision['stride'])
    # Context is allowed to trigger a correction, but is never itself a label.
    weight = valid*supervision['reliability']*support['confidence']*active[:, None, None]*details['anchor_hit'][:, ts]
    logits = details['contact_logits'][:, ts]
    bce = F.binary_cross_entropy_with_logits(logits, positive.to(logits), reduction='none',
        pos_weight=prediction.new_tensor(cp['positive_class_weight']))
    residual, residual_components = residual_penalty(details['rotation_residual'], model.refiner.limits, active, policy['residual'])
    terms = dict(motion=sum(policy['motion'][k]*v for k, v in components.items()),
        contact=masked_mean(bce, weight), geometry=sum(policy['geometry'][k]*v for k, v in geometry.items()), residual=residual)
    terms['total'] = sum(policy['weights'][k]*v for k, v in terms.items())
    with torch.no_grad():
        labelled = weight > 0; guessed = logits.sigmoid() >= .5; tp = (guessed & positive & labelled).sum()
        stats.update(contact_valid_count=labelled.sum(), contact_positive_count=(positive & labelled).sum(),
            contact_precision=tp/(guessed & labelled).sum().clamp_min(1), contact_recall=tp/(positive & labelled).sum().clamp_min(1),
            residual_rms_deg=details['rotation_residual'].square().mean().sqrt()*180/torch.pi,
            near_hit_fraction=details['anchor_near_hit'].float().mean(),
            context_or_ground_fraction=details['anchor_hit'].float().mean(), ground_valid_fraction=details['ground_valid'].float().mean())
        stats.update({'motion_'+k: v.detach() for k, v in components.items()})
        stats.update({'geometry_'+k: v.detach() for k, v in geometry.items()})
        stats.update({k: v.detach() for k, v in residual_components.items()})
    return terms, stats
