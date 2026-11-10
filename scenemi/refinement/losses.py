"""Training-only motion, contact and observed-geometry supervision.

No target geometry, shape, rotation or contact mask is an inference feature.
Geometric terms are evaluated after the final, differentiable B readout.
"""
import torch
from torch.nn import functional as F

from scenemi.generalization_losses import geodesic_squared
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.geometry.surface_query import query_surface, plane_gap
from scenemi.refinement.model import LOWER, CONTACT_PARTS
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def masked_mean(value, mask):
    """No valid label means exactly zero loss and zero gradient."""
    return (value*mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def contact_targets(model, batch, policy):
    teacher = dict(rest=batch['target_rest'], predicted_native_betas=batch['target_native_betas'],
                   predicted_template=batch['target_template'])
    points, labels = model.skin_points(batch['motion'], teacher, batch['body_format'])
    patch = query_surface(points, batch, model.geometry_policy, dynamic=False)
    gap = plane_gap(points, patch)
    difference = torch.diff(points, dim=1)*30
    velocity = torch.cat((difference[:, :1], difference), 1)
    speed = velocity.norm(dim=-1)
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for part in CONTACT_PARTS:
        selected |= labels == part
    native = (batch['body_format']>0)[:, None, None]
    valid = patch['valid'] & native & selected[:, None]
    valid &= patch['normal'][..., 1] >= policy['support_normal_y']
    valid &= gap >= -model.geometry_policy['source_penetration_tolerance_m']
    positive = (gap.abs() <= policy['positive_gap_m']) & (speed <= policy['positive_speed_mps'])
    negative = (gap >= policy['negative_gap_m']) | (speed >= policy['negative_speed_mps'])
    valid &= positive | negative  # Ambiguous gap/speed bands are not negatives.
    feet = ((labels==7)|(labels==8)|(labels==10)|(labels==11))[:, None]
    foot_id = ((labels==8)|(labels==11)).long()[:, None].expand_as(valid)
    source_weight = batch['contact_weight'].gather(2, foot_id).clamp(0, 1)
    # Pelvis labels require some independently reliable support evidence too;
    # do not invent seat contacts in source windows already flagged unreliable.
    reliability = torch.where(feet, source_weight, batch['contact_weight'].amax(-1, keepdim=True))
    weight = reliability * valid
    return positive.to(points), weight


def objective(model, prediction, details, batch, policy):
    target = batch['motion']
    position = forward_kinematics(prediction, details['rest'])
    truth = forward_kinematics(target, batch['target_rest']).detach()
    local = rotation_from_6d(prediction[..., 3:135].reshape(*prediction.shape[:2], 22, 6))
    gt_local = rotation_from_6d(target[..., 3:135].reshape(*target.shape[:2], 22, 6)).detach()
    active = details['scene_active'].to(prediction)
    mean = lambda value: masked_mean(value.flatten(1).mean(1), active)
    lower = position[:, :, LOWER]-position[:, :, :1]
    gt_lower = truth[:, :, LOWER]-truth[:, :, :1]
    obs = details['observations']
    head_rotation = global_rotations(prediction)[:, :, 15]
    observed_rotation = rotation_from_6d(obs['camera'][..., 3:] - prediction.new_tensor([1.,0,0,0,1,0]))
    predicted_camera = position[:, :, 15]+(head_rotation @ details['head_camera_translation'][:, None, :, None])[..., 0]
    label, contact_weight = contact_targets(model, batch, policy['contact'])
    contact_weight = contact_weight * details['anchor_hit'] * active[:, None, None]
    contact = F.binary_cross_entropy_with_logits(details['contact_logits'], label, reduction='none',
        pos_weight=prediction.new_tensor(policy['contact']['positive_class_weight']))
    limit = model.refiner.limits[None, None, :, None].clamp_min(1e-6)
    head_position_mask = obs['camera_available'][..., 0]*active[:, None]
    head_rotation_mask = obs['camera_available'][..., 1]*active[:, None]
    terms = dict(
        pose_rotation=mean(geodesic_squared(local, gt_local)),
        fk_position=mean(((position-truth)/policy['position_scale_m']).square()),
        lower_fk=mean(((lower-gt_lower)/policy['position_scale_m']).square()),
        lower_velocity=mean(((torch.diff(lower, dim=1)-torch.diff(gt_lower, dim=1))*30/policy['velocity_scale_mps']).square()),
        head_position=masked_mean(((predicted_camera-obs['camera'][..., :3]*2)/policy['head_position_scale_m']).square().mean(-1), head_position_mask),
        head_orientation=masked_mean(geodesic_squared(head_rotation, observed_rotation), head_rotation_mask),
        contact=masked_mean(contact, contact_weight),
        residual=mean((details['rotation_residual']/limit).square()))
    geometry, stats = geometry_terms(model, prediction, details, batch, active, model.geometry_policy)
    terms.update(geometry)
    terms['total'] = sum(policy['weights'][name]*term for name, term in terms.items())
    with torch.no_grad():
        predicted_contact = details['contact_logits'].sigmoid() >= .5
        positive = label > .5
        valid = contact_weight > 0
        tp = (predicted_contact & positive & valid).sum()
        stats.update(contact_valid_count=valid.sum(), contact_positive_count=(positive & valid).sum(),
            contact_precision=tp/(predicted_contact & valid).sum().clamp_min(1),
            contact_recall=tp/(positive & valid).sum().clamp_min(1))
    return terms, stats
