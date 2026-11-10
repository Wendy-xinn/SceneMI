"""Restored lower-body supervision and independently normalized scene losses.

Measured patches, fitted candidates and contact labels retain separate masks.
All physical denominators are detached effective-batch weight sums. Empty or
undersized groups yield exactly zero loss/gradient. GT is training-only.
"""
import numpy as np
import torch
from torch.nn import functional as F

from scenemi.generalization_losses import geodesic_squared
from scenemi.geometry.support_evidence import query_evidence
from scenemi.refinement.losses import masked_mean
from scenemi.refinement.model import ANCHOR_JOINTS, CONTACT_PARTS, LOWER
from scenemi.refinement.spatial_losses import residual_penalty
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

FEET = tuple(j in (7, 8, 10, 11) for j in ANCHOR_JOINTS)
GROUPS = ('penetration', 'support', 'sliding', 'contact')


@torch.no_grad()
def evidence(model, points, batch, *, dynamic):
    pending = []
    for i, view in enumerate(batch['scene_views']):
        if batch['body_format'][i] <= 0:
            view = None
        moving = (tuple(batch[k][i].detach().cpu().numpy() for k in
            ('frame_scene_points', 'frame_scene_owner', 'frame_normals', 'frame_normal_valid'))
            if dynamic and view is not None else None)
        options = dict(dynamic=moving, ground_mask=np.asarray(FEET))
        anchors = points[i].detach().cpu().numpy()
        executor = getattr(model, 'query_executor', None)
        pending.append(executor.submit('evidence', view, anchors, model.spatial_policy, **options)
            if executor is not None and view is not None else
            query_evidence(view, anchors, model.spatial_policy, **options))
    rows = [v.result() if hasattr(v, 'result') else v for v in pending]
    return {channel: {key: torch.as_tensor(np.stack([row[channel][key] for row in rows]),
        device=points.device, dtype=torch.bool if rows[0][channel][key].dtype == bool else points.dtype)
        for key in rows[0][channel]} for channel in rows[0]}


def weighted_group(values, weights, eligible, *, sample_axes=1):
    """Sufficient statistics; multiple sources don't inflate sample counts."""
    weights = weights.detach()
    valid = weights > 0
    samples = valid.any(-1) if sample_axes == 2 else valid
    nominal = eligible.detach().bool()
    return dict(numerator=(torch.where(valid, values, 0.)*weights).sum(),
        denominator=weights.sum(), count=samples.sum(), nominal=nominal.sum())


def normalize_group(groups, minimum, epsilon=1e-8):
    numerator = sum(g['numerator'] for g in groups)
    denominator = sum(g['denominator'] for g in groups)
    count = sum(g['count'] for g in groups)
    nominal = sum(g['nominal'] for g in groups)
    enabled = (count >= minimum) & (denominator > epsilon)
    loss = torch.where(enabled, numerator/denominator.clamp_min(epsilon), numerator*0)
    return loss, dict(valid_count=float(count), weight_sum=float(denominator),
        eligible_count=float(nominal), valid_fraction=float(count/nominal.clamp_min(1)),
        enabled=float(enabled))


def geometry_components(model, prediction, details, batch, policy):
    g = model.geometry_policy
    stride = g['loss_frame_stride'];ts = slice(None, None, stride)
    formats = batch['body_format'];active = details['scene_active'].bool()[:, None, None]
    skin, parts = model.skin_points(prediction[:, ts], details, formats)
    observed = dict(batch, **{k: batch[k][:, ts] for k in
        ('frame_scene_points', 'frame_scene_owner', 'frame_normals', 'frame_normal_valid')})
    with torch.no_grad():
        teacher = dict(rest=batch['target_rest'], predicted_native_betas=batch['target_native_betas'],
                       predicted_template=batch['target_template'])
        target, _ = model.skin_points(batch['motion'][:, ts], teacher, formats)
        measured = evidence(model, skin, observed, dynamic=True)['observed_geometry']
        sources = evidence(model, target, observed, dynamic=False)
        direct, fitted = sources['observed_geometry'], sources['support_surface']
        native = (formats > 0)[:, None, None].expand(skin.shape[:-1]) & active
        feet = torch.as_tensor(FEET, device=skin.device)[None, None].expand_as(native)
        velocity = torch.diff(target, dim=1)*(30/stride)
        speed = torch.cat((velocity[:, :1], velocity), 1).norm(dim=-1)
        side = ((parts == 8) | (parts == 11)).long()[:, None].expand_as(native)
        foot_reliability = batch['contact_weight'][:, ts].gather(2, side).clamp(0, 1)
        reliability = torch.where(feet, foot_reliability,
            batch['contact_weight'][:, ts].amax(-1, keepdim=True))
        teacher_measured_gap = ((target-measured['centre'])*measured['normal']).sum(-1)
        collision_valid = measured['valid'] & native & (teacher_measured_gap >= -g['source_penetration_tolerance_m'])
        centres = torch.stack((direct['centre'], fitted['centre']), -2)
        normals = torch.stack((direct['normal'], fitted['normal']), -2)
        confidence = torch.stack((direct['confidence'], fitted['confidence']), -1)
        # A raw 20cm neighbour is a feature. Strong contact requires a close,
        # horizontal patch and suitable vertical relation, checked separately.
        direct_ok = direct['valid'] & (direct['distance'] <= policy['measured_support_max_distance_m'])
        fitted_ok = fitted['valid'] & fitted['reliable'] & ~fitted['ambiguous'] & feet
        source_valid = torch.stack((direct_ok, fitted_ok), -1) & native[..., None]
        gap_gt = ((target[..., None, :]-centres)*normals).sum(-1)
        source_valid &= normals[..., 1] >= g['support_normal_y']
        source_valid &= gap_gt >= -g['source_penetration_tolerance_m']
        stance = feet & native & (speed <= g['stance_speed_mps'])
        support_valid = source_valid & stance[..., None] & (gap_gt.abs() <= g['source_contact_tolerance_m'])
        support_weight = confidence*support_valid*reliability[..., None]
        selected = torch.as_tensor([j in CONTACT_PARTS for j in ANCHOR_JOINTS], device=skin.device)[None, None]
        cp = policy['contact']
        positive = (gap_gt.abs() <= cp['positive_gap_m']) & (speed[..., None] <= cp['positive_speed_mps'])
        negative = (gap_gt >= cp['negative_gap_m']) | (speed[..., None] >= cp['negative_speed_mps'])
        contact_valid = source_valid & selected[..., None] & (positive | negative)
        # Contradictory observed/fitted labels are Unknown, not two targets.
        conflict = contact_valid.all(-1) & (positive[..., 0] != positive[..., 1])
        contact_valid &= ~conflict[..., None]
        contact_weight = confidence*reliability[..., None]*contact_valid
        interval = stance[:, 1:] & stance[:, :-1]
        interval_weight = torch.minimum(support_weight[:, 1:], support_weight[:, :-1])
        interval_weight *= (normals[:, 1:]*normals[:, :-1]).sum(-1) >= .95
    measured_gap = ((skin-measured['centre'])*measured['normal']).sum(-1)
    depth = (-measured_gap-g['penetration_tolerance_m']).clamp_min(0)
    crossing = F.smooth_l1_loss(depth/g['distance_scale_m'], torch.zeros_like(depth), reduction='none')
    support_gap = ((skin[..., None, :]-centres)*normals).sum(-1)
    excess = (support_gap.abs()-g['contact_tolerance_m']).clamp_min(0)/g['distance_scale_m']
    support_loss = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')
    tangent = torch.diff(skin, dim=1)[..., None, :]*(30/stride)
    tangent = tangent-(tangent*normals[:, 1:]).sum(-1, keepdim=True)*normals[:, 1:]
    sliding = ((tangent.norm(dim=-1)-g['sliding_tolerance_mps']).clamp_min(0)/g['velocity_scale_mps']).square()
    logits = details['contact_logits'][:, ts, :, None].expand_as(positive)
    bce = F.binary_cross_entropy_with_logits(logits, positive.to(logits), reduction='none',
        pos_weight=skin.new_tensor(cp['positive_class_weight']))
    groups = dict(penetration=weighted_group(crossing, collision_valid.to(skin), native),
        support=weighted_group(support_loss, support_weight, stance, sample_axes=2),
        sliding=weighted_group(sliding, interval_weight, interval, sample_axes=2),
        contact=weighted_group(bce, contact_weight, native & selected, sample_axes=2))
    with torch.no_grad():
        avg = lambda x, w: (x*w).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(1e-8)
        stats = dict(observed_surface_penetration_mm=avg(depth, collision_valid)*1000,
            observed_surface_penetrating_fraction=avg((depth > 0).to(skin), collision_valid),
            observed_surface_coverage=collision_valid.float().flatten(1).mean(1),
            observed_support_gap_mm=avg(support_gap.abs(), support_weight)*1000,
            observed_support_coverage=(support_weight > 0).any(-1).float().flatten(1).mean(1),
            observed_support_sliding_mps=avg(tangent.norm(dim=-1), interval_weight),
            measured_support_valid_count=(support_valid[..., 0] & (reliability > 0)).flatten(1).sum(1).float(),
            fitted_support_valid_count=(support_valid[..., 1] & (reliability > 0)).flatten(1).sum(1).float(),
            support_candidate_count=fitted['valid'].flatten(1).sum(1).float(),
            support_reliable_count=fitted['reliable'].flatten(1).sum(1).float(),
            contact_conflict_count=conflict.flatten(1).sum(1).float())
    return groups, stats


def components(model, prediction, details, batch, policy):
    target = batch['motion'];active = details['scene_active'].to(prediction)
    position = forward_kinematics(prediction, details['rest'])
    truth = forward_kinematics(target, batch['target_rest']).detach()
    local = rotation_from_6d(prediction[..., 3:135].reshape(*prediction.shape[:2], 22, 6))
    gt_local = rotation_from_6d(target[..., 3:135].reshape(*target.shape[:2], 22, 6)).detach()
    mean = lambda x: masked_mean(x.flatten(1).mean(1), active)
    obs = details['observations'];head_rotation = global_rotations(prediction)[:, :, 15]
    observed_rotation = rotation_from_6d(obs['camera'][..., 3:]-prediction.new_tensor([1.,0,0,0,1,0]))
    camera = position[:, :, 15]+(head_rotation @ details['head_camera_translation'][:, None, :, None])[..., 0]
    lower, gt_lower = position[:, :, LOWER]-position[:, :, :1], truth[:, :, LOWER]-truth[:, :, :1]
    motion = dict(rotation=mean(geodesic_squared(local, gt_local)),
        position=mean(((position-truth)/policy['position_scale_m']).square()),
        lower_fk=mean(((lower-gt_lower)/policy['position_scale_m']).square()),
        lower_velocity=mean(((torch.diff(lower, dim=1)-torch.diff(gt_lower, dim=1))*30/policy['velocity_scale_mps']).square()),
        head_position=masked_mean(((camera-obs['camera'][..., :3]*2)/policy['head_position_scale_m']).square().mean(-1), obs['camera_available'][..., 0]*active[:, None]),
        head_orientation=masked_mean(geodesic_squared(head_rotation, observed_rotation), obs['camera_available'][..., 1]*active[:, None]))
    residual, residual_stats = residual_penalty(details['rotation_residual'], model.refiner.limits, active, policy['residual'])
    groups, stats = geometry_components(model, prediction, details, batch, policy)
    stats.update({'motion_'+k: v.detach() for k,v in motion.items()})
    stats.update({k: v.detach() for k,v in residual_stats.items()})
    stats['residual_rms_deg'] = details['rotation_residual'].square().mean().sqrt().detach()*180/torch.pi
    stats['near_hit_fraction'] = details['anchor_near_hit'].float().mean()
    stats['ground_valid_fraction'] = details['ground_valid'].float().mean()
    stats['context_or_ground_fraction'] = details['anchor_hit'].float().mean()
    return dict(motion=sum(policy['motion'][k]*v for k,v in motion.items()), residual=residual), groups, stats


def combine(chunks, policy):
    """Physical groups normalize across the entire effective batch, not chunks."""
    total_examples = sum(n for n,_,_ in chunks)
    terms = {name: sum(n*local[name] for n,local,_ in chunks)/total_examples
             for name in ('motion','residual')}
    statistics = {}
    for name in GROUPS:
        terms[name], record = normalize_group([g[name] for _,_,g in chunks],
            policy['normalization']['minimum_valid_samples'][name], policy['normalization']['epsilon'])
        statistics.update({name+'_'+k: v for k,v in record.items()})
    weighted = {name: policy['weights'][name]*value for name,value in terms.items()}
    statistics.update({'weighted_'+name: float(value.detach()) for name,value in weighted.items()})
    terms['total'] = sum(weighted.values())
    return terms, statistics


class EffectiveBatchGradients:
    """Exact detached-denominator reduction while retaining one chunk graph.

For L=sum(numerator_i)/sum(weight_i), its parameter gradient is the sum
of numerator gradients divided by the final detached weight sum. Small
parameter-sized buffers replace full-batch activation storage.
"""
    def __init__(self, named_parameters, policy, examples):
        self.names, self.parameters = zip(*named_parameters)
        self.policy, self.examples = policy, examples
        self.buffers = {name:[None]*len(self.parameters) for name in (*GROUPS,'motion_residual')}
        self.chunks = []

    def add(self, examples, local, groups):
        scalars = {name:groups[name]['numerator'] for name in GROUPS}
        scalars['motion_residual'] = examples/self.examples*(
            self.policy['weights']['motion']*local['motion']+
            self.policy['weights']['residual']*local['residual'])
        for name, scalar in scalars.items():
            # A scene-free AMASS chunk has constant-zero geometric terms.
            # allow_unused handles unused inputs, not non-differentiable
            # outputs; the exact gradient of these constants is zero.
            if not scalar.requires_grad:
                continue
            gradients = torch.autograd.grad(scalar, self.parameters,
                retain_graph=name!='motion_residual', allow_unused=True)
            for i,gradient in enumerate(gradients):
                if gradient is None:
                    continue
                if self.buffers[name][i] is None:
                    self.buffers[name][i] = gradient.detach().clone()
                else:
                    self.buffers[name][i].add_(gradient.detach())
        self.chunks.append((examples,{k:v.detach() for k,v in local.items()},
            {name:{k:v.detach() for k,v in group.items()} for name,group in groups.items()}))

    def finish(self):
        terms, statistics = combine(self.chunks, self.policy)
        combined = [None]*len(self.parameters)
        for name,gradients in self.buffers.items():
            scale = (1. if name=='motion_residual' else
                self.policy['weights'][name]/statistics[name+'_weight_sum']
                if statistics[name+'_enabled'] else 0.)
            weighted = [None if g is None else g*scale for g in gradients]
            norm = sum(g.square().sum() for g in weighted if g is not None)
            statistics['weighted_gradient_norm_'+name] = float(norm.sqrt()) if torch.is_tensor(norm) else 0.
            probe_name = 'contact.2.weight' if name=='contact' else 'rotation.weight'
            if probe_name in self.names:
                probe = weighted[self.names.index(probe_name)]
                statistics['weighted_gradient_probe_'+name] = 0. if probe is None else float(probe.norm())
            for i,gradient in enumerate(weighted):
                if gradient is not None:
                    combined[i] = gradient.clone() if combined[i] is None else combined[i]+gradient
        for parameter,gradient in zip(self.parameters,combined):
            parameter.grad = gradient
        return terms, statistics
