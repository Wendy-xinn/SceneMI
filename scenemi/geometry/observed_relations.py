"""Body/surface supervision from observed points, independent of action labels.

Local planes are estimated from observed neighbours, with explicit degeneracy
and mixed-owner rejection. Their normals have no inside/outside interpretation.
Target motion/shape can define training labels, never model input. Unknown
geometry contributes zero loss; an anchor is not forced onto a surface.
"""
import torch
from torch.nn import functional as F

from scenemi.local_model import nearest_observed


def local_planes(points, owners, valid, *, minimum_points=6, maximum_ratio=.03,
                 minimum_spread_m=.005):
    # Keep the nearest observed owner's neighbourhood. Do not fit one plane
    # across a moving object and the static room.
    valid = valid & torch.isfinite(points).all(-1)
    valid = valid & (owners == owners[..., :1])
    count = valid.sum(-1)
    safe = torch.where(valid[..., None], points, 0.)
    centre = safe.sum(-2)/count.clamp_min(1)[..., None]
    delta = torch.where(valid[..., None], points-centre[..., None, :], 0.)
    covariance = delta.transpose(-1, -2)@delta/count.clamp_min(1)[..., None, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[..., 0]
    planar = (count >= minimum_points) & (eigenvalues[..., 1] >= minimum_spread_m**2)
    planar &= eigenvalues[..., 0]/eigenvalues.sum(-1).clamp_min(1e-12) <= maximum_ratio
    return dict(centre=centre.detach(), normal=normal.detach(), valid=planar.detach(),
                neighbours=count.detach(), owner=owners[..., 0].detach())


def observed_planes(target_anchors, batch, policy):
    """Targets only choose a supervised region in the *observed* point cloud."""
    rows = []
    for i, anchors in enumerate(target_anchors):
        points, owners, valid = nearest_observed(anchors,
            batch['local_history_points'][i], batch['local_history_valid'][i],
            batch['local_frame_points'][i], batch['local_frame_owner'][i],
            k=policy['neighbours'], radius=policy['query_radius_m'])
        plane = local_planes(points, owners, valid,
            minimum_points=policy['minimum_points'], maximum_ratio=policy['maximum_planarity_ratio'],
            minimum_spread_m=policy['minimum_spread_m'])
        offset = ((anchors-plane['centre'])*plane['normal']).sum(-1)
        plane['valid'] &= offset.abs() <= policy['maximum_target_offset_m']
        rows.append(plane)
    return {key:torch.stack([r[key] for r in rows]) for key in rows[0]}


def relation_loss(predicted, target, planes, *, scale_m=.05):
    """Match the target's surface offset, rather than forcing every point down."""
    valid = planes['valid']
    error = ((predicted-target.detach())*planes['normal']).sum(-1)/scale_m
    values = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
    # Retain an exact, differentiable zero for an empty observation.
    return (values*valid).sum()/valid.sum().clamp_min(1)


def gate_targets(target, proposal, labels, planes, *, near_m=.03,
                 correspondence_m=.05, neighbourhood_m=.25):
    """Weak geometric labels, NOT gold physical-contact or interaction labels."""
    offset = ((target-planes['centre'])*planes['normal']).sum(-1).abs()
    displacement = ((proposal.detach()-target)*planes['normal']).sum(-1).abs()
    nearby = (proposal.detach()-planes['centre']).norm(dim=-1) <= neighbourhood_m
    known = planes['valid'] & nearby
    proximity = torch.exp(-.5*(offset/near_m).square())
    correspondence = (displacement <= correspondence_m).to(target)
    result = {key:[] for key in ('near_surface', 'correspondence', 'valid')}
    for i in range(len(target)):
        for key, values in (('near_surface', proximity), ('correspondence', correspondence)):
            result[key].append(torch.stack([
                (values[i, :, labels[i]==j]*known[i, :, labels[i]==j]).sum(-1)
                /known[i, :, labels[i]==j].sum(-1).clamp_min(1) for j in range(22)], -1))
        result['valid'].append(torch.stack([known[i, :, labels[i]==j].any(-1) for j in range(22)], -1))
    return {key:torch.stack(value).detach() for key,value in result.items()}


def masked_bce(logits, target, valid):
    values = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (values*valid).sum()/valid.sum().clamp_min(1)


def supervision(model, prediction, details, batch, policy):
    """Label-only branch; the public inference payload is unchanged."""
    stride = policy['frame_stride']; indices = slice(None, None, stride)
    formats = batch['body_format']
    observed = dict(batch,
        local_frame_points=batch['local_frame_points'][:, indices],
        local_frame_owner=batch['local_frame_owner'][:, indices])
    with torch.no_grad():
        teacher = dict(rest=batch['target_rest'], predicted_native_betas=batch['target_native_betas'],
                       predicted_template=batch['target_template'])
        target, labels = model.skin(batch['motion'][:, indices], teacher, formats)
        proposal, _ = model.skin(details['baseline_motion'][:, indices], details, formats)
        planes = observed_planes(target, observed, policy)
        gates = gate_targets(target, proposal, labels, planes,
            near_m=policy['near_surface_m'], correspondence_m=policy['correspondence_m'],
            neighbourhood_m=policy['query_radius_m'])
    predicted, _ = model.skin(prediction[:, indices], details, formats)
    terms = dict(surface_relation=relation_loss(predicted, target, planes, scale_m=policy['loss_scale_m']),
        near_surface_gate=masked_bce(details['contact_logits'][:, indices], gates['near_surface'], gates['valid']),
        surface_correspondence_gate=masked_bce(details['correspondence_logits'][:, indices],
                                             gates['correspondence'], gates['valid']))
    coverage = dict(anchor_fraction=planes['valid'].float().mean((1,2)).detach().cpu().tolist(),
                    joint_fraction=gates['valid'].float().mean((1,2)).detach().cpu().tolist(),
                    target_normal_offset_mean_mm=[float(v.abs()[m].mean()*1000) if m.any() else None
                        for v,m in zip(((target-planes['centre'])*planes['normal']).sum(-1),planes['valid'])])
    return terms, coverage
