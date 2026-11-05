"""Observed-patch nonpenetration, static support and stance sliding objectives.

These are local observed-surface proxies, not watertight-mesh collision tests.
Unknown/nonplanar patches and source-body conflicts contribute no constraint.
"""
import torch
from torch.nn import functional as F

from scenemi.geometry.surface_query import query_surface,plane_gap


def _nominal_mean(value,weight,nominal):
    return (value*weight).flatten(1).sum(1)/nominal.flatten(1).sum(1).clamp_min(1)


def geometry_terms(model,prediction,details,batch,signal,policy):
    stride = policy['loss_frame_stride']
    time = slice(None,None,stride)
    formats = batch['body_format']
    predicted,labels = model.skin_points(prediction[:,time],details,formats)
    observed = dict(batch,frame_scene_points=batch['frame_scene_points'][:,time],
        frame_scene_owner=batch['frame_scene_owner'][:,time],
        frame_normals=batch['frame_normals'][:,time],frame_normal_valid=batch['frame_normal_valid'][:,time])
    with torch.no_grad():
        teacher = dict(rest=batch['target_rest'],predicted_native_betas=batch['target_native_betas'],
                       predicted_template=batch['target_template'])
        target,_ = model.skin_points(batch['motion'][:,time],teacher,formats)
        patch = query_surface(predicted.detach(),observed,policy)
        native = (formats>0)[:,None,None].expand_as(patch['valid'])
        teacher_gap = plane_gap(target,patch)
        valid = patch['valid'] & native & (teacher_gap>=-policy['source_penetration_tolerance_m'])
        support = query_surface(target,observed,policy,dynamic=False)
        target_gap = plane_gap(target,support)
        feet = ((labels==7)|(labels==10)|(labels==8)|(labels==11))[:,None].expand_as(valid)
        velocity = torch.zeros_like(target)
        velocity[:,1:] = torch.diff(target,dim=1)*(30/stride)
        velocity[:,0] = velocity[:,1]
        stance = feet & native & (velocity.norm(dim=-1)<=policy['stance_speed_mps'])
        contact_valid = support['valid'] & (support['normal'][...,1]>=policy['support_normal_y'])
        contact_valid &= (target_gap.abs()<=policy['source_contact_tolerance_m'])
        foot_id = ((labels==8)|(labels==11)).long()[:,None].expand_as(valid)
        source_weight = batch['contact_weight'][:,time].gather(2,foot_id).detach()
        weight = source_weight*stance*contact_valid
    gap = plane_gap(predicted,patch)
    depth = (-gap-policy['penetration_tolerance_m']).clamp_min(0.)
    crossing = F.smooth_l1_loss(depth/policy['distance_scale_m'],torch.zeros_like(depth),reduction='none')
    loss_crossing = _nominal_mean(crossing,valid,native)
    contact_gap = plane_gap(predicted,support)
    excess = (contact_gap.abs()-policy['contact_tolerance_m']).clamp_min(0.)/policy['distance_scale_m']
    contact = F.smooth_l1_loss(excess,torch.zeros_like(excess),reduction='none')
    loss_contact = _nominal_mean(contact,weight,stance)
    relative_velocity = torch.diff(predicted,dim=1)*(30/stride)
    normal = support['normal'][:,1:]
    tangent = relative_velocity-(relative_velocity*normal).sum(-1,keepdim=True)*normal
    with torch.no_grad():
        same_plane = (support['normal'][:,1:]*support['normal'][:,:-1]).sum(-1)>=.95
        interval = stance[:,1:]&stance[:,:-1]
        interval_weight = torch.minimum(weight[:,1:],weight[:,:-1])*same_plane
    slip = ((tangent.norm(dim=-1)-policy['sliding_tolerance_mps']).clamp_min(0.)/policy['velocity_scale_mps']).square()
    loss_sliding = _nominal_mean(slip,interval_weight,interval)
    terms = {name:(values*signal.square()).mean() for name,values in (
        ('scene_nonpenetration',loss_crossing),('scene_support',loss_contact),('scene_sliding',loss_sliding))}
    with torch.no_grad():
        count = valid.flatten(1).sum(1).clamp_min(1)
        contact_count = weight.flatten(1).sum(1).clamp_min(1)
        interval_count = interval_weight.flatten(1).sum(1).clamp_min(1)
        statistics = dict(
            observed_surface_penetration_mm=(depth*valid).flatten(1).sum(1)/count*1000,
            observed_surface_penetrating_fraction=((depth>0)&valid).flatten(1).sum(1)/count,
            observed_surface_coverage=valid.float().flatten(1).mean(1),
            observed_support_gap_mm=(contact_gap.abs()*weight).flatten(1).sum(1)/contact_count*1000,
            observed_support_coverage=(weight>0).float().flatten(1).mean(1),
            observed_support_sliding_mps=(tangent.norm(dim=-1)*interval_weight).flatten(1).sum(1)/interval_count)
    return terms,statistics
