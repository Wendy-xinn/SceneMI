"""Observed-anchor supervision, with sparse-region semantics and GT calibration."""
import torch
from torch.nn import functional as F

from experiments.offline_camera_retrain_v1.native30_scene_support import region_contact_loss
from experiments.offline_camera_retrain_v1.native30_gt_reliability import gt_weight
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points


def relation_geometry_losses(prediction,details,batch,signal):
    zero=prediction.sum()*0;distances=[];slips=[];planes=[];relations=[];correspondence=[]
    quality=gt_weight(batch,'gt_position_weight',prediction)
    for b,(g,asset) in enumerate(zip(details['geometry'],batch['skin_assets'])):
        labels=asset['labels'];vertices=skin_points(prediction[b],asset)
        # These correspondences are supervision only and never enter forward.
        if 'support_points' not in batch:raise ValueError('Relation training requires observed TRAIN support targets')
        point=batch['support_points'][b];normal=batch['support_normals'][b]
        truth=batch['support_vertices'][b];known=batch['support_surface_valid'][b] & (quality[b,:,labels]>.75)
        dp=vertices-point;dt=truth-point
        pd=dp.norm(dim=-1);td=dt.norm(dim=-1)
        label_weight=batch['contact_weight'][b]*quality[b]
        distances.append(region_contact_loss(pd,td,labels,known,batch['contact_target'][b],
                                             batch['contact_valid'][b],label_weight,slack_m=.01)*signal[b].square())
        def mean(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)*signal[b].square()
        ps=(dp*normal).sum(-1);ts=(dt*normal).sum(-1)
        near=known & (td<.25)
        # GT-calibrated proxy avoids a force against perfectly valid bodies
        # on the hidden side of camera-facing patches. Not a closed-scene SDF.
        planes.append(mean(((-ps-.02).clamp_min(0)-(-ts-.02).clamp_min(0)).clamp_min(0).square(),near))
        relations.append(mean((ps-ts).square(),near))
        selected=torch.zeros_like(known)
        for j in range(22):
            ids=(labels==j).nonzero().flatten()
            score=torch.where(known[:,ids],td[:,ids],torch.inf)
            chosen=score.argmin(-1)
            active=score.gather(1,chosen[:,None])[:,0]<.05
            active &=batch['contact_valid'][b,:,j] & (batch['contact_target'][b,:,j]>.5)
            selected.scatter_(1,ids[chosen][:,None],active[:,None])
        vel=vertices.diff(dim=0)*30-batch['support_velocities'][b,1:]
        target_vel=truth.diff(dim=0)*30-batch['support_velocities'][b,1:]
        mask=selected[1:] & selected[:-1] & batch['support_velocity_valid'][b,1:] & batch['support_velocity_valid'][b,:-1]
        mask &=batch['support_owners'][b,1:]==batch['support_owners'][b,:-1]
        excess=(vel.norm(dim=-1)-target_vel.norm(dim=-1)-.05).clamp_min(0)
        slips.append(mean(excess.square(),mask))
        region_target=torch.stack([near[:,labels==j].any(-1) for j in range(22)],-1).to(prediction)
        region_mask=g['region_valid'].to(prediction)
        correspondence.append(mean(F.binary_cross_entropy_with_logits(details['correspondence_logits'][b],region_target,reduction='none'),region_mask))
    avg=lambda values:torch.stack(values).mean() if values else zero
    error=F.binary_cross_entropy_with_logits(details['contact_logits'],batch['contact_target'].to(prediction),
                                             reduction='none',pos_weight=prediction.new_tensor(2.))
    weights=batch['contact_valid'].to(prediction)*batch['contact_weight']*quality
    contact=(error*weights*signal.detach()[:,None,None]).sum()/batch['contact_valid'].sum().clamp_min(1)
    dr=details['root_delta_m'];rr=details['rotation_delta_rad']
    smooth=(dr.diff(dim=1)*30).square().mean()+(rr.diff(dim=1)*30).square().mean()
    return dict(contact=contact,contact_distance_m2=avg(distances),contact_relative_slip_m2_s2=avg(slips),
                observed_plane_proxy_m2=avg(planes),correspondence=avg(correspondence),
                feedback_smoothness=smooth,support_relation_m2=avg(relations))
