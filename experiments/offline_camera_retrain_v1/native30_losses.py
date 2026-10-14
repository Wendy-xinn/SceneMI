"""Native30 physical objectives and per-track/geometry uncertainty masks."""
import numpy as np
import torch
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.unified_losses import SCALES,JOINT_WEIGHTS
from experiments.offline_camera_retrain_v1.native30_observations import observation_derivatives,sanitize_observations,FPS
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1.contact_supervision import calibrated_contact_loss
from experiments.offline_camera_retrain_v1.native30_gt_reliability import gt_weight,derivative_valid,soft_mean


def weighted_mean(value,mask,gate):
    per=(value*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
    return (per*gate.detach().reshape(-1)).mean()


def base_losses(prediction,truth,batch,signal,raw_motion=None):
    joint=forward_kinematics(prediction,batch['rest']);target=forward_kinematics(truth,batch['rest']).detach()
    rotations=global_rotations(prediction);target_rot=global_rotations(truth).detach()
    weights=prediction.new_ones(201);weights[:3]=2
    qp=gt_weight(batch,'gt_position_weight',prediction)
    ql=gt_weight(batch,'gt_local_rotation_weight',prediction)
    qr=gt_weight(batch,'gt_global_rotation_weight',prediction)
    feature_weight=torch.cat((qp[:,:,0,None].expand(-1,-1,3),ql.repeat_interleave(6,-1),qp.repeat_interleave(3,-1)),-1)
    x0=soft_mean((prediction-truth).square()*weights,feature_weight)
    position=soft_mean((joint-target).square(),qp);rotation=soft_mean((rotations-target_rot).square(),qr)
    velocity=torch.diff(joint,dim=1)*FPS;true_velocity=torch.diff(target,dim=1)*FPS
    acceleration=torch.diff(velocity,dim=1)*FPS;true_acceleration=torch.diff(true_velocity,dim=1)*FPS
    vm=derivative_valid(qp);am=derivative_valid(qp,2)
    v=soft_mean((velocity-true_velocity).square(),vm,signal.square())
    a=soft_mean((acceleration-true_acceleration).square(),am,signal.square())
    # Stable absolute SO(3) orientation teaches constant yaw offsets as well
    # as turning speed. No target heading is inferred from camera or velocity.
    chordal=(rotations-target_rot).square().sum((-1,-2))*.5
    root=soft_mean(chordal[:,:,0],qr[:,:,0]);head=soft_mean(chordal[:,:,15],qr[:,:,15]);neck=soft_mean(chordal[:,:,12],qr[:,:,12])
    relative_p=rotations[:,:,0].transpose(-1,-2)@rotations[:,:,15]
    relative_t=target_rot[:,:,0].transpose(-1,-2)@target_rot[:,:,15]
    coordination=soft_mean((relative_p-relative_t).square(),torch.minimum(qr[:,:,0],qr[:,:,15]))
    turning=prediction.new_zeros(())
    for lag in (1,6,12):  # 1/30, .2, .4 seconds; no old20 lag constants.
        if prediction.shape[1]>lag:
            p=rotations[:,lag:,(0,15)]@rotations[:,:-lag,(0,15)].transpose(-1,-2)
            t=target_rot[:,lag:,(0,15)]@target_rot[:,:-lag,(0,15)].transpose(-1,-2)
            turning+=soft_mean((p-t).square(),derivative_valid(qr,lag)[:,:,(0,15)],signal.square())*(FPS/lag)**2/3
    feet=(10,11);stance=true_velocity[:,:,feet].norm(dim=-1)<.2
    excess=(velocity[:,:,feet].norm(dim=-1)-true_velocity[:,:,feet].norm(dim=-1)-.02).clamp_min(0).square()
    planting=weighted_mean(excess*vm[:,:,feet],stance.to(prediction),signal.square())
    feet_velocity=soft_mean((velocity[:,:,feet]-true_velocity[:,:,feet]).square().sum(-1),vm[:,:,feet],signal.square())
    head_pos=soft_mean((joint[:,:,15]-target[:,:,15]).square(),qp[:,:,15])
    raw_motion=prediction if raw_motion is None else raw_motion
    # Supervise raw Euclidean 6D too: normalizing every rotation before all
    # feature losses can leave an exact 180-degree chordal saddle stationary.
    # The shared output keeps its original direct target before FK projection.
    raw_x0=soft_mean((raw_motion-truth).square()*weights,feature_weight)
    raw_fk=forward_kinematics(raw_motion,batch['rest'])
    raw_consistency=(raw_motion[...,135:].reshape(*prediction.shape[:2],22,3)*2-raw_fk).square().mean()
    # v/a coefficients are physical equivalents of old20 10*dpos² and
    # 2*ddpos², not reused unchanged against m/s and m/s² values.
    total=x0+.5*raw_x0+.2*rotation+2*position+.025*v+.0000125*a+head_pos+.5*raw_consistency
    total+=.25*root+.5*head+.25*neck+.1*coordination+.02*turning+.075*feet_velocity+.1*planting
    return dict(total=total,x0=x0,raw_x0=raw_x0,fk_position_m2=position,fk_velocity_m2_s2=v,
                fk_acceleration_m2_s4=a,root_absolute_chordal=root,head_absolute_chordal=head,
                neck_absolute_chordal=neck,head_root_coordination=coordination,
                relative_turning_s2=turning,planting_m2_s2=planting,raw_fk_consistency_m2=raw_consistency)


def observation_losses(prediction,truth,batch,signal,observation_config=None):
    scales=SCALES if observation_config is None else observation_config['scales']
    joint_weights=JOINT_WEIGHTS if observation_config is None else observation_config['joint_weights']
    observed,meta=sanitize_observations(batch['trajectory'],batch['observation_meta'])
    p=forward_kinematics(prediction,batch['rest']);r=global_rotations(prediction)
    pm,rm=meta[...,0]>0,meta[...,1]>0
    target=observed[...,:3]*2
    six=torch.where(rm[...,None],observed[...,3:],observed.new_tensor(IDENTITY_6D))
    tr=rotation_from_6d(six-observed.new_tensor(IDENTITY_6D))
    tv,ta,tw,dm=observation_derivatives(observed,meta)
    pv=torch.zeros_like(p);pv[:,1:]=torch.diff(p,dim=1)*FPS
    pa=torch.zeros_like(p);pa[:,2:]=torch.diff(pv[:,1:],dim=1)*FPS
    pw=torch.zeros_like(p);pw[:,1:]=matrix_to_axis_angle(r[:,1:]@r[:,:-1].transpose(-1,-2))*FPS
    cp,cr=meta[...,2],meta[...,3]
    vc=torch.zeros_like(cp);vc[:,1:]=torch.minimum(cp[:,1:],cp[:,:-1])
    ac=torch.zeros_like(cp);ac[:,2:]=torch.minimum(vc[:,2:],vc[:,1:-1])
    wc=torch.zeros_like(cr);wc[:,1:]=torch.minimum(cr[:,1:],cr[:,:-1])
    importance=prediction.new_tensor(joint_weights)[None,None]
    qp=gt_weight(batch,'gt_position_weight',prediction);qr=gt_weight(batch,'gt_global_rotation_weight',prediction)
    vp=torch.zeros_like(qp);vp[:,1:]=derivative_valid(qp)
    ap=torch.zeros_like(qp);ap[:,2:]=derivative_valid(qp,2)
    vr=torch.zeros_like(qr);vr[:,1:]=derivative_valid(qr)
    def reduce(error,mask,confidence,reliability):
        mask=mask.to(prediction)*importance
        values=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')*mask*confidence*reliability
        return ((values.flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1))*signal.square()).mean()
    ps=torch.maximum(batch['observation_uncertainty'][...,0],p.new_tensor(scales['position']))
    rs=torch.maximum(batch['observation_uncertainty'][...,1],p.new_tensor(scales['rotation']))
    terms=dict(position=reduce((p-target).norm(dim=-1)/ps,pm,cp,qp),
        rotation=reduce(matrix_to_axis_angle(r@tr.transpose(-1,-2)).norm(dim=-1)/rs,rm,cr,qr),
        velocity=reduce((pv-tv).norm(dim=-1)/scales['velocity'],dm[...,0],vc,vp),
        acceleration=reduce((pa-ta).norm(dim=-1)/scales['acceleration'],dm[...,1],ac,ap),
        angular_velocity=reduce((pw-tw).norm(dim=-1)/scales['angular_velocity'],dm[...,2],wc,vr))
    root=r[:,:,0];true_root=global_rotations(truth).detach()[:,:,0]
    pred_w=matrix_to_axis_angle(root[:,1:]@root[:,:-1].transpose(-1,-2))*FPS
    true_w=matrix_to_axis_angle(true_root[:,1:]@true_root[:,:-1].transpose(-1,-2))*FPS
    e=(pred_w-true_w).norm(dim=-1)/scales['body_turn_rate']
    terms['body_turn_rate']=soft_mean(F.smooth_l1_loss(e,torch.zeros_like(e),reduction='none'),derivative_valid(qr)[:,:,0],signal.square())
    return terms


def geometry_losses(prediction,details,batch,signal,tolerance_m=.02,slip_tolerance_m_s=.2):
    zero=prediction.sum()*0;distance=[];slip=[];plane=[];correspondence=[]
    qp=gt_weight(batch,'gt_position_weight',prediction)
    for b,(g,asset) in enumerate(zip(details['geometry'],batch['skin_assets'])):
        vertices=skin_points(prediction[b],asset);labels=g['labels']
        delta=vertices-g['closest_points'];normal=g['closest_normals']
        signed=(delta*normal).sum(-1);tangent=(delta-signed[...,None]*normal).norm(dim=-1)
        reliable=g['closest_valid']&g['closest_normal_valid']&(tangent<.09)&(delta.norm(dim=-1)<.25)
        annotated=batch['contact_valid'][b,:,labels]& (batch['contact_target'][b,:,labels]>.5)
        def mean(value,mask):
            return (value*mask).sum()/mask.sum().clamp_min(1)*signal[b].square()
        quality=qp[b,:,labels]
        distance.append(mean((delta.norm(dim=-1)-tolerance_m).clamp_min(0).square()*quality,reliable&annotated))
        # Weak observed-plane proxy with tolerance, never a claim of true
        # collision detection/closed-scene penetration. No unknown constraints.
        plane.append(mean((-signed-tolerance_m).clamp_min(0).square(),reliable))
        velocity=torch.diff(vertices,dim=0)*FPS
        surface_v=g['closest_velocity'][1:]
        mask=reliable[1:]&reliable[:-1]&annotated[1:]&annotated[:-1]
        mask &=g['closest_velocity_valid'][1:]&g['closest_velocity_valid'][:-1]
        mask &=g['closest_owner'][1:]==g['closest_owner'][:-1]
        relative=(velocity-surface_v).norm(dim=-1)
        slip.append(mean((relative-slip_tolerance_m_s).clamp_min(0).square()*derivative_valid(qp[b:b+1])[0,:,labels],mask))
        region_target=torch.stack([reliable[:,labels==j].any(-1) for j in range(22)],1).to(prediction)
        region_mask=g['region_valid'].to(prediction)
        logits=details['correspondence_logits'][b]
        correspondence.append(mean(F.binary_cross_entropy_with_logits(logits,region_target,reduction='none'),region_mask))
    avg=lambda values:torch.stack(values).mean() if values else zero
    dr=details['root_delta_m'];rotation=details['rotation_delta_rad']
    smooth=(torch.diff(dr,dim=1)*FPS).square().mean()+(torch.diff(rotation,dim=1)*FPS).square().mean()
    contact_error=F.binary_cross_entropy_with_logits(details['contact_logits'],batch['contact_target'].to(prediction),reduction='none')
    contact_mask=batch['contact_valid'].to(prediction)
    contact=(contact_error*contact_mask*qp*signal.detach()[:,None,None]).sum()/contact_mask.sum().clamp_min(1)
    return dict(contact=contact,
        contact_distance_m2=avg(distance),contact_relative_slip_m2_s2=avg(slip),
        observed_plane_proxy_m2=avg(plane),correspondence=avg(correspondence),feedback_smoothness=smooth)


def objective(prediction,truth,batch,signal,details,config,observation_weights):
    base=base_losses(prediction,truth,batch,signal,details['raw_motion'])
    observation=observation_losses(prediction,truth,batch,signal,config['observation'])
    if config.get('scene_relations',{}).get('enabled'):
        from experiments.offline_camera_retrain_v1.native30_relation_losses import relation_geometry_losses
        geometry=relation_geometry_losses(prediction,details,batch,signal)
    else:
        geometry=geometry_losses(prediction,details,batch,signal,config['geometry']['tolerance_m'],config['geometry']['slip_tolerance_m_s'])
    total=base['total']+sum(observation_weights[k]*v for k,v in observation.items())
    total+=sum(config['geometry']['coefficients'][k]*v for k,v in geometry.items())
    return total,dict(base=base,observation=observation,geometry=geometry)
