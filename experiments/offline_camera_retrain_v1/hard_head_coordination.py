"""Known anatomical head pose projection and experimental whole-body repair.

Inputs contain generated motion, configured rest body, calibrated head track,
and observed support geometry only. No target pose/stance/contact parameters.
Projection is exact; anatomical/contact feasibility is checked separately.
"""
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

def project_head_pose(motion,rest,position_m,rotation,position_valid,rotation_valid):
    b,t=motion.shape[:2]
    if position_m.shape!=(b,t,3) or rotation.shape!=(b,t,3,3):raise ValueError('Anatomical head poses required in meters/SO(3)')
    if position_valid.shape!=(b,t) or rotation_valid.shape!=(b,t):raise ValueError('Separate observation masks required')
    result=motion.clone();parent=global_rotations(result)[:,:,12]
    local=parent.transpose(-1,-2)@rotation
    six=local.transpose(-1,-2)[...,:2,:].flatten(-2)-motion.new_tensor(IDENTITY_6D)
    result[...,93:99]=torch.where(rotation_valid[...,None],six,motion[...,93:99])
    delta=position_m-forward_kinematics(result,rest)[:,:,15]
    result[...,:3]=motion[...,:3]+torch.where(position_valid[...,None],delta/2,torch.zeros_like(delta))
    # Keep the redundant joint channels consistent with corrected native FK.
    result[...,135:]=forward_kinematics(result,rest).flatten(2)/2
    return result

def _chord_angle(r):
    return (r-torch.eye(3,device=r.device,dtype=r.dtype)).square().sum((-1,-2)).clamp_min(1e-10).sqrt()/2**.5

def _yaw_path(rotation):
    # Gravity-axis swing/twist heading from the complete rotation. Forward
    # projection alone becomes singular when a seated wearer looks down.
    sine=rotation[...,0,2]-rotation[...,2,0]
    cosine=rotation[...,0,0]+rotation[...,2,2]
    wrapped=torch.atan2(sine,cosine)
    difference=wrapped[...,1:]-wrapped[...,:-1]
    increment=torch.atan2(difference.sin(),difference.cos())
    return torch.cat((wrapped[...,:1],wrapped[...,:1]+increment.cumsum(-1)),-1)

def coordinate_hard_head(generated,rest,head_position_m,head_rotation,*,floor,skin,iterations=160,turn_scaffold=False):
    if len(generated)!=1:raise ValueError('Experimental solver processes one complete sequence')
    if floor is None:raise ValueError('No observed support patch; do not extrapolate free space')
    base=generated.detach();valid=torch.ones(base.shape[:2],dtype=torch.bool,device=base.device)
    coef,tree,_=floor
    def height(v):return v[...,1]-v[...,0]*float(coef[0])-v[...,2]*float(coef[1])-float(coef[2])
    with torch.no_grad():
        vertices=skin(base[0]);base_h=height(vertices);base_r=rotation_from_6d(base[...,3:135].reshape(1,-1,22,6))
        joints=forward_kinematics(base,rest);joint_velocity=torch.diff(joints,dim=1)
        initial=base[...,3:135].clone()
        head_path=_yaw_path(head_rotation);root_path=_yaw_path(base_r[:,:,0])
        if turn_scaffold:
            heading_strength=((head_rotation[...,0,2]-head_rotation[...,2,0]).square()+(head_rotation[...,0,0]+head_rotation[...,2,2]).square()).sqrt()
            if (heading_strength<.3).any():raise ValueError('Gravity twist unobservable: no automatic turn scaffold')
            offset=torch.atan2(torch.sin(root_path[:,0]-head_path[:,0]),torch.cos(root_path[:,0]-head_path[:,0])).clamp(-np.deg2rad(60),np.deg2rad(60))
            desired=head_path+offset[:,None];delta=desired-root_path
            ry=base.new_zeros(*delta.shape,3,3);ry[...,0,0]=ry[...,2,2]=delta.cos();ry[...,0,2]=delta.sin();ry[...,2,0]=-delta.sin();ry[...,1,1]=1
            root=ry@base_r[:,:,0]
            initial[...,:6]=root.transpose(-1,-2)[...,:2,:].flatten(-2)-base.new_tensor(IDENTITY_6D)
            # Preserve intrinsic gait velocity instead of forcing the world
            # velocity of a body that originally turned the wrong way.
            relative=base_r[:,:,0].transpose(-1,-2)[:, :,None]@(joints-joints[:,:,:1])[...,None]
            local_velocity=torch.diff(relative[...,0],dim=1)
        nearest=tree.query(vertices[..., [0,2]].cpu().numpy().reshape(-1,2))[0].reshape(vertices.shape[:2]);known=torch.tensor(nearest<.6,device=base.device)
        witnesses=torch.zeros_like(known);stable=torch.zeros_like(known[1:])
        target_h=torch.zeros_like(base_h)
        for labels in ((7,10),(8,11)):
            ids=(skin.labels==labels[0])|(skin.labels==labels[1]);h=base_h[:,ids];ok=known[:,ids]
            low=h.masked_fill(~ok,100).min(-1).values
            sole=vertices[:,ids].mean(1);speed=torch.diff(sole,dim=0)[...,[0,2]].norm(dim=-1)
            eligible=(low>-.10)&(low<.04)&ok.any(-1)
            stance=(speed<.03)&eligible[1:]&eligible[:-1]
            phase=torch.zeros_like(eligible);phase[1:]|=stance;phase[:-1]|=stance
            witness=(h<=low[:,None]+.015)&ok&phase[:,None]
            witnesses[:,ids]=witness;stable[:,ids]=witness[1:]&witness[:-1]&stance[:,None]
            target_h[:,ids]=(h-low[:,None]).clamp(0,.015)
    rotations=initial.requires_grad_(True)
    optimizer=torch.optim.Adam([rotations],lr=.006)
    trace=[]
    def masked(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)
    for step in range(iterations):
        optimizer.zero_grad();trial=torch.cat((base[...,:3],rotations,base[...,135:]),-1)
        motion=project_head_pose(trial,rest,head_position_m,head_rotation,valid,valid)
        v=skin(motion[0]);h=height(v);r=rotation_from_6d(motion[...,3:135].reshape(1,-1,22,6));g=global_rotations(motion)
        prior=(r-base_r).square().sum((-1,-2)).mean()
        plant=masked(((h-target_h)/.03).square(),witnesses)
        velocity=torch.diff(v,dim=0)[...,[0,2]]
        # Witnesses are selected once from generated geometry, never disabled
        # when the corrected feet lift. Zero velocity is a support hypothesis,
        # not a ground-truth gait phase. Wrong phase inference may invalidate it.
        slide=masked((velocity/.01).square().sum(-1),stable)
        penetration=masked(((-h-.01).relu()/.03).square(),known)
        j=forward_kinematics(motion,rest)
        natural=((torch.diff(j,dim=1)-joint_velocity)/.03).square().mean()
        path_penalty=motion.new_zeros(())
        if turn_scaffold:
            relative=g[:,:,0].transpose(-1,-2)[:,:,None]@(j-j[:,:,:1])[...,None]
            natural=((torch.diff(relative[...,0],dim=1)-local_velocity)/.03).square().mean()
            path=_yaw_path(g[:,:,0]);deviation=(path-path[:,:1])-(head_path-head_path[:,:1])
            # Unwrapped, elapsed-path bounds distinguish a left turn from a
            # 360-degree-equivalent right turn. This is a candidate hypothesis,
            # not a claim that head and pelvis always have identical heading.
            path_penalty=((deviation.abs()-np.deg2rad(45)).relu()/np.deg2rad(15)).square().mean()
        # Broad anatomical priors allow looking sideways; head==pelvis is not
        # imposed. Penalize excessive total head-root yaw and head-neck bend.
        f=g[0,:,(0,15),:,2][...,[0,2]]
        yaw=torch.atan2(f[:,0,1]*f[:,1,0]-f[:,0,0]*f[:,1,1],(f[:,0]*f[:,1]).sum(-1))
        yaw_valid=(f.norm(dim=-1)>.3).all(-1)
        neck=masked(((yaw.abs()-np.deg2rad(65)).relu()/np.deg2rad(15)).square(),yaw_valid)
        head_joint=_chord_angle(r[0,:,15]);neck=neck+((head_joint-2*np.sin(np.deg2rad(45)/2)).relu()/.25).square().mean()
        dr=g[:,1:,0]@g[:,:-1,0].transpose(-1,-2)
        rate=((_chord_angle(dr)-2*np.sin(np.deg2rad(10)/2)).relu()/.05).square().mean()
        total=.15*prior+.5*plant+slide+.5*penetration+.15*natural+.5*neck+.2*rate+path_penalty
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite coordinate repair')
        total.backward();torch.nn.utils.clip_grad_norm_([rotations],10);optimizer.step()
        if step%40==0 or step==iterations-1:trace.append({k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in dict(step=step,total=total,prior=prior,support=plant,slide=slide,neck=neck,turn_path=path_penalty).items()})
    with torch.no_grad():
        result=project_head_pose(torch.cat((base[...,:3],rotations,base[...,135:]),-1),rest,head_position_m,head_rotation,valid,valid)
    return result,dict(iterations=iterations,turn_scaffold=turn_scaffold,inferred_support_vertex_frames=int(witnesses.sum()),inferred_support_vertex_pairs=int(stable.sum()),trace=trace,GT_body_contact_or_history_consumed=False,solver='post-sampling feasibility prototype; not a learned diffusion improvement')
