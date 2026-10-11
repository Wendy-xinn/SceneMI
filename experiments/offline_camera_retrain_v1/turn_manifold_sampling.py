"""Frozen55k feasibility probe: re-denoise legs along a generated turn hypothesis.

Head poses are calibrated observations. Root heading is an inferred candidate,
never a measured/GT pelvis track. Initial diffusion body remains random noise.
"""
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.hard_head_coordination import _yaw_path,project_head_pose
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d,IDENTITY_6D
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.known_input_generation import camera_inputs_only
from experiments.offline_camera_retrain_v1.control import fixed_control_mask

def inferred_turn_path(generated_reference,head_rotation):
    root=rotation_from_6d(generated_reference[...,3:9]);head=_yaw_path(head_rotation);rp=_yaw_path(root)
    strength=((head_rotation[...,0,2]-head_rotation[...,2,0]).square()+(head_rotation[...,0,0]+head_rotation[...,2,2]).square()).sqrt()
    if (strength<.3).any():raise ValueError('Head gravity heading not observable')
    offset=torch.atan2((rp[:,0]-head[:,0]).sin(),(rp[:,0]-head[:,0]).cos()).clamp(-np.deg2rad(60),np.deg2rad(60))
    return head+offset[:,None]

def project_turn_head(motion,rest,head_position_m,head_rotation,root_heading):
    result=motion.clone();r=rotation_from_6d(result[...,3:9]);delta=root_heading-_yaw_path(r)
    ry=motion.new_zeros(*delta.shape,3,3);ry[...,0,0]=ry[...,2,2]=delta.cos();ry[...,0,2]=delta.sin();ry[...,2,0]=-delta.sin();ry[...,1,1]=1
    root=ry@r;result[...,3:9]=root.transpose(-1,-2)[...,:2,:].flatten(-2)-motion.new_tensor(IDENTITY_6D)
    valid=torch.ones(motion.shape[:2],dtype=torch.bool,device=motion.device)
    return project_head_pose(result,rest,head_position_m,head_rotation,valid,valid)

@torch.no_grad()
def sample_turn_manifold(model,observations,head_position_m,head_rotation,generated_reference,*,seed,steps=20,projection_start=500):
    inputs=camera_inputs_only(observations);device=inputs['camera'].device
    if steps<2 or steps>1000:raise ValueError('Invalid DDIM steps')
    b,t=inputs['camera'].shape[:2];root_heading=inferred_turn_path(generated_reference,head_rotation)
    sample=torch.randn((b,t,201),device=device,generator=torch.Generator(device=device).manual_seed(seed));alphas=cosine_alphas().to(device)
    schedule=torch.linspace(999,0,steps,device=device).round().long().unique_consecutive()
    was_training=model.training;model.eval();projected_steps=0
    try:
        for index,value in enumerate(schedule):
            raw=model(sample,torch.full((b,),int(value),device=device,dtype=torch.long),inputs,control_mask=fixed_control_mask(b,t,'head',device),use_scene=True)
            alpha=alphas[value]
            # Keep the prior's epsilon. Recomputing epsilon from projected x0
            # would cancel part of the spatial correction in the DDIM update.
            epsilon=(sample-alpha.sqrt()*raw)/(1-alpha).sqrt().clamp_min(1e-6)
            if int(value)<=projection_start:
                raw=project_turn_head(raw,inputs['rest'],head_position_m,head_rotation,root_heading);projected_steps+=1
            if index+1==len(schedule):sample=raw
            else:
                following=alphas[schedule[index+1]];sample=following.sqrt()*raw+(1-following).sqrt()*epsilon
    finally:model.train(was_training)
    return sample,dict(steps=steps,projected_steps=projected_steps,projection_start=projection_start,body_initialization='random noise',root_heading_source='known head track plus original55k generated initial heading; inferred hypothesis, not GT',denoiser_weights_updated=False)
