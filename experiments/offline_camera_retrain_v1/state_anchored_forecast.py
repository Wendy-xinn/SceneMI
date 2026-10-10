"""Native-body forecast alignment using trusted measured past, no future GT.

Translation/yaw rigid variants preserve within-forecast foot-step lengths and
bone lengths. First-state alignment is imposed by construction; its score
must be assessed together with future world/head/support errors.
"""
import torch
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,rotation_from_6d,IDENTITY_6D


def anchor_forecast(motion,history,rest,*,mode='translation'):
    if mode not in ('translation','rigid'):raise ValueError(mode)
    h=history.shape[1]
    if h<4 or h>=motion.shape[1]:raise ValueError('Need four measured past states and forecast')
    output=motion.clone();past=forward_kinematics(history,rest)
    # A robust three-step measured velocity; no future target derivative.
    velocity=(past[:,-3:,0]-past[:,-4:-1,0]).median(1).values
    desired=past[:,-1,0]+velocity
    future=forward_kinematics(motion[:,h:],rest)
    rotation=torch.eye(3,device=motion.device,dtype=motion.dtype)[None].expand(len(motion),-1,-1).clone()
    if mode=='rigid':
        past_r=rotation_from_6d(history[...,3:9]);new_r=rotation_from_6d(motion[:,h,3:9])
        past_yaw=torch.atan2(past_r[...,0,2],past_r[...,2,2])
        dy=past_yaw[:,-3:]-past_yaw[:,-4:-1];dy=torch.atan2(dy.sin(),dy.cos()).median(1).values
        target=past_yaw[:,-1]+dy;delta=target-torch.atan2(new_r[:,0,2],new_r[:,2,2])
        upright=(past_r[:,-1,[0,2],2].norm(dim=-1)>.3)&(new_r[:,[0,2],2].norm(dim=-1)>.3)
        delta=torch.where(upright,delta,torch.zeros_like(delta))
        rotation[:,0,0]=rotation[:,2,2]=delta.cos();rotation[:,0,2]=delta.sin();rotation[:,2,0]=-delta.sin()
        local=rotation_from_6d(output[:,h:,3:135].reshape(len(motion),-1,22,6))
        local[:,:,0]=rotation[:,None]@local[:,:,0]
        output[:,h:,3:135]=(torch.cat((local[..., :,0],local[..., :,1]),-1)-motion.new_tensor(IDENTITY_6D)).flatten(2)
    relative=future[:,:,0]-future[:,:1,0]
    pelvis=desired[:,None]+(rotation[:,None]@relative[...,None]).squeeze(-1)
    output[:,h:,:3]=(pelvis-rest[:,None,0])/2
    output[:,h:,135:]=(forward_kinematics(output[:,h:],rest)/2).flatten(2)
    assert torch.equal(output[:,:h],motion[:,:h])
    return output
