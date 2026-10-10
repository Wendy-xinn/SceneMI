"""Separate, bounded observation ablations; previous simulator stays frozen."""
import torch
from experiments.offline_camera_retrain_v1.typed_head_condition import prepare_observation

PROTOCOLS=('joint','rotation_only','mild')


def quota_mask(batch_size, step, seed, device='cpu'):
    """Exactly 20% across each five-step block, before original control masks."""
    if step<1:raise ValueError('step starts at one')
    block,slot=divmod(step-1,5)
    g=torch.Generator(device='cpu').manual_seed(seed+19731+block*104729)
    selection=torch.zeros(5*batch_size,dtype=torch.bool)
    selection[torch.randperm(5*batch_size,generator=g)[:batch_size]]=True
    return selection.reshape(5,batch_size)[slot].to(device)


def prepare_bounded_observation(batch, protocol, *, generator=None,
                                evaluation='clean', step=None, seed=2026):
    if protocol not in PROTOCOLS:raise ValueError(protocol)
    if generator is None and evaluation!='mild_drift':
        return prepare_observation(batch,'joint',evaluation=evaluation)
    out=prepare_observation(batch,'joint')
    track=out['trajectory'];meta=out['observation_meta'];b,t=track.shape[:2]
    phase=torch.linspace(0,1,t,device=track.device)
    if generator is None:
        selected=torch.ones(b,dtype=torch.bool,device=track.device)
        offset=track.new_zeros(b,t,3);offset[...,0]=.01*phase
        yaw=(3*phase).expand(b,-1)
    else:
        if step is None:raise ValueError('Training augmentation requires step for fixed quota')
        selected=quota_mask(b,step,seed,track.device)
        # All protocols consume identical dedicated RNG, leaving global/diffusion
        # RNG untouched. Random endpoints stay within a 1cm Euclidean ball.
        draw=torch.rand(b,10,generator=generator,device='cpu').to(track.device)
        start=draw[:,:3]*2-1;end=draw[:,3:6]*2-1
        start=start/start.norm(dim=-1,keepdim=True).clamp_min(1e-8)*(.01*draw[:,6:7])
        end=end/end.norm(dim=-1,keepdim=True).clamp_min(1e-8)*(.01*draw[:,7:8])
        offset=(1-phase[None,:,None])*start[:,None]+phase[None,:,None]*end[:,None]
        yaw=(1-phase[None])*(draw[:,8:9]*6-3)+phase[None]*(draw[:,9:10]*6-3)
        if protocol=='rotation_only':meta[selected,:,15,1]=0
        if protocol!='mild':selected=torch.zeros_like(selected)
    if generator is None or protocol=='mild':
        offset=offset*selected[:,None,None];yaw=yaw*selected[:,None]
        track[:,:,15,:3]+=offset/2
        angle=torch.deg2rad(yaw);r=track.new_zeros(b,t,3,3)
        r[...,0,0]=r[...,2,2]=angle.cos();r[...,0,2]=angle.sin();r[...,2,0]=-angle.sin();r[...,1,1]=1
        columns=track[:,:,15,3:].reshape(b,t,2,3).transpose(-1,-2)
        track[:,:,15,3:]=(r@columns).transpose(-1,-2).reshape(b,t,6)
    return out
