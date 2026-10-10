"""Experimental plan reuse at lower noise; never a future GT motion input."""
import torch
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D


def shift_prior(previous, executed_history, rest, execute_frames=8):
    """Shift the previous prediction; extrapolate only its far lookahead tail."""
    shifted=previous[:,execute_frames:].clone(); b,t=previous.shape[:2]
    rotations=rotation_from_6d(previous[:,-2:,3:135].reshape(b,2,22,6))
    delta=rotations[:,1]@rotations[:,0].transpose(-1,-2)
    step=previous[:,-1,:3]-previous[:,-2,:3]
    tail=[]; last=previous[:,-1].clone(); r=rotations[:,1]
    for _ in range(execute_frames):
        r=delta@r; last=last.clone();last[:,:3]+=step
        last[:,3:135]=(torch.cat((r[..., :,0],r[..., :,1]),-1)-last.new_tensor(IDENTITY_6D)).flatten(1)
        tail.append(last)
    prior=torch.cat((shifted,torch.stack(tail,1)),1)
    prior[:,:executed_history.shape[1]]=executed_history
    prior[...,135:]=(forward_kinematics(prior,rest)/2).flatten(2)
    assert prior.shape==previous.shape and torch.isfinite(prior).all()
    return prior


@torch.no_grad()
def sample_warm_prefix(model,batch,history,prior,*,seed,steps=20,start_timestep=400):
    """Noisy previous planned x0, with actual past reinserted at every DDIM step."""
    if not 1<=start_timestep<=999 or batch.get('history_confidence',1.)<.999:
        raise ValueError('Require valid timestep and trusted past')
    b,t=batch['trajectory'].shape[:2];h=history.shape[1]
    if prior.shape!=(b,t,201) or not torch.isfinite(prior).all():raise ValueError('Invalid planned prior')
    alpha=cosine_alphas().to(prior);g=torch.Generator(device=prior.device).manual_seed(seed)
    noise=torch.randn(prior.shape,generator=g,device=prior.device,dtype=prior.dtype)
    clean=history.clone();clean[...,135:]=(forward_kinematics(clean,batch['rest'])/2).flatten(2)
    sample=alpha[start_timestep].sqrt()*prior+(1-alpha[start_timestep]).sqrt()*noise
    schedule=torch.linspace(start_timestep,0,steps,device=prior.device).round().long().unique_consecutive()
    for i,tvalue in enumerate(schedule):
        a=alpha[tvalue];sample[:,:h]=a.sqrt()*clean+(1-a).sqrt()*noise[:,:h]
        x0=model(sample,torch.full((b,),int(tvalue),device=prior.device,dtype=torch.long),batch)
        x0[:,:h]=clean;eps=(sample-a.sqrt()*x0)/(1-a).sqrt().clamp_min(1e-6)
        if i+1==len(schedule):sample=x0
        else:
            following=alpha[schedule[i+1]];sample=following.sqrt()*x0+(1-following).sqrt()*eps
    assert torch.equal(sample[:,:h],clean)
    return sample
