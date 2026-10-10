"""Training-only winding and tail-rate loss; head path stays supervised at high noise."""
import math
import torch
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

def winding_guard_losses(prediction,truth,signal):
    p=global_rotations(prediction);t=global_rotations(truth).detach()
    # Known head trajectory is available even at high diffusion noise. Avoid
    # silencing its winding target with alpha_bar squared near t=999.
    weight=.25+.75*signal.detach().reshape(-1)
    def reduce(x):return (x.flatten(1).mean(1)*weight).mean()
    def increments(r):
        f=r[..., :,2][...,[0,2]];a,b=f[:,:-1],f[:,1:]
        return torch.atan2(a[...,1]*b[...,0]-a[...,0]*b[...,1],(a*b).sum(-1)+1e-6),f
    paths=[]
    for j in (15,0):
        dp,pf=increments(p[:,:,j]);dt,tf=increments(t[:,:,j]);valid=(tf[:,:-1].norm(dim=-1)>.3)&(tf[:,1:].norm(dim=-1)>.3)
        error=dp-dt;error=torch.atan2(error.sin(),error.cos())*valid
        paths.append(reduce(error.cumsum(1).square()))
    joints=(0,12,15,1,2,4,5,7,8)
    rp=p[:,1:,joints]@p[:,:-1,joints].transpose(-1,-2);rt=t[:,1:,joints]@t[:,:-1,joints].transpose(-1,-2)
    eye=torch.eye(3,device=p.device)
    step=(.5*(rp-eye).square().sum((-1,-2))+1e-8).sqrt();true_step=(.5*(rt-eye).square().sum((-1,-2))+1e-8).sqrt()
    # Top10% per joint, not a mean over all frames that hides a single spike.
    k=max(1,math.ceil(step.shape[1]*.1))
    excess=((step-true_step-.05236).relu()/.08727).square()
    rate_error=.5*(rp-rt).square().sum((-1,-2))/.08727**2
    tail=reduce(excess.topk(k,dim=1).values);rate=reduce(rate_error.topk(k,dim=1).values)
    head=reduce(.5*(p[:,:,15]-t[:,:,15]).square().sum((-1,-2)))
    total=paths[0]+.5*paths[1]+.05*tail+.05*rate+.2*head
    return dict(winding_guard_total=total,head_winding_path=paths[0],pelvis_winding_path=paths[1],rotation_excess_tail=tail,signed_relative_rotation_tail=rate,head_rotation_all_noise=head)
