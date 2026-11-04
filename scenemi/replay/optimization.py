"""Source-balanced motion losses for AMASS replay and native body formats."""
import math
import torch
from scenemi import native_joint_training as stage
from scenemi.data.exo_initialization import repair_payload
from scenemi.exo_denoise import channel_reliability, denoising_loss
from scenemi.losses import motion_loss
from scenemi.native_joint_data import ego_payload
from scenemi.scene_model import native_motion_loss


def subset(value,mask):
    if isinstance(value,dict):return {k:subset(v,mask) for k,v in value.items()}
    if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==len(mask):return value[mask]
    return value


def optimizer_factor(c,step):
    if 'sampled_windows' in c['training']:
        exposure=step*c['training']['batch_size']
        warmup=c['optimizer']['warmup_windows'];total=c['training']['sampled_windows']
    else:exposure=step;warmup=c['optimizer']['warmup_updates'];total=c['training']['updates']
    return min(exposure/warmup,1.)*(.1+.9*.5*(1+math.cos(math.pi*exposure/total)))


def source_total(losses,weights):
    if len(losses)!=4 or len(weights)!=4 or any(w<=0 for w in weights) or abs(sum(weights)-1)>1e-8:
        raise ValueError('Four positive, normalized source weights required')
    return sum(w*v for w,v in zip(weights,losses))


def update(model,optimizer,ego,exo,c,reference,step,diagnostic=False):
    model.train();optimizer.zero_grad(set_to_none=True)
    weights=c.get('source_weights',[.25]*4)
    count=len(ego['motion']);t=ego.get('diffusion_t')
    if t is None:t=torch.randint(0,1000,(count,),device=ego['motion'].device)
    alpha=model.alphas[t,None,None]
    noise=ego['diffusion_noise'] if 'diffusion_noise' in ego else torch.randn_like(ego['motion'])
    noisy=alpha.sqrt()*ego['motion']+(1-alpha).sqrt()*noise
    prediction,details=model(noisy,t,ego_payload(ego),return_details=True)
    ego_losses=[]
    for source in range(4):
        mask=ego['dataset_id']==source
        loss=motion_loss if source==0 else native_motion_loss
        terms=loss(prediction[mask],ego['motion'][mask],subset(ego,mask),alpha[mask,0,0].sqrt(),
            details=subset(details,mask),config=reference)
        ego_losses.append(terms['total'])
    loss_ego=source_total(ego_losses,weights);(.5*loss_ego).backward()
    shared=[p for p in model.base.parameters() if p.requires_grad]
    ego_grad=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in shared] if diagnostic else None
    t=exo.get('diffusion_t')
    if t is None:t=torch.randint(0,c['diffusion']['start_timestep']+1,(len(exo['motion']),),device=exo['motion'].device)
    alpha=model.alphas[t,None,None]
    noise=exo['diffusion_noise'] if 'diffusion_noise' in exo else torch.randn_like(exo['motion'])
    payload=repair_payload(exo)
    clean=alpha.sqrt()*exo['motion']+(1-alpha).sqrt()*noise
    observed=alpha.sqrt()*model.anchor(payload)+(1-alpha).sqrt()*noise
    noisy=torch.lerp(clean,observed,channel_reliability(payload))
    prediction,details=model(noisy,t,payload,return_details=True)
    if not details['reference_x0'].requires_grad:raise AssertionError('Shared reference gradient detached')
    loss_exo=source_total([denoising_loss(prediction[exo['dataset_id']==source],
        subset(exo,exo['dataset_id']==source),subset(details,exo['dataset_id']==source))['total'] for source in range(4)],weights)
    (.5*loss_exo).backward()
    gradient_diagnostic={}
    if diagnostic:
        exo_grad=[(p.grad.detach() if p.grad is not None else torch.zeros_like(p))-g for p,g in zip(shared,ego_grad)]
        a=torch.stack([g.square().sum() for g in ego_grad]).sum().sqrt()
        b=torch.stack([g.square().sum() for g in exo_grad]).sum().sqrt()
        dot=torch.stack([(g*h).sum() for g,h in zip(ego_grad,exo_grad)]).sum()
        v=torch.stack((a,b,dot/(a*b).clamp_min(1e-20))).cpu().tolist()
        gradient_diagnostic=dict(ego_shared_gradient_norm=v[0],exo_shared_gradient_norm=v[1],shared_gradient_cosine=v[2])
    group_norms=torch.stack([torch.stack([p.grad.detach().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt()
        for g in optimizer.param_groups])
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['optimizer']['clip_norm'],error_if_nonfinite=True)
    factor=optimizer_factor(c,step)
    for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
    optimizer.step()
    values=torch.cat((torch.stack((loss_ego.detach(),loss_exo.detach(),norm)),group_norms)).cpu().tolist()
    if not all(math.isfinite(v) for v in values) or not all(v>0 for v in values[3:]):raise FloatingPointError('Invalid loss or optimizer group')
    return dict(loss_ego=values[0],loss_exo=values[1],gradient_norm=values[2],gradient_groups=dict(zip([g['name'] for g in optimizer.param_groups],values[3:])),**gradient_diagnostic)
