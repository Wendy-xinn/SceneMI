"""DDIM inpainting of trusted executed BODY history, never future head poses."""
import torch
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


@torch.no_grad()
def sample_with_executed_prefix(model,batch,executed_native_motion,*,steps=20,seed=777,control_mask=None,use_scene=True):
    if batch.get('history_confidence',1.) < .999:
        raise ValueError('Hard prefix requires trusted executed body state; use soft conditions for uncertain estimates')
    if executed_native_motion.ndim!=3 or executed_native_motion.shape[-1]!=201:
        raise ValueError('Native executed state must be [B,H,201]')
    b,t=batch['trajectory'].shape[:2];h=executed_native_motion.shape[1]
    if executed_native_motion.shape[0]!=b or not 0<h<t or not torch.isfinite(executed_native_motion).all():
        raise ValueError('Invalid executed prefix')
    if not 2<=steps<=1000:raise ValueError('Invalid DDIM steps')
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    sample=torch.randn((b,t,201),generator=generator,device=device)
    fixed_noise=sample[:,:h].clone()
    clean=executed_native_motion.to(sample).clone()
    # Do not trust a controller's predicted/direct joint channels.
    clean[...,135:]=(forward_kinematics(clean,batch['rest'])/2).flatten(2)
    alphas=cosine_alphas().to(device)
    schedule=torch.linspace(999,0,steps,device=device).round().long().unique_consecutive()
    was_training=model.training;model.eval()
    try:
        for i,tvalue in enumerate(schedule):
            alpha=alphas[tvalue]
            sample[:,:h]=alpha.sqrt()*clean+(1-alpha).sqrt()*fixed_noise
            timestep=torch.full((b,),int(tvalue),device=device,dtype=torch.long)
            x0=model(sample,timestep,batch,control_mask=control_mask,use_scene=use_scene)
            x0[:,:h]=clean
            epsilon=(sample-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-6)
            if i+1==len(schedule):sample=x0
            else:
                following=alphas[schedule[i+1]]
                sample=following.sqrt()*x0+(1-following).sqrt()*epsilon
        assert torch.equal(sample[:,:h],clean)
        return sample
    finally:model.train(was_training)
