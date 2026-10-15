"""Confidence-aware motion references and shared residual ego/exo diffusion.

The built-in initializer uses observed tracks and supplied rest bones only.
A real frontend can instead supply all 22 joints via reference_motion, with
low confidence for occluded estimates. Neither route reads motion labels.
"""
import torch
from torch import nn
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.native30_model import Native30SceneMI,reconcile_motion,sample_native30
from experiments.offline_camera_retrain_v1.native30_observations import inference_payload,sanitize_observations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D,PARENTS,rotation_from_6d,forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas


def align_vectors(source,target):
    a=source/source.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    b=target/target.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v=torch.cross(a,b,dim=-1);c=(a*b).sum(-1).clamp(-1,1)
    z=torch.zeros_like(v[...,0]);x,y,w=v.unbind(-1)
    k=torch.stack((z,-w,y,w,z,-x,-y,x,z),-1).reshape(*v.shape[:-1],3,3)
    eye=torch.eye(3,device=a.device,dtype=a.dtype).expand_as(k)
    r=eye+k+k@k/(1+c)[...,None,None].clamp_min(1e-6)
    basis=torch.nn.functional.one_hot(a.abs().argmin(-1),3).to(a)
    axis=torch.cross(a,basis,dim=-1);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    opposite=2*axis[..., :,None]*axis[...,None,:]-eye
    r=torch.where((c<-.9999)[...,None,None],opposite,r)
    valid=(source.norm(dim=-1)>1e-6)&(target.norm(dim=-1)>1e-6)
    return torch.where(valid[...,None,None],r,eye)


def fit_rotation(x,y,weights,center=False):
    total=weights.sum(-1,keepdim=True).clamp_min(1e-8)
    weights=weights/total
    cx=(x*weights[...,None]).sum(-2);cy=(y*weights[...,None]).sum(-2)
    if center:x=x-cx[...,None,:];y=y-cy[...,None,:]
    u,s,vt=torch.linalg.svd(x.transpose(-1,-2)@(y*weights[...,None]))
    correction=torch.eye(3,device=x.device,dtype=x.dtype).expand_as(u).clone()
    correction[...,2,2]=torch.linalg.det(vt.transpose(-1,-2)@u.transpose(-1,-2))
    r=vt.transpose(-1,-2)@correction@u.transpose(-1,-2)
    return r,s,cx,cy


def interpolate_evidence(values,valid,rotations=False):
    """Offline interpolation/endpoint hold; gaps remain low confidence."""
    result=values.clone()
    for b in range(len(values)):
        ids=valid[b].nonzero().flatten()
        if not len(ids):continue
        clock=torch.arange(values.shape[1],device=values.device)
        right=torch.searchsorted(ids,clock).clamp_max(len(ids)-1)
        left=(right-1).clamp_min(0)
        left=torch.where(clock>ids[-1],right,left)
        lo,hi=ids[left],ids[right]
        weight=((clock-lo)/(hi-lo).clamp_min(1)).clamp(0,1)
        if rotations:
            delta=matrix_to_axis_angle(values[b,lo].transpose(-1,-2)@values[b,hi])
            filled=values[b,lo]@axis_angle_to_matrix(delta*weight[:,None])
        else:filled=values[b,lo]*(1-weight[:,None])+values[b,hi]*weight[:,None]
        result[b]=torch.where(valid[b].reshape(-1,*([1]*(values.ndim-2))),values[b],filled)
    return result


@torch.no_grad()
def reference_from_observations(batch):
    trajectory,meta=sanitize_observations(batch['trajectory'],batch['observation_meta'])
    pos=trajectory[...,:3]*2;cp=meta[...,0]*meta[...,2]
    rest=batch['rest'];b,t=pos.shape[:2]
    eye=torch.eye(3,device=pos.device,dtype=pos.dtype).expand(b,t,3,3)
    observed_r=rotation_from_6d(trajectory[...,3:]-trajectory.new_tensor(IDENTITY_6D))
    head_valid=meta[:,:,15,1]*meta[:,:,15,3]>0
    fallback=torch.where(head_valid[...,None,None],observed_r[:,:,15],eye)
    x=rest[:,None,:4].expand(b,t,4,3)
    r,s,cx,cy=fit_rotation(x,pos[:,:,:4],cp[:,:,:4],center=True)
    root_valid=((cp[:,:,:4]>0).sum(-1)>=3)&(s[...,1]>1e-7)
    root_r=torch.where(root_valid[...,None,None],r,fallback)
    root_r=interpolate_evidence(root_r,root_valid|head_valid,rotations=True)
    root_pos=cy-(root_r@(cx-rest[:,None,0])[...,None]).squeeze(-1)
    all_w=cp/cp.sum(-1,keepdim=True).clamp_min(1e-8)
    offsets=rest[:,None]-rest[:,None,:1]
    expected=(root_r[:,:,None]@offsets[...,None]).squeeze(-1)
    fallback_pos=((pos-expected)*all_w[...,None]).sum(2)
    root_pos=torch.where(root_valid[...,None],root_pos,fallback_pos)
    root_pos=interpolate_evidence(root_pos,cp.sum(-1)>0)
    global_r=[root_r]
    local=[root_r]
    rotation_confidence=[torch.where(root_valid,cp[:,:,:4].amax(-1),meta[:,:,15,3])]
    for j in range(1,22):
        parent=int(PARENTS[j]);inherited=global_r[parent]
        children=[k for k in range(j+1,22) if int(PARENTS[k])==j]
        candidate=inherited;confidence=pos.new_full((b,t),.05)
        if children:
            weights=cp[:,:,children]*cp[:,:,j,None]
            bones=(rest[:,children]-rest[:,j,None])[:,None].expand(b,t,-1,-1)
            delta=pos[:,:,children]-pos[:,:,j,None]
            source=(inherited[:,:,None]@bones[...,None]).squeeze(-1)
            weighted_source=(source*weights[...,None]).sum(2)
            weighted_delta=(delta*weights[...,None]).sum(2)
            candidate=align_vectors(weighted_source,weighted_delta)@inherited
            if len(children)>1:
                fitted,singular,_,_=fit_rotation(bones,delta,weights)
                candidate=torch.where((singular[...,1]>1e-7)[...,None,None],fitted,candidate)
            confidence=torch.where(weights.sum(-1)>0,weights.amax(-1).sqrt()*.5,confidence)
        observed_valid=meta[:,:,j,1]*meta[:,:,j,3]>0
        candidate=torch.where(observed_valid[...,None,None],observed_r[:,:,j],candidate)
        confidence=torch.where(observed_valid,meta[:,:,j,3],confidence)
        global_r.append(candidate);local.append(inherited.transpose(-1,-2)@candidate)
        rotation_confidence.append(confidence)
    local=torch.stack(local,2)
    six=torch.cat((local[..., :,0],local[..., :,1]),-1)-pos.new_tensor(IDENTITY_6D)
    reference=torch.cat(((root_pos-rest[:,None,0])/2,six.flatten(2),pos.new_zeros(b,t,66)),-1)
    reference=reconcile_motion(reference,rest)
    confidence=torch.stack((cp.clamp_min(.05),torch.stack(rotation_confidence,2).clamp_min(.05)),-1)
    return reference,confidence


def refinement_payload(batch):
    result=inference_payload(batch)
    if 'reference_motion' in batch:
        reference=batch['reference_motion'];confidence=batch['reference_confidence']
        if reference.shape!=(*batch['camera'].shape[:2],201) or confidence.shape!=(*reference.shape[:2],22,2):
            raise ValueError('Full frontend motion/reference confidence shape mismatch')
        if not torch.isfinite(reference).all() or not torch.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():
            raise ValueError('Reference must be finite and confidence in [0,1]')
        result.update(reference_motion=reconcile_motion(reference.detach(),batch['rest']),reference_confidence=confidence.detach())
    else:
        reference,confidence=reference_from_observations(result)
        result.update(reference_motion=reference,reference_confidence=confidence)
    # Separate observed anchors from a possibly inaccurate full frontend prior.
    # This is inference evidence, never a target/body-pose label.
    observed_reference=reference_from_observations(result)[0] if 'reference_motion' in batch else reference
    trajectory,meta=sanitize_observations(result['trajectory'],result['observation_meta'])
    weights=meta[:,:,:4,0]*meta[:,:,:4,2]
    _,singular,_,_=fit_rotation(result['rest'][:,None,:4].expand(*weights.shape,3),trajectory[:,:,:4,:3]*2,weights,center=True)
    valid=((weights>0).sum(-1)>=3)&(singular[...,1]>1e-7)
    uncertainty=result['observation_uncertainty'][:,:,:4,0].clamp_min(0)
    confidence=(weights*.05**2/(.05**2+uncertainty.square())).sum(-1)/(weights>0).sum(-1).clamp_min(1)
    result.update(observation_anchor_reference=observed_reference,torso_anchor_confidence=confidence*valid)
    return result


def confidence_anchor(motion,batch,signal,gain):
    """Soft observation proximal step within diffusion; preserves fixed bones.

    Exo torso positions constrain rigid root pose. Ego head position constrains
    translation only. Gaps/uncertain evidence weaken or disable the constraint;
    camera orientation never forces body heading. Raw learned x0 remains under
    GT supervision before this constraint in Native30SceneMI.forward.
    """
    if gain==0:return motion
    trajectory,meta=sanitize_observations(batch['trajectory'],batch['observation_meta'])
    exo=torch.as_tensor([m=='exo' for m in batch['mode']],device=motion.device)[:,None]
    strength=signal[:,None]*gain
    weight=strength*batch['torso_anchor_confidence']*exo
    reference=batch['observation_anchor_reference']
    current=global_rotations(motion)[:,:,0];anchor=global_rotations(reference)[:,:,0]
    rotation_delta=motion.new_zeros(*motion.shape[:2],22,3)
    rotation_delta[:,:,0]=matrix_to_axis_angle(anchor@current.transpose(-1,-2))*weight[...,None]
    result=reconcile_motion(motion,batch['rest'],(reference[...,:3]-motion[...,:3])*2*weight[...,None],rotation_delta)
    head_error=trajectory[:,:,15,:3]*2-forward_kinematics(result,batch['rest'])[:,:,15]
    uncertainty=batch['observation_uncertainty'][:,:,15,0].clamp_min(0)
    head_weight=strength*meta[:,:,15,0]*meta[:,:,15,2]*.05**2/(.05**2+uncertainty.square())*(~exo)
    return reconcile_motion(result,batch['rest'],head_error*head_weight[...,None])


def model_inputs(config,batch):
    if config.get('diffusion_target','absolute_x0')=='reference_residual':
        return refinement_payload(batch)
    return inference_payload(batch)


def training_motion(config,batch,known):
    if config.get('diffusion_target','absolute_x0')=='reference_residual':
        return batch['motion']-known['reference_motion']
    return batch['motion']


def sample_motion(model,batch,config,**kwargs):
    sampling=sample_refinement if config.get('diffusion_target','absolute_x0')=='reference_residual' else sample_native30
    return sampling(model,batch,**kwargs)


class Native30Refinement(Native30SceneMI):
    def __init__(self,*args,observation_anchor_gain=0.,joint_anchor_gain=0.,
                 joint_anchor_iterations=6,joint_anchor_damping_m2=.0025,
                 joint_anchor_angle_limit_rad=.35,joint_anchor_compact=False,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<=observation_anchor_gain<=1:raise ValueError('Observation anchor gain must be in [0,1]')
        self.observation_anchor_gain=observation_anchor_gain
        if not 0<=joint_anchor_gain<=1:raise ValueError('Joint anchor gain must be in [0,1]')
        self.joint_anchor_gain=joint_anchor_gain
        if isinstance(joint_anchor_iterations,bool) or not isinstance(joint_anchor_iterations,int) or joint_anchor_iterations<1:
            raise ValueError('IK iterations must be a positive integer')
        if not 0<joint_anchor_damping_m2<float('inf') or not 0<joint_anchor_angle_limit_rad<float('inf'):
            raise ValueError('IK damping and angle limit must be positive finite')
        if not isinstance(joint_anchor_compact,bool):raise ValueError('Compact IK must be boolean')
        self.joint_anchor_iterations=joint_anchor_iterations
        self.joint_anchor_damping_m2=joint_anchor_damping_m2
        self.joint_anchor_angle_limit_rad=joint_anchor_angle_limit_rad
        self.joint_anchor_compact=joint_anchor_compact
        self.reference_encoder=nn.Sequential(nn.Linear(27,64),nn.SiLU(),nn.Linear(64,64))
        self.reference_attention=nn.MultiheadAttention(64,4,batch_first=True)
        self.reference_projection=nn.Linear(22*64,64,bias=False)

    def constrain_motion(self,motion,timestep,batch):
        result=confidence_anchor(motion,batch,self.alphas[timestep],self.observation_anchor_gain)
        if self.joint_anchor_gain:
            from experiments.offline_camera_retrain_v1.native30_observation_ik import position_proximal
            result=position_proximal(result,batch,self.alphas[timestep],self.joint_anchor_gain,
                iterations=self.joint_anchor_iterations,damping_m2=self.joint_anchor_damping_m2,
                angle_limit_rad=self.joint_anchor_angle_limit_rad,compact=self.joint_anchor_compact)
        return result

    def observation_condition(self,batch,control_mask=None):
        condition,trajectory,meta=super().observation_condition(batch,control_mask)
        reference=batch['reference_motion'];confidence=batch['reference_confidence']
        ids=self.joint_embedding.weight[None,None].expand(*reference.shape[:2],-1,-1)
        features=torch.cat((reference[...,135:].reshape(*reference.shape[:2],22,3)*confidence[...,0,None],
            reference[...,3:135].reshape(*reference.shape[:2],22,6)*confidence[...,1,None],confidence,ids),-1)
        tokens=self.reference_encoder(features)*confidence.amax(-1)[...,None]
        flattened=tokens.flatten(0,1)
        attended,_=self.reference_attention(flattened,flattened,flattened,need_weights=False)
        attended=attended.reshape_as(tokens)+tokens
        return condition+self.reference_projection(attended.flatten(2)),trajectory,meta

    def _core_pass(self,noisy,timestep,batch,scene,control_mask,use_scene,use_control,query_motion=None):
        reference=batch['reference_motion']
        query=reference+noisy if query_motion is None else query_motion
        residual=super()._core_pass(noisy,timestep,batch,scene,control_mask,use_scene,use_control,query)
        return reference+residual


@torch.no_grad()
def sample_refinement(model,batch,*,steps=20,seed=777,use_scene=True):
    if not 2<=steps<=1000:raise ValueError('Unsupported DDIM steps')
    reference=batch['reference_motion'];device=reference.device
    generator=torch.Generator(device=device).manual_seed(seed)
    state=torch.randn(reference.shape,device=device,generator=generator)
    alphas=cosine_alphas().to(device);schedule=torch.linspace(999,0,steps,device=device).round().long().unique_consecutive()
    was_training=model.training;model.eval()
    if model._sampling_scene is not None:raise ValueError('Nested sampling cache')
    try:
        if use_scene:model._sampling_scene=(batch['occupancy'],model.scene_tokens(batch))
        for i,step in enumerate(schedule):
            times=torch.full((len(state),),int(step),device=device,dtype=torch.long)
            absolute=model(state,times,batch,use_scene=use_scene)
            residual=absolute-reference;alpha=alphas[step]
            epsilon=(state-alpha.sqrt()*residual)/(1-alpha).sqrt().clamp_min(1e-6)
            state=residual if i+1==len(schedule) else alphas[schedule[i+1]].sqrt()*residual+(1-alphas[schedule[i+1]]).sqrt()*epsilon
        return reconcile_motion(reference+state,batch['rest'])
    finally:
        model._sampling_scene=None;model.train(was_training)
