"""Frozen legacy absolute-x0 prior followed by label-free EXO SO(3) residuals.

The legacy prior is never reinterpreted as a residual predictor. Its optional
low-noise projection takes an observation-only reference, with sparse controls
disabled because EXO positions do not supply measured orientations.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.native30_refinement import reference_from_observations
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


@torch.no_grad()
def frozen_initial_pose(model, batch, seed, schedule=(99, 66, 33, 0)):
    """SDEdit-style four absolute-x0 passes; no hidden body labels or scene.

    Legacy 20Hz weights transfer to a 30Hz clock explicitly for this pilot.
    Translation is fitted to available positions after denoising. Unavailable
    rotations never become legacy sparse controls, even via guessed rotations.
    """
    reference, _ = reference_from_observations(batch)
    alphas = cosine_alphas().to(reference)
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    state = alphas[schedule[0]].sqrt()*reference + (1-alphas[schedule[0]]).sqrt()*torch.randn(
        reference.shape, device=reference.device, generator=generator)
    # Explicit whitelist: legacy.forward needs no motion/contact/joints labels.
    known = {k: batch[k] for k in ('trajectory','rest','body_type','body_scale','occupancy','bps','bps_valid')}
    known['trajectory']=torch.zeros_like(batch['trajectory'])
    mask = torch.zeros(*reference.shape[:2],22,device=reference.device,dtype=torch.bool)
    model.eval()
    for index, timestep in enumerate(schedule):
        pred = model(state, torch.full((len(state),), timestep, device=state.device, dtype=torch.long),
                     known, control_mask=mask, use_scene=False, use_control=False)
        a = alphas[timestep]
        if index+1 == len(schedule):
            state = pred
        else:
            epsilon = (state-a.sqrt()*pred)/(1-a).sqrt().clamp_min(1e-6)
            next_a = alphas[schedule[index+1]]
            state = next_a.sqrt()*pred+(1-next_a).sqrt()*epsilon
    state = reconcile_motion(state,batch['rest'])
    trajectory, meta = sanitize_observations(batch['trajectory'],batch['observation_meta'])
    weight = meta[...,0]*meta[...,2]
    delta = ((trajectory[...,:3]*2-forward_kinematics(state,batch['rest']))*weight[...,None]).sum(2)
    delta = delta/weight.sum(2,keepdim=True).clamp_min(1e-8)
    return reconcile_motion(state,batch['rest'],delta), reference


def scene_features(scene, joints, radius=1.5):
    """Nearest observed static/dynamic patch at the FROZEN initial joints.

    Computed once per input, never on GT. Missing surfaces produce zero tokens
    with an explicit zero validity bit, not an inferred floor or free space.
    Features: displacement3, normal3, distance1, valid1, reliable-normal1,
    static/rigid/other-person3. Target owner100 must never occur.
    """
    joints = np.asarray(joints)
    if hasattr(scene,'base') and hasattr(scene,'rotation'):
        values=scene_features(scene.base,joints@scene.rotation,radius).reshape(len(joints),22,2,12)
        values[...,:3]=values[...,:3]@scene.rotation.T
        values[...,3:6]=values[...,3:6]@scene.rotation.T
        return values.reshape(len(joints),-1)
    result = np.zeros((*joints.shape[:2],2,12),np.float32)
    for t in range(len(joints)):
        for slot,cloud in enumerate((scene.static,scene.dynamic[t])):
            if not len(cloud['points']): continue
            if np.any(cloud['owners']==100): raise ValueError('Target body in scene input')
            tree = scene._static_tree if slot==0 else cKDTree(cloud['points'])
            distance, ids = tree.query(joints[t],k=1)
            valid = distance<=radius
            owner=cloud['owners'][ids]
            features = np.concatenate((cloud['points'][ids]-joints[t],cloud['normals'][ids],
                distance[:,None],np.ones((22,1)),cloud['reliable'][ids,None],
                (owner==0)[:,None],((owner>0)&(owner<100))[:,None],(owner>100)[:,None]),-1)
            result[t,:,slot]=np.where(valid[:,None],features,0)
    return result.reshape(len(joints),-1)


def residual_inputs(initial, batch, scene):
    """Whitelist and sanitize before constructing a train/inference feature set."""
    trajectory,meta=sanitize_observations(batch['trajectory'],batch['observation_meta'])
    joints=forward_kinematics(initial,batch['rest'])
    pm=meta[...,0];cp=meta[...,2]
    error=(trajectory[...,:3]*2-joints)*pm[...,None]
    rest=(batch['rest']-batch['rest'][:,:1]).flatten(1)[:,None].expand(-1,initial.shape[1],-1)
    values=torch.cat((initial,error.flatten(2),pm,cp,rest),-1)
    return dict(initial=initial,features=values,scene=scene,rest=batch['rest'])


def correction_payload(batch):
    return {k:batch[k] for k in ('initial','features','scene','rest')}


class ExoResidual(nn.Module):
    def __init__(self,width=192,root_limit=.3,rotation_limit=1.):
        super().__init__()
        self.input=nn.Conv1d(377+528,width,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),
            nn.Conv1d(width,width,5,padding=2*d,dilation=d)) for d in (1,2,4,8,16)])
        self.output=nn.Conv1d(width,69,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.root_limit,self.rotation_limit=root_limit,rotation_limit

    def forward(self,known,use_scene=True):
        if set(known)!={'initial','features','scene','rest'}:
            raise ValueError('Residual inference accepts only initial/features/scene/rest')
        scene=known['scene'] if use_scene else torch.zeros_like(known['scene'])
        h=self.input(torch.cat((known['features'],scene),-1).transpose(1,2))
        for block in self.blocks:h=h+block(h)
        raw=self.output(h).transpose(1,2)
        translation=self.root_limit*torch.tanh(raw[...,:3])
        rotation=self.rotation_limit*torch.tanh(raw[...,3:].reshape(*raw.shape[:2],22,3))
        motion=reconcile_motion(known['initial'],known['rest'],translation,rotation)
        return motion,dict(translation=translation,rotation=rotation)
