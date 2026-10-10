"""Native sole support training targets; no GT body/contact inputs at generation."""
import json
from collections import OrderedDict
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

class SoleCache:
    def __init__(self,limit=64):self.cache=OrderedDict();self.limit=limit
    def get(self,body):
        key=json.dumps(body,sort_keys=True)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        skin=NativeSurfacePoints(body);old_n=len(skin.v);old_feet=skin.sole_indices;ids=np.unique(sum(old_feet,[]))
        skin.v=skin.v[ids];skin.weights=skin.weights[ids];skin.pose_dirs=skin.pose_dirs.reshape(-1,old_n,3)[:,ids].reshape(skin.pose_dirs.shape[0],-1)
        skin.sole_indices=[[int(np.flatnonzero(ids==i)[0]) for i in foot] for foot in old_feet]
        self.cache[key]=skin
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return skin
    def soles(self,motion,identities):return torch.stack([s.soles(s(m)) for m,s in zip(motion,[self.get(i['native_body']) for i in identities])])

def sole_support_losses(prediction,truth,predicted_soles,target_soles,turn_teacher,signal_weight):
    weight=signal_weight.detach().reshape(-1).square();target_soles=target_soles.detach()
    def reduce(x,mask=None):
        if mask is None:v=x.flatten(1).mean(1)
        else:v=(x*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
        return (v*weight).mean()
    hp=predicted_soles[...,1];ht=target_soles[...,1];ground=ht.min(-1).values
    # Match changing clean sole support envelope; jumps/stairs retain their
    # clean changing height. This is a training target, not a scene floor SDF.
    difference=hp.min(-1).values-ground
    envelope=reduce((difference/.05).square())
    float_band=reduce(((difference-.015).clamp_min(0)/.05).square())
    sink_band=reduce(((-difference-.015).clamp_min(0)/.05).square())
    pv=torch.diff(predicted_soles,dim=1)*20;tv=torch.diff(target_soles,dim=1)*20
    stance=(tv.norm(dim=-1)<.2)&(ht[:,1:]<=ground[:,1:,None]+.03)
    plant_velocity=reduce(((pv[...,[0,2]]-tv[...,[0,2]])/.2).square().sum(-1),stance)
    swing_velocity=reduce(((pv-tv)/.4).square().sum(-1),~stance)
    support_height=reduce(((hp[:,1:]-ht[:,1:])/.05).square(),stance)
    # Keep the already improved teacher turning, without freezing root Y or
    # projecting head/feet. Sole supervision can still move the whole body.
    pred_r=global_rotations(prediction);teacher_r=global_rotations(turn_teacher).detach()
    root_keep=reduce(.5*(pred_r[:,:,0]-teacher_r[:,:,0]).square().sum((-1,-2)))
    head_keep=reduce(.5*(pred_r[:,:,15]-teacher_r[:,:,15]).square().sum((-1,-2)))
    def leg_steps(m):
        r=rotation_from_6d(m[...,3:135].reshape(*m.shape[:2],22,6))[:,:,(1,2,4,5,7,8)]
        return r[:,1:]@r[:,:-1].transpose(-1,-2)
    pa=torch.diff(leg_steps(prediction),dim=1);ta=torch.diff(leg_steps(truth).detach(),dim=1)
    angular_accel=reduce(((pa-ta)/.035).square().sum((-1,-2))*.5)
    total=.10*envelope+.15*float_band+.10*sink_band+.04*support_height+.03*plant_velocity+.01*swing_velocity+.15*root_keep+.10*head_keep+.015*angular_accel
    return dict(sole_envelope_5cm_units=envelope,sole_floating_15mm_band=float_band,sole_sinking_15mm_band=sink_band,sole_stance_height=support_height,sole_plant_velocity=plant_velocity,sole_swing_velocity=swing_velocity,teacher_root_rotation_keep=root_keep,teacher_head_rotation_keep=head_keep,leg_accel_2deg_units=angular_accel,sole_total=total)
