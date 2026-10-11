"""Experimental body queries -> surfaces -> body parts -> frame conditioning.

This is an untrained architecture prototype, not an improved checkpoint.
The original occupancy/camera branches remain. Additional contact predictions
read the same body-surface relations directly. No target poses are queried.
"""
import numpy as np
import torch
from torch import nn
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

PARTS=((0,3,6,9),(1,4,7,10),(2,5,8,11),(12,15),(13,16,18,20),(14,17,19,21))

class BodySurfaceAttention(nn.Module):
    def __init__(self,width=64,heads=4,scales=3):
        super().__init__()
        self.query=nn.Linear(21,width);self.surface=nn.Linear(6,width)
        self.joint=nn.Embedding(22,width);self.scale=nn.Embedding(scales,width)
        self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.part_output=nn.Linear(len(PARTS)*width,32)
        self.contact_output=nn.Linear(width,1)
        nn.init.zeros_(self.part_output.weight);nn.init.zeros_(self.part_output.bias)
        nn.init.zeros_(self.contact_output.weight);nn.init.zeros_(self.contact_output.bias)

    def forward(self,queries,surfaces):
        b,t,j,s,k,_=surfaces.shape
        valid=surfaces[...,4].bool();active=valid.flatten(-2).any(-1)
        q=self.query(queries)+self.joint.weight[None,None]
        kv=self.surface(surfaces)+self.scale.weight[None,None,None,:,None]
        kv=kv.reshape(b*t*j,s*k,-1);mask=~valid.reshape(b*t*j,s*k)
        # An all-unknown neighbourhood attends a zero null token, then is
        # explicitly zeroed. It can never turn unknown space into free space.
        kv=torch.cat((kv,torch.zeros_like(kv[:,:1])),1)
        mask=torch.cat((mask,active.reshape(-1,1)),1)
        context,weights=self.attention(q.reshape(b*t*j,1,-1),kv,kv,key_padding_mask=mask)
        context=context.reshape(b,t,j,-1)*active[...,None]
        parts=[]
        for ids in PARTS:
            parts.append(context[:,:,ids].sum(2)/active[:,:,ids].sum(2).clamp_min(1)[...,None])
        frame=self.part_output(torch.cat(parts,-1))*active.any(-1)[...,None]
        contact=self.contact_output(context)[...,0]*active
        return frame,contact,weights.reshape(b,t,j,s*k+1),active

class HierarchicalBodySceneMI(OfflineSceneMI):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.random.fork_rng(devices=[]):self.body_surface_attention=BodySurfaceAttention()

    def relations(self,motion,batch):
        if 'body_scene_query' not in batch:raise ValueError('Known timestamped scene required')
        joints=forward_kinematics(motion,batch['rest']);rot=global_rotations(motion)
        surfaces=np.stack([scene.neighbourhoods(p,r) for scene,p,r in zip(batch['body_scene_query'],joints.detach().cpu().numpy(),rot.detach().cpu().numpy())])
        velocity=torch.cat((torch.zeros_like(joints[:,:1]),joints[:,1:]-joints[:,:-1]),1)*20
        camera=batch['camera'];relative=camera[...,None,:3]*2-joints
        queries=torch.cat((joints,rot[...,:,:2].transpose(-1,-2).flatten(-2),velocity,relative,camera[...,None,3:].expand(-1,-1,22,-1)),-1)
        surfaces=torch.as_tensor(surfaces,device=motion.device,dtype=motion.dtype)
        self.last_surface_available_by_scale=surfaces[...,4].bool().any(-1).detach()
        return self.body_surface_attention(queries,surfaces)

    def forward(self,noisy,timestep,batch,*,control_mask=None,use_scene=True,use_control=True):
        if not use_scene:return super().forward(noisy,timestep,batch,control_mask=control_mask,use_scene=False,use_control=use_control)
        training=self.training
        try:
            self.eval()
            with torch.no_grad():preliminary=super().forward(noisy,timestep,batch,control_mask=control_mask,use_scene=True,use_control=use_control)
        finally:self.train(training)
        frame,_,weights,active=self.relations(preliminary,batch)
        self.last_surface_attention=weights.detach();self.last_surface_available=active.detach()
        return super().forward(noisy,timestep,dict(batch,body_local_embedding=frame),control_mask=control_mask,use_scene=True,use_control=use_control)

    def contact_logits(self,motion,batch,*,use_scene=True):
        legacy=super().contact_logits(motion,batch,use_scene=use_scene)
        if not use_scene:return legacy
        _,contact,_,_=self.relations(motion,batch)
        return legacy+contact
