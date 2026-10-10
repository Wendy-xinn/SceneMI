"""Differentiable sparse native SMPL/SMPL-X skin points, neutral missing hands."""
import numpy as np
import torch
from smplx.lbs import blend_shapes,vertices2joints,batch_rigid_transform
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d

class NativeSurfacePoints:
    def __init__(self,body_info,device='cuda',full_regions=None):
        model=load_model(body_info['model'],body_info['gender']).to(device)
        beta=torch.tensor([body_info['betas']],device=device,dtype=torch.float32)
        with torch.no_grad():
            shaped=model.v_template[None]+blend_shapes(beta,model.shapedirs[:,:,:len(body_info['betas'])]);self.joints=vertices2joints(model.J_regressor,shaped)[0].detach();v=shaped[0]
        weights=model.lbs_weights.detach();regions=weights.argmax(-1).cpu().numpy();verts=v.cpu().numpy();selected=[]
        for joint in [0,1,2,4,5,7,8,10,11,3,6,9,12,15]:
            ids=np.flatnonzero(regions==joint)
            if len(ids):selected.extend(ids[np.linspace(0,len(ids)-1,min(32,len(ids))).astype(int)].tolist())
        soles=[]
        for j in [10,11]:
            ids=np.flatnonzero((regions==j)|(regions==j-3));low=ids[np.argsort(verts[ids,1])[:12]];soles.append(low.tolist());selected.extend(low.tolist())
        if full_regions is not None:
            # Opt-in only: preserve previously scored sparse objectives exactly.
            selected.extend(np.flatnonzero(np.isin(regions,full_regions)).tolist())
        self.indices=np.unique(selected);self.v=v[self.indices].detach();self.weights=weights[self.indices].detach();self.parents=model.parents.detach();self.pose_dirs=model.posedirs.reshape(-1,len(v),3)[:,self.indices].reshape(model.posedirs.shape[0],-1).detach();self.scale=float(body_info['scale']);self.sole_indices=[[int(np.flatnonzero(self.indices==i)[0]) for i in foot] for foot in soles]
        self.labels=torch.tensor(regions[self.indices],device=device);self.vertex_ids=self.indices.copy();self.model_info=dict(body_info)
    def __call__(self,motion):
        local=rotation_from_6d(motion[:,3:135].reshape(-1,22,6));t=len(motion);eye=torch.eye(3,device=motion.device,dtype=motion.dtype);extra=eye.expand(t,len(self.parents)-22,3,3);rot=torch.cat((local,extra),1)
        posed=self.v[None]+((rot[:,1:]-eye).reshape(t,-1)@self.pose_dirs).reshape(t,len(self.v),3)
        _,transforms=batch_rigid_transform(rot,self.joints[None].expand(t,-1,-1),self.parents)
        transform=torch.einsum('vj,tjkl->tvkl',self.weights,transforms)
        homogeneous=torch.cat((posed,torch.ones_like(posed[:,:,:1])),-1)
        points=(transform@homogeneous[...,None])[:,:,:3,0]*self.scale+motion[:,:3,None].transpose(1,2)*2
        return points
    def soles(self,points):return torch.stack([points[:,ix].mean(1) for ix in self.sole_indices],1)
