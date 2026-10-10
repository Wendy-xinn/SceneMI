"""Conservative static support patches from known observations, training-only physics."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.native_surface_points import NativeSurfacePoints

class FloorCache:
    def __init__(self,limit=64):self.cache=OrderedDict();self.limit=limit
    def get(self,identity,frames=128):
        key=(identity['memory_bundle'],float(identity['source_start_30fps']),frames)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        folder=Path(identity['scene_bundle']);memory=Path(identity['memory_bundle']);ids=np.load(folder/'source_frame_ids.npy',mmap_mode='r');first=identity['source_start_30fps'];ix=int(np.searchsorted(ids,first));assert abs(ids[ix]-first)<1e-4
        camera=np.asarray(np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[ix:ix+frames]);rotation=np.asarray(np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[ix:ix+frames]);a=anchor_rotation(rotation);origin=camera[0];cam=(camera-origin)@a.T
        times=np.load(memory/'first_observed_source_frames.npy',mmap_mode='r');points=np.asarray(np.load(memory/'static_points.npy',mmap_mode='r')[times<=first]);p=(points-origin)@a.T
        roi=((p[:,[0,2]]>=cam[:,[0,2]].min(0)-1)&(p[:,[0,2]]<=cam[:,[0,2]].max(0)+1)).all(-1)
        roi&=(p[:,1]>cam[:,1].min()-2.4)&(p[:,1]<cam[:,1].min()-.65);p=p[roi];result=None
        if len(p)>=300:
            bottom=np.quantile(p[:,1],.15)+.04;bins=np.arange(p[:,1].min()-.01,bottom+.021,.02);hist,edges=np.histogram(p[:,1],bins=bins)
            candidates=[k for k,n in enumerate(hist) if n>=150 and n>=hist[max(0,k-1):min(len(hist),k+2)].max()]
            for k in candidates:
                mode=edges[k:k+2].mean();patch=p[abs(p[:,1]-mode)<.025]
                if len(patch)<200:continue
                for _ in range(3):
                    x=np.column_stack((patch[:,0],patch[:,2],np.ones(len(patch))));coef=np.linalg.lstsq(x,patch[:,1],rcond=None)[0];patch=patch[abs(patch[:,1]-x@coef)<.012]
                if len(patch)<200 or np.linalg.norm(coef[:2])>.08 or np.ptp(patch[:,[0,2]],axis=0).min()<.6:continue
                result=(coef,cKDTree(patch[:,[0,2]]),dict(points=len(patch),plane=coef.tolist(),time_policy='start-causal static only',dynamic_used=False,GT_or_generated_body_used_for_plane=False));break
        self.cache[key]=result
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return result

class FootCache:
    def __init__(self,limit=64):self.cache=OrderedDict();self.limit=limit
    def get(self,body):
        import json
        key=json.dumps(body,sort_keys=True)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        skin=NativeSurfacePoints(body);n=len(skin.v);ids=np.flatnonzero(np.isin(skin.labels.cpu().numpy(),[7,8,10,11]));skin.v=skin.v[ids];skin.weights=skin.weights[ids];skin.pose_dirs=skin.pose_dirs.reshape(-1,n,3)[:,ids].reshape(skin.pose_dirs.shape[0],-1);skin.labels=skin.labels[ids]
        self.cache[key]=skin
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return skin

def physics_losses(pred,truth,rest,identities,signal,floors,feet):
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    true_velocity=torch.diff(forward_kinematics(truth.detach(),rest)[:,:,(10,11)],dim=1).norm(dim=-1)
    values=[];counts=0
    for b,identity in enumerate(identities):
        field=floors.get(identity,len(pred[b]))
        if field is None:values.append(pred[b].sum()*0);continue
        coef,tree,_=field;skin=feet.get(identity['native_body']);v=skin(pred[b])
        with torch.no_grad():gt=skin(truth[b])
        def height(vertices):return vertices[...,1]-vertices[...,0]*float(coef[0])-vertices[...,2]*float(coef[1])-float(coef[2])
        def known(vertices):
            xz=vertices.detach()[...,[0,2]].cpu().numpy();d=tree.query(xz.reshape(-1,2),workers=2)[0].reshape(xz.shape[:2]);return torch.tensor(d<.6,device=pred.device)
        hp=height(v);ht=height(gt);kp=known(v);kt=known(gt);counts+=1
        def masked(x,mask):return (x*mask).sum()/mask.sum().clamp_min(1)
        penetration=masked(((-hp-.01).relu()/.03).square(),kp)
        ground=pred.new_zeros(());plant=pred.new_zeros(())
        for f,labels in enumerate(([7,10],[8,11])):
            ix=(skin.labels==labels[0])|(skin.labels==labels[1]);ph=hp[:,ix];th=ht[:,ix];pk=kp[:,ix];tk=kt[:,ix]
            gt_low=th.masked_fill(~tk,100).min(-1).values
            near=(gt_low>-.10)&(gt_low<.04)&tk.any(-1)
            stance=(true_velocity[b,:,f]<.01)&near[1:]&near[:-1]
            lowest=ph.masked_fill(~pk,100).min(-1).values
            ground+=masked(((lowest[1:].abs()-.015).relu()/.05).square(),stance&pk[1:].any(-1))/2
            # Persistent sampled points near the actual support patch. GT only
            # supplies the training stance tag, not the target contact height.
            contact=(ph.abs()<.025)&pk;stable=contact[1:]&contact[:-1]&stance[:,None]
            speed=torch.diff(v[:,ix][:,:,[0,2]],dim=0)*20
            plant+=masked((speed/.2).square().sum(-1),stable)/2
        values.append((.10*penetration+.08*ground+.08*plant)*signal[b].detach().square())
    return dict(scene_physics_total=torch.stack(values).mean(),scene_floor_valid_examples=pred.new_tensor(counts))
