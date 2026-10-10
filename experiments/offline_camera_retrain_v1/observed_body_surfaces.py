"""Local observed tangent planes, not a closed-scene signed distance field.

Static memory is cut at each query time; dynamic points only at that time.
Normals face their observing camera. Unknown/poorly sampled patches excluded.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.data import anchor_rotation


def oriented_planes(points,cameras,*,neighbors=16):
    points=np.asarray(points,np.float32);cameras=np.asarray(cameras,np.float32)
    if len(points)<neighbors:return points,np.zeros_like(points),np.zeros(len(points),bool)
    tree=cKDTree(points);dist,ix=tree.query(points,k=neighbors,workers=2)
    patch=points[ix];center=patch.mean(1);relative=patch-center[:,None]
    cov=np.einsum('nki,nkj->nij',relative,relative)/neighbors
    eig,vec=np.linalg.eigh(cov);normal=vec[:,:,0]
    rays=cameras-points;normal*=np.where((normal*rays).sum(-1)>=0,1.,-1.)[:,None]
    valid=(eig[:,0]<.12*eig[:,1])&(eig[:,1]>1e-5)&(dist[:,-1]<.18)&(np.abs((normal*rays).sum(-1))>.1*np.linalg.norm(rays,axis=-1))
    return points,normal.astype(np.float32),valid

class ObservedPlanes:
    def __init__(self,frames):
        self.frames=[]
        for points,normals,owners in frames:
            points=np.asarray(points,np.float32);normals=np.asarray(normals,np.float32);owners=np.asarray(owners,np.int32)
            if not np.isfinite(points).all() or not np.isfinite(normals).all():raise ValueError('Nonfinite observed geometry')
            assert owners.shape==(len(points),) and normals.shape==points.shape
            assert np.all((owners>=0)&(owners<100)), 'Wearer geometry cannot be a collision surface'
            self.frames.append((points,normals,owners,cKDTree(points) if len(points) else None))
    def query(self,vertices):
        # Frozen nearest correspondence, differentiable displacement to plane.
        array=vertices.detach().cpu().numpy();locations=[];normals=[];owners=[];known=[]
        if len(array)!=len(self.frames):raise ValueError('Geometry/frame mismatch')
        for v,(points,norm,owner,tree) in zip(array,self.frames):
            if tree is None:
                locations.append(np.zeros_like(v));normals.append(np.zeros_like(v));owners.append(np.full(len(v),-1));known.append(np.zeros(len(v),bool));continue
            distance,ix=tree.query(v,workers=2);locations.append(points[ix]);normals.append(norm[ix]);owners.append(owner[ix]);known.append(distance<.25)
        tensor=lambda x:torch.as_tensor(np.asarray(x),device=vertices.device)
        p=tensor(locations).to(vertices);n=tensor(normals).to(vertices);owner=tensor(owners);valid=tensor(known)
        delta=vertices-p;signed=(delta*n).sum(-1);tangent=delta-signed[...,None]*n
        valid=valid&(tangent.norm(dim=-1)<.09)
        return signed,n,owner,valid,p


def from_identity(identity,frames=128,*,history_frames=16):
    folder=Path(identity['scene_bundle']);memory=Path(identity['memory_bundle'])
    ids=np.load(folder/'source_frame_ids.npy');first=float(identity['source_start_30fps']);ix=int(np.argmin(abs(ids-first)));assert abs(ids[ix]-first)<1e-4
    q=ids[ix:ix+frames];assert len(q)==frames
    camera=np.load(folder/'camera_position_scenemi_yup.npy');rotation=np.load(folder/'camera_rotation_scenemi_yup.npy');origin=camera[ix];anchor=anchor_rotation(rotation[ix:ix+frames])
    points=np.load(memory/'static_points.npy');times=np.load(memory/'first_observed_source_frames.npy');keep=times<=q[-1];points=points[keep];times=times[keep]
    # Orient first-observation planes; previous EgoBody segments use the same
    # recording frame indices. Those not covered by this bundle remain unknown.
    camix=np.searchsorted(ids,times);covered=(camix<len(ids))
    camix=np.clip(camix,0,len(ids)-1);covered&=abs(ids[camix]-times)<1e-4
    points=points[covered];times=times[covered];camix=camix[covered]
    points=(points-origin)@anchor.T;cameras=(camera[camix]-origin)@anchor.T
    # Do not use future static points to fit normals of past surfaces.
    clouds=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');owners=np.load(folder/'visible_frame_owner.npy',mmap_mode='r')
    result=[];counts=[]
    for k in range(history_frames,frames):
        known=times<=q[k];sp=points[known];sc=cameras[known]
        if len(sp)>30000:
            select=np.linspace(0,len(sp)-1,30000).astype(int);sp=sp[select];sc=sc[select]
        p,n,valid=oriented_planes(sp,sc);parts=[p[valid]];normals=[n[valid]];labels=[np.zeros(valid.sum(),np.int32)]
        obs=np.asarray(clouds[ix+k]);own=np.asarray(owners[ix+k]);cam=(camera[ix+k]-origin)@anchor.T
        for owner in np.unique(own[(own>0)&(own<100)]):
            dynamic=(obs[own==owner]-origin)@anchor.T;dp,dn,dv=oriented_planes(dynamic,np.broadcast_to(cam,dynamic.shape),neighbors=8)
            parts.append(dp[dv]);normals.append(dn[dv]);labels.append(np.full(dv.sum(),owner,np.int32))
        result.append((np.concatenate(parts),np.concatenate(normals),np.concatenate(labels)));counts.append({'static':len(parts[0]),'dynamic':sum(len(x) for x in parts[1:])})
    return ObservedPlanes(result),dict(source='observed first-hit points only; no complete scene mesh',counts=counts,static_time_policy='first observation <= query, fit normals only from known points',dynamic_time_policy='current frame owner1..99 only, never memory union',normal_policy='PCA planar patch facing observing camera',distance_semantics='local observed half-space proxy; unknown holes/deep interiors not guaranteed',older_segment_points_without_observer_camera_excluded=True)
