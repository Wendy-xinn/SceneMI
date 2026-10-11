"""Pose-centred observed-surface queries: causal static/current dynamic only."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.data import anchor_rotation

class BodySceneWindow:
    def __init__(self,static_points,static_times,queries,dynamic_frames,radius=.75):
        self.points=np.asarray(static_points,np.float64);self.times=np.asarray(static_times);self.queries=np.asarray(queries);self.dynamic=[np.asarray(x,np.float64).reshape(-1,3) for x in dynamic_frames];self.radius=radius;self.trees={};self.dynamic_trees={}
        if len(self.points)!=len(self.times) or len(self.dynamic)!=len(self.queries) or np.any(np.diff(self.times)<0):raise ValueError('Scene timestamps invalid')
        if not np.isfinite(self.points).all() or any(not np.isfinite(x).all() for x in self.dynamic):raise ValueError('Nonfinite known geometry')
    def features(self,joints,rotation=None):
        joints=np.asarray(joints);out=[]
        if len(joints)!=len(self.queries):raise ValueError('Body query time mismatch')
        for t,v in enumerate(joints):
            n=int(np.searchsorted(self.times,self.queries[t],side='right'))
            if n not in self.trees:self.trees[n]=cKDTree(self.points[:n],copy_data=False) if n else None
            if t not in self.dynamic_trees:self.dynamic_trees[t]=cKDTree(self.dynamic[t]) if len(self.dynamic[t]) else None
            nearest=np.zeros_like(v);distance=np.full(len(v),np.inf);moving=np.zeros(len(v),bool)
            for tree,is_dynamic in ((self.trees[n],False),(self.dynamic_trees[t],True)):
                if tree is None:continue
                d,ix=tree.query(v);take=d<distance;nearest[take]=tree.data[ix[take]];distance[take]=d[take];moving[take]=is_dynamic
            valid=distance<=self.radius;delta=nearest-v
            if rotation is not None:delta=np.einsum('jik,ji->jk',rotation[t],delta)
            feature=np.column_stack((delta/self.radius,np.minimum(distance,self.radius)/self.radius,valid,moving));feature[~valid]=0
            out.append(feature)
        return np.asarray(out,np.float32)

    def neighbourhoods(self,joints,rotation=None,*,radii=(.25,.75,2.),k=8):
        """Multi-surface, multi-scale tokens. Unknown slots remain masked.

        Static observations obey each query's timestamp. Moving surfaces are
        queried only at that exact time, never accumulated into static memory.
        Features: body-local displacement/radius, distance/radius, valid,
        dynamic. Selection is discrete; callers must not claim a signed SDF.
        """
        if k<1 or any(r<=0 for r in radii):raise ValueError('Invalid neighbourhood')
        joints=np.asarray(joints)
        if len(joints)!=len(self.queries):raise ValueError('Body query time mismatch')
        out=np.zeros((*joints.shape[:2],len(radii),k,6),np.float32)
        for t,v in enumerate(joints):
            n=int(np.searchsorted(self.times,self.queries[t],side='right'))
            if n not in self.trees:self.trees[n]=cKDTree(self.points[:n]) if n else None
            if t not in self.dynamic_trees:self.dynamic_trees[t]=cKDTree(self.dynamic[t]) if len(self.dynamic[t]) else None
            # Separate radii must see different spatial extents, rather than
            # re-encoding the same nearest k points three times. Bounded,
            # deterministic candidate sampling + farthest-point selection.
            for j,point in enumerate(v):
                for s,r in enumerate(radii):
                    patches=[];tags=[]
                    for tree,moving in ((self.trees[n],False),(self.dynamic_trees[t],True)):
                        if tree is None:continue
                        ix=tree.query_ball_point(point,r,return_sorted=True)
                        if not len(ix):continue
                        if len(ix)>128:ix=np.asarray(ix)[np.linspace(0,len(ix)-1,128).astype(int)]
                        patches.append(tree.data[ix]);tags.extend([moving]*len(ix))
                    if not patches:continue
                    points=np.concatenate(patches);delta=points-point;d=np.linalg.norm(delta,axis=1)
                    selected=[int(d.argmin())];distance=np.full(len(points),np.inf)
                    for _ in range(min(k,len(points))-1):
                        distance=np.minimum(distance,((points-points[selected[-1]])**2).sum(-1));distance[selected]=-1
                        selected.append(int(distance.argmax()))
                    delta=delta[selected]
                    if rotation is not None:delta=delta@rotation[t,j]
                    count=len(selected)
                    out[t,j,s,:count]=np.column_stack((delta/r,d[selected]/r,np.ones(count),np.asarray(tags)[selected]))
        return out

class BodySceneCache:
    def __init__(self,limit=4):self.limit=limit;self.cache=OrderedDict()
    def get(self,identity,frames=128):
        key=(identity['scene_bundle'],identity['memory_bundle'],float(identity['source_start_30fps']),frames)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        folder=Path(identity['scene_bundle']);memory=Path(identity['memory_bundle']);ids=np.load(folder/'source_frame_ids.npy',mmap_mode='r');first=identity['source_start_30fps'];ix=int(np.searchsorted(ids,first));q=np.asarray(ids[ix:ix+frames]);assert len(q)==frames and abs(q[0]-first)<1e-4
        camera=np.asarray(np.load(folder/'camera_position_scenemi_yup.npy',mmap_mode='r')[ix:ix+frames]);rotation=np.asarray(np.load(folder/'camera_rotation_scenemi_yup.npy',mmap_mode='r')[ix:ix+frames]);origin=camera[0];anchor=anchor_rotation(rotation)
        times=np.load(memory/'first_observed_source_frames.npy',mmap_mode='r');n=int(np.searchsorted(times,q[-1],side='right'));points=np.asarray(np.load(memory/'static_points.npy',mmap_mode='r')[:n]);times=np.asarray(times[:n]);points=(points-origin)@anchor.T
        clouds=np.load(folder/'visible_frame_points_scenemi_yup.npy',mmap_mode='r');owners=np.load(folder/'visible_frame_owner.npy',mmap_mode='r');dynamic=[]
        for t in range(ix,ix+frames):
            owner=np.asarray(owners[t]);keep=(owner>0)&(owner<100);dynamic.append((np.asarray(clouds[t])[keep]-origin)@anchor.T)
        value=BodySceneWindow(points,times,q,dynamic);self.cache[key]=value
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return value
