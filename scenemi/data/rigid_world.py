"""Read-only rigid world views before the retained EGO canonicalizers."""
from collections import OrderedDict
import copy
import numpy as np
from scipy.spatial.transform import Rotation

from scenemi.data.amass_dataset import AmassDataset
from scenemi.native_joint_data import SceneFreeEgoDataset
from scenemi.geometry.body import AMASS_ZUP_TO_YUP
from scenemi.data.scene_geometry_cache import static_geometry


class _SourceView:
    def __init__(self,base,rotation,translation):
        self.base=base
        self.rotation=np.asarray(rotation,np.float64)
        self.translation=np.asarray(translation,np.float64)
        if (self.rotation.shape!=(3,3) or self.translation.shape!=(3,)
                or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-12)
                or not np.isclose(np.linalg.det(self.rotation),1.,atol=1e-12)
                or not np.isfinite(self.translation).all()
                or not np.allclose(self.rotation[1],[0,1,0],atol=1e-12)
                or self.translation[1]!=0):
            raise ValueError('Use gravity-preserving yaw and horizontal translation')
        self.cache=OrderedDict()

    def __getattr__(self,name):return getattr(self.base,name)

    def positions(self,value):return np.asarray(value,np.float64)@self.rotation.T+self.translation

    def remember(self,key,value):
        self.cache[key]=value;self.cache.move_to_end(key)
        while len(self.cache)>4:self.cache.popitem(last=False)
        return value


class RigidAmassSource(_SourceView):
    sample=AmassDataset.sample

    def load(self,relative,digest):
        key=(relative,digest)
        if key not in self.cache:
            original=self.base.load(relative,digest)
            z_to_y=AMASS_ZUP_TO_YUP
            rotation=z_to_y.T@self.rotation@z_to_y
            translation=self.translation@z_to_y
            poses=np.asarray(original['poses'],np.float64).copy()
            root=Rotation.from_rotvec(poses[:,:3]).as_matrix()
            poses[:,:3]=Rotation.from_matrix(rotation@root).as_rotvec()
            joints=np.asarray(original['joints'],np.float64)@rotation.T+translation
            self.remember(key,dict(original,poses=poses,joints=joints))
        return self.cache[key]


class RigidSceneSource(_SourceView):
    sample=SceneFreeEgoDataset.sample

    def __init__(self,*args,rebuild_static=False,**kwargs):
        super().__init__(*args,**kwargs);self.rebuild_static=rebuild_static

    def cached(self,row):
        path=row['path']
        if path not in self.cache:
            original=self.base.cached(row);saved=dict(original)
            rest=np.asarray(original['rest'],np.float64)
            motion=np.asarray(original['motion_world'],np.float64).copy()
            motion[:,:3]=(self.positions(motion[:,:3]*2+rest[0])-rest[0])/2
            six=motion[:,3:135].reshape(-1,22,6)
            root=six[:,0]+[1,0,0,0,1,0]
            first=root[:,:3]@self.rotation.T;second=root[:,3:]@self.rotation.T
            six[:,0]=np.concatenate((first,second),-1)-[1,0,0,0,1,0]
            motion[:,135:]=self.positions(motion[:,135:].reshape(-1,22,3)*2).reshape(-1,66)/2
            saved['motion_world']=motion
            for name in ('camera_position_scenemi_yup','static_full','static_points','frame_points'):
                saved[name]=self.positions(original[name])
            saved['frame_points'][original['frame_owner']<0]=0
            saved['camera_rotation_scenemi_yup']=self.rotation@np.asarray(original['camera_rotation_scenemi_yup'],np.float64)
            for name in ('static_normals','frame_normals'):
                saved[name]=np.asarray(original[name],np.float64)@self.rotation.T
            if self.rebuild_static:
                meta=self.base.selection_bundle.metadata(row,self.base.split)
                points=self.base.selection_bundle.array(meta,'visible_frame_points_scenemi_yup')
                owners=self.base.selection_bundle.array(meta,'visible_frame_owner')
                valid=self.base.selection_bundle.array(meta,'observation_valid')
                full,queries,normals,normal_valid=static_geometry(self.positions(points),owners,valid,
                    saved['camera_position_scenemi_yup'],self.base.geometry_policy)
                saved.update(static_full=full,static_points=queries,static_normals=normals,static_normal_valid=normal_valid)
            self.remember(path,saved)
        return self.cache[path]

    def array(self,meta,name):
        return self.cached(self.base._by_path[meta['path']])[name]


def world_view(data,rotation,translation,*,rebuild_static=False):
    """Keep original sampling/canonicalization methods and immutable sources."""
    result=copy.copy(data)
    if hasattr(data,'amass'):
        result.amass=RigidAmassSource(data.amass,rotation,translation)
    if getattr(data,'native',None) is not None:
        result.native=RigidSceneSource(data.native,rotation,translation,rebuild_static=rebuild_static)
    result.datasets={name:result.amass if name=='amass' else result.native for name in data.datasets}
    if hasattr(result,'ground_evidence'):result.ground_evidence={}
    return result
