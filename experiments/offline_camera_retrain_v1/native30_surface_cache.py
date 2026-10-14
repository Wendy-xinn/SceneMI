"""Observation-only canonical geometry; rigid yaw never resamples patches.

Persistent files contain observed surfaces, not GT bodies or contact labels.
The existing pilot's YawSurface protocol is shared by formal random windows.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.data import anchor_rotation
from experiments.offline_camera_retrain_v1.native30_exo import external_transforms
from experiments.offline_camera_retrain_v1.native30_memory import observer_stream_key
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow

FIELDS=('points','normals','reliable','owners','velocities','velocity_valid')


class CanonicalSurfaceCache:
    def __init__(self, data, limit=8):
        self.data, self.limit = data, limit
        self.root=data.runtime_cache_root/'surfaces';self.root.mkdir(parents=True,exist_ok=True)
        self.windows=OrderedDict();self.hits=self.misses=0
        self.code={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in
            ('native30_surface_cache.py','native30_geometry.py','native30_memory.py',
             'observed_body_surfaces.py','native30_exo.py','data.py')}

    def get(self, identity, mode, frames):
        sources=[dict(path=m['path'],hashes=m.get('output_sha256')) for m in self.data.scene_memory.entries
                 if observer_stream_key(m,mode)==observer_stream_key(identity,mode)]
        signature=dict(code=self.code,sources=sources,path=identity['path'],start=identity['start'],
            frames=frames,mode=mode,memory=self.data.scene_memory.protocol,voxel=self.data.scene_memory.voxel)
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        if key in self.windows:
            self.windows.move_to_end(key);base,anchor=self.windows[key];self.hits+=1
        else:
            file=self.root/(key+'.npz'); folder=identity['path']; start=identity['start']; sl=slice(start,start+frames)
            if mode=='exo':
                transform=external_transforms(self.data.load,folder,start,frames)
                camera,rot=transform[:,:3,3],transform[:,:3,:3]
                point_key,owner_key='exo_frame_points_world','exo_frame_owner'
            else:
                camera=np.asarray(self.data.load(folder,'camera_position_scenemi_yup')[sl])
                rot=np.asarray(self.data.load(folder,'camera_rotation_scenemi_yup')[sl])
                point_key,owner_key='visible_frame_points_scenemi_yup','visible_frame_owner'
            origin=camera[0];anchor=np.asarray(anchor_rotation(rot))
            if file.exists():
                with np.load(file,allow_pickle=False) as saved:
                    base=Native30SurfaceWindow.__new__(Native30SurfaceWindow)
                    base.fps=30.;base.cameras=saved['cameras'];base.static={k:saved['static_'+k] for k in FIELDS}
                    offsets=saved['offsets'];flat={k:saved['dynamic_'+k] for k in FIELDS}
                    base.dynamic=[{k:v[offsets[t]:offsets[t+1]] for k,v in flat.items()} for t in range(frames)]
                    for name in ('dynamic_owner_ids','dynamic_features','dynamic_valid'):setattr(base,name,saved[name])
                base.points=base.static['points'];base._static_tree=cKDTree(base.points) if len(base.points) else None
                base._dynamic_trees=[None]*frames;self.hits+=1
            else:
                self.misses+=1
                canonical=dict(identity,origin=origin.tolist(),anchor=anchor.tolist())
                kwargs={}
                if (Path(folder)/'object_positions.npy').exists():
                    kwargs=dict(object_positions=(np.asarray(self.data.load(folder,'object_positions')[sl])-origin)@anchor.T,
                        object_rotations=anchor[None,None]@np.asarray(self.data.load(folder,'object_rotations')[sl]),
                        object_valid=np.asarray(self.data.load(folder,'object_valid')[sl]))
                points=(np.asarray(self.data.load(folder,point_key)[sl])-origin)@anchor.T
                owners=np.asarray(self.data.load(folder,owner_key)[sl])
                base=Native30SurfaceWindow(points,owners,(camera-origin)@anchor.T,
                    static_memory=self.data.scene_memory.window(canonical,mode,frames),**kwargs)
                arrays={'static_'+k:v for k,v in base.static.items()}
                arrays.update({'dynamic_'+k:np.concatenate([cloud[k] for cloud in base.dynamic]) for k in FIELDS})
                arrays.update(offsets=np.r_[0,np.cumsum([len(c['points']) for c in base.dynamic])],cameras=base.cameras,
                    dynamic_owner_ids=base.dynamic_owner_ids,dynamic_features=base.dynamic_features,dynamic_valid=base.dynamic_valid)
                temporary=file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    with temporary.open('wb') as stream:
                        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
                    temporary.replace(file)
                finally:temporary.unlink(missing_ok=True)
            self.windows[key]=(base,anchor)
            while len(self.windows)>self.limit:self.windows.popitem(last=False)
        rotation=(np.asarray(identity['anchor'])@anchor.T).astype(np.float32)
        if np.allclose(rotation,np.eye(3),atol=1e-7):return base
        from experiments.offline_camera_retrain_v1.native30_overnight import YawSurface
        return YawSurface(base,rotation)


def cached_surface(data,identity,mode,frames):
    if not hasattr(data,'surface_cache'):data.surface_cache=CanonicalSurfaceCache(data)
    return data.surface_cache.get(identity,mode,frames)
