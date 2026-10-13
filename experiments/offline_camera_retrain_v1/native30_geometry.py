"""Observed metric surfaces with owner identity and optional rigid velocities."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.observed_body_surfaces import oriented_planes
from experiments.offline_camera_retrain_v1.metric_surface_tokens import SCALES


def rigid_surface_velocity(point, centre, translation_velocity, angular_velocity):
    return np.asarray(translation_velocity) + np.cross(angular_velocity, np.asarray(point)-np.asarray(centre))


class Native30SurfaceWindow:
    """Fixed-budget offline static memory plus current dynamics, target excluded.

    Normals are oriented towards observing cameras. A local half-plane is a
    reliable surface-patch description, never a complete signed scene SDF.
    """
    def __init__(self, frame_points, frame_owners, cameras, *, static_memory=None, object_positions=None,
                 object_rotations=None, object_valid=None, fps=30., voxel=.025):
        if fps != 30:
            raise ValueError('This surface protocol requires native30')
        self.fps, self.cameras = fps, np.asarray(cameras, np.float32)
        points, owners = np.asarray(frame_points), np.asarray(frame_owners)
        if points.shape[:2] != owners.shape or len(points) != len(cameras) or np.any(owners == 100):
            raise ValueError('Surface shape/time mismatch or target skin leaked into environment')
        if not np.isfinite(points[owners>=0]).all():
            raise ValueError('Nonfinite observed surface')
        static = owners == 0
        p = points[static]; cam = np.broadcast_to(self.cameras[:,None], points.shape)[static]
        if static_memory is not None:
            p, cam = (np.asarray(v, np.float32) for v in static_memory)
            if p.shape != cam.shape or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite([p, cam]).all():
                raise ValueError('Invalid observed static memory points/cameras')
        if len(p):
            _, ix = np.unique(np.floor(p/voxel).astype(np.int32), axis=0, return_index=True)
            ix.sort(); p, cam = p[ix], cam[ix]
        self.points = p.astype(np.float32)
        self.static = self._cloud(p, cam, np.zeros(len(p),np.int64), np.zeros_like(p), np.ones(len(p),bool))
        self._static_tree = cKDTree(self.static['points']) if len(p) else None
        self.dynamic = []
        if object_positions is not None:
            pos = np.asarray(object_positions); rot = np.asarray(object_rotations); ov = np.asarray(object_valid,bool)
            if pos.shape[:2] != ov.shape or rot.shape != (*ov.shape,3,3) or len(pos) != len(points):
                raise ValueError('Rigid source clock mismatch')
            v = np.zeros_like(pos); w = np.zeros_like(pos); vm = np.zeros_like(ov)
            v[1:] = np.diff(pos,axis=0)*fps
            w[1:] = Rotation.from_matrix((rot[1:]@rot[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(*pos[1:].shape)*fps
            vm[1:] = ov[1:]&ov[:-1]
        for t in range(len(points)):
            select = owners[t]>0; dp, do = points[t][select], owners[t][select]
            velocity = np.zeros_like(dp); velocity_valid = np.zeros(len(dp),bool)
            if object_positions is not None:
                for owner in np.unique(do):
                    if 0 < owner < 100 and owner <= pos.shape[1] and vm[t,owner-1]:
                        mask = do == owner
                        velocity[mask] = rigid_surface_velocity(dp[mask],pos[t,owner-1],v[t,owner-1],w[t,owner-1])
                        velocity_valid[mask] = True
            self.dynamic.append(self._cloud(dp, np.broadcast_to(self.cameras[t], dp.shape), do, velocity, velocity_valid))
        # Current observed-owner summaries also provide context outside the
        # local vertex radius. Centroids/extents describe VISIBLE patches,
        # never reconstructed whole objects or target GT bodies.
        ids=sorted({int(owner) for cloud in self.dynamic for owner in cloud['owners']})
        self.dynamic_owner_ids=np.asarray(ids,np.int64)
        self.dynamic_features=np.zeros((len(points),len(ids),11),np.float32)
        self.dynamic_valid=np.zeros((len(points),len(ids)),bool)
        for t,cloud in enumerate(self.dynamic):
            for j,owner in enumerate(ids):
                mask=cloud['owners']==owner;p=cloud['points'][mask]
                if not len(p):continue
                vm=cloud['velocity_valid'][mask]
                v=cloud['velocities'][mask][vm].mean(0) if vm.any() else np.zeros(3)
                self.dynamic_features[t,j]=np.r_[p.mean(0)/2,p.max(0)-p.min(0),v/.2,
                                                  min(len(p)/32,1),float(vm.any())]
                self.dynamic_valid[t,j]=True
        self._dynamic_trees = [None] * len(self.dynamic)

    @staticmethod
    def _cloud(points, cameras, owners, velocities, velocity_valid):
        points = np.asarray(points,np.float32).reshape(-1,3)
        normals = np.zeros_like(points); reliable = np.zeros(len(points),bool)
        for owner in np.unique(owners):
            mask = owners == owner
            if mask.sum() >= 8:
                _, n, r = oriented_planes(points[mask], np.asarray(cameras)[mask], neighbors=8)
                normals[mask], reliable[mask] = n, r
        return dict(points=points, normals=normals, reliable=reliable, owners=np.asarray(owners,np.int64),
                    velocities=velocities, velocity_valid=velocity_valid)

    def query(self, vertices, rotations, neighbors=8):
        """Discrete neighbour choice; returned world samples permit differentiable deltas."""
        frames, anchors = vertices.shape[:2]
        if len(self.dynamic) != frames or rotations.shape != (frames,anchors,3,3):
            raise ValueError('Predicted-body query does not match observed scene clock')
        shape = (frames,anchors,3,neighbors)
        result = {k:np.zeros((*shape,3),np.float32) for k in ('points','normals','velocities')}
        result.update({k:np.zeros(shape,bool) for k in ('valid','normal_reliable','velocity_valid')})
        result['owners'] = np.zeros(shape,np.int64)
        static_count=len(self.static['points'])
        for t in range(frames):
            dynamic=self.dynamic[t]
            if not static_count and not len(dynamic['points']):
                continue
            if len(dynamic['points']) and self._dynamic_trees[t] is None:
                self._dynamic_trees[t]=cKDTree(dynamic['points'])
            def gather(key,ids):
                value=np.empty((len(ids),*self.static[key].shape[1:]),dtype=self.static[key].dtype)
                static=ids<static_count
                value[static]=self.static[key][ids[static]]
                value[~static]=dynamic[key][ids[~static]-static_count]
                return value
            for si,radius in enumerate(SCALES):
                # Concatenating sorted static IDs then offset dynamic IDs
                # preserves exactly the combined-tree neighborhood order.
                static_hits=self._static_tree.query_ball_point(vertices[t],radius,return_sorted=True) if self._static_tree is not None else [[] for _ in range(anchors)]
                dynamic_hits=self._dynamic_trees[t].query_ball_point(vertices[t],radius,return_sorted=True) if self._dynamic_trees[t] is not None else [[] for _ in range(anchors)]
                for a,(sids,dids) in enumerate(zip(static_hits,dynamic_hits)):
                    if not len(sids) and not len(dids):
                        continue
                    ids=np.r_[np.asarray(sids,np.int64),np.asarray(dids,np.int64)+static_count]
                    delta=gather('points',ids)-vertices[t,a]
                    chosen=[int(np.argmin((delta**2).sum(-1)))]; spread=np.full(len(ids),np.inf)
                    for _ in range(min(neighbors,len(ids))-1):
                        spread=np.minimum(spread,((delta-delta[chosen[-1]])**2).sum(-1));spread[chosen]=-1
                        chosen.append(int(spread.argmax()))
                    ix=ids[chosen]; n=len(ix); dest=(t,a,si,slice(0,n))
                    for key in ('points','normals','owners','velocities','velocity_valid'):
                        result[key][dest]=gather(key,ix)
                    result['normal_reliable'][dest]=gather('reliable',ix); result['valid'][dest]=True
        return result
