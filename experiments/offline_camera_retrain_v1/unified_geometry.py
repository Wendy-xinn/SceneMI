"""Fixed native skin vertex identities and observed-only scene neighbours."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from smplx.lbs import blend_shapes, vertices2joints, batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix

from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
from experiments.offline_camera_retrain_v1.data import anchor_rotation


class SkinAnchorCache:
    """67 fixed vertex IDs per native topology, articulated by generated pose.

    IDs are selected on the neutral template once, independently of subject,
    trajectory and labels. Native shape is declared body calibration in this
    simulated experiment; no initial pose is passed through this cache.
    """
    def __init__(self, device='cuda', limit=64):
        self.device, self.limit = device, limit
        self.models, self.ids, self.cache = {}, {}, OrderedDict()

    def get(self, info):
        key = (info['model'], info['gender'], tuple(info['betas']), float(info['scale']))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        model_key = key[:2]
        if model_key not in self.models:
            self.models[model_key] = load_model(*model_key).cpu()
        model = self.models[model_key]
        if model_key not in self.ids:
            v = model.v_template.detach().numpy()
            regions = model.lbs_weights.argmax(1).numpy()
            parents = model.parents.numpy()
            mapped = regions.copy()
            for j in range(22, len(parents)):
                ancestor = j
                while ancestor >= 22:
                    ancestor = int(parents[ancestor])
                mapped[regions == j] = ancestor
            ids, labels = [], []
            for joint in range(22):
                candidates = np.flatnonzero(mapped == joint)
                if len(candidates) < 3:
                    raise ValueError(f'Insufficient skin anchors in region {joint}')
                centre = v[candidates].mean(0)
                first = int(np.argmin(v[candidates, 1])) if joint in (7, 8, 10, 11) else int(np.argmin(((v[candidates] - centre) ** 2).sum(1)))
                chosen = [first]
                distance = np.full(len(candidates), np.inf)
                for _ in range(2):
                    distance = np.minimum(distance, ((v[candidates] - v[candidates[chosen[-1]]]) ** 2).sum(1))
                    distance[chosen] = -1
                    chosen.append(int(distance.argmax()))
                ids.extend(candidates[chosen].tolist())
                labels.extend([joint] * 3)
            # One additional head marker gives the legacy 67-point BPS width.
            head = np.flatnonzero(mapped == 15)
            extra = head[np.argmax(v[head, 2])]
            if extra in ids:
                extra = next(int(i) for i in head if i not in ids)
            ids.append(int(extra)); labels.append(15)
            self.ids[model_key] = np.asarray(ids), np.asarray(labels)
        ids, labels = self.ids[model_key]
        beta = torch.as_tensor(info['betas'], dtype=torch.float32)[None]
        with torch.no_grad():
            shaped = model.v_template[None] + blend_shapes(beta, model.shapedirs[:, :, :beta.shape[-1]])
            joints = vertices2joints(model.J_regressor, shaped)[0]
        value = {
            'vertices': shaped[0, ids].to(self.device).detach(),
            'joints': joints.to(self.device).detach(),
            'weights': model.lbs_weights[ids].to(self.device).detach(),
            'parents': model.parents.to(self.device).detach(),
            'posedirs': model.posedirs.reshape(-1, len(model.v_template), 3)[:, ids].reshape(model.posedirs.shape[0], -1).to(self.device).detach(),
            'labels': torch.as_tensor(labels, device=self.device),
            'vertex_ids': ids.copy(), 'scale': float(info['scale']),
            # SMPL-X's zero hand parameters include its registered mean hand
            # pose. Match the native decoder instead of assuming flat hands.
            'extra_rotations': axis_angle_to_matrix(model.pose_mean.reshape(-1, 3)[22:]).to(self.device).detach()
            if hasattr(model, 'pose_mean') else torch.eye(3, device=self.device).expand(len(model.parents) - 22, 3, 3),
        }
        self.cache[key] = value
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return value


def skin_points(motion, asset):
    t = len(motion)
    local = rotation_from_6d(motion[:, 3:135].reshape(t, 22, 6))
    eye = torch.eye(3, device=motion.device, dtype=motion.dtype)
    rotations = torch.cat((local, asset['extra_rotations'][None].expand(t, -1, -1, -1)), 1)
    posed = asset['vertices'][None] + ((rotations[:, 1:] - eye).reshape(t, -1) @ asset['posedirs']).reshape(t, 67, 3)
    _, transforms = batch_rigid_transform(rotations, asset['joints'][None].expand(t, -1, -1), asset['parents'])
    transform = torch.einsum('vj,tjkl->tvkl', asset['weights'], transforms)
    homogeneous = torch.cat((posed, torch.ones_like(posed[:, :, :1])), -1)
    return (transform @ homogeneous[..., None])[:, :, :3, 0] * asset['scale'] + motion[:, None, :3] * 2


PROBES = np.asarray([[0., 0., 0.]] + [
    [distance if axis == k else 0. for axis in range(3)]
    for distance in (-.08, .08, -.3, .3) for k in range(3)], np.float32)


class ObservedSurfaceWindow:
    """Offline observed static union plus current observed dynamic objects.

    Nearest surface samples do not constitute a signed SDF. A missing
    neighbourhood is unknown and has an explicit mask/null attention token.
    """
    def __init__(self, points, dynamic=None, radius=1.):
        self.points = np.asarray(points, np.float32).reshape(-1, 3)
        if not np.isfinite(self.points).all():
            raise ValueError('Nonfinite observed scene')
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.dynamic = dynamic
        self.radius = radius
        self.dynamic_trees = {}

    def query(self, vertices, rotations):
        vertices = np.asarray(vertices, np.float32)
        rotations = np.asarray(rotations, np.float32)
        probes = vertices[:, :, None] + np.einsum('tjik,pk->tjpi', rotations, PROBES)
        flat = probes.reshape(-1, 3)
        nearest = np.zeros_like(flat)
        distance = np.full(len(flat), np.inf)
        moving = np.zeros(len(flat), bool)
        if self.tree is not None:
            d, idx = self.tree.query(flat, distance_upper_bound=self.radius)
            valid = np.isfinite(d)
            distance[valid] = d[valid]
            nearest[valid] = self.points[idx[valid]]
        nearest = nearest.reshape(probes.shape)
        distance = distance.reshape(probes.shape[:-1])
        moving = moving.reshape(distance.shape)
        if self.dynamic is not None:
            if len(self.dynamic) != len(vertices):
                raise ValueError('Dynamic observation time mismatch')
            for t, cloud in enumerate(self.dynamic):
                if not len(cloud):
                    continue
                if t not in self.dynamic_trees:
                    self.dynamic_trees[t] = cKDTree(cloud)
                d, idx = self.dynamic_trees[t].query(probes[t].reshape(-1, 3), distance_upper_bound=self.radius)
                d = d.reshape(distance[t].shape); idx = idx.reshape(d.shape)
                take = d < distance[t]
                nearest[t][take] = cloud[idx[take]]
                distance[t][take] = d[take]; moving[t][take] = True
        return nearest, np.isfinite(distance), moving


class OfflineSceneCache:
    def __init__(self, limit=4):
        self.limit, self.cache = limit, OrderedDict()

    def ego(self, identity, frames):
        key = (identity['scene_bundle'], float(identity['source_start_30fps']), frames)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        folder = Path(identity['scene_bundle']); memory = Path(identity['memory_bundle'])
        ids = np.load(folder / 'source_frame_ids.npy', mmap_mode='r')
        ix = int(np.searchsorted(ids, identity['source_start_30fps']))
        expected = identity['source_start_30fps'] + np.arange(frames) * 1.5
        if not np.allclose(ids[ix:ix + frames], expected):
            raise ValueError('Observed scene/window timestamp mismatch')
        positions = np.asarray(np.load(folder / 'camera_position_scenemi_yup.npy', mmap_mode='r')[ix:ix + frames])
        rotations = np.asarray(np.load(folder / 'camera_rotation_scenemi_yup.npy', mmap_mode='r')[ix:ix + frames])
        origin, anchor = positions[0], anchor_rotation(rotations)
        # The declared offline task permits all surfaces observed in this stream.
        points = (np.asarray(np.load(memory / 'static_points.npy', mmap_mode='r')) - origin) @ anchor.T
        cloud = np.load(folder / 'visible_frame_points_scenemi_yup.npy', mmap_mode='r')
        owners = np.load(folder / 'visible_frame_owner.npy', mmap_mode='r')
        dynamic = []
        for t in range(ix, ix + frames):
            keep = (owners[t] > 0) & (owners[t] < 100)
            dynamic.append((np.asarray(cloud[t])[keep] - origin) @ anchor.T)
        value = ObservedSurfaceWindow(points, dynamic)
        self.cache[key] = value
        if len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return value


def occupancy_from_points(points):
    points = np.asarray(points)
    index = np.floor((points - [0., -.7, 0.] + [6.4, 2.4, 6.4]) /
                     [.26666667, .2, .26666667]).astype(np.int32)
    index = index[((index >= 0) & (index < [48, 24, 48])).all(1)]
    occupancy = np.zeros((24, 48, 48), np.float32)
    occupancy[index[:, 1], index[:, 0], index[:, 2]] = 1.
    return occupancy
