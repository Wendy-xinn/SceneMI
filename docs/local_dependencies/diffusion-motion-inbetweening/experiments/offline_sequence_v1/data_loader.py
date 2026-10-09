"""Strict mixed-source loader for offline camera/scene motion experiments."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from sample.prepare_trumans_static_scene import camera_cloud

HERE = Path(__file__).resolve().parent
TRUMANS = HERE.parents[2] / 'TRUMANS'
LENGTHS = (64, 128, 192)


def anchor_rotation(camera_rotation):
    forward = camera_rotation[0, :, 2]
    theta = np.arctan2(forward[0], forward[2])
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float32)


class MixedSequences:
    def __init__(self, split, *, trumans_dir=None, egobody_dir=None,
                 global_points=1024, local_points=64, seed=2026):
        self.split = split
        self.global_points = global_points
        self.local_points = local_points
        self.rng = np.random.default_rng(seed)
        self.cache = OrderedDict()
        self.object_cache = {}
        source = [json.loads(line) for line in (HERE / 'data/intervals.jsonl').read_text().splitlines() if line]
        audit = {(r['dataset'], r['sequence_id']): r for r in source}
        self.groups = {'trumans': [], 'camera_wearer': [], 'interactee': []}
        for dataset, directory in (('trumans', trumans_dir), ('egobody', egobody_dir)):
            if directory is None:
                continue
            index = Path(directory) / f'{split}.jsonl'
            if not index.is_file():
                raise FileNotFoundError(index)
            for line in index.read_text().splitlines():
                if not line:
                    continue
                meta = json.loads(line)
                if meta['dataset'] != dataset or meta['split'] != split:
                    raise ValueError(f'{index}: wrong dataset or split')
                source_id = (meta['sequence_id'] if dataset == 'trumans'
                             else f'{meta["recording"]}/{meta["role"]}')
                key = (dataset, source_id)
                if key not in audit or audit[key]['split'] != split:
                    raise ValueError(f'{source_id}: source split mismatch')
                if audit[key]['scene_family'] != meta['scene_family']:
                    raise ValueError(f'{source_id}: scene family mismatch')
                compatible_camera = {
                    'real_PV_pose': 'real_PV_pose_synthetic_scene_depth',
                    'synthetic_head_from_smpl': 'synthetic_head_xright_yup_zforward',
                }.get(audit[key]['camera_source'], audit[key]['camera_source'])
                if compatible_camera != meta['camera_source']:
                    raise ValueError(f'{source_id}: camera source mismatch')
                valid = {}
                first = meta.get('source_first_frame', 0)
                for length in LENGTHS:
                    starts = []
                    for source_start in audit[key]['valid_starts_30fps'][str(length)]:
                        offset = (source_start - first) / 1.5
                        if abs(offset - round(offset)) > 1e-5:
                            continue
                        index20 = int(round(offset))
                        if index20 >= 0 and index20 + length <= meta['frames_20fps']:
                            starts.append((index20, source_start))
                    valid[length] = starts
                if not any(valid.values()):
                    continue
                group = 'trumans' if dataset == 'trumans' else meta['role']
                self.groups[group].append((meta, valid))
        for group, members in self.groups.items():
            if not members:
                raise ValueError(f'{split}: no usable sequences for {group}')

    def summary(self):
        return {group: {str(length): {'sequences': sum(bool(v[length]) for _, v in members),
                                      'starts': sum(len(v[length]) for _, v in members)}
                        for length in LENGTHS}
                for group, members in self.groups.items()}

    def _load(self, meta):
        folder = Path(meta['folder'])
        key = str(folder)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        arrays = tuple(np.load(folder / filename, mmap_mode='r') for filename in
                       ('joints_world.npy', 'camera_position_world.npy',
                        'camera_rotation_world.npy', 'visible_static_points_world.npy'))
        joints, pos, rot, points = arrays
        if not (joints.shape == (meta['frames_20fps'], 22, 3)
                and pos.shape == (len(joints), 3) and rot.shape == (len(joints), 3, 3)
                and points.ndim == 2 and points.shape[1] == 3):
            raise ValueError(f'{key}: tensor shape mismatch')
        self.cache[key] = arrays
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return arrays

    def _object_points(self, meta, source_start, camera_position, camera_rotation):
        if meta['dataset'] != 'trumans' or not meta['object_tracks']:
            return np.zeros((0, 3), np.float32)
        track_path = meta['object_tracks']
        if track_path not in self.object_cache:
            tracks = np.load(track_path, allow_pickle=True).item()
            pieces = {}
            for name, track in tracks.items():
                mesh_path = TRUMANS / 'Object_all/Object_mesh' / f'{name}.obj'
                mesh = trimesh.load(mesh_path, force='mesh', process=False)
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
                points, _ = trimesh.sample.sample_surface(mesh, 512, seed=seed)
                pieces[name] = (points.astype(np.float32), track)
            self.object_cache[track_path] = pieces
        visible = []
        for local_points, track in self.object_cache[track_path].values():
            frame = source_start
            rot = Rotation.from_euler('xyz', np.asarray(track['rotation'][frame])).as_matrix()
            pos = np.asarray(track['location'][frame])
            world = local_points @ rot.T + pos
            # Only surfaces actually projected into the known video interval are used.
            for t in range(0, len(camera_position), 20):
                cloud, mask = camera_cloud(world, np.zeros_like(world),
                                           camera_position[t], camera_rotation[t],
                                           128, 90, 70, .05, 4.)
                if mask.any():
                    visible.append(cloud[mask, :3] @ camera_rotation[t].T + camera_position[t])
        return np.concatenate(visible).astype(np.float32) if visible else np.zeros((0, 3), np.float32)

    def sample(self, length, group=None, sequence_index=None, start_index=None):
        if length not in LENGTHS:
            raise ValueError(f'Unknown length {length}')
        if group is None:
            group = self.rng.choice(list(self.groups))
        members = [(meta, valid) for meta, valid in self.groups[group] if valid[length]]
        if not members:
            raise ValueError(f'{group}: no {length}-frame sequences')
        if sequence_index is None:
            sequence_index = int(self.rng.integers(len(members)))
        meta, valid = members[sequence_index]
        if start_index is None:
            start_index = int(self.rng.integers(len(valid[length])))
        start, source_start = valid[length][start_index]
        joints_full, pos_full, rot_full, static_points = self._load(meta)
        joints = np.asarray(joints_full[start:start + length])
        position = np.asarray(pos_full[start:start + length])
        rotation = np.asarray(rot_full[start:start + length])
        dynamic = self._object_points(meta, source_start, position, rotation)
        points = np.concatenate((np.asarray(static_points), dynamic))
        if len(points) < 32:
            raise ValueError(f'{meta["sequence_id"]}: no usable scene')
        tree = cKDTree(points)
        global_ids = np.unique(tree.query(position[::8], k=min(64, len(points)))[1].reshape(-1))
        if len(global_ids) > self.global_points:
            global_ids = global_ids[np.linspace(0, len(global_ids) - 1,
                                                 self.global_points).astype(int)]
        if len(global_ids) < self.global_points:
            supplement = np.linspace(0, len(points) - 1,
                                     min(self.global_points - len(global_ids), len(points))).astype(int)
            global_ids = np.unique(np.r_[global_ids, supplement])[:self.global_points]
        global_world = points[global_ids]
        local_ids = tree.query(position, k=min(self.local_points, len(points)))[1]
        local_ids = np.asarray(local_ids).reshape(length, -1)
        local_world = points[local_ids]
        world_to_anchor = anchor_rotation(rotation)
        origin = position[0]
        target = (joints - origin) @ world_to_anchor.T / 2.
        camera_position = (position - origin) @ world_to_anchor.T / 2.
        camera_rotation = np.einsum('ij,tjk->tik', world_to_anchor, rotation)
        camera = np.concatenate((camera_position, camera_rotation[:, :, 0],
                                 camera_rotation[:, :, 1]), axis=-1)
        global_scene = (global_world - origin) @ world_to_anchor.T / 2.
        local_scene = np.einsum('tpi,tij->tpj', (local_world - position[:, None]), rotation) / 2.
        global_data = np.zeros((self.global_points, 3), np.float32)
        global_mask = np.zeros(self.global_points, bool)
        global_data[:len(global_scene)] = global_scene
        global_mask[:len(global_scene)] = True
        local_data = np.zeros((length, self.local_points, 3), np.float32)
        local_mask = np.zeros((length, self.local_points), bool)
        local_data[:, :local_scene.shape[1]] = local_scene
        local_mask[:, :local_scene.shape[1]] = True
        output = {'motion': target.astype(np.float32), 'camera': camera.astype(np.float32),
                  'scene_global': global_data, 'scene_global_mask': global_mask,
                  'scene_local': local_data, 'scene_local_mask': local_mask}
        if not all(np.isfinite(value).all() for value in output.values()):
            raise ValueError(f'{meta["sequence_id"]}: nonfinite batch')
        return output, dict(dataset=meta['dataset'], group=group,
                            sequence_id=meta['sequence_id'], source_start_30fps=source_start,
                            camera_source=meta['camera_source'], dynamic_object_points=len(dynamic))


def collate(items):
    return {key: torch.from_numpy(np.stack([item[key] for item in items]))
            for key in items[0]}
