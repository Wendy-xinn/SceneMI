"""Whole-recording observed static maps for explicitly offline scene stages.

Uses saved first-hit observations only. Dynamic surfaces stay per-frame, and
the existing versioned motion-quality selection remains unchanged. This
adapter is separate from the immutable causal loaders of retained runs.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

from scenemi.config import ROOT
from scenemi.data.checked_scene_dataset import CheckedLocalSceneDataset
from scenemi.data.grail_visibility import sha256
from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points


DEFAULT_POLICY = ROOT / 'configs/recording_scene_observed.yaml'


def load_policy(path=DEFAULT_POLICY):
    p = yaml.safe_load(Path(path).read_text())
    if (p['version'] != 'observed-recording-scene-v1' or p['causal'] is not False
            or p['observed_only'] is not True or p['scope'] != 'complete_source_recording'
            or p['static_owner_id'] != 0 or p['full_source_mesh_as_input'] is not False
            or p['dynamic_geometry'] != 'current_frame_only' or p['voxel_m'] <= 0
            or p['local_point_limit'] < 1):
        raise ValueError('Expected explicit offline, observed-only recording map policy')
    return p


def observed_recording(points, owners, valid_frames, voxel_m):
    """Deterministic first-observation voxel union; no complete mesh or GT pose."""
    points, owners = np.asarray(points), np.asarray(owners)
    valid_frames = np.asarray(valid_frames, bool)
    if points.shape != owners.shape + (3,) or valid_frames.shape != (len(points),):
        raise ValueError('Observation points/owners/frame validity have different clocks')
    valid = (owners == 0) & np.isfinite(points).all(-1) & valid_frames[:, None]
    selected = points[valid]
    if not len(selected):
        return np.empty((0,3), np.float32)
    _, ids = np.unique(np.floor(selected / voxel_m).astype(np.int64), axis=0, return_index=True)
    return np.asarray(selected[np.sort(ids)], np.float32)


def bounded_scene_points(points, count):
    points = np.asarray(points, np.float32)
    if len(points) > count:
        points = points[np.linspace(0,len(points)-1,count).astype(int)]
    return points


class RecordingSceneDataset(CheckedLocalSceneDataset):
    """Same motion/camera sampler, explicitly replaced offline scene memory."""
    def __init__(self, *args, map_policy=DEFAULT_POLICY, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording_policy = load_policy(map_policy)
        self.recording_policy_sha256 = sha256(map_policy)
        if self.config['local']['history_points'] != self.recording_policy['local_point_limit']:
            raise ValueError('Model point budget differs from the declared recording-map budget')
        self.recording_maps = OrderedDict()

    def sample(self, rng, **kwargs):
        sample, identity = super().sample(rng, **kwargs)
        record = self.records[identity['member']]
        meta = self.selection_bundle.metadata(record,self.split)
        path = record['path']
        if path not in self.recording_maps:
            observed = observed_recording(self.array(meta,'visible_frame_points_scenemi_yup'),
                self.array(meta,'visible_frame_owner'),self.array(meta,'observation_valid'),
                self.recording_policy['voxel_m'])
            self.recording_maps[path] = observed
            while len(self.recording_maps)>4:
                self.recording_maps.popitem(last=False)
        self.recording_maps.move_to_end(path)
        observed = self.recording_maps[path]
        synthetic = record.get('synthetic_chair')
        if synthetic:
            if self.split != 'train' or sha256(synthetic['condition_file']) != synthetic['condition_sha256']:
                raise ValueError('Synthetic chair is not an authorized unchanged train condition')
            with np.load(synthetic['condition_file'],allow_pickle=False) as data:
                observed = np.concatenate((observed,data['synthetic_chair_points_m']))
        origin, anchor = np.asarray(identity['origin']), np.asarray(identity['anchor'])
        local = ((observed-origin) @ anchor.T).astype(np.float32)
        sample['occupancy'] = occupancy_from_points(local)
        points = bounded_scene_points(local,self.recording_policy['local_point_limit'])
        hp = np.zeros_like(sample['local_history_points']); hv = np.zeros_like(sample['local_history_valid'])
        hp[:len(points)], hv[:len(points)] = points, True
        sample.update(local_history_points=hp,local_history_valid=hv)
        identity.update(scene_map_protocol='observed-recording-scene-v1',
            scene_map_policy_sha256=self.recording_policy_sha256,
            scene_history_cutoff=meta['frames']-1,local_surface_history_cutoff=meta['frames']-1,
            local_future_history=True,scene_map_causal=False,
            scene_map_observed_frames=[0,meta['frames']-1],scene_map_observed_only=True,
            scene_map_voxel_m=self.recording_policy['voxel_m'],
            scene_map_points_before_limit=len(local),scene_map_points_after_limit=len(points),
            source_scene_mesh_is_input=False)
        return sample, identity
