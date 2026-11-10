"""Existing four-source sampler with public CPU scene-map views.

Motion membership, GT, camera normalization and source quotas are unchanged.
EgoBody maps span the original video; TRUMANS and GRAIL require verified
full source clips. No full point map is copied by the tensor collator.
"""
from collections import OrderedDict
import json

import numpy as np

from scenemi.config import ROOT
from scenemi.data.scene_geometry_data import GeometryEgoData, collate as tensor_collate, payload as tensor_payload
from scenemi.data.scene_geometry_cache import fit_normals
from scenemi.data.video_scene import VideoMapStore, voxel_observations
from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView
from experiments.offline_camera_retrain_v1.data import anchor_rotation


def collate(samples, device, *, pin=False):
    result = tensor_collate(samples, device, pin=pin)
    result['scene_views'] = tuple(s['scene_view'] for s in samples)
    return result


def payload(batch):
    return dict(tensor_payload(batch), scene_views=batch['scene_views'])


class IndexedGeometryEgoData(GeometryEgoData):
    def __init__(self, c, paths, split, *, spatial_policy, video_manifest, index_cache=None):
        super().__init__(c, paths, split)
        self.spatial_policy = spatial_policy
        self.video_maps = VideoMapStore(video_manifest)
        if self.video_maps.manifest['split'] != split:
            raise ValueError('Original-video map split differs from the sampler')
        self.spatial_indices = OrderedDict()
        self.index_cache = None
        if index_cache is not None:
            from scenemi.geometry.spatial_execution import SceneIndexCache
            from scenemi.data.amass_prepare import sha256
            self.index_cache = SceneIndexCache(index_cache)
            self.index_implementation = {str(p.relative_to(ROOT)): sha256(p) for p in
                (ROOT/'scenemi/data/spatial_scene_data.py', ROOT/'scenemi/data/video_scene.py',
                 ROOT/'scenemi/data/scene_geometry_cache.py')}
        from experiments.offline_camera_retrain_v1.data import PROJECTS
        self.trumans_frames = {r['clip_name']: r['num_frames'] for r in
            map(json.loads, (PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}

    def source_index(self, row):
        path = row['path']
        if row['dataset'] == 'egobody':
            if self.index_cache is not None:
                if path not in self.spatial_indices:
                    entry = self.video_maps.by_path[path]
                    self.spatial_indices[path] = self.index_cache.get(dict(version='complete-observed-index-v1',
                        split=self.split, map_sha256=entry['sha256']), lambda: self.video_maps.get(path)[0])
                self.spatial_indices.move_to_end(path)
                while len(self.spatial_indices) > 4:
                    self.spatial_indices.popitem(last=False)
                return self.spatial_indices[path], 'original_video_same_observer'
            return self.video_maps.get(path)[0], 'original_video_same_observer'
        if path not in self.spatial_indices:
            bundle = self.native.selection_bundle
            meta = bundle.metadata(row, self.split)
            if row['dataset'] == 'grail':
                if meta.get('scope') != 'full' or meta.get('source_first') != 0:
                    raise ValueError('GRAIL scene memory requires the complete original source animation')
            elif row['dataset'] == 'trumans':
                if meta['source_first'] != 0 or meta['frames'] != self.trumans_frames[meta['sequence_id']]:
                    raise ValueError('TRUMANS scene memory does not cover the original video clip')
            else:
                raise ValueError('Unsupported scene source')
            if self.index_cache is not None:
                identity = dict(version='complete-observed-index-v1', source=path, split=self.split,
                    input_sha256=meta['input_sha256'],
                    arrays={name: meta['output_sha256'][name+'.npy'] for name in
                        ('visible_frame_points_scenemi_yup', 'visible_frame_owner', 'observation_valid',
                         'camera_position_scenemi_yup')},
                    voxel_m=self.spatial_policy['map']['voxel_m'], normals=self.spatial_policy['normals'],
                    implementation_sha256=self.index_implementation)
                self.spatial_indices[path] = self.index_cache.get(identity, lambda: self._build_source_index(meta))
            else:
                self.spatial_indices[path] = self._build_source_index(meta)
            while len(self.spatial_indices) > 4:
                self.spatial_indices.popitem(last=False)
        self.spatial_indices.move_to_end(path)
        return self.spatial_indices[path], 'complete_original_source_clip'

    def _build_source_index(self, meta):
        bundle = self.native.selection_bundle
        points = bundle.array(meta, 'visible_frame_points_scenemi_yup')
        owners = bundle.array(meta, 'visible_frame_owner')
        valid = bundle.array(meta, 'observation_valid').astype(bool)
        camera = bundle.array(meta, 'camera_position_scenemi_yup')
        active = (owners == 0) & valid[:, None] & np.isfinite(points).all(-1)
        frames, _ = np.nonzero(active)
        full, origins, observations = voxel_observations(points[active], frames, camera[frames],
            self.spatial_policy['map']['voxel_m'])
        normals, normal_valid = fit_normals(full, full, origins, self.spatial_policy['normals'])
        # Separately declared synthetic-chair context is not an observed
        # surface and is therefore excluded from this observed-map index.
        return ObservedSceneIndex(full, normals, normal_valid, observations)

    def sample(self, source, rng, *, member=None, start=None):
        sample, identity = super().sample(source, rng, member=member, start=start)
        view = None
        if source != 'amass':
            row = self.native.records[identity['member']]
            index, scope = self.source_index(row)
            saved = self.native.cached(row)
            first = identity['start']
            origin = np.asarray(saved['camera_position_scenemi_yup'][first], np.float64).copy();origin[1] = 0.
            rotation = anchor_rotation(saved['camera_rotation_scenemi_yup'][first:first+128])
            view = SceneView(index, origin, rotation)
            identity = dict(identity, spatial_map_scope=scope, spatial_map_points=len(index.points),
                spatial_map_global_point_limit=None, spatial_query_after_B=True,
                synthetic_context_in_observed_index=False)
        return dict(sample, scene_view=view), identity

    def sample_camera_case(self, case):
        sample, identity = super().sample_camera_case(case)
        return dict(sample, scene_view=None), identity
