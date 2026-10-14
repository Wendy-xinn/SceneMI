"""Source-keyed, TRAIN-only weak labels; no cached truth in model inputs.

Labels are computed in the clean canonical window frame, then transformed
with the sampled world yaw. Window boundaries (velocity/persistence), observer,
source hashes, shape, quality and code are all part of the cache identity.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np

from experiments.offline_camera_retrain_v1.native30_scene_support import training_support_targets
from experiments.offline_camera_retrain_v1.native30_memory import observer_stream_key
from experiments.offline_camera_retrain_v1.native30_cache_io import mapped_sha256

PROTOCOL = 'native30-canonical-window-support-v1'
VECTOR_KEYS = ('points', 'normals', 'velocities', 'vertices')
ARRAY_KEYS = ('target', 'valid', 'weight', 'surface_valid', 'owners', 'velocity_valid') + VECTOR_KEYS


class TrainingSupportCache:
    def __init__(self, data, builder):
        if data.split != 'train': raise ValueError('Weak label cache is TRAIN-only')
        self.data, self.builder = data, builder
        self.root = data.runtime_cache_root / 'support'
        self.root.mkdir(parents=True, exist_ok=True)
        files = ('native30_support_cache.py', 'native30_scene_support.py', 'native30_geometry.py',
                 'native30_memory.py', 'native30_assets.py', 'unified_geometry.py',
                 'orientation_supervision.py', 'supervision.py', 'native30_gt_reliability.py',
                 'native30_observations.py', 'native30_data.py', 'native30_query_runtime.py')
        files += ('native30_surface_cache.py',)
        self.code = {f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest() for f in files}
        here=Path(__file__).parent
        body_files=list((here.parents[1]/'body_models/smplx').glob('SMPLX_*.*'))
        body_files+=list((here.parents[2]/'ProtoMotions/data/smpl').glob('SMPL_*.pkl'))
        self.body_hashes={str(p):mapped_sha256(p) for p in body_files if p.suffix in ('.npz','.pkl')}
        self.hits = self.misses = 0

    def path(self, identity):
        sources = [dict(path=str(Path(m['path']).resolve()), hashes=m.get('output_sha256'))
                   for m in self.data.scene_memory.entries
                   if observer_stream_key(m, identity['mode']) == observer_stream_key(identity, identity['mode'])]
        signature = dict(protocol=PROTOCOL, code=self.code, body_models=self.body_hashes, sources=sources,
                         mode=identity['mode'], start=identity['start'], frames=self.data.frames,
                         path=str(Path(identity['path']).resolve()), body=identity['native_body'],
                         memory=self.data.scene_memory.protocol, voxel=self.data.scene_memory.voxel,
                         quality=self.data.quality_config)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        return self.root / (digest + '.npz')

    def get(self, identity):
        path = self.path(identity)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                result = {key: saved[key] for key in ARRAY_KEYS}
                anchor, origin = saved['anchor'], saved['origin']
            self.hits += 1
        else:
            self.misses += 1
            group, mode = identity['group'], identity['mode']
            member = next(i for i, (m, _) in enumerate(self.data.members[group]) if m['path'] == identity['path'])
            sample, base = self.data.sample(group, mode, member=member, start=identity['start'], yaw=0)
            sample, scene, asset = self.builder.build(self.data, sample, base, mode, (15,) if mode == 'ego' else None)
            computed = training_support_targets(sample, scene, asset)
            result = {key: computed[key] for key in ARRAY_KEYS}
            anchor, origin = np.asarray(base['anchor']), np.asarray(base['origin'])
            temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.savez(stream, **result, anchor=anchor, origin=origin)
                    stream.flush(); os.fsync(stream.fileno())
                temporary.replace(path)
            finally: temporary.unlink(missing_ok=True)
        rotation = np.asarray(identity['anchor']) @ anchor.T
        translation = (origin - np.asarray(identity['origin'])) @ np.asarray(identity['anchor']).T
        result = dict(result)
        for key in VECTOR_KEYS:
            result[key] = (result[key] @ rotation.T + (translation if key in ('points', 'vertices') else 0)).astype(np.float32)
        return result

    def attach(self, sample, identity):
        targets = self.get(identity)
        result = dict(sample, contact_target=targets['target'], contact_valid=targets['valid'], contact_weight=targets['weight'])
        for key in ('points', 'normals', 'surface_valid', 'owners', 'velocities', 'velocity_valid', 'vertices'):
            result['support_' + key] = targets[key]
        return result


def cached_support(data, builder, sample, identity):
    if not hasattr(data, 'support_cache'): data.support_cache = TrainingSupportCache(data, builder)
    return data.support_cache.attach(sample, identity)
