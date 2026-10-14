"""World-space observed static maps, independent of motion-window sampling.

Only owner=0 first-hit observations enter memory. Offline reconstruction uses
the supplied recording's observed union; a timestamp-filtered prefix remains
available as a separate control. Dynamic surfaces remain per-frame.
"""
from experiments.offline_camera_retrain_v1.native30_exo import external_transforms
from collections import OrderedDict
from pathlib import Path
import json
import hashlib
import os
import uuid
import re
import numpy as np

MEMORY_PROTOCOL = 'observed-static-prefix-through-window-end-v2'
OFFLINE_MEMORY_PROTOCOL = 'observed-static-recording-union-v3'
MEMORY_PROTOCOLS = (MEMORY_PROTOCOL, OFFLINE_MEMORY_PROTOCOL)


def observer_stream_key(meta, mode):
    if mode not in ('ego', 'exo'):
        raise ValueError('Unknown observer mode')
    sequence = meta['sequence_id']
    recording = meta.get('recording')
    if not recording and meta['group'] in ('camera_wearer', 'interactee'):
        match = re.match(r'(recording_\d{8}_S\d+_S\d+_\d+)_', sequence)
        recording = match.group(1) if match else None
    # Different observers, virtual target cameras, splits and world frames
    # must never exchange maps, even when the physical room is the same.
    return (meta['split'], meta['group'], mode, recording or sequence,
            meta.get('scene_id', meta.get('scene')), meta.get('world_frame_id'),
            meta.get('exo_camera',{}).get('view_id') if mode=='exo' else None)


class ObservedStaticMemory:
    def __init__(self, load, entries, *, protocol=OFFLINE_MEMORY_PROTOCOL, voxel=.025, cached_streams=8, cache_root=None):
        if protocol not in MEMORY_PROTOCOLS:
            raise ValueError('Unsupported observed scene-memory protocol')
        self.protocol = protocol
        self.load = load
        self.entries = entries
        self.voxel = voxel
        self.cached_streams = cached_streams
        self.maps = OrderedDict()
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.cache_hits = self.cache_misses = 0

    def _cache_path(self, key, mode, entries):
        if self.cache_root is None: return None
        signature = dict(protocol=self.protocol, voxel=self.voxel, stream=key,
            implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sources=[dict(path=str(Path(m['path']).resolve()), output_sha256=m.get('output_sha256'),
                          input_sha256=m.get('input_sha256'), exo_camera=m.get('exo_camera')) for m in entries])
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root / (digest + '.npz')

    def _map(self, identity, mode):
        key = observer_stream_key(identity, mode)
        if key in self.maps:
            self.maps.move_to_end(key)
            return self.maps[key]
        entries = sorted((m for m in self.entries if observer_stream_key(m, mode) == key and
                          mode in m.get('allowed_modes',['ego','exo'])),
                         key=lambda m: (m['source_first'], m['path']))
        if not any(Path(m['path']).resolve() == Path(identity['path']).resolve() for m in entries):
            raise ValueError('Current observation bundle absent from its memory stream')
        cache = self._cache_path(key, mode, entries)
        if cache is not None and cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                result = saved['points'], saved['cameras'], saved['first_seen']
            self.cache_hits += 1
            return self._remember(key, result)
        self.cache_misses += 1
        point_key, owner_key = (('exo_frame_points_world', 'exo_frame_owner') if mode == 'exo'
            else ('visible_frame_points_scenemi_yup', 'visible_frame_owner'))
        parts = []
        for meta in entries:
            folder = meta['path']
            points, owners = self.load(folder, point_key), self.load(folder, owner_key)
            ids = np.asarray(self.load(folder, 'source_frame_ids'))
            valid = np.asarray(self.load(folder, 'observation_valid'), bool)
            if points.shape[:2] != owners.shape or len(points) != len(ids) or valid.shape != ids.shape:
                raise ValueError('Scene memory observation clocks disagree')
            if mode == 'exo':
                camera = external_transforms(self.load,folder,0,len(ids))[:,:3,3]
            else:
                camera = self.load(folder, 'camera_position_scenemi_yup')
            for start in range(0, len(ids), 128):
                sl = slice(start, start+128)
                own = np.asarray(owners[sl])
                if np.any(own == 100):
                    raise ValueError('Target body leaked into scene memory')
                mask = (own == 0) & valid[sl, None]
                p = np.asarray(points[sl])[mask]
                c = np.broadcast_to(camera[sl, None], points[sl].shape)[mask]
                time = np.broadcast_to(ids[sl, None], own.shape)[mask]
                if not len(p):
                    continue
                if not np.isfinite(p).all() or not np.isfinite(c).all():
                    raise ValueError('Nonfinite observed static map')
                _, ix = np.unique(np.floor(p/self.voxel).astype(np.int64), axis=0, return_index=True)
                parts.append((p[ix], c[ix], time[ix]))
        if parts:
            p, c, time = (np.concatenate([part[i] for part in parts]) for i in range(3))
            order = np.argsort(time, kind='stable')
            p, c, time = p[order], c[order], time[order]
            _, ix = np.unique(np.floor(p/self.voxel).astype(np.int64), axis=0, return_index=True)
            result = p[ix].astype(np.float32), c[ix].astype(np.float32), time[ix]
        else:
            result = np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty(0)
        if cache is not None:
            temporary = cache.with_name(cache.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.savez(stream, points=result[0], cameras=result[1], first_seen=result[2])
                    stream.flush(); os.fsync(stream.fileno())
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
        return self._remember(key, result)

    def _remember(self, key, result):
        self.maps[key] = result
        while len(self.maps) > self.cached_streams:
            self.maps.popitem(last=False)
        return result

    def window(self, identity, mode, frames):
        ids = self.load(identity['path'], 'source_frame_ids')
        last = identity['start']+frames-1
        if identity['start'] < 0 or last >= len(ids):
            raise ValueError('Memory window outside observed source')
        p, c, first_seen = self._map(identity, mode)
        select = first_seen <= ids[last] if self.protocol == MEMORY_PROTOCOL else np.ones(len(p),bool)
        origin, anchor = np.asarray(identity['origin']), np.asarray(identity['anchor'])
        return ((p[select]-origin)@anchor.T).astype(np.float32), ((c[select]-origin)@anchor.T).astype(np.float32)


class ObservedStaticPrefix(ObservedStaticMemory):
    """Explicit legacy prefix control; preserve reproducibility of v2 checks."""
    def __init__(self, load, entries, *, voxel=.025, cached_streams=2):
        super().__init__(load,entries,protocol=MEMORY_PROTOCOL,voxel=voxel,cached_streams=cached_streams)


def frozen_memory_entries(folder, meta, manifest=None):
    """Read observation identities only; never open motion/contact arrays."""
    folder = Path(folder).resolve()
    if manifest is None:
        # Full export is root/split/bundle. A standalone bundle stays valid,
        # with memory bounded by its own exported observation coverage.
        candidate = folder.parent.parent/'manifest.jsonl'
        manifest = candidate if candidate.exists() else None
    entries = [dict(meta, path=str(folder))]
    if manifest is not None:
        entries = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
        entries = [m for m in entries if m['split'] == meta['split'] and
                   m.get('split_protocol') == meta.get('split_protocol') and m['fps'] == 30]
    return entries
