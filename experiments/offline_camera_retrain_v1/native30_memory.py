"""World-space observed static maps, independent of motion-window sampling.

Only owner=0 first-hit observations enter memory. First-seen timestamps keep
later windows from using future observations. Dynamic surfaces remain per-frame.
"""
from collections import OrderedDict
from pathlib import Path
import json
import re
import numpy as np

MEMORY_PROTOCOL = 'observed-static-prefix-through-window-end-v2'


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
            meta.get('scene_id', meta.get('scene')), meta.get('world_frame_id'))


class ObservedStaticPrefix:
    def __init__(self, load, entries, *, voxel=.025, cached_streams=2):
        self.load = load
        self.entries = entries
        self.voxel = voxel
        self.cached_streams = cached_streams
        self.maps = OrderedDict()

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
                camera = np.broadcast_to(self.load(folder, 'exo_sensor_to_world')[:3, 3], (len(ids), 3))
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
        select = first_seen <= ids[last]
        origin, anchor = np.asarray(identity['origin']), np.asarray(identity['anchor'])
        return ((p[select]-origin)@anchor.T).astype(np.float32), ((c[select]-origin)@anchor.T).astype(np.float32)


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
