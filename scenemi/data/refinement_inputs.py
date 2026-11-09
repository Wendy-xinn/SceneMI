"""Selective verified motion/current-frame inputs for indexed scene refinement.

Legacy static tensors are unused by the frozen camera-only prior and the
indexed refiner. Their payloads stay on disk; CPU query workers own the full
original-source observed map separately.
"""
from collections import OrderedDict
import json
from pathlib import Path

from scenemi.geometry.query_runtime import read_verified_array


REFINEMENT_ARRAYS = ('rest','motion_world','camera_position_scenemi_yup',
    'camera_rotation_scenemi_yup','camera_intrinsics_128x96','frame_points',
    'frame_owner','frame_normals','frame_normal_valid','contact_weight')


class RefinementInputStore:
    """One bounded owner; checks only the ten fields actually consumed."""
    def __init__(self, root, *, limit=32, byte_limit=256*2**20):
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError('Refinement input cache supports one to 32 entries')
        if type(byte_limit) is not int or not 0 < byte_limit <= 256*2**20:
            raise ValueError('Refinement input cache must be bounded by 256MiB')
        self.root = Path(root).resolve()
        if str(self.root).startswith('/mnt/'):
            raise ValueError('Refinement input caches must be on the Linux filesystem')
        self.limit,self.byte_limit = limit,byte_limit
        self.opened = OrderedDict()
        self.statistics = dict(loads=0,hits=0,evictions=0,hash_bytes=0,rejected_oversize_entries=0)

    def resident_bytes(self):
        return sum(row[2] for row in self.opened.values())

    def read(self, key, identity):
        if not isinstance(key,str) or not key or Path(key).name != key or key in ('.','..'):
            raise ValueError('Invalid prepared refinement input key')
        if key in self.opened:
            saved_identity,arrays,_ = self.opened[key]
            if saved_identity != identity:
                raise ValueError('Prepared refinement input identity differs')
            self.opened.move_to_end(key);self.statistics['hits'] += 1
            return arrays
        folder = self.root/key
        file = folder/'CACHE.json'
        if file.stat().st_size > 2**20:
            raise ValueError(f'Oversize refinement input manifest: {file}')
        meta = json.loads(file.read_bytes())
        if meta.get('identity') != identity or meta.get('complete') is not True:
            raise ValueError(f'Prepared refinement input identity differs: {file}')
        names = [name+'.npy' for name in REFINEMENT_ARRAYS]
        if not set(names) <= set(meta.get('arrays',{})):
            raise ValueError(f'Prepared refinement input fields are missing: {folder}')
        # Do not stat, hash, map or transform any static_* payload, including
        # when other arrays in this legacy cache are being verified.
        sizes = {name:(folder/name).stat().st_size for name in names}
        required = sum(sizes.values())
        if required > self.byte_limit:
            self.statistics['rejected_oversize_entries'] += 1
            raise MemoryError(f'Complete refinement input entry exceeds cache budget before loading: '
                              f'{folder}; required={required}, limit={self.byte_limit}')
        while self.opened and (len(self.opened) >= self.limit
                or self.resident_bytes()+required > self.byte_limit):
            self.opened.popitem(last=False);self.statistics['evictions'] += 1
        arrays = {Path(name).stem:read_verified_array(folder/name,meta['arrays'][name],
            max_bytes=sizes[name],statistics=self.statistics) for name in names}
        self.opened[key] = identity,arrays,required
        self.statistics['loads'] += 1
        return arrays

    def stats(self):
        return dict(self.statistics,resident_bytes=self.resident_bytes(),resident_entries=len(self.opened),
            cache_limit_bytes=self.byte_limit,cache_entry_limit=self.limit,
            selected_arrays=list(REFINEMENT_ARRAYS),legacy_static_geometry_read=False)
