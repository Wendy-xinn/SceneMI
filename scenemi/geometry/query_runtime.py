"""Bounded, verified, read-only map ownership for CPU scene queries.

The parent can carry a map reference without constructing spatial trees.
Workers own complete maps and fitted support indices together. Neither this
module nor query workers import Torch or perform surface fitting.
"""
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

import numpy as np

from scenemi.data.numeric_mmap import _fields, map_numeric_npy
from scenemi.data.support_cache import PreparedSupportIndex, SupportCache
from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView
from scenemi.geometry.support_surfaces import SupportSurfaceAtlas

QUERY_WORKSPACE_BYTES = 64*2**20
MAX_NEIGHBOURS = 256


def validate_query(action, anchors, policy, *, dynamic=None, final_anchors=None, **options):
    """Reject oversized expansion before constructing any neighbourhood arrays."""
    anchors = np.asarray(anchors)
    if (anchors.ndim != 3 or anchors.shape[-1] != 3
            or anchors.dtype.kind not in 'fiu'):
        raise ValueError('Scene queries expect finite numeric T,A,3 anchors')
    count = policy['query']['neighbours_per_scale']
    if type(count) is not int or not 1 <= count <= MAX_NEIGHBOURS:
        raise ValueError(f'Scene neighbour count must be in 1..{MAX_NEIGHBOURS}')
    size = anchors.shape[0]*anchors.shape[1]
    # Two scales, conversion temporaries and ground channels. Comparisons
    # additionally retain two sets of Python point/owner identities.
    estimate = size*(256*count+512)
    if action == 'compare': estimate += size*count*4*320
    if estimate > QUERY_WORKSPACE_BYTES:
        raise MemoryError(f'Scene query expansion exceeds workspace budget: estimated={estimate}, '
                          f'limit={QUERY_WORKSPACE_BYTES}')
    if not np.isfinite(anchors).all():
        raise ValueError('Scene queries expect finite numeric T,A,3 anchors')
    if action == 'compare':
        final = np.asarray(final_anchors)
        if final.shape != anchors.shape or final.dtype.kind not in 'fiu' or not np.isfinite(final).all():
            raise ValueError('Both B queries require matching finite T,A,3 anchors')
    if dynamic is not None:
        if not isinstance(dynamic,(tuple,list)) or len(dynamic) != 4:
            raise ValueError('Expected four current-frame dynamic geometry arrays')
        points,owners,normals,valid = map(np.asarray,dynamic)
        if (points.ndim != 3 or points.shape[0] != anchors.shape[0] or points.shape[-1] != 3
                or normals.shape != points.shape or owners.shape != points.shape[:-1]
                or valid.shape != owners.shape
                or any(value.dtype.kind not in 'fibu' for value in (points,owners,normals,valid))):
            raise ValueError('Dynamic geometry shape or numeric dtype differs from query frames')
        estimate += points.shape[1]*192+sum(value.nbytes for value in (points,owners,normals,valid))
        if estimate > QUERY_WORKSPACE_BYTES:
            raise MemoryError(f'Dynamic scene query expansion exceeds workspace budget: '
                              f'estimated={estimate}, limit={QUERY_WORKSPACE_BYTES}')
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError('Dynamic scene points and normals must be finite')
    return estimate


class BoundedSupportAtlas(SupportSurfaceAtlas):
    """Keep historical fitting/selection unchanged; bound proposal allocation first."""
    def __init__(self, arrays, points, policy, *, statistics=None):
        self.workspace_statistics = {} if statistics is None else statistics
        super().__init__(arrays,points,policy)

    def _check_workspace(self, estimate):
        stats = self.workspace_statistics
        stats['support_query_workspace_peak_bytes'] = max(
            stats.get('support_query_workspace_peak_bytes',0),estimate)
        if estimate > QUERY_WORKSPACE_BYTES:
            stats['support_query_workspace_rejections'] = stats.get('support_query_workspace_rejections',0)+1
            raise MemoryError(f'Support proposals exceed query workspace before expansion: '
                              f'estimated={estimate}, limit={QUERY_WORKSPACE_BYTES}')

    def query(self, anchors):
        anchors = np.asarray(anchors)
        if (anchors.ndim not in (2,3) or anchors.shape[-1] != 3
                or anchors.dtype.kind not in 'fiu'):
            raise ValueError('Support queries expect finite numeric Q,3 or T,A,3 anchors')
        count = self.policy['query']['maximum_candidates']
        if type(count) is not int or not 1 <= count <= MAX_NEIGHBOURS:
            raise ValueError('Invalid support candidate count')
        size = anchors.size//3
        # Includes candidate tables and the candidate-pair temporal workspace.
        fixed = size*(256+count*128+count*count*32)
        self._check_workspace(fixed)
        if not np.isfinite(anchors).all():
            raise ValueError('Support queries expect finite anchors')
        flat = np.asarray(anchors,np.float32).reshape(-1,3)
        radius = self.policy['extraction']['radius_m']+self.policy['query']['horizontal_radius_m']
        # Counts do not allocate the ragged proposal lists. The original
        # method will run unchanged only after its expansion is affordable.
        lengths = self.centres.query_ball_point(flat[:,[0,2]],radius,workers=1,return_length=True)
        hull_faces = int(np.diff(self.arrays['hull_offsets']).max(initial=0))
        proposal_bytes = 2048+hull_faces*8
        self._check_workspace(fixed+int(np.asarray(lengths).sum())*proposal_bytes)
        return super().query(anchors)


def stamp(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def read_verified_array(path, expected, *, max_bytes, statistics=None):
    """Decode the same immutable RAM bytes whose complete NPY digest passed."""
    path = Path(path)
    before = stamp(path)
    if before[2] > max_bytes:
        raise MemoryError(f'Numeric array exceeds read budget before loading: {path}')
    with path.open('rb') as stream:
        if os.fstat(stream.fileno()).st_size != before[2]:
            raise ValueError(f'Numeric array changed before loading: {path}')
        raw = stream.read(before[2])
        if len(raw) != before[2] or stream.read(1):
            raise ValueError(f'Numeric array changed while loading: {path}')
    if statistics is not None:
        statistics['hash_bytes'] = statistics.get('hash_bytes',0)+len(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f'Numeric array checksum differs: {path}; expected={expected}, actual={actual}')
    if stamp(path) != before:
        raise ValueError(f'Numeric array changed while loading: {path}')
    if raw[:6] != b'\x93NUMPY' or raw[6:8] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
        raise ValueError(f'Unsupported numeric NPY format: {path}')
    width = 2 if raw[6] == 1 else 4
    if len(raw) < 8+width:
        raise ValueError(f'Truncated numeric NPY header: {path}')
    length = struct.unpack('<H' if width == 2 else '<I', raw[8:8+width])[0]
    offset = 8+width+length
    if not 0 < length <= 65536 or offset > len(raw):
        raise ValueError(f'Invalid numeric NPY header length: {path}')
    fields = _fields(raw[8+width:offset].decode('utf-8' if raw[6] == 3 else 'latin1'))
    dtype = np.dtype(fields['descr']); shape = fields['shape']
    if dtype.hasobject or dtype.fields or dtype.kind not in 'biufc':
        raise ValueError(f'Only numeric NPY arrays are supported: {path}')
    if offset+math.prod(shape)*dtype.itemsize != len(raw):
        raise ValueError(f'NPY payload length differs from its header: {path}')
    return np.ndarray(shape, dtype=dtype, buffer=raw, offset=offset,
                      order='F' if fields['fortran_order'] else 'C')


def manifest(reference):
    folder, expected = reference
    path = Path(folder)/'CACHE.json'
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f'Map manifest checksum differs: {path}; expected={expected}, actual={actual}')
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f'Invalid verified map manifest: {path}') from error
    if value.get('complete') is not True:
        raise ValueError(f'Incomplete map: {path}')
    return value


class MapReference:
    """Parent-side reference. Point mapping is used only for its real shape.

    All arrays consumed by queries are independently verified by MapRuntime.
    Deliberately offers no query/tree API, preventing accidental parent work.
    """
    def __init__(self, reference):
        self.execution_reference = tuple(reference)
        manifest(self.execution_reference)
        self.points = map_numeric_npy(Path(reference[0])/'points.npy')


class MapRuntime:
    def __init__(self, support_root, policy, manifests=(), *, cache_entries=8, cache_mib=256,
                 load_lock_path=None, load_timeout_seconds=120):
        if type(cache_entries) is not int or not 1 <= cache_entries <= 32:
            raise ValueError('Query cache entries must be in 1..32')
        if type(cache_mib) is not int or not 16 <= cache_mib <= 1024:
            raise ValueError('Query cache memory must be in 16..1024 MiB per worker')
        self.support = SupportCache(support_root, policy, manifests)
        self.limit, self.byte_limit = cache_entries, cache_mib*2**20
        if not 0 < load_timeout_seconds <= 600:
            raise ValueError('Map loading timeout must be in (0,600] seconds')
        self.load_lock_path = None if load_lock_path is None else Path(load_lock_path)
        self.load_timeout_seconds = load_timeout_seconds
        self.opened = OrderedDict()
        self.statistics = dict(cache_hits=0, cache_misses=0, evictions=0,
            hash_bytes=0, map_load_seconds=0., query_seconds=0., queries=0,
            rejected_oversize_maps=0, map_load_lock_wait_seconds=0.,
            query_workspace_peak_bytes=0,support_query_workspace_peak_bytes=0,
            support_query_workspace_rejections=0,rejected_queries=0)

    @contextmanager
    def _load_lock(self):
        """Serialize cold loads across workers; cached queries never take this lock."""
        if self.load_lock_path is None:
            yield
            return
        started = time.monotonic()
        with self.load_lock_path.open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = self.load_timeout_seconds-(time.monotonic()-started)
                    if remaining <= 0:
                        raise TimeoutError(f'Map loading lock exceeded {self.load_timeout_seconds}s')
                    time.sleep(min(.01, remaining))
            self.statistics['map_load_lock_wait_seconds'] += time.monotonic()-started
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _support_metadata(self, reference):
        folder, identity = self.support.location(reference)
        path = folder/'CACHE.json'
        raw = path.read_bytes()
        if self.support.allowed is not None:
            row = self.support.allowed.get(reference)
            if (row is None or Path(row['path']) != folder
                    or hashlib.sha256(raw).hexdigest() != row['sha256']):
                raise ValueError('Support map is outside the prepared input contract')
        meta = json.loads(raw)
        if meta.get('complete') is not True or meta.get('identity') != identity:
            raise ValueError(f'Support cache identity differs: {path}')
        return folder, meta

    def _layout(self, reference):
        """Bound allocation using verified manifests and NPY headers only.

        The historical writer publishes these exact numeric layouts. No
        payload is copied and no spatial tree is built during this check.
        Reservation includes backing files, RAM copies and tree workspace;
        even a single map must fit the configured per-worker budget.
        """
        folder = Path(reference[0]); meta = manifest(reference)
        map_fields = dict(points=(np.dtype('float32'), 3),
            normals=(np.dtype('float32'), 3), normal_valid=(np.dtype('bool'), None),
            observations=(np.dtype('int32'), None))

        def headers(root, metadata, fields):
            if set(metadata.get('arrays', {})) != {name+'.npy' for name in fields}:
                raise ValueError(f'Unexpected numeric map fields: {root}')
            values = {}
            for name, (dtype, width) in fields.items():
                path = root/(name+'.npy')
                if path.stat().st_size > self.byte_limit:
                    self.statistics['rejected_oversize_maps'] += 1
                    raise MemoryError(f'Map array exceeds query cache budget before loading: {path}')
                value = map_numeric_npy(path)
                if value.dtype != dtype or value.ndim != (1 if width is None else 2):
                    raise ValueError(f'Invalid numeric map dtype or dimensions: {path}')
                if width is not None and value.shape[1] != width:
                    raise ValueError(f'Invalid numeric map width: {path}')
                values[name] = value
            return values

        arrays = headers(folder, meta, map_fields)
        points = len(arrays['points'])
        if any(len(value) != points for value in arrays.values()):
            raise ValueError(f'Map arrays disagree on their point count: {folder}')
        raw_bytes = sum(value.nbytes for value in arrays.values())
        if meta.get('bytes') != raw_bytes:
            raise ValueError(f'Map byte count differs from numeric headers: {folder}')
        support_folder, support_meta = self._support_metadata(reference)
        support_fields = dict(coefficients=(np.dtype('float64'), 3),
            normals=(np.dtype('float32'), 3), centres=(np.dtype('float64'), 2),
            quality=(np.dtype('float32'), 4), fit_error=(np.dtype('float32'), None),
            sample_offsets=(np.dtype('int64'), None), sample_ids=(np.dtype('int32'), None),
            hull_offsets=(np.dtype('int64'), None), hull_equations=(np.dtype('float64'), 3))
        support = headers(support_folder, support_meta, support_fields)
        patches = len(support['coefficients'])
        if (any(len(support[name]) != patches for name in
                ('normals', 'centres', 'quality', 'fit_error'))
                or any(len(support[name]) != patches+1 for name in ('sample_offsets', 'hull_offsets'))):
            raise ValueError(f'Support arrays disagree on their patch count: {support_folder}')
        support_bytes = sum(value.nbytes for value in support.values())
        if support_meta.get('bytes') != support_bytes or support_meta.get('patches') != patches:
            raise ValueError(f'Support byte/patch count differs from numeric headers: {support_folder}')
        # Include temporary/file-backed copies, full-map 3D/2D tree coordinates
        # and topology, patch tree copies and per-patch Python/tree overhead.
        return (2*raw_bytes + points*128 + 2*support_bytes
                + len(support['sample_ids'])*112 + patches*2048)

    def _read_array(self, path, expected):
        """Hash and decode the exact same immutable RAM bytes, without reopening payloads."""
        return read_verified_array(path,expected,max_bytes=self.byte_limit,statistics=self.statistics)

    def _arrays_in(self, folder, meta):
        arrays = {}
        for name, expected in meta['arrays'].items():
            if Path(name).name != name or not name.endswith('.npy'):
                raise ValueError(f'Invalid map array path: {folder/name}')
            arrays[Path(name).stem] = self._read_array(folder/name, expected)
        return arrays

    def _arrays(self, reference):
        return self._arrays_in(Path(reference[0]), manifest(reference))

    def view(self, reference, origin, rotation):
        reference = tuple(reference)
        if reference not in self.opened:
            start = time.monotonic();self.statistics['cache_misses'] += 1
            with self._load_lock():
                reservation = self._layout(reference)
                if reservation > self.byte_limit:
                    self.statistics['rejected_oversize_maps'] += 1
                    raise MemoryError(f'Complete map exceeds query cache budget before loading: '
                        f'{reference[0]}; estimated={reservation}, limit={self.byte_limit}')
                # Evict before allocation so old and incoming trees cannot
                # briefly exceed this worker's complete-map reservation.
                while self.opened and (len(self.opened) >= self.limit
                        or self.resident_bytes()+reservation > self.byte_limit):
                    self.opened.popitem(last=False);self.statistics['evictions'] += 1
                index = ObservedSceneIndex(**self._arrays(reference))
                index.execution_reference = reference
                folder, metadata = self._support_metadata(reference)
                atlas = BoundedSupportAtlas(self._arrays_in(folder, metadata), index.points,
                    self.support.policy,statistics=self.statistics)
                # Own tree coordinates explicitly, rather than retaining any
                # borrowed mapping through SciPy's float64 zero-copy path.
                atlas.centres = type(atlas.centres)(atlas.arrays['centres'],copy_data=True)
                pair = PreparedSupportIndex(index, atlas)
                size = sum(v.nbytes for v in (index.points,index.normals,index.normal_valid,index.observations))
                size += len(index.points)*96 + sum(v.nbytes for v in atlas.arrays.values())
                size += sum(tree.data.nbytes+tree.n*64 for tree in atlas.samples)
                if size > reservation:
                    raise MemoryError(f'Loaded map exceeds its preallocation estimate: {reference[0]}')
                self.opened[reference] = pair, reservation
            self.statistics['map_load_seconds'] += time.monotonic()-start
        else:
            self.statistics['cache_hits'] += 1
        self.opened.move_to_end(reference)
        return SceneView(self.opened[reference][0], origin, rotation)

    def resident_bytes(self):
        return sum(size for _,size in self.opened.values())

    def stats(self):
        return dict(self.statistics, resident_maps=len(self.opened),
            estimated_resident_bytes=self.resident_bytes(), cache_limit_bytes=self.byte_limit,
            query_workspace_limit_bytes=QUERY_WORKSPACE_BYTES)


def compare_neighbourhoods(view, anchors, final_anchors, policy, *, dynamic=None):
    """Stable point/owner set identities, independent of nearest-K ordering."""
    from scenemi.geometry.spatial_scene import query_view
    from scenemi.geometry.spatial_execution import nearest_patches
    if np.asarray(anchors).shape != np.asarray(final_anchors).shape:
        raise ValueError('Both B queries must use the same frame and anchor axes')
    none = np.zeros(np.asarray(anchors).shape[1], bool)
    first = query_view(view, anchors, policy, ground_mask=none, dynamic=dynamic)
    last = query_view(view, final_anchors, policy, ground_mask=none, dynamic=dynamic)

    def sets(patch):
        valid = patch['valid'].reshape(-1, patch['valid'].shape[-1])
        points = patch['points'].reshape(*valid.shape, 3)
        owners = patch['owner'].reshape(valid.shape)
        return [{(*map(float,p),int(o)) for p,o in zip(ps[mask],os[mask])}
                for ps,os,mask in zip(points,owners,valid)]
    def changes(a,b):
        unions = [x|y for x,y in zip(a,b)]
        return (np.array([1-len(x&y)/len(u) if u else 0. for x,y,u in zip(a,b,unions)]),
                np.array([bool(u) for u in unions]))
    shape = np.asarray(anchors).shape[:-1];result = {}; groups=[]
    for name in ('near','context'):
        a,b=sets(first[name]),sets(last[name]);groups.append((a,b))
        change,valid=changes(a,b)
        result[name+'_change']=change.reshape(shape);result[name+'_union']=valid.reshape(shape)
    for label,which in (('before',0),('after',1)):
        a,b=groups[0][which],groups[1][which]
        result['scale_overlap_'+label]=np.array([len(x&y)/min(len(x),len(y)) if x and y else 0.
            for x,y in zip(a,b)]).reshape(shape)
        result['scale_overlap_valid_'+label]=np.array([bool(x and y) for x,y in zip(a,b)]).reshape(shape)
        result['scale_jaccard_'+label]=np.array([len(x&y)/len(x|y) if x or y else 0.
            for x,y in zip(a,b)]).reshape(shape)
        result['context_novel_fraction_'+label]=np.array([len(y-x)/len(y) if y else 0.
            for x,y in zip(a,b)]).reshape(shape)
        result['context_present_'+label]=np.array([bool(y) for y in b]).reshape(shape)
        patch=nearest_patches(view, anchors if which==0 else final_anchors, policy,dynamic=dynamic)
        distances=first['near']['nearest_distance'] if which==0 else last['near']['nearest_distance']
        result['nearest_'+label]=distances
        result['_centre_'+label]=patch['centre']
    finite_before=np.isfinite(result['nearest_before']);finite_after=np.isfinite(result['nearest_after'])
    result['nearest_surface_union']=finite_before|finite_after
    # Identity of the nearest measured point centre, not a mesh-face ID.
    result['nearest_surface_changed']=(finite_before!=finite_after)|(
        (result.pop('_centre_before')!=result.pop('_centre_after')).any(-1)&finite_before&finite_after)
    return result
