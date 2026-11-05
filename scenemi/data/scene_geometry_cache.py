"""Versioned Linux read caches: observed maps, oriented patches and motion arrays.

No full scene mesh enters a cache. Maps retain the existing offline recording
union exactly. Normals describe small observed patches, never a complete SDF.
"""
from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from scenemi.config import ROOT
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import sha256
from scenemi.data.recording_scene import observed_recording, bounded_scene_points
from scenemi.data.scene_dataset import bounded_frame_points
from scenemi.native_joint_data import SceneFreeEgoDataset

VERSION = 'observed-geometry-cache-v1'
CAMERA_ARRAYS = ('rest', 'motion_world', 'camera_position_scenemi_yup',
                 'camera_rotation_scenemi_yup', 'camera_intrinsics_128x96')
CACHE_POLICY_KEYS = ('voxel_m','static_points','frame_points','normal_neighbours',
    'normal_radius_m','minimum_neighbours','minimum_spread_m','maximum_planarity_ratio')


def fit_normals(surface, queries, origins, policy):
    """Compact PCA patches, oriented towards their actual first observer."""
    normals = np.zeros_like(queries, dtype=np.float32)
    good = np.zeros(len(queries), bool)
    if len(surface) < policy['minimum_neighbours'] or not len(queries):
        return normals, good
    k = min(policy['normal_neighbours'], len(surface))
    distance, ids = cKDTree(surface).query(queries, k=k,
        distance_upper_bound=policy['normal_radius_m'], workers=1)
    valid = np.isfinite(distance)
    neighbours = np.asarray(surface)[np.minimum(ids, len(surface)-1)]
    count = valid.sum(-1)
    centre = np.where(valid[..., None], neighbours, 0.).sum(-2)/count.clip(1)[:, None]
    delta = np.where(valid[..., None], neighbours-centre[:, None], 0.)
    covariance = np.einsum('nki,nkj->nij', delta, delta)/count.clip(1)[:, None, None]
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[..., 0]
    normal *= np.where(((origins-queries)*normal).sum(-1) < 0, -1., 1.)[:, None]
    good = ((count >= policy['minimum_neighbours']) &
            (values[:, 1] >= policy['minimum_spread_m']**2) &
            (values[:, 0]/values.sum(-1).clip(1e-12) <= policy['maximum_planarity_ratio']))
    normals[good] = normal[good]
    return normals, good


def static_geometry(points, owners, valid_frames, cameras, policy):
    valid = (owners == 0) & np.isfinite(points).all(-1) & valid_frames[:, None]
    selected = np.asarray(points[valid], np.float32)
    if not len(selected):
        return selected, selected, selected, np.empty(0, bool)
    _, ids = np.unique(np.floor(selected/policy['voxel_m']).astype(np.int64),
                       axis=0, return_index=True)
    ids = np.sort(ids)
    full = selected[ids]
    frame_ids = np.nonzero(valid)[0][ids]
    keep = np.arange(len(full))
    if len(keep) > policy['static_points']:
        keep = np.linspace(0, len(full)-1, policy['static_points']).astype(int)
    queries = full[keep]
    normal, good = fit_normals(full, queries, cameras[frame_ids[keep]], policy)
    return full, queries, normal, good


def _publish(folder, identity, arrays):
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    required = sum(np.asarray(v).nbytes for v in arrays.values())
    if shutil.disk_usage(folder.parent).free-required < 10*2**30:
        raise OSError('Geometry cache would leave less than 10GiB free')
    temporary = Path(tempfile.mkdtemp(prefix=folder.name+'.', dir=folder.parent))
    try:
        hashes = {}
        for name, value in arrays.items():
            file = temporary/(name+'.npy')
            with file.open('wb') as stream:
                np.save(stream, np.asarray(value), allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            hashes[file.name] = sha256(file)
        meta = dict(identity=identity, arrays=hashes, bytes=required, complete=True)
        with (temporary/'CACHE.json').open('w') as stream:
            json.dump(meta, stream, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class ArrayStore:
    """Bounded mappings, verified once per opened content-addressed bundle."""
    def __init__(self, root, limit=32):
        self.root = Path(root).resolve()
        if str(self.root).startswith('/mnt/'):
            raise ValueError('Prepared geometry cache must be on the Linux filesystem')
        self.limit, self.opened = limit, OrderedDict()
        self.verified = {}

    def read(self, key, identity):
        if key not in self.opened:
            folder = self.root/key
            try:
                meta = json.loads((folder/'CACHE.json').read_text())
            except FileNotFoundError as error:
                raise FileNotFoundError(f'Unprepared cache {key}; run preflight --prepare-data') from error
            if meta['identity'] != identity or meta.get('complete') is not True:
                raise ValueError('Prepared data identity differs')
            arrays = {}
            stamps = {name:( (folder/name).stat().st_size,(folder/name).stat().st_mtime_ns)
                      for name in meta['arrays']}
            verify = self.verified.get(key)!=(identity,stamps)
            for name, digest in meta['arrays'].items():
                file = folder/name
                if verify and sha256(file) != digest:
                    raise ValueError(f'Prepared array checksum differs: {file}')
                arrays[Path(name).stem] = np.load(file, mmap_mode='r', allow_pickle=False)
            self.verified[key] = identity,stamps
            self.opened[key] = arrays
            while len(self.opened) > self.limit:
                self.opened.popitem(last=False)
        self.opened.move_to_end(key)
        return self.opened[key]


class PreparedSceneDataset(SceneFreeEgoDataset):
    """Original camera/GT sampler plus cached observed geometry; no prefix work."""
    def __init__(self, *args, cache_root, geometry_policy, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_policy = {k:geometry_policy[k] for k in CACHE_POLICY_KEYS}
        self.store = ArrayStore(Path(cache_root)/self.split, limit=32)
        self.identities, self._preparing = {}, False

    def cache_identity(self, row):
        path = row['path']
        if path not in self.identities:
            meta = self.selection_bundle.metadata(row, self.split)
            arrays = {name:meta['output_sha256'][name+'.npy'] for name in CAMERA_ARRAYS+
                ('visible_frame_points_scenemi_yup','visible_frame_owner','observation_valid')}
            identity = dict(version=VERSION, source=path, split=self.split, arrays=arrays,
                input_sha256=meta['input_sha256'], policy=self.geometry_policy,
                implementation_sha256=sha256(__file__),
                synthetic=row.get('synthetic_chair'),
                contact_policy_sha256=sha256(ROOT/self.config['scene_stage']['contact_policy']),
                review_sha256=self.config['scene_stage']['quality_review']['sha256'])
            key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            self.identities[path] = key, identity
        return self.identities[path]

    def cached(self, row):
        key, identity = self.cache_identity(row)
        return self.store.read(key, identity)

    def array(self, meta, name):
        if getattr(self, '_preparing', False) or not hasattr(self, 'store'):
            return super().array(meta, name)
        row = self._by_path[meta['path']]
        arrays = self.cached(row)
        if name not in arrays:
            raise KeyError(f'{name} was not prepared for camera/geometry sampling')
        return arrays[name]

    def index_paths(self):
        self._by_path = {r['path']:r for r in self.records}

    def prepare_record(self, row, labels=None):
        key, identity = self.cache_identity(row)
        folder = self.store.root/key
        folder.parent.mkdir(parents=True, exist_ok=True)
        with (folder.parent/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if folder.exists():
                self.store.read(key, identity)
                return False
            self._preparing = True
            try:
                meta = dict(self.selection_bundle.metadata(row,self.split), path=row['path'])
                source_array = super().array
                arrays = {name:np.asarray(source_array(meta,name)) for name in CAMERA_ARRAYS}
                points = np.asarray(source_array(meta,'visible_frame_points_scenemi_yup'))
                owners = np.asarray(source_array(meta,'visible_frame_owner'))
                valid = np.asarray(source_array(meta,'observation_valid'),bool)
                full, queries, normals, normal_valid = static_geometry(points,owners,valid,
                    arrays['camera_position_scenemi_yup'],self.geometry_policy)
                if not np.array_equal(full,observed_recording(points,owners,valid,self.geometry_policy['voxel_m'])):
                    raise AssertionError('Cached static union differs from the original map')
                synthetic = row.get('synthetic_chair')
                if synthetic:
                    if self.split != 'train' or sha256(synthetic['condition_file']) != synthetic['condition_sha256']:
                        raise ValueError('Changed synthetic training condition')
                    with np.load(synthetic['condition_file'],allow_pickle=False) as saved:
                        extra = saved['synthetic_chair_points_m']
                        full = np.concatenate((full,extra))
                    # Synthetic points are context, without claimed observed normals.
                    all_queries = bounded_scene_points(full,self.geometry_policy['static_points'])
                    take = cKDTree(queries).query(all_queries)[1] if len(queries) else np.zeros(len(all_queries),int)
                    hit = (np.linalg.norm(queries[take]-all_queries,axis=-1)<1e-7) if len(queries) else np.zeros(len(all_queries),bool)
                    normals = np.where(hit[:,None],normals[take] if len(normals) else 0.,0.).astype(np.float32)
                    normal_valid = hit & (normal_valid[take] if len(normal_valid) else False)
                    queries = all_queries
                frame_points, frame_owner = bounded_frame_points(points,owners,self.geometry_policy['frame_points'])
                frame_normals = np.zeros_like(frame_points)
                frame_valid = np.zeros_like(frame_owner,dtype=bool)
                cameras = arrays['camera_position_scenemi_yup']
                for t in range(len(points)):
                    for owner in np.unique(frame_owner[t]):
                        if not 0 < owner < 100:
                            continue
                        selected = frame_owner[t] == owner
                        pool = points[t,(owners[t]==owner)&np.isfinite(points[t]).all(-1)]
                        n,v = fit_normals(pool,frame_points[t,selected],
                            np.broadcast_to(cameras[t],(int(selected.sum()),3)),self.geometry_policy)
                        frame_normals[t,selected],frame_valid[t,selected] = n,v
                arrays.update(static_full=full,static_points=queries,static_normals=normals,
                    static_normal_valid=normal_valid,frame_points=frame_points,frame_owner=frame_owner,
                    frame_normals=frame_normals,frame_normal_valid=frame_valid)
                weight = np.ones((meta['frames'],2),np.float32)
                if self.split=='train' and row['dataset']=='grail':
                    if labels is None:
                        raise ValueError('Training cache needs the private source-contact labels')
                    _, evidence = labels.window(row['path'],0)
                    with np.load(evidence['contact_label_cache'],allow_pickle=False) as saved:
                        weight = saved['weight'].copy()
                arrays['contact_weight'] = weight
                _publish(folder,identity,arrays)
            finally:
                self._preparing = False
        return True


class PreparedAmass(AmassDataset):
    """Checksum once, decode once, then read Linux NPY arrays by source digest."""
    def __init__(self, original, root):
        self.__dict__.update(original.__dict__)
        self.store = ArrayStore(root,limit=16)
        self.identities = {}

    def prepare_variant(self, relative, digest):
        if digest in self.identities:
            return self.identities[digest]
        identity = dict(version='amass-array-cache-v1',variant_sha256=digest,
                        preprocessing_sha256=sha256(ROOT/'scenemi/data/amass_dataset.py'))
        folder = self.store.root/digest
        folder.parent.mkdir(parents=True,exist_ok=True)
        with (folder.parent/(digest+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not folder.exists():
                arrays = AmassDataset.load.__wrapped__(self,relative,digest)
                _publish(folder,identity,arrays)
        self.identities[digest] = identity
        return identity

    def load(self, relative, digest):
        identity = self.prepare_variant(relative,digest)
        return self.store.read(digest,identity)
