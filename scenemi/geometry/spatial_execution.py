"""Ordered CPU processes for exact scene queries; CUDA stays in the parent.

Maps are published once as verified, read-only arrays. Workers keep bounded
indices, receive only detached coordinates/current-frame observations and
return the same queries as the serial implementation. No random state is used.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import hashlib
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from scenemi.geometry.spatial_scene import ObservedSceneIndex, SceneView, query_view

FIELDS = ('points', 'normals', 'normal_valid', 'observations')
_INDICES = OrderedDict()
_BLAS_LIMIT = None


def _sha256(file):
    # Workers must not import the training/data stack just to hash a file.
    digest = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class SceneIndexCache:
    """Exact map/normal cache; never resamples or caps the complete map."""
    def __init__(self, root):
        # Reuse the established atomic array writer and verified mmap reader.
        from scenemi.data.scene_geometry_cache import ArrayStore
        self.root = Path(root).resolve()
        self.store = ArrayStore(self.root, limit=4)

    def get(self, identity, build):
        from scenemi.data.scene_geometry_cache import _publish
        from scenemi.data.amass_prepare import sha256
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = self.root/key;self.root.mkdir(parents=True, exist_ok=True)
        with (self.root/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not folder.exists():
                index = build()
                _publish(folder, identity, {k: getattr(index, k) for k in FIELDS})
        arrays = self.store.read(key, identity)
        index = ObservedSceneIndex(**arrays)
        index.execution_reference = (str(folder), sha256(folder/'CACHE.json'))
        return index


def _initialize():
    global _BLAS_LIMIT
    # Independent processes remove Python-loop/GIL contention and do not share
    # Qhull/LAPACK state. Each process uses one BLAS thread, not four nested ones.
    _BLAS_LIMIT = threadpool_limits(limits=1, user_api='blas')


def _view(reference, origin, rotation):
    folder, digest = reference
    if reference not in _INDICES:
        path = Path(folder)
        if _sha256(path/'CACHE.json') != digest:
            raise ValueError('CPU worker map manifest changed')
        meta = json.loads((path/'CACHE.json').read_text())
        if meta.get('complete') is not True:
            raise ValueError('Incomplete CPU worker map')
        arrays = {}
        for name, expected in meta['arrays'].items():
            if _sha256(path/name) != expected:
                raise ValueError(f'CPU worker map array changed: {name}')
            arrays[Path(name).stem] = np.load(path/name, mmap_mode='r', allow_pickle=False)
        _INDICES[reference] = ObservedSceneIndex(**arrays)
        while len(_INDICES) > 8:
            _INDICES.popitem(last=False)
    _INDICES.move_to_end(reference)
    return SceneView(_INDICES[reference], origin, rotation)


def nearest_patches(view, anchors, policy, *, dynamic=None, ground_mask=None):
    """Serial reference for detached physical-patch selection (not a loss)."""
    radius = policy['query']['near_radius_m'];anchors = np.asarray(anchors)
    shape = anchors.shape[:-1]
    centre, normal = np.zeros_like(anchors), np.zeros_like(anchors)
    valid = np.zeros(shape, bool);confidence = np.zeros(shape, np.float32)
    distance = np.full(shape, np.inf, np.float32)
    if view is not None:
        world = view.to_world(anchors)
        if len(view.index.points):
            dist, ids = view.index.tree.query(world.reshape(-1, 3), workers=1)
            ids = ids.reshape(shape);distance = dist.reshape(shape)
            centre = view.to_local(view.index.points[ids]).astype(np.float32)
            normal = (view.index.normals[ids] @ view.rotation.T).astype(np.float32)
            valid = (distance <= radius) & view.index.normal_valid[ids]
            confidence = valid.astype(np.float32)
        if dynamic is not None:
            ps, os, ns, vs = dynamic
            for t in range(len(anchors)):
                keep = (os[t] > 0) & (os[t] < 100)
                if not keep.any():
                    continue
                d, at = cKDTree(ps[t, keep]).query(anchors[t], workers=1)
                take = d < distance[t]
                centre[t, take] = ps[t, keep][at[take]]
                normal[t, take] = ns[t, keep][at[take]]
                valid[t, take] = (d[take] <= radius) & vs[t, keep][at[take]]
                confidence[t, take] = valid[t, take]
                distance[t, take] = d[take]
        if ground_mask is not None:
            feet = np.asarray(ground_mask, bool)
            patch = view.index.support_planes(world[:, feet], policy['ground'])
            use = patch['valid'] & ~valid[:, feet] & (distance[:, feet] > radius)
            candidate = world[:, feet].copy();candidate[..., 1] = patch['height']
            local = view.to_local(candidate).astype(np.float32)
            normals = (patch['normal'] @ view.rotation.T).astype(np.float32)
            centre[:, feet] = np.where(use[..., None], local, centre[:, feet])
            normal[:, feet] = np.where(use[..., None], normals, normal[:, feet])
            confidence[:, feet] = np.where(use, patch['confidence'], confidence[:, feet])
            valid[:, feet] |= use
    return dict(centre=centre, normal=normal, valid=valid, confidence=confidence)


def _execute(task):
    action, reference, origin, rotation, anchors, policy, options = task
    view = _view(reference, origin, rotation)
    if action == 'features':
        return query_view(view, anchors, policy, **options)
    if action == 'patches':
        return nearest_patches(view, anchors, policy, **options)
    raise ValueError(f'Unknown spatial action: {action}')


class SpatialQueryExecutor:
    """Independent CPU-only executables; parent threads do pipe I/O only.

    A multiprocessing spawn pool reimports the training __main__ (including
    PyTorch) in every worker. Fresh module executables avoid that memory cost
    and start without inheriting CUDA state. All helpers remain in the job's
    CPU/RAM cgroup. Geometry itself never runs in the parent's I/O threads.
    """
    def __init__(self, workers, cache_root, *, array_backend='numpy'):
        if type(workers) is not int or workers < 1 or workers > 4:
            raise ValueError('Use one to four CPU processes')
        if array_backend not in ('numpy', 'numeric_header_v1'):
            raise ValueError('Unknown numeric array backend')
        self.cache = SceneIndexCache(cache_root)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='scene-query-io')
        self.workers = [];self.available = queue.Queue()
        self.closed = False
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
        try:
            for _ in range(workers):
                parent, child = socket.socketpair()
                try:
                    process = subprocess.Popen([sys.executable, '-m', 'scenemi.geometry.spatial_execution',
                        '--worker-fd', str(child.fileno()), '--array-backend', array_backend],
                        pass_fds=(child.fileno(),), env=env)
                    connection = Connection(parent.detach())
                finally:
                    parent.close();child.close()
                self.workers.append((connection, process));self.available.put((connection, process))
        except BaseException:
            self.close()
            raise

    def _dispatch(self, task):
        connection, process = self.available.get()
        try:
            connection.send(task)
            success, value = connection.recv()
            if not success:
                raise value
            return value
        except (EOFError, BrokenPipeError, ConnectionError) as error:
            raise RuntimeError(f'CPU scene worker exited unexpectedly (pid={process.pid}, status={process.poll()})') from error
        finally:
            self.available.put((connection, process))

    def submit(self, action, view, anchors, policy, **options):
        if self.closed:
            raise RuntimeError('Spatial executor has been closed')
        if not hasattr(view.index, 'execution_reference'):
            # Compatibility for ad-hoc/public views. Training data normally
            # already carries the source-addressed persistent cache reference.
            digest = hashlib.sha256()
            for name in FIELDS:
                array = np.ascontiguousarray(getattr(view.index, name))
                digest.update(name.encode());digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode());digest.update(array.tobytes())
            index = self.cache.get(dict(version='exact-map-arrays-v1', sha256=digest.hexdigest()), lambda: view.index)
            view.index.execution_reference = index.execution_reference
        return self.pool.submit(self._dispatch, (action, view.index.execution_reference,
            view.origin, view.rotation, anchors, policy, options))

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            for connection, process in self.workers:
                try:
                    if process.poll() is None: connection.send(None)
                except (OSError, EOFError):
                    pass
                finally:
                    connection.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _worker(fd):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _initialize()
    if 'torch' in sys.modules:
        raise RuntimeError('CPU geometry workers must not import PyTorch')
    with Connection(fd) as connection:
        while True:
            try:
                task = connection.recv()
            except EOFError:
                break
            if task is None: break
            try:
                result = _execute(task)
            except Exception as error:
                connection.send((False, error))
            else:
                connection.send((True, result))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Internal CPU geometry worker')
    parser.add_argument('--worker-fd', type=int, required=True)
    parser.add_argument('--array-backend', choices=('numpy', 'numeric_header_v1'), default='numpy')
    args = parser.parse_args()
    if args.array_backend == 'numeric_header_v1':
        from scenemi.data.numeric_mmap import numeric_mmap_backend
        backend = numeric_mmap_backend()
    else:
        backend = nullcontext()
    with backend:
        _worker(args.worker_fd)
