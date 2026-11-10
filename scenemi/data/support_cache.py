"""Content-addressed sequence support caches and isolated CPU query workers.

This module is deliberately usable without PyTorch. Preparation is explicit;
training workers only open verified read-only arrays and cannot fit a plane.
"""
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from scenemi.data.numeric_mmap import map_numeric_npy
from scenemi.geometry import spatial_execution as execution
from scenemi.geometry.spatial_scene import SceneView, query_view
from scenemi.geometry.support_surfaces import (VERSION, SupportSurfaceAtlas,
    extract_support_surfaces, validate_policy)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class SupportCache:
    """Cache identity binds observed input, extraction, gravity and software."""
    def __init__(self, root, policy, manifests=()):
        self.root = Path(root).resolve()
        self.policy = validate_policy(policy)
        self.opened = OrderedDict()
        self.software = dict(implementation_sha256=digest(
            Path(__file__).parents[1]/'geometry/support_surfaces.py'),
            open3d_version=importlib.metadata.version('open3d'))
        self.allowed = None
        if manifests:
            self.allowed = {}
            for path in manifests:
                manifest = json.loads(Path(path).read_text())
                if (manifest['version'] != VERSION or manifest['state'] != 'complete'
                        or manifest['test_used'] or manifest['extraction_identity'] != self.extraction_identity):
                    raise ValueError('A completed matching support-cache manifest is required')
                for row in manifest['maps']:
                    reference = tuple(row['map_reference'])
                    if reference in self.allowed and self.allowed[reference] != row:
                        raise ValueError('Support manifests disagree on an observed map')
                    self.allowed[reference] = row

    @property
    def extraction_identity(self):
        return dict(version=VERSION, gravity=self.policy['gravity'],
                    extraction=self.policy['extraction'], **self.software)

    def identity(self, reference):
        return dict(self.extraction_identity, map_manifest_sha256=reference[1])

    def location(self, reference):
        identity = self.identity(reference)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.root/'maps'/key, identity

    def prepare(self, reference, index):
        """Atomically publish a sequence once; resumed preparation reuses it."""
        folder, identity = self.location(reference)
        folder.parent.mkdir(parents=True, exist_ok=True)
        with folder.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not folder.exists():
                arrays = extract_support_surfaces(index, self.policy,
                    seed=int(reference[1][:8], 16))
                required = sum(a.nbytes for a in arrays.values())
                if shutil.disk_usage(folder.parent).free-required < 10*2**30:
                    raise OSError('Support cache would leave less than 10GiB free')
                temporary = Path(tempfile.mkdtemp(prefix=folder.name+'.', dir=folder.parent))
                try:
                    hashes = {}
                    for name, value in arrays.items():
                        path = temporary/(name+'.npy')
                        with path.open('wb') as stream:
                            np.save(stream, value, allow_pickle=False)
                            stream.flush()
                            os.fsync(stream.fileno())
                        hashes[path.name] = digest(path)
                    meta = dict(identity=identity, arrays=hashes, complete=True,
                                patches=len(arrays['coefficients']), bytes=required)
                    with (temporary/'CACHE.json').open('w') as stream:
                        json.dump(meta, stream, sort_keys=True)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(folder)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
        # Verify a reused publication as strictly as a newly written one.
        atlas = self.load(reference, index)
        return dict(map_reference=list(reference), path=str(folder),
                    sha256=digest(folder/'CACHE.json'), patches=len(atlas.arrays['coefficients']))

    def load(self, reference, index):
        reference = tuple(reference)
        if reference not in self.opened:
            folder, identity = self.location(reference)
            if self.allowed is not None:
                row = self.allowed.get(reference)
                if row is None or Path(row['path']) != folder or digest(folder/'CACHE.json') != row['sha256']:
                    raise ValueError('Support map is outside the prepared input contract')
            try:
                meta = json.loads((folder/'CACHE.json').read_text())
            except FileNotFoundError as error:
                raise FileNotFoundError('Support map is unprepared; run the prepare action first') from error
            if meta['identity'] != identity or meta.get('complete') is not True:
                raise ValueError('Support cache identity differs')
            arrays = {}
            for name, expected in meta['arrays'].items():
                if Path(name).name != name or not name.endswith('.npy'):
                    raise ValueError('Invalid support array path')
                path = folder/name
                if digest(path) != expected:
                    raise ValueError('Support array checksum differs: '+str(path))
                arrays[Path(name).stem] = map_numeric_npy(path)
            self.opened[reference] = SupportSurfaceAtlas(arrays, index.points, self.policy)
            while len(self.opened) > 8:
                self.opened.popitem(last=False)
        self.opened.move_to_end(reference)
        return self.opened[reference]


class PreparedSupportIndex:
    """Observed-map adapter retaining the established near/context queries."""
    def __init__(self, index, atlas):
        self.index, self.atlas = index, atlas

    def __getattr__(self, name):
        return getattr(self.index, name)

    def support_planes(self, anchors, policy):
        return self.atlas.query(anchors)


def support_patches(view, anchors, policy, *, dynamic=None, ground_mask=None):
    """Measured collision patches; only reliable cached planes support feet.

    Surface confidence describes geometry, not whether a foot is in stance.
    Stance/reliability labels remain private to the established training loss.
    """
    result = execution.nearest_patches(view, anchors, policy, dynamic=dynamic, ground_mask=None)
    if view is None or ground_mask is None:
        return result
    feet = np.asarray(ground_mask, bool)
    world = view.to_world(np.asarray(anchors))
    support = view.index.support_planes(world[:, feet], policy['ground'])
    point = world[:, feet].copy()
    point[..., 1] = support['height']
    result['centre'][:, feet] = view.to_local(point).astype(np.float32)
    result['normal'][:, feet] = (support['normal'] @ view.rotation.T).astype(np.float32)
    result['valid'][:, feet] = support['reliable']
    result['confidence'][:, feet] = support['confidence']*support['reliable']
    return result


class SupportQueryExecutor(execution.SpatialQueryExecutor):
    """Reuse ordered IPC and error handling with a versioned support worker."""
    def __init__(self, workers, index_cache, support_root, support_policy, *, manifests=(), mode='query',
                 log_root=None):
        if type(workers) is not int or not 1 <= workers <= 4 or mode not in ('query', 'prepare'):
            raise ValueError('Use one to four bounded support workers')
        validate_policy(support_policy)
        self.cache = execution.SceneIndexCache(index_cache)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='support-query-io')
        self.workers, self.available = [], queue.Queue()
        self.closed = False
        self._mode = mode
        self._fatal = None
        self._worker_lock = threading.Lock()
        self.log_root = Path(log_root) if log_root else Path(support_root)/'worker_logs'
        self.log_root.mkdir(parents=True, exist_ok=True)
        self._environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   PYTHONFAULTHANDLER='1')
        self._command = [sys.executable, '-m', 'scenemi.data.support_cache',
            '--cache-root', str(support_root), '--policy-json', json.dumps(support_policy), '--mode', mode]
        for path in manifests:
            self._command += ['--manifest', str(path)]
        try:
            for _ in range(workers):
                self.available.put(self._spawn_worker())
        except BaseException:
            self.close()
            raise

    def _spawn_worker(self):
        parent, child = socket.socketpair()
        try:
            with self._worker_lock:
                log = self.log_root/f'worker_{os.getpid()}_{len(self.workers):04d}.log'
                with log.open('ab', buffering=0) as stream:
                    process = subprocess.Popen(self._command+['--worker-fd', str(child.fileno())],
                        pass_fds=(child.fileno(),), env=self._environment, stdout=stream, stderr=stream)
                process.scene_log = str(log)
                connection = Connection(parent.detach())
                worker = (connection, process)
                self.workers.append(worker)
            return worker
        finally:
            parent.close()
            child.close()

    def _record_failure(self, task, process, attempt):
        action, reference, origin, rotation, anchors, policy, options = task
        folder = self.log_root/f'failure_{process.pid}_{time.time_ns()}'
        folder.mkdir()
        arrays = dict(origin=origin, rotation=rotation)
        if anchors is not None:
            arrays['anchors'] = anchors
        moving = options.get('dynamic')
        if moving is not None:
            arrays.update({f'dynamic_{i}': value for i, value in enumerate(moving)})
        if options.get('ground_mask') is not None:
            arrays['ground_mask'] = options['ground_mask']
        np.savez_compressed(folder/'QUERY.npz', **arrays)
        info = dict(action=action, reference=list(reference), policy=policy, attempt=attempt,
                    worker_pid=process.pid, worker_log=process.scene_log,
                    exit_code=process.poll(), worker_command=self._command,
                    query_sha256=digest(folder/'QUERY.npz'))
        (folder/'FAILURE.json').write_text(json.dumps(info, indent=2)+'\n')
        return folder

    @staticmethod
    def _retire(worker):
        connection, process = worker
        connection.close()
        try:
            return process.wait(timeout=1), False
        except subprocess.TimeoutExpired:
            # Only this abandoned helper is killed; never an external job.
            process.kill()
            return process.wait(timeout=5), True

    def _dispatch(self, task):
        worker = None
        while worker is None:
            if self._fatal is not None:
                raise RuntimeError('Support worker could not be replaced') from self._fatal
            try:
                worker = self.available.get(timeout=.1)
            except queue.Empty:
                pass
        try:
            retries = 1 if self._mode == 'query' else 0
            for attempt in range(retries+1):
                connection, process = worker
                try:
                    connection.send(task)
                    success, value = connection.recv()
                except (EOFError, BrokenPipeError, ConnectionError) as error:
                    folder = self._record_failure(task, process, attempt)
                    code, forced = self._retire(worker)
                    info = json.loads((folder/'FAILURE.json').read_text())
                    info.update(observed_exit_code=code, forced_termination=forced,
                                replay_scheduled=attempt < retries)
                    (folder/'FAILURE.json').write_text(json.dumps(info, indent=2)+'\n')
                    try:
                        worker = self._spawn_worker()
                    except BaseException as spawn_error:
                        self._fatal = spawn_error
                        worker = None
                        raise
                    if attempt == retries:
                        raise RuntimeError(f'CPU support worker failed after bounded replay '
                            f'(pid={process.pid}, exit={code}, evidence={folder})') from error
                    print(f'CPU support worker {process.pid} exited ({code}); '
                          f'replaying the same query once. Evidence: {folder}', file=sys.stderr, flush=True)
                    continue
                if not success:
                    # Invalid data/checksums are ordinary errors: no retry.
                    raise value
                return value
        finally:
            if worker is not None:
                self.available.put(worker)


class SerialSupportQueries:
    """Reference execution with the same feature and physical-query semantics."""
    def submit(self, action, view, anchors, policy, **options):
        future = Future()
        try:
            if action == 'features':
                value = query_view(view, anchors, policy, **options)
            elif action == 'patches':
                value = support_patches(view, anchors, policy, **options)
            else:
                raise ValueError('Unknown support query action: '+action)
            future.set_result(value)
        except Exception as error:
            future.set_exception(error)
        return future

    def close(self):
        pass


def _worker(fd, cache, mode):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    execution._initialize()
    if mode == 'query' and 'torch' in sys.modules:
        raise RuntimeError('Support query workers must not import PyTorch')
    with Connection(fd) as connection:
        while True:
            try:
                task = connection.recv()
            except (EOFError, ConnectionError):
                break
            if task is None:
                break
            try:
                action, reference, origin, rotation, anchors, policy, options = task
                view = execution._view(reference, origin, rotation)
                if action == 'prepare' and mode == 'prepare':
                    result = cache.prepare(reference, view.index)
                else:
                    if mode != 'query':
                        raise ValueError('Preparation workers cannot run model queries')
                    index = PreparedSupportIndex(view.index, cache.load(reference, view.index))
                    view = SceneView(index, origin, rotation)
                    if action == 'features':
                        result = query_view(view, anchors, policy, **options)
                    elif action == 'patches':
                        result = support_patches(view, anchors, policy, **options)
                    else:
                        raise ValueError('Unknown support query action: '+action)
                response = (True, result)
            except Exception as error:
                response = (False, error)
            try:
                connection.send(response)
            except (EOFError, BrokenPipeError, ConnectionError):
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Internal sequence-support worker')
    parser.add_argument('--worker-fd', type=int, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--policy-json', required=True)
    parser.add_argument('--manifest', action='append', type=Path, default=[])
    parser.add_argument('--mode', choices=('prepare', 'query'), default='query')
    args = parser.parse_args()
    from scenemi.data.numeric_mmap import numeric_mmap_backend
    with numeric_mmap_backend():
        _worker(args.worker_fd, SupportCache(args.cache_root, json.loads(args.policy_json), args.manifest), args.mode)
