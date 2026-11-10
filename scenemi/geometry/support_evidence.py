"""Independent measured geometry and fitted support, with read-only workers.

The retained v1 worker and caches stay immutable. A fitted candidate never
overwrites a measured patch. Contact/stance eligibility belongs to the loss,
not to this public geometric query.
"""
import argparse
from concurrent.futures import Future
import hashlib
import json
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from scenemi.data.numeric_mmap import numeric_mmap_backend
from scenemi.data.support_cache import PreparedSupportIndex, SupportCache, SupportQueryExecutor
from scenemi.geometry import spatial_execution as execution
from scenemi.geometry.spatial_scene import SceneView, query_view
from scenemi.geometry.query_runtime import MapReference, MapRuntime, compare_neighbourhoods, validate_query


def query_evidence(view, anchors, policy, *, dynamic=None, ground_mask=None):
    """Return both sources on the same anchors; validity is source-specific."""
    anchors = np.asarray(anchors, np.float32)
    measured = execution.nearest_patches(view, anchors, policy, dynamic=dynamic)
    measured['distance'] = (np.linalg.norm(anchors-measured['centre'], axis=-1)
                            if view is not None else np.full(anchors.shape[:-1], np.inf))
    shape = anchors.shape[:-1]
    support = dict(centre=np.zeros_like(anchors), normal=np.zeros_like(anchors),
        confidence=np.zeros(shape, np.float32), valid=np.zeros(shape, bool),
        reliable=np.zeros(shape, bool), ambiguous=np.zeros(shape, bool),
        fit_error=np.zeros(shape, np.float32))
    if view is not None and ground_mask is not None:
        feet = np.asarray(ground_mask, bool)
        world = view.to_world(anchors)
        candidate = view.index.support_planes(world[:, feet], policy['ground'])
        point = world[:, feet].copy()
        point[..., 1] = candidate['height']
        support['centre'][:, feet] = view.to_local(point)
        support['normal'][:, feet] = candidate['normal'] @ view.rotation.T
        for name in ('confidence', 'valid', 'reliable', 'ambiguous', 'fit_error'):
            support[name][:, feet] = candidate[name]
    return dict(observed_geometry=measured, support_surface=support)


def execute(action, view, anchors, policy, *, _validated=False, **options):
    if not _validated: validate_query(action,anchors,policy,**options)
    if action == 'compare':
        return compare_neighbourhoods(view, anchors, options['final_anchors'], policy,
            dynamic=options.get('dynamic'))
    if action == 'evidence':
        return query_evidence(view, anchors, policy, **options)
    if action == 'features':
        result = query_view(view, anchors, policy, **options)
        # Preserve the trained encoder's tensor layout and parameter names.
        # These named channels expose their different physical semantics.
        result['observed_geometry'] = dict(near=result['near'], context=result['context'])
        result['support_surface'] = result['ground']
        return result
    if action == 'patches':
        return execution.nearest_patches(view, anchors, policy, dynamic=options.get('dynamic'))
    raise ValueError('Unknown evidence query action: '+action)


class EvidenceQueries(SupportQueryExecutor):
    """Bound submissions, complete IPC deadlines and verified worker ownership."""
    MAX_MESSAGE_BYTES = 64*2**20
    MAX_FAILURE_ARRAY_BYTES = 512*2**10

    def __init__(self, workers, index_cache, support_root, policy, *, cache_entries=8,
                 cache_mib=256, timeout_seconds=120, **kwargs):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError('Use one to four bounded support workers')
        if not 0 < timeout_seconds <= 600:
            raise ValueError('Query timeout must be in (0,600] seconds')
        self.runtime_options = dict(cache_entries=cache_entries,cache_mib=cache_mib)
        self.timeout_seconds = timeout_seconds
        if type(cache_entries) is not int or not 1 <= cache_entries <= 32 or type(cache_mib) is not int or not 16 <= cache_mib <= 1024:
            raise ValueError('Invalid bounded query cache')
        self.worker_locks = [threading.Lock() for _ in range(workers)]
        self._state_lock = threading.RLock()
        self.max_pending = workers*2
        self._permits = threading.BoundedSemaphore(self.max_pending)
        self._pending = set()
        self._counters = dict(submitted=0,completed=0,request_bytes=0,response_bytes=0,
                              peak_pending=0,submission_wait_seconds=0.)
        super().__init__(workers,index_cache,support_root,policy,**kwargs)

    def _spawn_worker(self):
        self._command[2] = 'scenemi.geometry.support_evidence'
        if '--cache-entries' not in self._command:
            self._command += ['--cache-entries',str(self.runtime_options['cache_entries']),
                '--cache-mib',str(self.runtime_options['cache_mib']), '--trace-root',str(self.log_root),
                '--timeout-seconds',str(self.timeout_seconds)]
        parent, child = socket.socketpair()
        process = None
        try:
            with self._worker_lock:
                log = self.log_root/f'worker_{os.getpid()}_{len(self.workers):04d}.log'
                with log.open('ab',buffering=0) as stream:
                    process = subprocess.Popen(self._command+['--worker-fd',str(child.fileno())],
                        pass_fds=(child.fileno(),),env=self._environment,stdout=stream,stderr=stream)
                process.scene_log = str(log)
                descriptor = parent.detach()
                try:
                    connection = Connection(descriptor)
                except BaseException:
                    os.close(descriptor)
                    raise
                worker = connection,process
                self.workers.append(worker)
                return worker
        except BaseException:
            if process is not None:
                if process.poll() is None: process.kill()
                process.wait(timeout=5)
            raise
        finally:
            parent.close();child.close()

    @staticmethod
    def _remaining(deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Scene query exceeded its complete IPC deadline')
        return remaining

    @staticmethod
    def _payload_size(value):
        if isinstance(value,np.ndarray): return value.nbytes
        if isinstance(value,(bytes,bytearray,memoryview,str)): return len(value)
        if isinstance(value,dict):
            return sum(EvidenceQueries._payload_size(k)+EvidenceQueries._payload_size(v)
                       for k,v in value.items())
        if isinstance(value,(tuple,list)):
            return sum(EvidenceQueries._payload_size(v) for v in value)
        return 16

    def _duplicate_channel(self, connection, *, allow_closed=False):
        # Closing an original descriptor and duplicating it must share this
        # lock: its integer could otherwise be reused by an unrelated file.
        with self._state_lock:
            if self.closed and not allow_closed:
                raise RuntimeError('Spatial executor has been closed')
            descriptor = os.dup(connection.fileno())
            try:
                return socket.socket(fileno=descriptor)
            except BaseException:
                try: os.close(descriptor)
                except OSError: pass
                raise

    def _exchange(self, connection, task, deadline):
        """Connection-compatible frames; the deadline covers sends and partial receives."""
        self._remaining(deadline)
        if self._payload_size(task) > self.MAX_MESSAGE_BYTES:
            raise MemoryError(f'Scene request exceeds {self.MAX_MESSAGE_BYTES} bytes before serialization')
        payload = ForkingPickler.dumps(task)
        if len(payload) > self.MAX_MESSAGE_BYTES:
            raise MemoryError(f'Scene request exceeds {self.MAX_MESSAGE_BYTES} bytes')
        # A private descriptor permits shutdown/close to interrupt this I/O
        # without a thread closing and later reusing the same descriptor.
        with self._duplicate_channel(connection) as channel:
            def send(value):
                channel.settimeout(self._remaining(deadline));channel.sendall(value)
            def receive(size):
                result = bytearray(size);offset = 0
                while offset < size:
                    channel.settimeout(self._remaining(deadline))
                    count = channel.recv_into(memoryview(result)[offset:])
                    if not count: raise EOFError('Scene worker closed a partial IPC frame')
                    offset += count
                return result
            send(struct.pack('!i',len(payload)));send(payload)
            size = struct.unpack('!i',receive(4))[0]
            if size == -1: size = struct.unpack('!Q',receive(8))[0]
            if not 0 <= size <= self.MAX_MESSAGE_BYTES:
                raise MemoryError(f'Scene response frame exceeds {self.MAX_MESSAGE_BYTES} bytes')
            result = ForkingPickler.loads(receive(size))
            self._remaining(deadline)
        with self._state_lock:
            self._counters['request_bytes'] += len(payload)
            self._counters['response_bytes'] += size
        if not isinstance(result,tuple) or len(result) != 2 or type(result[0]) is not bool:
            raise ValueError('Invalid scene worker response')
        return result

    def _stop_workers(self):
        """Only own helpers; socket shutdown wakes a blocked sender or receiver."""
        for connection, process in self.workers:
            if process.poll() is None:
                try: process.kill()
                except ProcessLookupError: pass
            try:
                with self._duplicate_channel(connection,allow_closed=True) as channel:
                    channel.shutdown(socket.SHUT_RDWR)
            except (OSError,ValueError):
                pass

    def _record_failure(self, task, process, attempt):
        """Small evidence only; never compress or re-read dynamic scene maps on failure."""
        action,reference,origin,rotation,anchors,policy,options = task
        folder = self.log_root/f'failure_{process.pid}_{time.time_ns()}'
        folder.mkdir()
        budget = self.MAX_FAILURE_ARRAY_BYTES
        arrays = {}
        for name,value in (('anchors',anchors),('final_anchors',options.get('final_anchors'))):
            if value is None: continue
            value = np.asarray(value)
            row = dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes)
            if value.nbytes <= budget and not value.dtype.hasobject:
                path = folder/(name+'.npy')
                np.save(path,value,allow_pickle=False)
                row.update(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                budget -= value.nbytes
            else:
                row['omitted'] = 'bounded failure evidence budget'
            arrays[name] = row
        dynamic = options.get('dynamic')
        info = dict(action=action,reference=list(reference),policy=policy,attempt=attempt,
            worker_pid=process.pid,worker_log=process.scene_log,exit_code=process.poll(),
            worker_command=self._command,origin=np.asarray(origin).tolist(),
            rotation=np.asarray(rotation).tolist(),arrays=arrays,replay_scheduled=False,
            dynamic_arrays=[] if dynamic is None else [dict(shape=list(value.shape),dtype=str(value.dtype),
                bytes=value.nbytes,omitted='no dynamic-scene copying on failure') for value in dynamic],
            failure_array_budget_bytes=self.MAX_FAILURE_ARRAY_BYTES)
        (folder/'FAILURE.json').write_text(json.dumps(info,indent=2)+'\n')
        return folder

    def _fail(self, task, process, error):
        with self._state_lock:
            if self.closed:
                # Shutdown-induced EOF is expected. It must not initiate a
                # second diagnostic pass or keep close waiting on scene I/O.
                return RuntimeError(f'Scene executor closed during query: {type(error).__name__}: {error}')
            if self._fatal is None:
                self._fatal = (f'Scene worker failed without retry: pid={process.pid}; '
                    f'log={process.scene_log}; {type(error).__name__}: {error}')
                first = True
            else:
                first = False
        if first:
            self._stop_workers()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            if task[1] is not None:
                try:
                    if self.closed: return RuntimeError(self._fatal)
                    folder = self._record_failure(task,process,0)
                    info=json.loads((folder/'FAILURE.json').read_text())
                    info.update(replay_scheduled=False,error=f'{type(error).__name__}: {error}')
                    (folder/'FAILURE.json').write_text(json.dumps(info,indent=2)+'\n')
                except Exception as evidence_error:
                    # Preserve the original failure even if its filesystem is
                    # unavailable; never retry a query to recreate evidence.
                    self._fatal += f'; evidence write failed: {evidence_error}'
        return RuntimeError(self._fatal)

    def _request(self, task, slot, *, deadline=None):
        deadline = time.monotonic()+self.timeout_seconds if deadline is None else deadline
        connection, process = self.workers[slot]
        acquired = False
        try:
            acquired = self.worker_locks[slot].acquire(timeout=self._remaining(deadline))
            if not acquired: raise TimeoutError('Scene query timed out waiting for its worker')
            if self.closed: raise RuntimeError('Spatial executor has been closed')
            if self._fatal is not None: raise RuntimeError(self._fatal)
            try:
                if process.poll() is not None:
                    raise EOFError('Scene worker is no longer alive')
                success,value = self._exchange(connection,task,deadline)
                if not success: raise value
                return value
            except Exception as error:
                raise self._fail(task,process,error) from error
        except TimeoutError as error:
            raise self._fail(task,process,error) from error
        finally:
            if acquired: self.worker_locks[slot].release()

    def _dispatch(self, task, deadline=None):
        # Stable map affinity keeps one video's complete index hot in one
        # worker instead of duplicating it randomly across all workers.
        slot = int.from_bytes(hashlib.sha256(str(task[1]).encode()).digest()[:4],'little') % len(self.workers)
        return self._request(task,slot,deadline=deadline)

    def submit(self, action, view, anchors, policy, **options):
        started = time.monotonic();deadline = started+self.timeout_seconds
        with self._state_lock:
            if self.closed: raise RuntimeError('Spatial executor has been closed')
            if self._fatal is not None: raise RuntimeError(self._fatal)
        # Runtime queries never create/rebuild a map in the parent process.
        if not hasattr(view.index,'execution_reference'):
            raise ValueError('Scene query requires a prepared persistent map reference')
        validate_query(action,anchors,policy,**options)
        task = (action,view.index.execution_reference,view.origin,view.rotation,anchors,policy,options)
        if self._payload_size(task) > self.MAX_MESSAGE_BYTES:
            raise MemoryError(f'Scene request exceeds {self.MAX_MESSAGE_BYTES} bytes before submission')
        if not self._permits.acquire(timeout=self._remaining(deadline)):
            raise TimeoutError('Bounded scene submission queue timed out')
        try:
            with self._state_lock:
                if self.closed: raise RuntimeError('Spatial executor has been closed')
                if self._fatal is not None: raise RuntimeError(self._fatal)
                future = self.pool.submit(self._dispatch,task,deadline)
                self._pending.add(future)
                self._counters['submitted'] += 1
                self._counters['peak_pending'] = max(self._counters['peak_pending'],len(self._pending))
                self._counters['submission_wait_seconds'] += time.monotonic()-started
        except BaseException:
            self._permits.release()
            raise
        def complete(done):
            with self._state_lock:
                self._pending.discard(done);self._counters['completed'] += 1
            self._permits.release()
        future.add_done_callback(complete)
        return future

    def stats(self):
        task = ('stats',None,None,None,None,None,{})
        rows = [self._request(task,slot) for slot in range(len(self.workers))]
        with self._state_lock:
            snapshot = dict(self._counters,pending=len(self._pending),pending_limit=self.max_pending,
                message_limit_bytes=self.MAX_MESSAGE_BYTES,cold_loads_serialized=True)
        return [dict(row,executor=snapshot) for row in rows]

    def close(self):
        with self._state_lock:
            if self.closed: return
            self.closed = True
        # Stop helpers before waiting for I/O threads, including failed starts.
        self.pool.shutdown(wait=False,cancel_futures=True)
        self._stop_workers()
        deadline = time.monotonic()+5
        unreaped = []
        for connection,process in self.workers:
            with self._state_lock:
                connection.close()
            try: process.wait(timeout=max(.001,deadline-time.monotonic()))
            except subprocess.TimeoutExpired: unreaped.append(process.pid)
        self.pool.shutdown(wait=True,cancel_futures=True)
        if unreaped:
            raise RuntimeError(f'Scene helpers did not exit after termination: {unreaped}')


class SerialEvidenceQueries:
    def __init__(self, support_root=None, policy=None, manifests=(), **runtime_options):
        self.runtime = MapRuntime(support_root,policy,manifests,**runtime_options) if support_root is not None else None

    def submit(self, action, view, anchors, policy, **options):
        future = Future()
        try:
            if isinstance(view.index,MapReference):
                if self.runtime is None: raise ValueError('A map-reference query needs a verified runtime')
                view = self.runtime.view(view.index.execution_reference,view.origin,view.rotation)
            future.set_result(execute(action, view, anchors, policy, **options))
        except Exception as error:
            future.set_exception(error)
        return future

    def close(self):
        if self.runtime is not None:
            self.runtime.opened.clear()

    def stats(self):
        return [] if self.runtime is None else [self.runtime.stats()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-fd', type=int, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--policy-json', required=True)
    parser.add_argument('--manifest', type=Path, action='append', default=[])
    parser.add_argument('--mode', choices=('query',), default='query')
    parser.add_argument('--cache-entries', type=int, default=8)
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--trace-root', type=Path)
    parser.add_argument('--timeout-seconds',type=float,default=120)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    execution._initialize()
    if 'torch' in sys.modules:
        raise RuntimeError('Geometry worker must remain Torch-free')
    runtime = MapRuntime(args.cache_root,json.loads(args.policy_json),args.manifest,
        cache_entries=args.cache_entries,cache_mib=args.cache_mib,
        load_lock_path=None if args.trace_root is None else args.trace_root/'map_load.lock',
        load_timeout_seconds=args.timeout_seconds)
    def trace(value):
        if args.trace_root is not None:
            path=args.trace_root/f'CURRENT_QUERY_{__import__("os").getpid()}.json'
            temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    with numeric_mmap_backend(), Connection(args.worker_fd) as connection:
        while True:
            try:
                task = connection.recv()
            except (EOFError, ConnectionError):
                break
            if task is None:
                break
            try:
                action, reference, origin, rotation, anchors, policy, options = task
                if action == 'stats':
                    connection.send((True,runtime.stats()));continue
                estimate = validate_query(action,anchors,policy,**options)
                runtime.statistics['query_workspace_peak_bytes'] = max(
                    runtime.statistics['query_workspace_peak_bytes'],estimate)
                trace(dict(state='running',action=action,map_reference=list(reference),anchor_shape=list(anchors.shape),at=time.time()))
                started=time.monotonic()
                view=runtime.view(reference,origin,rotation)
                mark=time.monotonic()
                result = execute(action, view, anchors, policy, _validated=True, **options)
                runtime.statistics['query_seconds'] += time.monotonic()-mark
                runtime.statistics['queries'] += 1
                trace(dict(state='complete',action=action,map_reference=list(reference),
                    seconds=time.monotonic()-started,statistics=runtime.stats(),at=time.time()))
                response = True, result
            except Exception as error:
                runtime.statistics['rejected_queries'] += 1
                trace(dict(state='failed',action=action,map_reference=list(reference),
                    error=f'{type(error).__name__}: {error}',statistics=runtime.stats(),at=time.time()))
                response = False, error
            try:
                connection.send(response)
            except (EOFError, ConnectionError):
                break
            if not response[0]:
                break


if __name__ == '__main__':
    main()
