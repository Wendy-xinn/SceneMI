"""Verified whole-map ownership, reuse, deterministic IPC and hard failures."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from multiprocessing.connection import Connection
import signal
import socket
import struct
import threading
import time

import numpy as np
import pytest

from scenemi.data.support_cache import PreparedSupportIndex
from scenemi.geometry.query_runtime import (BoundedSupportAtlas, MapReference, MapRuntime,
    QUERY_WORKSPACE_BYTES, compare_neighbourhoods, validate_query)
from scenemi.geometry.spatial_scene import SceneView
from scenemi.geometry.support_evidence import EvidenceQueries, SerialEvidenceQueries, execute
from tests.model.test_support_surfaces import plane, policy, index_for
from tests.model.test_spatial_scene import policy as query_policy
from tests.training.test_support_cache import prepared
from tests.training.test_spatial_execution import equal_tree


def test_hot_map_is_read_and_built_once_with_one_shared_owner(tmp_path, monkeypatch):
    index, _, _ = prepared(tmp_path, plane())
    runtime = MapRuntime(tmp_path/'support', policy(), cache_entries=1, cache_mib=16)
    reference = index.execution_reference
    first = runtime.view(reference, np.zeros(3), np.eye(3))
    hashed = runtime.stats()['hash_bytes']
    def fail(*args, **kwargs):
        raise AssertionError('Hot map was reread or rebuilt')
    monkeypatch.setattr(runtime, '_arrays', fail)
    again = runtime.view(reference, np.ones(3), np.eye(3))
    assert first.index is again.index
    assert runtime.stats()['hash_bytes'] == hashed > 0
    assert runtime.stats()['cache_hits'] == 1 and len(runtime.opened) == 1
    assert not runtime.support.opened
    assert not isinstance(first.index.points, np.memmap)
    assert not first.index.points.flags.writeable
    assert all(not isinstance(a,np.memmap) for a in first.index.atlas.arrays.values())
    assert first.index.atlas.centres.data.flags.owndata
    assert not np.shares_memory(first.index.atlas.centres.data,first.index.atlas.arrays['centres'])


def test_corrupted_array_and_manifest_fail_with_filename_without_repair(tmp_path):
    index, _, _ = prepared(tmp_path, plane())
    reference = index.execution_reference
    path = Path(reference[0])/'points.npy'
    with path.open('r+b') as stream:
        stream.seek(-1,2); value=stream.read(1);stream.seek(-1,2);stream.write(bytes([value[0]^1]))
    with pytest.raises(ValueError, match='points.npy'):
        MapRuntime(tmp_path/'support',policy()).view(reference,np.zeros(3),np.eye(3))
    manifest = Path(reference[0])/'CACHE.json'
    manifest.write_bytes(b'\xff')
    with pytest.raises(ValueError,match='CACHE.json'):
        MapReference(reference)


def test_reference_serial_and_affine_worker_match_all_query_actions(tmp_path):
    index, cache, _ = prepared(tmp_path, plane())
    reference = MapReference(index.execution_reference)
    assert not hasattr(reference,'tree')
    local = SceneView(reference,np.zeros(3),np.eye(3))
    original = SceneView(PreparedSupportIndex(index,cache.load(index.execution_reference,index)),np.zeros(3),np.eye(3))
    anchors=np.zeros((4,2,3),np.float32);anchors[...,1]=.03
    serial=SerialEvidenceQueries(tmp_path/'support',policy())
    with EvidenceQueries(2,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs') as workers:
        for action in ('features','evidence','patches','compare'):
            options = {'final_anchors':anchors+np.array([.1,0.,0.],np.float32)} if action=='compare' else {'ground_mask':np.ones(2,bool)}
            expected=execute(action,original,anchors,query_policy(),**options)
            equal_tree(expected,serial.submit(action,local,anchors,query_policy(),**options).result())
            equal_tree(expected,workers.submit(action,local,anchors,query_policy(),**options).result(timeout=60))
        stats=workers.stats()
        assert sum(s['cache_misses'] for s in stats)==1
        assert sum(s['cache_hits'] for s in stats)==3
        assert sum(s['queries'] for s in stats)==4
        assert len(list((tmp_path/'logs').glob('CURRENT_QUERY_*.json')))==1


def test_dead_worker_fails_without_replay_and_leaves_evidence(tmp_path):
    index, _, _ = prepared(tmp_path,plane())
    local=SceneView(MapReference(index.execution_reference),np.zeros(3),np.eye(3))
    with EvidenceQueries(1,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs') as workers:
        process=workers.workers[0][1];process.kill();process.wait()
        future=workers.submit('patches',local,np.zeros((2,1,3),np.float32),query_policy())
        with pytest.raises(RuntimeError,match='without retry'):future.result(timeout=10)
        assert workers.workers[0][1].pid==process.pid
        assert list((tmp_path/'logs').rglob('FAILURE.json'))


def test_neighbour_changes_and_identical_scales_are_explicit():
    view=SceneView(index_for(plane()),np.zeros(3),np.eye(3))
    before=np.zeros((2,1,3),np.float32);before[...,1]=.03
    same=compare_neighbourhoods(view,before,before,query_policy())
    assert not same['near_change'].any() and not same['nearest_surface_changed'].any()
    assert (same['scale_overlap_before']==1).all()
    moved=before+np.array([.25,.02,0.],np.float32)
    changed=compare_neighbourhoods(view,before,moved,query_policy())
    assert (changed['near_change']>0).all()
    assert changed['nearest_surface_changed'].all()
    assert (changed['nearest_after']>changed['nearest_before']).all()
    empty=SceneView(index_for(np.empty((0,3),np.float32)),np.zeros(3),np.eye(3))
    missing=compare_neighbourhoods(empty,before,before,query_policy())
    assert not missing['near_union'].any() and not missing['nearest_surface_union'].any()


def test_oversize_map_is_rejected_before_payload_copy_or_tree_construction(tmp_path, monkeypatch):
    index, _, _ = prepared(tmp_path, plane())
    runtime = MapRuntime(tmp_path/'support', policy(), cache_entries=1, cache_mib=16)
    reservation = runtime._layout(index.execution_reference)
    runtime.byte_limit = reservation-1
    def fail(*args, **kwargs):
        raise AssertionError('Oversize input reached allocation or tree construction')
    monkeypatch.setattr(runtime, '_arrays', fail)
    monkeypatch.setattr('scenemi.geometry.query_runtime.ObservedSceneIndex', fail)
    with pytest.raises(MemoryError, match='before loading'):
        runtime.view(index.execution_reference, np.zeros(3), np.eye(3))
    assert not runtime.opened and not runtime.support.opened
    assert runtime.stats()['rejected_oversize_maps'] == 1


def test_cache_evicts_old_map_before_allocating_incoming_map(tmp_path, monkeypatch):
    from scenemi.geometry.spatial_execution import SceneIndexCache
    index, cache, _ = prepared(tmp_path, plane())
    second = SceneIndexCache(tmp_path/'indices').get(
        {'source':'second_observed_map'}, lambda:index_for(plane(.35)))
    cache.prepare(second.execution_reference, second)
    runtime = MapRuntime(tmp_path/'support', policy(), cache_entries=1, cache_mib=16)
    runtime.view(index.execution_reference, np.zeros(3), np.eye(3))
    build = runtime._arrays
    checked = []
    def check(reference):
        assert not runtime.opened
        checked.append(reference)
        return build(reference)
    monkeypatch.setattr(runtime, '_arrays', check)
    runtime.view(second.execution_reference, np.zeros(3), np.eye(3))
    assert checked == [second.execution_reference]
    assert list(runtime.opened) == [second.execution_reference]
    assert runtime.stats()['evictions'] == 1
    assert runtime.resident_bytes() <= runtime.byte_limit


def test_preallocation_rejects_wrong_numeric_layout_before_copy(tmp_path, monkeypatch):
    import hashlib
    import json
    index, _, _ = prepared(tmp_path, plane())
    folder = Path(index.execution_reference[0])
    points = folder/'points.npy'
    np.save(points, np.asarray(index.points, np.float64), allow_pickle=False)
    metadata = json.loads((folder/'CACHE.json').read_text())
    metadata['arrays']['points.npy'] = hashlib.sha256(points.read_bytes()).hexdigest()
    raw = json.dumps(metadata, sort_keys=True).encode()
    (folder/'CACHE.json').write_bytes(raw)
    reference = (str(folder), hashlib.sha256(raw).hexdigest())
    runtime = MapRuntime(tmp_path/'support', policy(), cache_entries=1, cache_mib=16)
    def fail(*args, **kwargs):
        raise AssertionError('Invalid numeric layout reached payload copy')
    monkeypatch.setattr(runtime, '_arrays', fail)
    with pytest.raises(ValueError, match='dtype or dimensions.*points.npy'):
        runtime.view(reference, np.zeros(3), np.eye(3))
    assert not runtime.opened


def test_map_arrays_use_the_same_verified_ram_bytes_not_a_second_file_read(tmp_path, monkeypatch):
    index, _, _ = prepared(tmp_path,plane())
    runtime = MapRuntime(tmp_path/'support',policy())
    folder = Path(index.execution_reference[0])
    meta = json.loads((folder/'CACHE.json').read_text())
    expected_points = np.array(index.points,copy=True)
    def fail(*args,**kwargs):
        raise AssertionError('Verified payload was reopened through a separate mapping')
    monkeypatch.setattr('scenemi.geometry.query_runtime.map_numeric_npy',fail)
    arrays = runtime._arrays(index.execution_reference)
    assert np.array_equal(arrays['points'],expected_points)
    assert all(not value.flags.writeable and not isinstance(value,np.memmap) for value in arrays.values())
    path = folder/'points.npy'
    raw = bytearray(path.read_bytes());raw[-1] ^= 1;path.write_bytes(raw)
    assert np.array_equal(arrays['points'],expected_points)
    with pytest.raises(ValueError,match='checksum differs.*points.npy'):
        runtime._read_array(path,meta['arrays']['points.npy'])


def test_support_payload_checksum_is_checked_before_building_atlas(tmp_path):
    index, cache, _ = prepared(tmp_path,plane())
    folder,_ = cache.location(index.execution_reference)
    path = folder/'normals.npy'
    raw = bytearray(path.read_bytes());raw[-1] ^= 1;path.write_bytes(raw)
    runtime = MapRuntime(tmp_path/'support',policy())
    with pytest.raises(ValueError,match='checksum differs.*normals.npy'):
        runtime.view(index.execution_reference,np.zeros(3),np.eye(3))
    assert not runtime.opened and not runtime.support.opened


def test_cold_load_lock_serializes_builds_and_never_blocks_hot_queries(tmp_path,monkeypatch):
    index, _, _ = prepared(tmp_path,plane())
    runtimes = [MapRuntime(tmp_path/'support',policy(),load_lock_path=tmp_path/'load.lock',
                           load_timeout_seconds=2) for _ in range(2)]
    active = 0;peak = 0;guard = threading.Lock()
    def wrap(original):
        def build(reference):
            nonlocal active,peak
            with guard:
                active += 1;peak = max(peak,active)
            try:
                time.sleep(.02)
                return original(reference)
            finally:
                with guard: active -= 1
        return build
    for runtime in runtimes: monkeypatch.setattr(runtime,'_arrays',wrap(runtime._arrays))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runtime.view,index.execution_reference,np.zeros(3),np.eye(3))
                   for runtime in runtimes]
        for future in futures: future.result(timeout=5)
    assert peak == 1
    runtimes[1].load_timeout_seconds = .02
    with runtimes[0]._load_lock():
        # Already cached queries remain parallel while a different map loads.
        runtimes[1].view(index.execution_reference,np.zeros(3),np.eye(3))
        with pytest.raises(TimeoutError,match='loading lock'):
            with runtimes[1]._load_lock():
                raise AssertionError('Concurrent cold loads entered together')
    with runtimes[1]._load_lock(): pass


@pytest.mark.parametrize('partial_response',[False,True])
def test_complete_ipc_deadline_covers_blocked_send_and_partial_receive(partial_response):
    executor = object.__new__(EvidenceQueries)
    executor.closed = False
    executor._state_lock = threading.RLock()
    executor._counters = dict(request_bytes=0,response_bytes=0)
    parent,child = socket.socketpair()
    parent.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4096)
    connection = Connection(parent.detach())
    try:
        if partial_response:
            # A readable header must not make a partially sent body unbounded.
            child.sendall(struct.pack('!i',64)+b'x')
            task = ('tiny',)
        else:
            # The peer deliberately never reads a request larger than the pipe.
            task = (np.zeros(1024*1024,np.uint8),)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            executor._exchange(connection,task,started+.05)
        assert time.monotonic()-started < 2
    finally:
        connection.close();child.close()


def test_response_frame_limit_rejects_oversize_before_allocating_body():
    executor = object.__new__(EvidenceQueries)
    executor.closed = False
    executor._state_lock = threading.RLock()
    executor._counters = dict(request_bytes=0,response_bytes=0)
    parent,child = socket.socketpair()
    connection = Connection(parent.detach())
    try:
        child.sendall(struct.pack('!i',executor.MAX_MESSAGE_BYTES+1))
        with pytest.raises(MemoryError,match='response frame exceeds'):
            executor._exchange(connection,('tiny',),time.monotonic()+1)
    finally:
        connection.close();child.close()


@pytest.mark.parametrize('remote_error',[ValueError('bad map'),MemoryError('budget exceeded')])
def test_worker_reported_errors_trip_executor_and_stop_followup_tasks(tmp_path,monkeypatch,remote_error):
    index, _, _ = prepared(tmp_path,plane())
    local = SceneView(MapReference(index.execution_reference),np.zeros(3),np.eye(3))
    with EvidenceQueries(1,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs') as executor:
        monkeypatch.setattr(executor,'_exchange',lambda *args:(False,remote_error))
        future = executor.submit('patches',local,np.zeros((2,1,3),np.float32),query_policy())
        with pytest.raises(RuntimeError,match=f'without retry.*{type(remote_error).__name__}'):
            future.result(timeout=5)
        with pytest.raises(RuntimeError,match='without retry'):
            executor.submit('patches',local,np.zeros((2,1,3),np.float32),query_policy())
        assert executor.workers[0][1].poll() is not None
        assert list((tmp_path/'logs').rglob('FAILURE.json'))


def test_bounded_submissions_complete_without_consuming_results(tmp_path,monkeypatch):
    index, _, _ = prepared(tmp_path,plane())
    local = SceneView(MapReference(index.execution_reference),np.zeros(3),np.eye(3))
    release = threading.Event()
    def work(task,deadline):
        assert release.wait(3)
        return task[0]
    with EvidenceQueries(1,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs') as executor:
        monkeypatch.setattr(executor,'_dispatch',work)
        first = executor.submit('patches',local,np.zeros((2,1,3),np.float32),query_policy())
        second = executor.submit('patches',local,np.zeros((2,1,3),np.float32),query_policy())
        submitted = []
        thread = threading.Thread(target=lambda:submitted.append(executor.submit(
            'patches',local,np.zeros((2,1,3),np.float32),query_policy())))
        thread.start()
        try:
            time.sleep(.02)
            assert not submitted and executor._counters['peak_pending'] == executor.max_pending == 2
        finally:
            release.set();thread.join(timeout=3)
        assert not thread.is_alive() and len(submitted) == 1
        assert [f.result(timeout=3) for f in (first,second,*submitted)] == ['patches']*3
        assert executor._counters['peak_pending'] <= executor.max_pending


def test_close_interrupts_pending_ipc_and_reaps_only_owned_helper(tmp_path,monkeypatch):
    index, _, _ = prepared(tmp_path,plane())
    local = SceneView(MapReference(index.execution_reference),np.zeros(3),np.eye(3))
    executor = EvidenceQueries(1,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs')
    process = executor.workers[0][1]
    process.send_signal(signal.SIGSTOP)
    entered = threading.Event();exchange = executor._exchange
    def blocked(*args):
        entered.set()
        return exchange(*args)
    monkeypatch.setattr(executor,'_exchange',blocked)
    future = executor.submit('patches',local,np.zeros((2,1,3),np.float32),query_policy())
    try:
        assert entered.wait(3)
        started = time.monotonic();executor.close()
        assert time.monotonic()-started < 3
        assert executor.closed and process.poll() is not None and future.done()
        executor.close()
    finally:
        executor.close()


def test_partial_worker_start_failure_reaps_previously_started_helpers(tmp_path,monkeypatch):
    started = [];spawn = EvidenceQueries._spawn_worker
    def fail_second(executor):
        if started: raise RuntimeError('synthetic startup failure')
        worker = spawn(executor);started.append(worker[1]);return worker
    monkeypatch.setattr(EvidenceQueries,'_spawn_worker',fail_second)
    with pytest.raises(RuntimeError,match='synthetic startup failure'):
        EvidenceQueries(2,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs')
    assert started and all(process.poll() is not None for process in started)


def test_worker_start_failure_after_popen_reaps_unregistered_helper(tmp_path,monkeypatch):
    import scenemi.geometry.support_evidence as module
    processes = [];popen = module.subprocess.Popen
    def remember(*args,**kwargs):
        process = popen(*args,**kwargs);processes.append(process);return process
    def fail_connection(*args,**kwargs):
        raise RuntimeError('synthetic connection construction failure')
    monkeypatch.setattr(module.subprocess,'Popen',remember)
    monkeypatch.setattr(module,'Connection',fail_connection)
    with pytest.raises(RuntimeError,match='synthetic connection construction failure'):
        EvidenceQueries(1,tmp_path/'indices',tmp_path/'support',policy(),log_root=tmp_path/'logs')
    assert processes and all(process.poll() is not None for process in processes)


def test_descriptor_duplication_is_atomic_with_original_connection_close(monkeypatch):
    import os
    executor = object.__new__(EvidenceQueries)
    executor.closed = False;executor._state_lock = threading.RLock()
    parent,child = socket.socketpair();connection = Connection(parent.detach())
    entered = threading.Event();release = threading.Event();closed = threading.Event()
    duplicate = os.dup;channels = []
    def blocked_duplicate(descriptor):
        entered.set()
        assert release.wait(3)
        return duplicate(descriptor)
    monkeypatch.setattr(os,'dup',blocked_duplicate)
    def close_original():
        with executor._state_lock:
            connection.close();executor.closed = True
        closed.set()
    read = threading.Thread(target=lambda:channels.append(executor._duplicate_channel(connection)))
    close = threading.Thread(target=close_original)
    try:
        read.start();assert entered.wait(3);close.start()
        assert not closed.wait(.02)
        release.set();read.join(timeout=3);close.join(timeout=3)
        assert closed.is_set() and channels and not read.is_alive() and not close.is_alive()
        child.sendall(b'x')
        channels[0].settimeout(1)
        assert channels[0].recv(1) == b'x'
    finally:
        release.set()
        if read.ident is not None: read.join(timeout=3)
        if close.ident is not None: close.join(timeout=3)
        for channel in channels: channel.close()
        connection.close();child.close()


def test_failure_capsule_never_copies_dynamic_scene_arrays_and_obeys_budget(tmp_path):
    from types import SimpleNamespace
    executor = object.__new__(EvidenceQueries)
    executor.log_root = tmp_path;executor._command = ['synthetic-worker']
    executor.MAX_FAILURE_ARRAY_BYTES = 64
    process = SimpleNamespace(pid=123,scene_log='synthetic.log',poll=lambda:-9)
    anchors = np.zeros((2,1,3),np.float32)
    dynamic = tuple(np.zeros((8,64,3),np.float32) for _ in range(4))
    folder = executor._record_failure(('compare',('synthetic-map','sha'),np.zeros(3),np.eye(3),
        anchors,{},dict(final_anchors=np.zeros((100,3),np.float32),dynamic=dynamic)),process,0)
    info = json.loads((folder/'FAILURE.json').read_text())
    assert info['arrays']['anchors']['file'] == 'anchors.npy'
    assert 'omitted' in info['arrays']['final_anchors']
    assert all('omitted' in row for row in info['dynamic_arrays'])
    assert not list(folder.glob('*.npz')) and not list(folder.glob('dynamic*'))
    assert sum(np.load(path,allow_pickle=False).nbytes for path in folder.glob('*.npy')) <= 64


def test_shutdown_induced_failure_does_not_start_diagnostics(monkeypatch):
    executor = object.__new__(EvidenceQueries)
    executor.closed = True;executor._state_lock = threading.RLock()
    def fail(*args): raise AssertionError('Closing executor started diagnostic I/O')
    monkeypatch.setattr(executor,'_record_failure',fail)
    monkeypatch.setattr(executor,'_stop_workers',fail)
    error = executor._fail(None,None,EOFError('closed'))
    assert 'closed during query' in str(error)


def test_normal_training_queries_fit_workspace_but_huge_expansion_is_rejected():
    import copy
    anchors = np.zeros((128,67,3),np.float32)
    p = query_policy()
    assert validate_query('features',anchors,p) < QUERY_WORKSPACE_BYTES
    assert validate_query('compare',anchors[::4],p,final_anchors=anchors[::4]) < QUERY_WORKSPACE_BYTES
    large_k = copy.deepcopy(p);large_k['query']['neighbours_per_scale'] = 256
    with pytest.raises(MemoryError,match='query expansion exceeds workspace'):
        validate_query('features',anchors,large_k)
    enormous_k = copy.deepcopy(p);enormous_k['query']['neighbours_per_scale'] = 10**9
    with pytest.raises(ValueError,match='neighbour count'):
        validate_query('features',np.zeros((1,1,3),np.float32),enormous_k)
    invalid = anchors[:1,:1].copy();invalid[0,0,0] = np.nan
    with pytest.raises(ValueError,match='finite'):
        validate_query('features',invalid,p)
    with pytest.raises(ValueError,match='T,A,3'):
        validate_query('features',anchors[0],p)


@pytest.mark.parametrize('proposals,hull_faces',[(10**6,4),(1,10**7)])
def test_dense_support_or_large_hull_rejected_before_ragged_expansion(monkeypatch,proposals,hull_faces):
    from types import SimpleNamespace
    atlas = object.__new__(BoundedSupportAtlas)
    atlas.policy = policy();atlas.workspace_statistics = {}
    atlas.arrays = dict(hull_offsets=np.array([0,hull_faces],np.int64))
    counted = []
    def counts(points,radius,*,workers,return_length):
        assert return_length is True and workers == 1
        counted.append(True)
        return np.full(len(points),proposals,np.int64)
    atlas.centres = SimpleNamespace(query_ball_point=counts)
    def forbidden(*args,**kwargs):
        raise AssertionError('Oversize support reached historical ragged allocation')
    monkeypatch.setattr('scenemi.geometry.support_surfaces.SupportSurfaceAtlas.query',forbidden)
    with pytest.raises(MemoryError,match='before expansion'):
        atlas.query(np.zeros((1,1,3),np.float32))
    assert counted == [True]
    assert atlas.workspace_statistics['support_query_workspace_rejections'] == 1
