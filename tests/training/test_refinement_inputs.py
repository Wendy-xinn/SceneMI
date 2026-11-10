"""Selective prepared input reads preserve motion, labels and window RNG."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scenemi.data.ego import ALL_KEYS
from scenemi.data.refinement_inputs import REFINEMENT_ARRAYS, RefinementInputStore
from scenemi.data.scene_geometry_cache import PreparedSceneDataset
from scenemi.data.spatial_scene_data import IndexedGeometryEgoData
from scenemi.geometry.query_runtime import MapReference
from scenemi.refinement.contact_training import ReferenceSupportData


def arrays_for():
    frames,count = 132,3
    time = np.arange(frames,dtype=np.float32)
    angle = time*.002
    rotation = np.zeros((frames,3,3),np.float32)
    rotation[:,0,0] = rotation[:,2,2] = np.cos(angle)
    rotation[:,0,2] = np.sin(angle);rotation[:,2,0] = -np.sin(angle);rotation[:,1,1] = 1.
    motion = np.zeros((frames,201),np.float32)
    motion[:,0] = time*.001
    rest = np.zeros((22,3),np.float32);rest[:,1] = np.arange(22)*.03
    camera = np.column_stack((time*.002,np.full(frames,1.6),time*.0003)).astype(np.float32)
    points = np.broadcast_to(camera[:,None],(frames,count,3)).copy()
    points += np.array([[[.1,-.7,.1],[.2,-.8,.2],[.3,-.9,.3]]],np.float32)
    normals = np.broadcast_to(np.array([0.,1.,0.],np.float32),points.shape).copy()
    return dict(rest=rest,motion_world=motion,camera_position_scenemi_yup=camera,
        camera_rotation_scenemi_yup=rotation,
        camera_intrinsics_128x96=np.tile(np.array([60.,60.,64.,48.],np.float32),(frames,1)),
        frame_points=points,frame_owner=np.tile(np.array([-1,0,1],np.int64),(frames,1)),
        frame_normals=normals,frame_normal_valid=np.ones((frames,count),bool),
        contact_weight=np.linspace(0.,1.,frames*2,dtype=np.float32).reshape(frames,2))


def publish(root, key='one'):
    arrays = arrays_for();identity = dict(source=key,split='train')
    folder = root/key;folder.mkdir(parents=True)
    hashes = {}
    for name,value in arrays.items():
        file = folder/(name+'.npy');np.save(file,value,allow_pickle=False)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    # Deliberately invalid legacy payloads. They must never even be statted.
    for name in ('static_full','static_points','static_normals','static_normal_valid'):
        (folder/(name+'.npy')).write_bytes(b'forbidden legacy static geometry')
        hashes[name+'.npy'] = 'not-an-accepted-digest'
    (folder/'CACHE.json').write_text(json.dumps(dict(identity=identity,arrays=hashes,complete=True)))
    return arrays,identity


def forbid_static_files(monkeypatch):
    original_open,original_stat = Path.open,Path.stat
    def open_file(path,*args,**kwargs):
        if path.name.startswith('static_'): raise AssertionError('Legacy static payload was opened')
        return original_open(path,*args,**kwargs)
    def stat_file(path,*args,**kwargs):
        if path.name.startswith('static_'): raise AssertionError('Legacy static payload was statted')
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',open_file);monkeypatch.setattr(Path,'stat',stat_file)


def test_store_reads_only_ten_fields_into_verified_immutable_ram(tmp_path,monkeypatch):
    expected,identity = publish(tmp_path)
    forbid_static_files(monkeypatch)
    store = RefinementInputStore(tmp_path)
    actual = store.read('one',identity)
    assert set(actual) == set(REFINEMENT_ARRAYS)
    for name,value in actual.items():
        assert np.array_equal(value,expected[name])
        assert not isinstance(value,np.memmap) and not value.flags.writeable
    hashed = store.stats()['hash_bytes']
    assert store.read('one',identity) is actual
    stats = store.stats()
    assert stats['hash_bytes'] == hashed > 0 and stats['loads'] == 1 and stats['hits'] == 1
    assert stats['resident_bytes'] <= stats['cache_limit_bytes'] == 256*2**20
    assert stats['legacy_static_geometry_read'] is False


def test_store_rejects_oversize_entry_before_read_and_evicts_before_next_copy(tmp_path,monkeypatch):
    _,identity = publish(tmp_path)
    _,second_identity = publish(tmp_path,'two')
    required = sum((tmp_path/'one'/(name+'.npy')).stat().st_size for name in REFINEMENT_ARRAYS)
    import scenemi.data.refinement_inputs as module
    original = module.read_verified_array
    def forbidden(*args,**kwargs): raise AssertionError('Oversize entry reached a payload read')
    monkeypatch.setattr(module,'read_verified_array',forbidden)
    store = RefinementInputStore(tmp_path,byte_limit=required-1)
    with pytest.raises(MemoryError,match='before loading'):
        store.read('one',identity)
    assert store.stats()['hash_bytes'] == 0 and not store.opened
    monkeypatch.setattr(module,'read_verified_array',original)
    store = RefinementInputStore(tmp_path,limit=1)
    store.read('one',identity)
    reads = []
    def after_eviction(*args,**kwargs):
        assert not store.opened
        reads.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(module,'read_verified_array',after_eviction)
    store.read('two',second_identity)
    assert len(reads) == 10 and list(store.opened) == ['two']
    assert store.stats()['evictions'] == 1


def test_store_rejects_corruption_without_publishing_partial_entry(tmp_path):
    _,identity = publish(tmp_path)
    path = tmp_path/'one'/'motion_world.npy'
    raw = bytearray(path.read_bytes());raw[-1] ^= 1;path.write_bytes(raw)
    store = RefinementInputStore(tmp_path)
    with pytest.raises(ValueError,match='checksum differs.*motion_world.npy'):
        store.read('one',identity)
    assert not store.opened and store.stats()['loads'] == 0


def native_fixture(store,identity):
    native = object.__new__(PreparedSceneDataset)
    path = 'synthetic/grail/recording'
    row = dict(path=path,dataset='grail',family='synthetic-family',starts=np.arange(5))
    metadata = dict(path=path,input_sha256='synthetic-source-sha',group='synthetic',
        native_body=dict(model='smpl',gender='neutral',betas=np.zeros(10,np.float32)))
    native.records = [row];native.split = 'train'
    native.common_pelvis = np.array([.03,.01,-.02],np.float32)
    native.families = dict(grail=['synthetic-family'])
    native.groups = dict(grail={'synthetic-family':[0]})
    native.selection_bundle = SimpleNamespace(metadata=lambda row,split:metadata)
    native.store = store;native.identities = {path:('one',identity)};native._by_path = {path:row}
    return native


def test_reference_sampler_matches_old_camera_frame_labels_identity_and_rng(tmp_path,monkeypatch):
    arrays,identity = publish(tmp_path)
    new = object.__new__(ReferenceSupportData)
    new.native = native_fixture(RefinementInputStore(tmp_path),identity)
    new.c = dict(geometry=dict(static_points=7,frame_points=3))
    reference = object.__new__(MapReference)
    reference.points = np.zeros((11,3),np.float32)
    reference.execution_reference = ('synthetic-map','sha')
    new.source_index = lambda row:(reference,'complete_original_source_clip')
    # The control follows the old concrete sampler with tiny in-memory
    # legacy tensors, so no actual map file is read by either execution.
    full = dict(arrays,static_full=np.array([[0.,0.,0.],[.3,0.,.2]],np.float32),
        static_points=np.array([[0.,0.,0.],[.3,0.,.2]],np.float32),
        static_normals=np.tile(np.array([0.,1.,0.],np.float32),(2,1)),
        static_normal_valid=np.ones(2,bool))
    old = object.__new__(IndexedGeometryEgoData)
    old.native = native_fixture(SimpleNamespace(read=lambda key,identity:full),identity)
    old.c = new.c;old.source_index = new.source_index
    forbid_static_files(monkeypatch)
    rng_old,rng_new = np.random.default_rng(42),np.random.default_rng(42)
    expected,old_identity = old.sample('grail',rng_old)
    actual,new_identity = new.sample('grail',rng_new)
    assert deepcopy(rng_old.bit_generator.state) == deepcopy(rng_new.bit_generator.state)
    for name in ALL_KEYS+('frame_scene_points','frame_scene_owner','frame_normals',
                          'frame_normal_valid','contact_weight'):
        assert np.array_equal(actual[name],expected[name]),name
    assert np.array_equal(actual['scene_view'].origin,expected['scene_view'].origin)
    assert np.array_equal(actual['scene_view'].rotation,expected['scene_view'].rotation)
    changed = {'scene_map_points_before_limit','scene_map_points_after_limit','scene_geometry'}
    for name,value in old_identity.items():
        if name not in changed: assert new_identity[name] == value,name
    assert new_identity['scene_input'] is True and new_identity['legacy_static_geometry_skipped'] is True
    assert new_identity['spatial_map_points'] == new_identity['scene_map_points_after_limit'] == 11
    assert not actual['occupancy'].any() and not actual['static_valid'].any()
    assert not actual['static_points'].any() and not actual['static_normals'].any()
    assert not np.array_equal(actual['contact_weight'],np.ones((128,2),np.float32))
    assert actual['scene_view'].index is reference
