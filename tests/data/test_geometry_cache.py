import json
import numpy as np
import pytest

from scenemi.data.recording_scene import observed_recording
from scenemi.data.scene_geometry_cache import static_geometry,ArrayStore,_publish


POLICY = dict(voxel_m=.025,static_points=64,normal_neighbours=16,
    normal_radius_m=.1,minimum_neighbours=6,minimum_spread_m=.005,maximum_planarity_ratio=.03)


def test_union_preserves_original_and_orients_only_measured_static_patches():
    x,z = np.meshgrid(np.linspace(-.1,.1,11),np.linspace(-.1,.1,11))
    points = np.stack((x.ravel(),np.zeros(x.size),z.ravel()),-1).astype(np.float32)
    points = np.stack((points,points+.001))
    owners = np.zeros(points.shape[:-1],np.int64);owners[1,:20]=1
    frames = np.array([True,False]);cameras=np.array([[0,1,0],[0,-1,0]],np.float32)
    full,query,normal,valid = static_geometry(points,owners,frames,cameras,POLICY)
    assert np.array_equal(full,observed_recording(points,owners,frames,.025))
    assert valid.any()
    assert np.all(normal[valid,1]>.99)
    assert len(query)<=64


def test_cache_refuses_changed_bytes_or_identity(tmp_path):
    folder=tmp_path/'key';identity={'version':'test','source':'train/one'}
    _publish(folder,identity,{'a':np.arange(8,dtype=np.float32)})
    assert np.array_equal(ArrayStore(tmp_path).read('key',identity)['a'],np.arange(8))
    with pytest.raises(ValueError,match='identity'):
        ArrayStore(tmp_path).read('key',dict(identity,source='other'))
    with (folder/'a.npy').open('r+b') as f:
        f.seek(-1,2);f.write(b'X')
    with pytest.raises(ValueError,match='checksum'):
        ArrayStore(tmp_path).read('key',identity)
