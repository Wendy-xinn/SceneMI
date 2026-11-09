import numpy as np
from pathlib import Path
import tempfile
import unittest
from scipy.spatial.transform import Rotation

from scenemi.data.gimo import (recover_vertices, fit_rigid, mount_calibration,
    split_rows, read_pv, window_starts, read_scan_obj, valid_cameras)


def test_mesh_permutation_is_recovered_and_nonbijective_mapping_refused():
    vertices=np.arange(12,dtype=np.float32).reshape(4,3)
    faces=np.array([[0,1,2],[0,2,3]])
    permutation=np.array([2,0,3,1])
    shuffled=np.empty_like(vertices);shuffled[permutation]=vertices
    np.testing.assert_array_equal(recover_vertices(shuffled,permutation[faces],faces),vertices)
    corrupt=permutation[faces].copy();corrupt[1,0]=corrupt[1,1]
    with unittest.TestCase().assertRaises(ValueError):recover_vertices(shuffled,corrupt,faces)


def test_rigid_head_mount_recovers_known_transform_without_camera_modification():
    template=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    r=Rotation.from_rotvec([.2,.1,-.3]).as_matrix()
    actual=template@r.T+[2,3,4]
    rotation,center=fit_rigid(template,actual[None])
    np.testing.assert_allclose(rotation[0],r,atol=1e-12)
    np.testing.assert_allclose(center[0],actual.mean(0))
    cameras=np.broadcast_to(np.eye(4),(3,4,4)).copy();cameras[:,:3,3]=[[0,0,0],[1,0,0],[0,2,0]]
    mount=np.eye(4);mount[:3,:3]=r;mount[:3,3]=[.1,.2,.3]
    before=cameras.copy()
    fitted,_,position,angle=mount_calibration(cameras,cameras@mount)
    np.testing.assert_allclose(fitted,mount,atol=1e-12)
    np.testing.assert_allclose(position,0,atol=1e-12);np.testing.assert_allclose(angle,0,atol=1e-12)
    np.testing.assert_array_equal(cameras,before)


def test_split_overlap_test_wins_duplicates_removed_and_recording_val_grouped():
    def row(i,record,first,end,split):
        return dict(annotation=i,scene='room',recording=record,first=first,end=end,official_split=split)
    rows=[row(0,'shared',0,160,'train'),row(1,'shared',150,300,'test'),
          row(2,'onlytrain',0,256,'train'),row(3,'onlytrain',128,384,'train')]
    config={'data':{'seed':1,'validation_fraction_of_train_only_recordings':1.}}
    selected,meta=split_rows(rows,config)
    assert selected[0]['kept_ranges']==[[0,150]]
    assert selected[1]['kept_ranges']==[[150,300]]
    assert selected[2]['split']==selected[3]['split']=='validation'
    assert selected[3]['kept_ranges']==[[256,384]]
    assert meta['mixed_train_test_recordings']==1


def test_windows_never_cross_gaps_or_invalid_frames():
    ids=np.r_[np.arange(256),np.arange(400,656)]
    valid=np.ones(512,bool);valid[20]=False
    starts=window_starts(ids,valid)
    assert starts==[21,256,384]
    for start in starts:assert np.all(np.diff(ids[start:start+128])==1)


def test_pv_timestamp_keeps_100ns_precision():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'pv.txt';stamp=132872969651868997
        matrix=','.join(map(str,np.eye(4).ravel()))
        path.write_text(f'374,199,760,428\n{stamp},587,588,{matrix}\n{stamp+1},587,588,{matrix}\n')
        times,camera,intr,size=read_pv(path)
        assert times[0]==stamp and times[1]-times[0]==1
        assert camera.shape==(2,4,4) and intr.shape==(2,4) and size==(760,428)


def test_scan_rgb_uv_polygons_and_relative_indices_preserve_geometry():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'scan.obj'
        path.write_text('v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 1 0 1 0 0 1\nv 0 0 1 1 1 1\n'
                        'f -4/1 -3/2 -2/3 -1/4\n')
        vertices,faces=read_scan_obj(path)
        np.testing.assert_array_equal(vertices,[[0,0,0],[1,0,0],[1,0,1],[0,0,1]])
        np.testing.assert_array_equal(faces,[[0,1,2],[0,2,3]])
        path.write_text('v 0 0 0\nv 1 0 0\nv 1 0 1\nf 1 2 9\n')
        with unittest.TestCase().assertRaises(ValueError):read_scan_obj(path)


def test_broken_camera_frames_do_not_discard_other_valid_windows():
    cameras=np.broadcast_to(np.eye(4),(400,4,4)).copy()
    cameras[204:260,:3,:3]*=1e8
    cameras[300,:3,:3]=np.diag([-1,1,1])
    valid,errors=valid_cameras(cameras)
    assert not valid[204:260].any() and not valid[300]
    assert valid[:204].all() and valid[260:300].all()
    assert window_starts(np.arange(400),valid)==[0]


if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in list(globals().items()) if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
