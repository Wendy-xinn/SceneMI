"""Existing query regression suite plus exact-radius/culling edge cases."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1 import test_native30_query_runtime as tests
import experiments.offline_camera_retrain_v1.native30_query_runtime as runtime
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.prepare_native30 import sha256,write_json
from exact_gpu_query import query_surfaces_cuda

ROOT=Path(__file__).resolve().parent


def equal(scene,probes,rotations,neighbors):
    expected=scene.query_reference(probes,rotations,neighbors)
    actual=query_surfaces_cuda(scene,probes,rotations,'cuda',neighbors)
    assert all(np.array_equal(expected[k],actual[k]) for k in expected), [k for k in expected if not np.array_equal(expected[k],actual[k])]


if __name__=='__main__':
    runtime.query_surfaces_cuda=query_surfaces_cuda
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not result.wasSuccessful():raise SystemExit(1)
    cases=0
    scene,probes,rotations=tests.fixture()
    # A large off-window map must not affect any radius membership or tie.
    extra=100000
    for k,v in scene.static.items():
        values=np.zeros((extra,*v.shape[1:]),dtype=v.dtype)
        if k=='points':values[:]=[1e5,1e5,1e5]
        scene.static[k]=np.concatenate((v,values))
    scene._static_tree=cKDTree(scene.static['points'])
    for dtype in (np.float32,np.float64):
        equal(scene,probes.astype(dtype),rotations,8);cases+=1
    boundary=np.array([[.08,0,0],[-.08,0,0],[.25,0,0],[-.25,0,0],
        [.6,0,0],[-.6,0,0],[0,.25,0],[0,-.25,0]],np.float32)
    scene=Native30SurfaceWindow(np.broadcast_to(boundary,(2,*boundary.shape)),
        np.zeros((2,len(boundary)),np.int64),np.zeros((2,3)))
    for dtype in (np.float32,np.float64):
        p=np.zeros((2,5,3),dtype=dtype)
        p[:,1,0]=np.float64(np.float32(.6))-.6
        p[:,2,0]=np.nextafter(p[0,1,0],-np.inf)
        p[:,3,0]=np.nextafter(p[0,1,0],np.inf)
        p[:,4]=1e6
        r=np.broadcast_to(np.eye(3),(2,5,3,3))
        for neighbors in (1,8,12):
            equal(scene,p,r,neighbors);cases+=1
    for far in (False,True):
        p=np.full((2,3,3),1e8 if far else 0.,np.float64)
        r=np.broadcast_to(np.eye(3),(2,3,3,3))
        equal(scene,p,r,8);cases+=1
    write_json(ROOT/'QUERY_EDGE_TESTS.json',dict(status='passed',existing_tests=result.testsRun,
        additional_all_field_cases=cases,candidate_sha256=sha256(ROOT/'exact_gpu_query.py'),
        coverage='float32/64, empty/missing, frame owners/velocity, yaw, sparse/excess neighbor slots, radius borders and rounded ties,100000 irrelevant map points, all points culled'))
    print('Additional exact query cases passed:',cases)
