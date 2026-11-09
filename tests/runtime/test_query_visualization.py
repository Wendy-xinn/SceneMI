"""Diagnostic geometry preserves metric radii, case identity and world axes."""
from types import SimpleNamespace
from collections import OrderedDict
from contextlib import nullcontext
import json
import threading

import numpy as np
import pytest

from scenemi.refinement.visualization import select_misses, sphere_segments, nearest_static, nearest_input
from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer


def test_selects_distinct_recordings_and_only_saved_no_hit_windows():
    rows=[]
    for recording in range(16):
        for start in range(3):
            rows.append(dict(identity=dict(dataset='egobody',family=f'r{recording:02}'),
                case=dict(case_id=recording*3+start,start=start),audit=dict(correct=dict(
                    zero_reason='no_anchor_neighbour_hits',gt_query_hit_fraction=float(start==2)))))
    selected=select_misses(rows,10)
    assert len({r['identity']['family'] for r in selected})==10
    assert all(r['case']['start']==2 for r in selected)
    assert select_misses(rows,10)==selected


def test_query_sphere_has_exact_radius_and_three_orthogonal_circles():
    centres=np.array([[0,0,0],[1,2,3]],np.float32)
    segments=sphere_segments(centres,.2,32)
    assert segments.shape==(2*3*32,2,3)
    for i,centre in enumerate(centres):
        points=np.concatenate([segments[k*64+i*32:k*64+(i+1)*32].reshape(-1,3) for k in range(3)])
        np.testing.assert_allclose(np.linalg.norm(points-centre,axis=-1),.2,atol=2e-7)


def test_nearest_query_keeps_static_union_and_current_dynamic_clock_separate():
    anchors=np.zeros((2,1,3),np.float32)
    static=np.array([[1,0,0]],np.float32)
    dynamic=np.array([[[.1,0,0]],[[.3,0,0]]],np.float32)
    owners=np.array([[1],[-1]])
    distance,closest=nearest_input(anchors,static,dynamic,owners)
    np.testing.assert_allclose(distance[:,0],[.1,1.])
    np.testing.assert_allclose(closest[:,0,0],[.1,1.])
    empty,_=nearest_static(anchors,np.empty((0,3),np.float32))
    assert np.isinf(empty).all()


def test_top_camera_preserves_gravity_axis_and_non_collinear_up_direction():
    viewer=MotionViewer.__new__(MotionViewer)
    viewer.lock=threading.RLock()
    viewer.runtime=dict(ego=dict(data=dict(center=np.array([1.,2.,3.]),radius=2.6)))
    viewer.follow=SimpleNamespace(value=False)
    viewer.camera_view=SimpleNamespace(value='俯视 · 沿重力向下')
    initial=SimpleNamespace()
    client=SimpleNamespace(camera=SimpleNamespace())
    viewer.server=SimpleNamespace(initial_camera=initial,get_clients=lambda:{1:client},atomic=nullcontext)
    viewer.camera()
    direction=np.array(client.camera.position)-client.camera.look_at
    assert direction[1]>0 and abs(direction[2]/direction[1])<.05
    assert client.camera.up_direction==(0.,1.,0.)
    assert np.linalg.norm(np.cross(direction,client.camera.up_direction))>0
    viewer.camera_view.value='斜视';viewer.camera()
    assert client.camera.up_direction==(0.,1.,0.)


def test_follow_translates_current_view_without_resetting_manual_orbit_or_zoom():
    viewer=MotionViewer.__new__(MotionViewer)
    viewer.lock=threading.RLock()
    joints=np.zeros((2,22,3));joints[1,0]=[1.,.2,-2.]
    viewer.runtime=dict(ego=dict(data=dict(center=np.zeros(3),radius=2.6,truth_joints=joints)))
    viewer.follow=SimpleNamespace(value=True);viewer.frame=SimpleNamespace(value=0)
    viewer.camera_view=SimpleNamespace(value='斜视')
    client=SimpleNamespace(camera=SimpleNamespace())
    viewer.server=SimpleNamespace(initial_camera=SimpleNamespace(),get_clients=lambda:{1:client},atomic=nullcontext)
    viewer.camera()
    client.camera.position=(4.,5.,6.);client.camera.look_at=(.2,.3,.4)
    viewer.camera(follow_only=True)
    assert client.camera.position==(4.,5.,6.)
    viewer.frame.value=1;viewer.camera(follow_only=True)
    np.testing.assert_allclose(client.camera.position,[5.,5.2,4.])
    np.testing.assert_allclose(client.camera.look_at,[1.2,.5,-1.6])
    np.testing.assert_allclose(np.subtract(client.camera.position,client.camera.look_at),[3.8,4.7,5.6])


def test_group_filter_retains_original_case_indices():
    viewer=MotionViewer.__new__(MotionViewer)
    viewer.labels=['old','new1','new2']
    viewer.cases=[{},dict(visualization_group='诊断'),dict(visualization_group='诊断')]
    viewer.case_group=SimpleNamespace(value='诊断')
    viewer.choice=SimpleNamespace(options=tuple(viewer.labels),value='old')
    viewer.select_group()
    assert viewer.choice.options==['new1','new2'] and viewer.choice.value=='new1'
    assert viewer.labels.index(viewer.choice.value)==1


def test_full_video_layer_preserves_old_prediction_and_rejects_modified_sidecar(tmp_path):
    from scenemi.data.amass_prepare import sha256
    from scenemi.refinement.visualization import QueryDiagnosticViewer, PROTOCOL
    from scenemi.visualization import prepared_playback

    original, extra = tmp_path/'motion.npz', tmp_path/'video.npz'
    np.savez(original, metadata_json=np.array(json.dumps(dict(identity=dict(family='video')))),
        truth_joints=np.ones((2,22,3)), visible_static=np.array([[1.,2.,3.]]))
    original_sha = sha256(original)
    np.savez(extra, video_observed_static=np.array([[4.,5.,6.]]),
        video_metadata_json=np.array(json.dumps(dict(points=1,used_by_historical_refiner=False))))
    manifest = tmp_path/'PREPARED_CASES.json'
    manifest.write_text(json.dumps(dict(protocol=PROTOCOL,cases=[dict(models=[dict(
        mode='ego',case=dict(cache=str(original),cache_sha256=original_sha))],
        scene_query_video_asset=dict(path=str(extra),sha256=sha256(extra)))])))
    cases, _ = prepared_playback(manifest)
    viewer = QueryDiagnosticViewer.__new__(QueryDiagnosticViewer)
    viewer.cache, viewer.cases = OrderedDict(), cases
    loaded = viewer.read_case(0)[0]
    np.testing.assert_array_equal(loaded['data']['visible_static'], [[1.,2.,3.]])
    np.testing.assert_array_equal(loaded['data']['video_observed_static'], [[4.,5.,6.]])
    assert not loaded['meta']['query_video_map']['used_by_historical_refiner']
    assert sha256(original) == original_sha
    extra.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Original-video'):
        prepared_playback(manifest)
