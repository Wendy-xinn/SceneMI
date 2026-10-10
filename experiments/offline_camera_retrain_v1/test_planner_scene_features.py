"""Known future plan is allowed; unobserved future/old dynamic geometry excluded."""
import json
import numpy as np
from experiments.offline_camera_retrain_v1.planner_scene_features import known_scene_bps
from experiments.offline_camera_retrain_v1.data import HERE


def main():
    camera=np.zeros((3,3),np.float32);rotation=np.repeat(np.eye(3)[None],3,0);anchors=np.zeros((1,3),np.float32)
    static=np.array([[0,0,.6],[0,0,.01]],np.float32)
    objects=[np.array([[.1,0,0]]),np.array([[.8,0,0]]),np.array([[.01,0,0]])]
    kwargs=dict(dynamic_known=[True,True,False],static_first_observed=[0,100],decision_source_frame=10)
    out=known_scene_bps(static,objects,camera,rotation,anchors,**kwargs)
    assert out['static_points_available']==1
    np.testing.assert_allclose(out['bps'][:,0],[[.05,0,0],[0,0,.3],[0,0,.3]],atol=1e-7)
    # Remove the static map: supplied future moving-object poses remain usable.
    out2=known_scene_bps(np.empty((0,3)),objects,camera,rotation,anchors,dynamic_known=[True,True,False])
    np.testing.assert_allclose(out2['bps'][:2,0,0],[.05,.4],atol=1e-7)
    assert not out2['bps_valid'][2].any()
    # A surveyed known map can include the .01m surface independent of observation time.
    surveyed=known_scene_bps(static,objects,camera,rotation,anchors,dynamic_known=[False]*3)
    np.testing.assert_allclose(surveyed['bps'][:,:,2],.005,atol=1e-7)
    try:known_scene_bps(static,objects,camera,rotation,anchors,dynamic_known=[True]*3,static_first_observed=[0,100])
    except ValueError:pass
    else:raise AssertionError('Undated recording memory accepted')
    report=dict(status='passed',known_future_object_pose_allowed=True,known_surveyed_static_map_allowed=True,recorded_unobserved_future_static_excluded=True,unknown_future_dynamic_not_frozen_or_accumulated=True,missing_decision_timestamp_rejected=True,
                integration='optional deployment scene adapter, existing training scene caches unchanged')
    (HERE/'runs/replan_transition_oct10/planner_scene_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
