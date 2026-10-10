"""Short-horizon planner using trusted executed body, known head and scene.

Only execute a short chunk; next call supplies actual tracked state, never the
unexecuted old plan. Camera-to-head calibration and simulator/native joint
mapping must be performed upstream. This API is not a simulator integration.
"""
from experiments.offline_camera_retrain_v1.body_history_condition import attach_executed_history
from experiments.offline_camera_retrain_v1.executed_prefix_sampling import sample_with_executed_prefix
from experiments.offline_camera_retrain_v1.state_relative_spline_refinement import refine_from_state

INPUT_KEYS=('trajectory','observation_meta','occupancy','bps','bps_valid','rest','body_type','body_scale','camera')


def generate_state_relative_plan(model,observations,executed_native_motion,*,seed=777,steps=20,
                                 execute_frames=4,history_confidence=1.,use_scene=True):
    """Return native201 future plan and its proposed short execution chunk.

    Head observations: typed [B,T,22,9] position/2 + global 6D, with confidence
    and availability in observation_meta. Native201 actual history: root/2,
    local residual22x6D; direct-joint channels ignored/recomputed. Same native
    body, rest, scale and camera-anchored frame are required.

    Future body/contact truth fields are stripped before either stage. Dynamic
    scene arrays must already obey per-time known_scene_bps/causal-cache rules.
    Missing dynamic future must not be filled with stale world-space surfaces.
    """
    if history_confidence<.999:
        raise ValueError('State-relative anchoring requires trusted actual body state; uncertain estimates need a soft-history policy')
    if executed_native_motion.shape[1]<4:
        raise ValueError('Need at least four executed states for measured velocity')
    future_frames=observations['trajectory'].shape[1]-executed_native_motion.shape[1]
    if future_frames<4 or not 1<=execute_frames<=future_frames:
        raise ValueError('Need at least four future frames and a valid execution chunk')
    inputs={key:observations[key] for key in INPUT_KEYS if key in observations}
    # This version is explicitly a head/scene planner; never forward other future body tracks.
    inputs['trajectory']=inputs['trajectory'].clone()
    inputs['observation_meta']=inputs['observation_meta'].clone()
    other_joints=[j for j in range(22) if j!=15]
    inputs['trajectory'][:,:,other_joints]=0
    inputs['observation_meta'][:,:,other_joints]=0
    inputs=attach_executed_history(inputs,executed_native_motion,confidence=history_confidence)
    original=sample_with_executed_prefix(model,inputs,executed_native_motion,steps=steps,seed=seed,use_scene=use_scene)
    refined,audit=refine_from_state(original,executed_native_motion,inputs,pose_frames=min(32,future_frames),iterations=60,relative_root=True)
    future=refined[:,executed_native_motion.shape[1]:]
    return dict(future_motion=future,execute_motion=future[:,:execute_frames],full_motion=refined,original_motion=original,
                audit={**audit,'execute_frames':execute_frames,'known_future_head_soft':True,'future_body_contact_fields_stripped':True,
                       'next_history_source':'actual tracked/executed states, not unexecuted prediction','physics_tracking_or_mesh_collision_verified':False})
