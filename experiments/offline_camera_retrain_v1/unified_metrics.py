"""Explicit world alignment, occluded-joint and native foot protocols."""
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
from experiments.offline_camera_retrain_v1.audit_full_validation_turns import hip_turn
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, IDENTITY_6D
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from pytorch3d.transforms import matrix_to_axis_angle
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points


def similarity_transform(source, target):
    x = source.reshape(-1, 3); y = target.reshape(-1, 3)
    centre_x, centre_y = x.mean(0), y.mean(0)
    x, y = x - centre_x, y - centre_y
    u, singular, vh = np.linalg.svd(x.T @ y)
    sign = np.ones(3); sign[-1] = np.linalg.det(u @ vh)
    rotation = (u * sign[None]) @ vh
    scale = max(0., float((singular * sign).sum() / max(float((x * x).sum()), 1e-12)))
    return scale, rotation, centre_y - scale * centre_x @ rotation


def aligned_world_metrics(prediction, truth, segment=100):
    """Full, nonoverlapping 100-frame segments; partial tail is excluded.

    Same alignment definitions as Human3R; 22 joints/20Hz remain our protocol,
    so this is not a reproduction of its official dataset/joint-regressor score.
    """
    errors = {'w_first2_aligned_100_mpjpe_mm': [], 'wa_full_aligned_100_mpjpe_mm': []}
    for first in range(0, len(truth) - segment + 1, segment):
        p, t = prediction[first:first + segment], truth[first:first + segment]
        for key, fit_frames in (('w_first2_aligned_100_mpjpe_mm', 2), ('wa_full_aligned_100_mpjpe_mm', segment)):
            scale, rotation, translation = similarity_transform(p[:fit_frames], t[:fit_frames])
            errors[key].append(float(np.linalg.norm(scale * p @ rotation + translation - t, axis=-1).mean() * 1000))
    return {key: float(np.mean(values)) if values else None for key, values in errors.items()}


@torch.no_grad()
def score(motion, batch, camera_region_available=None, camera_mount=None):
    pred = forward_kinematics(motion, batch['rest'])[0].cpu().numpy()
    truth = (batch['joints'][0] * 2).cpu().numpy()
    values = standard_motion_metrics(pred, truth)
    values['world_unaligned_mpjpe_mm'] = values.pop('w_mpjpe_mm')
    values.update(aligned_world_metrics(pred, truth))
    observed = batch['observation_meta'][0, ..., 0].bool().cpu().numpy()
    anatomical = batch['observation_meta'][0, ..., 4].cpu().numpy() > .5
    observed_anatomical = observed & anatomical
    hidden = ~observed_anatomical
    error = np.linalg.norm(pred - truth, axis=-1) * 1000
    values['unobserved_anatomical_mpjpe_mm'] = float(error[hidden].mean()) if hidden.any() else None
    if camera_region_available is not None:
        visible = np.asarray(camera_region_available, bool)
        if visible.shape != error.shape:
            raise ValueError('Camera-region visibility/target shape mismatch')
        unavailable = ~visible
        available_not_input = visible & ~observed_anatomical
        values['camera_unavailable_region_joint_mpjpe_mm'] = float(error[unavailable].mean()) if unavailable.any() else None
        values['camera_visible_unconditioned_joint_mpjpe_mm'] = float(error[available_not_input].mean()) if available_not_input.any() else None
        values['camera_unavailable_region_joint_entries'] = int(unavailable.sum())
        values['camera_visible_unconditioned_joint_entries'] = int(available_not_input.sum())
    obs = (batch['trajectory'][0, ..., :3] * 2).cpu().numpy()
    values['observed_anatomical_error_mm'] = float(np.linalg.norm(pred - obs, axis=-1)[observed_anatomical].mean() * 1000) if observed_anatomical.any() else None
    rotation = global_rotations(motion)[0]
    target_rotation = global_rotations(batch['motion'])[0]
    for label, joint in (('root', 0), ('head', 15)):
        angle = matrix_to_axis_angle(rotation[:, joint] @ target_rotation[:, joint].transpose(-1, -2)).norm(dim=-1)
        values[label + '_global_orientation_error_deg'] = float(torch.rad2deg(angle).mean())
    predicted_observation = pred.copy()
    if camera_mount is not None:
        offset = rotation[:, 15] @ rotation.new_tensor(camera_mount['translation_m'])[:, None]
        camera_prediction = pred[:, 15] + offset[:, :, 0].cpu().numpy()
        predicted_observation[:, 15] = np.where(anatomical[:, 15, None], pred[:, 15], camera_prediction)
    camera_input = observed & ~anatomical
    values['camera_track_position_error_mm'] = float(np.linalg.norm(predicted_observation - obs, axis=-1)[camera_input].mean() * 1000) if camera_input.any() and camera_mount is not None else None
    velocity_valid = observed[1:] & observed[:-1]
    acceleration_valid = velocity_valid[1:] & velocity_valid[:-1]
    observation_error = predicted_observation - obs
    velocity_error = np.linalg.norm(np.diff(observation_error, axis=0) * 20, axis=-1)
    acceleration_error = np.linalg.norm(np.diff(observation_error, n=2, axis=0) * 400, axis=-1)
    values['observed_linear_velocity_error_mm_s'] = float(velocity_error[velocity_valid].mean() * 1000) if velocity_valid.any() else None
    values['observed_linear_acceleration_error_mm_s2'] = float(acceleration_error[acceleration_valid].mean() * 1000) if acceleration_valid.any() else None
    rot_mask = batch['observation_meta'][0, ..., 1].bool()
    predicted_observation_rotation = rotation.clone()
    if camera_mount is not None:
        sensor_rotation = rotation[:, 15] @ rotation.new_tensor(camera_mount['rotation'])
        anatomical_head = batch['observation_meta'][0, :, 15, 4] > .5
        predicted_observation_rotation[:, 15] = torch.where(anatomical_head[:, None, None], rotation[:, 15], sensor_rotation)
    supplied6 = batch['trajectory'][0, ..., 3:]
    supplied6 = torch.where(rot_mask[..., None], supplied6, supplied6.new_tensor(IDENTITY_6D))
    supplied_rotation = rotation_from_6d(supplied6 - supplied6.new_tensor(IDENTITY_6D))
    angles = matrix_to_axis_angle(predicted_observation_rotation @ supplied_rotation.transpose(-1, -2)).norm(dim=-1)
    values['observed_orientation_error_deg'] = float(torch.rad2deg(angles[rot_mask]).mean()) if rot_mask.any() else None
    angular_mask = rot_mask[1:] & rot_mask[:-1]
    pw = matrix_to_axis_angle(predicted_observation_rotation[1:] @ predicted_observation_rotation[:-1].transpose(-1, -2)) * 20
    tw = matrix_to_axis_angle(supplied_rotation[1:] @ supplied_rotation[:-1].transpose(-1, -2)) * 20
    angular_error = (pw - tw).norm(dim=-1)
    values['observed_angular_velocity_error_deg_s'] = float(torch.rad2deg(angular_error[angular_mask]).mean()) if angular_mask.any() else None
    gt_turn, reliability = hip_turn(truth); turn, pred_reliability = hip_turn(pred)
    eligible = reliability >= .95 and abs(gt_turn) >= 30
    values.update(gt_hip_turn_deg=gt_turn, predicted_hip_turn_deg=turn,
                  turn_eligible=float(eligible), predicted_hip_projection_reliable_fraction=pred_reliability,
                  opposite_turn=float(turn * gt_turn < 0 and abs(turn) >= 15) if eligible else None,
                  under_half_turn=float(abs(turn) < .5 * abs(gt_turn)) if eligible else None)
    asset = batch['skin_assets'][0]
    generated_skin = skin_points(motion[0], asset).cpu().numpy()
    true_skin = skin_points(batch['motion'][0], asset).cpu().numpy()
    # One fixed lowest-template vertex per ankle/toe region, chosen without
    # per-frame truth. This is explicitly a marker metric, not a contact oracle.
    foot_ids = [int(torch.where(asset['labels'] == joint)[0][0]) for joint in (7, 8, 10, 11)]
    dp, dt = np.diff(generated_skin[:, foot_ids], axis=0), np.diff(true_skin[:, foot_ids], axis=0)
    stance = np.linalg.norm(dt, axis=-1) * 20 < .2
    horizontal = np.linalg.norm(dp[..., [0, 2]], axis=-1) * 20
    values['native_foot_marker_gt_stance_speed_mm_s'] = float(horizontal[stance].mean() * 1000) if stance.any() else None
    values['native_foot_marker_stance_entries'] = int(stance.sum())
    return values, pred


def metric_checks():
    rng = np.random.default_rng(401)
    truth = rng.normal(size=(128, 22, 3)); angle = .7
    r = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    pred = 1.7 * truth @ r + [3., 4., -2.]
    result = aligned_world_metrics(pred, truth)
    assert max(result.values()) < 1e-8
    pred[30:] += [1., 0., 0.]
    result = aligned_world_metrics(pred, truth)
    assert result['w_first2_aligned_100_mpjpe_mm'] > result['wa_full_aligned_100_mpjpe_mm'] > 0
    return dict(similarity_invariance=True, drift_distinguishes_first2_and_full=True,
                joints=22, fps=20, segment=100, partial_tail='excluded from aligned100 metrics')
