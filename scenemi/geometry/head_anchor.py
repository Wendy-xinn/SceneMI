"""Translation-only head anchoring from public camera/body predictions.

Filters act on the correction, never on the generated motion or rotations.
The quadratic solve spans one prediction window, with free endpoints.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from scenemi.metrics_batch import metric_tensors


def camera_residual(motion, rest, mount, camera):
    joints = forward_kinematics(motion, rest)
    head_rotation = global_rotations(motion)[:, :, 15]
    predicted_camera = joints[:, :, 15] + (head_rotation @ mount[:, None, :, None])[..., 0]
    return camera[..., :3].double() * 2 - predicted_camera.double()


def correction_operator(method, frames, fps):
    """Return [T,T]; all physical derivatives include their fps factors."""
    eye = np.eye(frames, dtype=np.float64)
    kind = method['kind']
    if kind == 'raw':
        return np.zeros_like(eye)
    if kind == 'hard':
        return eye
    if kind == 'constant':
        return np.ones_like(eye) / frames
    if kind in ('gaussian', 'savgol'):
        window = method['window']
        if window < 3 or window % 2 != 1 or window > frames:
            raise ValueError('Smoothing window must be odd and within the prediction window')
        if kind == 'gaussian':
            value = gaussian_filter1d(eye, (window - 1) / 6, axis=0,
                radius=window // 2, mode='reflect')
        else:
            value = savgol_filter(eye, window, method['polyorder'], axis=0, mode='interp')
        strength = method['strength']
        if not 0 <= strength <= 1:
            raise ValueError('Correction strength must be in [0,1]')
        return value * strength
    if kind == 'quadratic':
        data, acc, jerk = [method[k] for k in ('data_weight', 'acceleration_weight', 'jerk_weight')]
        if data <= 0 or acc < 0 or jerk < 0:
            raise ValueError('Quadratic weights require positive data and nonnegative regularizers')
        d2, d3 = np.diff(eye, n=2, axis=0) * fps**2, np.diff(eye, n=3, axis=0) * fps**3
        normal = data * eye + acc * d2.T @ d2 + jerk * d3.T @ d3
        return np.linalg.solve(normal, data * eye)
    raise ValueError(f'Unknown correction: {kind}')


def translate_motion(motion, rest, correction):
    if correction.shape != (*motion.shape[:2], 3) or not torch.isfinite(correction).all():
        raise ValueError('Expected finite [B,T,3] corrections in meters')
    result = motion.clone()
    result[..., :3] += correction.to(motion) / 2
    result[..., 135:] = forward_kinematics(result, rest).flatten(2) / 2
    if not torch.equal(result[..., 3:135], motion[..., 3:135]):
        raise AssertionError('Anchoring changed a predicted rotation')
    return result


def translation_grid_metrics(prediction, target, residual, operators, *, fps=30):
    """Exact translation-dependent metrics; reuse translation-invariant SVDs.

    Returns metrics [methods,batch], without moving any target into anchoring.
    """
    pred, gt = prediction.double(), target.double()
    base = metric_tensors(pred, gt, fps=fps)
    correction = torch.einsum('stq,bqc->sbtc', operators, residual.double())
    shifted = pred[None] + correction[..., None, :]
    error = (shifted - gt[None]).norm(dim=-1)
    relative = (pred - pred[:, :, 15:16]) - (gt - gt[:, :, 15:16])
    count = len(operators)
    shared = lambda x: x[None].expand(count, -1)
    values = dict(
        mpjpe_mm=error.mean((-1, -2)) * 1000,
        root_error_mm=error[..., 0].mean(-1) * 1000,
        head_error_mm=error[..., 15].mean(-1) * 1000,
        pa_mpjpe_mm=shared(base['pa_mpjpe_mm']),
        root_relative_mpjpe_mm=shared(base['root_relative_mpjpe_mm']),
        head_relative_mpjpe_mm=shared(relative.norm(dim=-1).mean((-1, -2)) * 1000),
        mpjve_mm_s=(torch.diff(shifted, dim=2) - torch.diff(gt[None], dim=2)).norm(dim=-1).mean((-1, -2)) * fps * 1000,
        acceleration_error_m_s2=(torch.diff(shifted, n=2, dim=2) - torch.diff(gt[None], n=2, dim=2)).norm(dim=-1).mean((-1, -2)) * fps**2,
        jerk_m_s3=torch.diff(shifted, n=3, dim=2).norm(dim=-1).mean((-1, -2)) * fps**3,
        jerk_error_m_s3=(torch.diff(shifted, n=3, dim=2) - torch.diff(gt[None], n=3, dim=2)).norm(dim=-1).mean((-1, -2)) * fps**3,
        gt_jerk_m_s3=shared(base['gt_jerk_m_s3']),
        camera_position_residual_mm=(residual[None] - correction).norm(dim=-1).mean(-1) * 1000,
        correction_mean_mm=correction.norm(dim=-1).mean(-1) * 1000,
        correction_jerk_m_s3=torch.diff(correction, n=3, dim=2).norm(dim=-1).mean(-1) * fps**3,
    )
    return values
