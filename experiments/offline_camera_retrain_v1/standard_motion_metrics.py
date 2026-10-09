"""Common 22-joint metrics. Positions in metres; fixed native20 clock."""
import numpy as np


def standard_motion_metrics(pred, truth, fps=20.0):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != 3 or len(pred) < 3:
        raise ValueError('Expected matched [T>=3,J,3] arrays')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Nonfinite positions or invalid frame rate')
    x = pred - pred.mean(axis=1, keepdims=True)
    y = truth - truth.mean(axis=1, keepdims=True)
    u, singular, vh = np.linalg.svd(np.einsum('tji,tjk->tik', x, y))
    d = np.ones_like(singular)
    d[:, -1] = np.linalg.det(u @ vh)
    rotation = (u * d[:, None, :]) @ vh
    scale = (singular * d).sum(axis=1) / np.maximum((x*x).sum(axis=(1, 2)), 1e-12)
    aligned = scale[:, None, None] * (x @ rotation) + truth.mean(axis=1, keepdims=True)
    error = pred - truth
    relative = (pred-pred[:, :1]) - (truth-truth[:, :1])
    return {
        'w_mpjpe_mm': float(np.linalg.norm(error, axis=-1).mean()*1000),
        'root_relative_mpjpe_mm': float(np.linalg.norm(relative, axis=-1).mean()*1000),
        'pa_mpjpe_mm': float(np.linalg.norm(aligned-truth, axis=-1).mean()*1000),
        'root_position_error_mm': float(np.linalg.norm(error[:, 0], axis=-1).mean()*1000),
        'mpjve_mm_s': float(np.linalg.norm(np.diff(error, axis=0)*fps, axis=-1).mean()*1000),
        'acceleration_error_mm_s2': float(np.linalg.norm(np.diff(error, n=2, axis=0)*fps**2, axis=-1).mean()*1000),
    }


METRIC_PROTOCOL = {
    'joints': '22 common native SMPL/SMPL-X body joints, including pelvis; not a dataset official 14/17-joint regressor',
    'w_mpjpe_mm': 'World coordinates, no alignment; existing mpjpe_cm times 10',
    'root_relative_mpjpe_mm': 'Subtract each frame pelvis separately from prediction and GT; no rotation or scale alignment',
    'pa_mpjpe_mm': 'Per-frame proper rotation (no reflection), positive similarity scale and translation; all 22 joints',
    'root_position_error_mm': 'World pelvis translation error',
    'mpjve_mm_s': 'World joint first-difference velocity error at 20Hz, mm/s; no alignment',
    'acceleration_error_mm_s2': 'World joint second-difference acceleration error at 20Hz, mm/s^2; no alignment; not mm/frame^2',
    'temporal_boundaries': 'Only within each saved window, no differences across window boundaries; overlapping windows retained',
}
