"""CPU evaluation of head following, body tilt, and foot support proxies.

Inputs use the native30 anchored frame (+Y up, metres after position * 2).
GT is used only in this evaluator. Slow GT foot joints are a *proxy*, not
measured contact. Joint heights are never described as sole heights; optional
native skin anchors provide a separate sparse surface-height measurement.
No scene/floor height or dynamics is inferred here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _motion(value: Any, name: str) -> torch.Tensor:
    array = _array(value)
    if array.ndim != 2 or array.shape[1] != 201 or len(array) < 3:
        raise ValueError(f'{name} must be [T>=3,201]')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains nonfinite values')
    return torch.as_tensor(np.array(array, dtype=np.float32, copy=True))[None]


def _mean(value: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    values = value if mask is None else value[mask]
    return float(np.mean(values)) if values.size else None


def _angle_to_up(vector: np.ndarray) -> np.ndarray:
    unit = vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)
    return np.rad2deg(np.arccos(np.clip(unit[..., 1], -1., 1.)))


def _angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.rad2deg(np.arccos(np.clip((first * second).sum(-1), -1., 1.)))


def support_metrics(
    prediction: Any,
    truth: Any,
    rest: Any,
    observed: Any | None = None,
    meta: Any | None = None,
    asset: dict[str, Any] | None = None,
    *,
    fps: float = 30.0,
) -> dict[str, float | int | None]:
    """Return JSON-safe flat metrics for one matched, unaligned motion window.

    ``observed`` is [T,22,D>=3] native trajectory (positions normalized /2),
    ``meta`` is [T,22,D>=1] observation metadata. Head input errors use
    availability (channel 0) and positive confidence (channel 2, if present).
    ``asset`` may be a Native30SkinAnchorCache entry; all tensor data are moved
    to CPU. Sparse-anchor minima need not equal a complete native mesh sole.

    Native ``raw_metrics`` is fixed at 30 Hz, so this evaluator rejects other
    rates rather than silently mixing velocity conventions.
    """
    if not np.isfinite(fps) or fps != 30.0:
        raise ValueError('Native30 raw metrics require fps=30.0')
    p, g = _motion(prediction, 'prediction'), _motion(truth, 'truth')
    if p.shape != g.shape:
        raise ValueError('Prediction and truth shapes differ')
    r = _array(rest)
    if r.shape != (22, 3) or not np.isfinite(r).all():
        raise ValueError('rest must be finite [22,3]')
    r = torch.as_tensor(np.array(r, dtype=np.float32, copy=True))[None]
    batch = {'motion': g, 'rest': r}
    frames = p.shape[1]
    if meta is None:
        if observed is not None:
            raise ValueError('Observed trajectory needs availability metadata')
        m = None
        conditioned = torch.zeros((1, frames, 22), dtype=torch.bool)
    else:
        m = _array(meta)
        if m.ndim != 3 or m.shape[:2] != (frames, 22) or m.shape[2] < 1:
            raise ValueError('meta must be [T,22,D>=1]')
        if not np.isfinite(m).all():
            raise ValueError('Nonfinite observation metadata')
        conditioned = torch.as_tensor(m[None, ..., 0] > 0)
    with torch.no_grad():
        jp = forward_kinematics(p, r)[0].numpy().astype(np.float64)
        jg = forward_kinematics(g, r)[0].numpy().astype(np.float64)
        raw = raw_metrics(p, batch, conditioned)
        result = {key: value['value'] for key, value in raw.items()}
        result.update(standard_motion_metrics(jp, jg, fps=fps))
        up_p = rotation_from_6d(p[:, :, 3:9])[0, :, :, 1].numpy().astype(np.float64)
        up_g = rotation_from_6d(g[:, :, 3:9])[0, :, :, 1].numpy().astype(np.float64)

    result['gt_head_position_error_mm'] = _mean(np.linalg.norm(jp[:, 15] - jg[:, 15], axis=-1) * 1000.)
    result['observed_head_position_error_mm'] = None
    result['observed_head_position_error_p95_mm'] = None
    result['observed_head_position_valid_frames'] = 0
    if observed is not None:
        o = _array(observed)
        if o.ndim != 3 or o.shape[:2] != (frames, 22) or o.shape[2] < 3:
            raise ValueError('observed must be [T,22,D>=3]')
        valid = m[:, 15, 0] > 0
        if m.shape[2] > 2:
            valid &= m[:, 15, 2] > 0
        if not np.isfinite(o[valid, 15, :3]).all():
            raise ValueError('Nonfinite valid head input positions')
        head_error = np.linalg.norm(jp[valid, 15] - o[valid, 15, :3] * 2., axis=-1) * 1000.
        result['observed_head_position_valid_frames'] = int(valid.sum())
        result['observed_head_position_error_mm'] = _mean(head_error)
        result['observed_head_position_error_p95_mm'] = float(np.percentile(head_error, 95)) if head_error.size else None

    tilt_p, tilt_g = _angle_to_up(up_p), _angle_to_up(up_g)
    result.update(
        root_tilt_deg=_mean(tilt_p),
        gt_root_tilt_deg=_mean(tilt_g),
        root_tilt_abs_error_deg=_mean(np.abs(tilt_p - tilt_g)),
        root_up_direction_error_deg=_mean(_angle_between(up_p, up_g)),
    )
    torso_p, torso_g = jp[:, 12] - jp[:, 0], jg[:, 12] - jg[:, 0]
    torso_tilt_p, torso_tilt_g = _angle_to_up(torso_p), _angle_to_up(torso_g)
    result.update(
        torso_pelvis_neck_tilt_deg=_mean(torso_tilt_p),
        gt_torso_pelvis_neck_tilt_deg=_mean(torso_tilt_g),
        torso_pelvis_neck_tilt_abs_error_deg=_mean(np.abs(torso_tilt_p - torso_tilt_g)),
        torso_pelvis_neck_direction_error_deg=_mean(_angle_between(torso_p, torso_g)),
    )

    feet = (10, 11)
    vp = np.diff(jp[:, feet], axis=0) * fps
    vg = np.diff(jg[:, feet], axis=0) * fps
    low = np.linalg.norm(vg, axis=-1) < .2
    y_error = jp[:, feet, 1] - jg[:, feet, 1]
    result.update(
        gt_low_speed_foot_joint_samples=int(low.sum()),
        gt_low_speed_foot_joint_fraction=float(low.mean()),
        foot_joint_horizontal_speed_gt_low_m_s=_mean(np.linalg.norm(vp[..., (0, 2)], axis=-1), low),
        gt_foot_joint_horizontal_speed_gt_low_m_s=_mean(np.linalg.norm(vg[..., (0, 2)], axis=-1), low),
        foot_joint_vertical_speed_gt_low_m_s=_mean(np.abs(vp[..., 1]), low),
        gt_foot_joint_vertical_speed_gt_low_m_s=_mean(np.abs(vg[..., 1]), low),
        foot_joint_horizontal_velocity_error_gt_low_m_s=_mean(np.linalg.norm((vp - vg)[..., (0, 2)], axis=-1), low),
        foot_joint_vertical_velocity_error_gt_low_m_s=_mean(np.abs((vp - vg)[..., 1]), low),
        foot_joint_y_signed_error_mm=_mean(y_error * 1000.),
        foot_joint_y_abs_error_mm=_mean(np.abs(y_error) * 1000.),
        foot_joint_y_signed_error_gt_low_mm=_mean(y_error[1:] * 1000., low),
        foot_joint_y_abs_error_gt_low_mm=_mean(np.abs(y_error[1:]) * 1000., low),
    )
    for side, column in (('left', 0), ('right', 1)):
        result[f'{side}_foot_joint_y_signed_error_mm'] = _mean(y_error[:, column] * 1000.)
        result[f'{side}_foot_joint_y_abs_error_mm'] = _mean(np.abs(y_error[:, column]) * 1000.)
        result[f'{side}_foot_joint_horizontal_speed_gt_low_m_s'] = _mean(np.linalg.norm(vp[:, column, (0, 2)], axis=-1), low[:, column])

    if asset is not None:
        # Import the native skinning path only when an asset was supplied.
        from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
        cpu_asset = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in asset.items()}
        labels = _array(cpu_asset['labels'])
        with torch.no_grad():
            skin_p = skin_points(p[0], cpu_asset).numpy()
            skin_g = skin_points(g[0], cpu_asset).numpy()
        both_errors = []
        for side, regions, column in (('left', (7, 10), 0), ('right', (8, 11), 1)):
            selected = np.isin(labels, regions)
            if not selected.any():
                raise ValueError(f'No {side} foot/ankle skin anchors in asset')
            difference = skin_p[:, selected, 1].min(1) - skin_g[:, selected, 1].min(1)
            both_errors.append(difference)
            result[f'{side}_foot_skin_anchor_low_y_signed_error_mm'] = _mean(difference * 1000.)
            result[f'{side}_foot_skin_anchor_low_y_abs_error_mm'] = _mean(np.abs(difference) * 1000.)
            result[f'{side}_foot_skin_anchor_low_y_signed_error_gt_low_mm'] = _mean(difference[1:] * 1000., low[:, column])
        difference = np.stack(both_errors, axis=-1)
        result['foot_skin_anchor_low_y_signed_error_mm'] = _mean(difference * 1000.)
        result['foot_skin_anchor_low_y_abs_error_mm'] = _mean(np.abs(difference) * 1000.)
        result['foot_skin_anchor_low_y_abs_error_gt_low_mm'] = _mean(np.abs(difference[1:]) * 1000., low)
    return result
