"""Independent NumPy diagnostics for metric EXO placement and body scale.

All positions, including ``rest_m``, are metres in one calibrated frame. Native
trajectory channels must be multiplied by 2 before calling this module; FK
already returns metres. No transform returned here is applied to predictions.

Visible fits use only prediction/measurement correspondences and confidence.
GT fits are named ``diagnostic_gt_*`` and use ONE transform for the complete
window, never a separate pose/scale/rotation fit for every frame. A low aligned
error is a diagnostic, not a replacement for the raw metric reconstruction.
"""
from __future__ import annotations

from typing import Any

import numpy as np


PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19)


def _joints(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (22, 3) or not len(result):
        raise ValueError(f'{name} must be [T>=1,22,3], in metres')
    return result


def _mean(value: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    if not value.size:
        return None
    if weights is None:
        return float(value.mean())
    return float(np.sum(value * weights) / weights.sum()) if weights.sum() > 0 else None


def _weighted_median(value: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(value, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(value[order[np.searchsorted(cumulative, cumulative[-1] * .5)]])


def _solve(source: np.ndarray, target: np.ndarray, weights: np.ndarray,
           similarity: bool) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Weighted proper Kabsch/Umeyama fit, with row-vector convention."""
    total = weights.sum()
    if len(source) < 3 or total <= 0:
        return None
    cx = (source * weights[:, None]).sum(0) / total
    cy = (target * weights[:, None]).sum(0) / total
    x, y = source - cx, target - cy
    root_weights = np.sqrt(weights[:, None] / total)
    for cloud in (x, y):
        singular = np.linalg.svd(cloud * root_weights, compute_uv=False)
        # Planar bodies are allowed; collinear or coincident points do not
        # identify a 3-D rigid orientation. Do not invent one in that case.
        if singular[0] <= 1e-8 or singular[1] <= max(1e-8, singular[0] * 1e-6):
            return None
    u, singular, vt = np.linalg.svd((x * weights[:, None]).T @ y)
    correction = np.ones(3)
    correction[-1] = 1. if np.linalg.det(u @ vt) >= 0 else -1.
    rotation = (u * correction[None]) @ vt
    variance = float(np.sum(weights[:, None] * x * x))
    scale = float(np.dot(singular, correction) / variance) if similarity else 1.
    if not np.isfinite(scale) or scale <= 1e-8 or scale >= 1e8:
        return None
    translation = cy - scale * cx @ rotation
    return rotation, translation, scale


def _fit(source: np.ndarray, target: np.ndarray, weights: np.ndarray,
         *, similarity: bool, robust: bool) -> dict[str, Any]:
    count = len(source)
    invalid = dict(valid=False, samples=count, scale=None, scale_minus_one=None,
                   rotation_deg=None, translation_norm_mm=None,
                   weighted_residual_mm=None, weighted_rms_residual_mm=None,
                   residual_median_mm=None,
                   robust_inlier_fraction=None)
    working = weights.copy()
    solved = _solve(source, target, working, similarity)
    if solved is None:
        return invalid
    for _ in range(30 if robust else 1):
        rotation, translation, scale = solved
        error = np.linalg.norm(scale * source @ rotation + translation - target, axis=-1)
        if not robust:
            break
        # Huber IRLS floor of 2 cm avoids an unstable zero noise estimate on
        # exact synthetic measurements. Base confidence is always retained.
        threshold = max(.02, 1.5 * _weighted_median(error, weights))
        next_weights = weights * np.minimum(1., threshold / np.maximum(error, 1e-12))
        next_solved = _solve(source, target, next_weights, similarity)
        if next_solved is None:
            return invalid
        difference = max(float(np.max(np.abs(a - b))) for a, b in
                         zip(solved[:2], next_solved[:2]))
        difference = max(difference, abs(solved[2] - next_solved[2]))
        solved = next_solved
        if difference < 1e-10:
            break
    rotation, translation, scale = solved
    error = np.linalg.norm(scale * source @ rotation + translation - target, axis=-1)
    threshold = max(.02, 1.5 * _weighted_median(error, weights))
    return dict(valid=True, samples=count, scale=scale, scale_minus_one=scale - 1.,
                rotation_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1.) / 2., -1., 1.)))),
                translation_norm_mm=float(np.linalg.norm(translation) * 1000.),
                weighted_residual_mm=_mean(error * 1000., weights),
                weighted_rms_residual_mm=float(np.sqrt(np.sum(weights * error ** 2) / weights.sum()) * 1000.),
                residual_median_mm=_weighted_median(error * 1000., weights),
                robust_inlier_fraction=float(np.sum(weights[error <= threshold]) / weights.sum()))


def _put_fit(result: dict[str, Any], prefix: str, fit: dict[str, Any]) -> None:
    result.update({f'{prefix}_{key}': value for key, value in fit.items()})


def _bone_metrics(prediction: np.ndarray, observed: np.ndarray, mask: np.ndarray,
                  confidence: np.ndarray, rest: np.ndarray) -> dict[str, Any]:
    children = np.arange(1, 22)
    parents = np.asarray(PARENTS[1:])
    length = np.linalg.norm(rest[children] - rest[parents], axis=-1)
    valid_rest = length > 1e-6
    predicted = np.linalg.norm(prediction[:, children] - prediction[:, parents], axis=-1)
    visible = mask[:, children] & mask[:, parents] & valid_rest[None]
    pair_weights = np.minimum(confidence[:, children], confidence[:, parents])
    measured = np.linalg.norm(observed[:, children] - observed[:, parents], axis=-1)
    valid = visible & np.isfinite(measured) & (measured > 1e-6)
    ratio = measured[valid] / np.broadcast_to(length, measured.shape)[valid]
    weights = pair_weights[valid]
    result = dict(observed_bone_valid_pairs=int(valid.sum()),
                  observed_bone_distinct_edges=int(valid.any(0).sum()),
                  observed_bone_to_rest_ratio_median=None,
                  observed_bone_to_rest_ratio_abs_deviation=None,
                  observed_bone_to_prediction_ratio_median=None,
                  prediction_bone_to_rest_ratio_mean=_mean((predicted[:, valid_rest] / length[None, valid_rest]).ravel()),
                  prediction_bone_rest_abs_error_mm=_mean(np.abs(predicted[:, valid_rest] - length[None, valid_rest]).ravel() * 1000.))
    if ratio.size:
        result['observed_bone_to_rest_ratio_median'] = _weighted_median(ratio, weights)
        result['observed_bone_to_rest_ratio_abs_deviation'] = _weighted_median(np.abs(ratio - 1.), weights)
        eligible = valid & (predicted > 1e-6)
        if eligible.any():
            result['observed_bone_to_prediction_ratio_median'] = _weighted_median(
                measured[eligible] / predicted[eligible], pair_weights[eligible])
    return result


def visible_alignment_metrics(prediction_m: Any, observed_m: Any, available: Any,
                              confidence: Any, rest_m: Any, *,
                              min_confidence: float = .05) -> dict[str, Any]:
    """GT-free, read-only diagnostics; unavailable values never enter a fit.

    Confidence weights are used once, not squared. Positive available samples
    with confidence >= ``min_confidence`` must be finite. Missing positions may
    be zero/NaN. Fits pool the entire window and estimate a single proper rigid
    or similarity transform. Cloud scale can include trajectory error; adjacent
    bone/rest ratios independently test body-size or unit consistency.
    """
    prediction = _joints(prediction_m, 'prediction_m')
    observed = _joints(observed_m, 'observed_m')
    rest = np.asarray(rest_m, dtype=np.float64)
    mask = np.array(available, dtype=bool, copy=True)
    conf = np.asarray(confidence, dtype=np.float64)
    if observed.shape != prediction.shape or mask.shape != prediction.shape[:2] or conf.shape != mask.shape:
        raise ValueError('Joint positions, availability and confidence must share their window clock')
    if rest.shape != (22, 3) or not np.isfinite(rest).all() or not np.isfinite(prediction).all():
        raise ValueError('Prediction and rest must be finite, rest [22,3]')
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError('min_confidence must be in [0,1]')
    if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
        raise ValueError('Confidence must be finite in [0,1]')
    mask &= (conf >= min_confidence) & (conf > 0)
    if not np.isfinite(observed[mask]).all():
        raise ValueError('Nonfinite available measured position')
    # Replace unavailable NaNs only in the local temporary for bone arithmetic.
    observed = np.where(mask[..., None], observed, 0.)
    source, target, weights = prediction[mask], observed[mask], conf[mask]
    error = np.linalg.norm(source - target, axis=-1)
    translation = ((target - source) * weights[:, None]).sum(0) / weights.sum() if len(source) else None
    result = dict(observed_valid_samples=int(mask.sum()),
                  observed_valid_joints=int(mask.any(0).sum()),
                  observed_valid_frames=int(mask.any(1).sum()),
                  observed_raw_weighted_mpjpe_mm=_mean(error * 1000., weights),
                  observed_raw_weighted_rms_mm=None if not len(error) else float(np.sqrt(np.sum(weights * error ** 2) / weights.sum()) * 1000.),
                  observed_window_translation_norm_mm=None if translation is None else float(np.linalg.norm(translation) * 1000.),
                  observed_window_translation_aligned_weighted_mpjpe_mm=None if translation is None else _mean(
                      np.linalg.norm(source + translation - target, axis=-1) * 1000., weights))
    for kind, similarity in [('rigid', False), ('similarity', True)]:
        _put_fit(result, f'observed_window_robust_{kind}_fit',
                 _fit(source, target, weights, similarity=similarity, robust=True))
    result.update(_bone_metrics(prediction, observed, mask, conf, rest))
    return result


def _root_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orientation from rigid root children (hips and spine1), not pose GT."""
    side = joints[:, 2] - joints[:, 1]
    up = joints[:, 3] - joints[:, 0]
    side_norm = np.linalg.norm(side, axis=-1)
    x = side / np.maximum(side_norm[:, None], 1e-12)
    orthogonal = up - (up * x).sum(-1, keepdims=True) * x
    up_norm = np.linalg.norm(orthogonal, axis=-1)
    y = orthogonal / np.maximum(up_norm[:, None], 1e-12)
    return np.stack((x, y, np.cross(x, y)), axis=-1), (side_norm > 1e-6) & (up_norm > 1e-6)


def alignment_metrics(prediction_m: Any, truth_m: Any, observed_m: Any, available: Any,
                      confidence: Any, rest_m: Any, *,
                      min_confidence: float = .05) -> dict[str, Any]:
    """Flat JSON-safe metrics, with explicitly labelled GT-only diagnostics.

    ``diagnostic_gt_root_relative_mpjpe_mm`` subtracts each frame's root for
    articulation error only; it does not rotate/scale poses. All fitted GT
    transformations are constant across the complete window. The API returns
    numbers, never corrected trajectories, and never mutates its inputs.
    """
    prediction = _joints(prediction_m, 'prediction_m')
    truth = _joints(truth_m, 'truth_m')
    if prediction.shape != truth.shape or not np.isfinite(truth).all():
        raise ValueError('Truth must be finite and match prediction shape')
    result = visible_alignment_metrics(prediction, observed_m, available, confidence,
                                       rest_m, min_confidence=min_confidence)
    error = np.linalg.norm(prediction - truth, axis=-1)
    result['raw_gt_world_mpjpe_mm'] = _mean(error.ravel() * 1000.)
    result['raw_gt_world_rms_mm'] = float(np.sqrt(np.mean(error ** 2)) * 1000.)
    result['raw_gt_root_translation_error_mm'] = _mean(error[:, 0] * 1000.)
    local = (prediction - prediction[:, :1]) - (truth - truth[:, :1])
    result['diagnostic_gt_root_relative_mpjpe_mm'] = _mean(np.linalg.norm(local[:, 1:], axis=-1).ravel() * 1000.)
    pr, pv = _root_frame(prediction)
    gr, gv = _root_frame(truth)
    root_valid = pv & gv
    cosine = ((pr * gr).sum((1, 2)) - 1.) / 2.
    result['diagnostic_gt_root_orientation_valid_frames'] = int(root_valid.sum())
    result['diagnostic_gt_root_orientation_error_deg'] = _mean(np.rad2deg(np.arccos(np.clip(cosine[root_valid], -1., 1.))))
    source, target = prediction.reshape(-1, 3), truth.reshape(-1, 3)
    weights = np.ones(len(source), dtype=np.float64)
    translation = (target - source).mean(0)
    result['diagnostic_gt_window_translation_norm_mm'] = float(np.linalg.norm(translation) * 1000.)
    result['diagnostic_gt_window_translation_aligned_mpjpe_mm'] = _mean(np.linalg.norm(source + translation - target, axis=-1) * 1000.)
    for kind, similarity in [('rigid', False), ('similarity', True)]:
        _put_fit(result, f'diagnostic_gt_window_{kind}_fit',
                 _fit(source, target, weights, similarity=similarity, robust=False))
    return result
