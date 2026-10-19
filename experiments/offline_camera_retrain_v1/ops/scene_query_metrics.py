"""Read-only diagnostics for observed scene queries during denoising.

All positions are metres in one fixed coordinate frame and times are adjacent
frames in a single motion window.  The optional ground truth is used only here
for post-hoc errors and low-speed masks; it must never be supplied to a model
query.  A nearest returned surface sample is not a tracked material point.
Its displacement (including the centre-compensated residual) is descriptive
and must not be called a contact switch, penetration, or a tracking error.

Only NumPy is required.  Callers collect detached CPU arrays; this module does
not load models, run FK, query geometry, or change predictions.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import numpy as np


SCHEMA = 'scene-query-diagnostics-v1'
ALPHA_BAR_EDGES = (0., .1, .5, .9, 1.)


def _array(value, *, shape=None, name, dtype=None):
    value = np.asarray(value, dtype=dtype)
    if shape is not None and value.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {value.shape}')
    return value


def _finite_valid(value, valid, name):
    if not np.isfinite(value[valid]).all():
        raise ValueError(f'{name} must be finite wherever its mask is true')


def _summary(value, valid):
    """Finite observations only; empty metrics retain count zero and None."""
    selected = np.asarray(value, dtype=np.float64)[valid]
    if not np.isfinite(selected).all():
        raise ValueError('Metric has non-finite values at valid observations')
    count = int(selected.size)
    return dict(count=count, mean=None if not count else float(selected.mean()),
                p50=None if not count else float(np.quantile(selected, .5)),
                p90=None if not count else float(np.quantile(selected, .9)),
                min=None if not count else float(selected.min()),
                max=None if not count else float(selected.max()))


def _fraction(condition, eligible):
    denominator = int(np.count_nonzero(eligible))
    numerator = int(np.count_nonzero(np.asarray(condition, dtype=bool) & eligible))
    return dict(numerator=numerator, denominator=denominator,
                fraction=None if not denominator else numerator / denominator)


def alpha_bar_bin(alpha_bar, edges=ALPHA_BAR_EDGES):
    """Return a fixed cumulative-alpha bin; low alpha means high noise."""
    alpha_bar = float(alpha_bar)
    edges = tuple(float(edge) for edge in edges)
    if (len(edges) < 2 or edges[0] != 0. or edges[-1] != 1.
            or any(not math.isfinite(edge) for edge in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError('Alpha-bin edges must increase strictly from zero to one')
    if not math.isfinite(alpha_bar) or not 0. <= alpha_bar <= 1.:
        raise ValueError('alpha_bar must be finite and between zero and one')
    index = min(int(np.searchsorted(edges, alpha_bar, side='right')) - 1, len(edges) - 2)
    return dict(index=index, lower=edges[index], upper=edges[index + 1],
                upper_inclusive=index == len(edges) - 2,
                label=f'[{edges[index]:g},{edges[index + 1]:g}'
                      + (']' if index == len(edges) - 2 else ')'))


def summarize_query_step(*, timestep, alpha_bar, proposal_joints_m,
                         query_centres_m, surface_points_m, surface_valid,
                         surface_owners, surface_normal_reliable, fps=30.,
                         scale_index=2, neighbor_index=0, gt_joints_m=None,
                         frame_valid=None, joint_valid=None, query_valid=None,
                         foot_indices=(10, 11), gt_low_speed_threshold_m_s=.2,
                         feature_values=None, feature_valid=None,
                         window_id=None, query_kind='near_surface'):
    """Summarize one DDIM step, without retaining or mutating input arrays.

    Shapes: proposal/GT [T,J,3], query centres [T,A,3], surface points
    [T,A,S,K,3], [T,A,K,3], or selected [T,A,3], with surface masks/owners
    matching points[:-1].
    The default scale=2, neighbor=0 exactly matches native30's existing
    ``closest_points`` (.6 m radius, first returned neighbour).  This selects
    that returned slot; it does not recompute a nearest neighbour or assume
    rank zero valid because another slot is valid.  Normal reliability is a
    supplied boolean flag, not a calibrated continuous probability.

    ``joint_valid`` and ``query_valid`` have shapes [T,J]/[T,A].  Invalid
    padded values may be NaN; all values at valid positions must be finite.
    Temporal metrics use only adjacent frames with both endpoints valid.
    Optional features [T,A,F] must be for the same ordered query centres.
    """
    proposal = _array(proposal_joints_m, name='proposal_joints_m', dtype=np.float64)
    centres = _array(query_centres_m, name='query_centres_m', dtype=np.float64)
    if proposal.ndim != 3 or proposal.shape[-1] != 3 or not all(proposal.shape[:2]):
        raise ValueError('proposal_joints_m must be nonempty [T,J,3]')
    frames, joints = proposal.shape[:2]
    if centres.ndim != 3 or centres.shape[0] != frames or centres.shape[-1] != 3 or not centres.shape[1]:
        raise ValueError('query_centres_m must be nonempty [T,A,3]')
    anchors = centres.shape[1]
    fps = float(fps)
    threshold = float(gt_low_speed_threshold_m_s)
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('fps must be positive; GT low-speed threshold nonnegative')
    if isinstance(timestep, bool) or int(timestep) != timestep or timestep < 0:
        raise ValueError('timestep must be a nonnegative integer')
    stage = alpha_bar_bin(alpha_bar)
    fv = np.ones(frames, dtype=bool) if frame_valid is None else _array(
        frame_valid, shape=(frames,), name='frame_valid', dtype=bool)
    jv = np.ones((frames, joints), dtype=bool) if joint_valid is None else _array(
        joint_valid, shape=(frames, joints), name='joint_valid', dtype=bool)
    qv = np.ones((frames, anchors), dtype=bool) if query_valid is None else _array(
        query_valid, shape=(frames, anchors), name='query_valid', dtype=bool)
    jv = jv & fv[:, None]
    qv = qv & fv[:, None]
    _finite_valid(proposal, jv, 'proposal_joints_m')
    _finite_valid(centres, qv, 'query_centres_m')
    feet = tuple(foot_indices)
    if (not feet or len(set(feet)) != len(feet)
            or any(isinstance(index, bool) or int(index) != index or not 0 <= index < joints for index in feet)):
        raise ValueError('foot_indices must be unique, in-range joint indices')
    feet = tuple(int(index) for index in feet)
    points = _array(surface_points_m, name='surface_points_m', dtype=np.float64)
    if points.ndim not in (3, 4, 5) or points.shape[:2] != (frames, anchors) or points.shape[-1] != 3:
        raise ValueError('surface_points_m must be [T,A,S,K,3], [T,A,K,3], or [T,A,3]')
    original_point_shape = list(points.shape)
    if not all(points.shape[2:-1]):
        raise ValueError('Surface scale/neighbor axes must be nonempty')
    valid = _array(surface_valid, shape=points.shape[:-1], name='surface_valid', dtype=bool)
    owners = _array(surface_owners, shape=points.shape[:-1], name='surface_owners')
    reliable = _array(surface_normal_reliable, shape=points.shape[:-1], name='surface_normal_reliable', dtype=bool)
    if not np.issubdtype(owners.dtype, np.integer):
        raise ValueError('surface_owners must have integer dtype')
    directly_selected = points.ndim == 3
    if directly_selected:
        if neighbor_index != 0:
            raise ValueError('An already selected [T,A,3] point requires neighbor_index=0')
        points = points[:, :, None]
        valid, owners, reliable = valid[:, :, None], owners[:, :, None], reliable[:, :, None]
    eligible = np.broadcast_to(qv.reshape(*qv.shape, *((1,) * (valid.ndim - 2))), valid.shape)
    valid = valid & eligible
    _finite_valid(points, valid, 'surface_points_m')
    if points.ndim == 5:
        if isinstance(scale_index, bool) or int(scale_index) != scale_index or not 0 <= scale_index < points.shape[2]:
            raise ValueError('scale_index must select an existing surface scale')
        selected = (slice(None), slice(None), int(scale_index), int(neighbor_index))
        selected_scale = int(scale_index)
    else:
        selected = (slice(None), slice(None), int(neighbor_index))
        selected_scale = None
    if (isinstance(neighbor_index, bool) or int(neighbor_index) != neighbor_index
            or not 0 <= neighbor_index < points.shape[-2]):
        raise ValueError('neighbor_index must select an existing returned neighbor')
    nearest, nearest_valid = points[selected], valid[selected]
    nearest_owner, nearest_reliable = owners[selected], reliable[selected]
    # Zero invalid slots before arithmetic so invalid padding does not produce
    # NaN intermediates or warnings.  No caller-owned array is modified.
    safe_points = np.where(valid[..., None], points, 0.)
    safe_centres = np.where(qv[..., None], centres, 0.)
    safe_proposal = np.where(jv[..., None], proposal, 0.)
    expanded_centres = safe_centres.reshape(*qv.shape, *((1,) * (valid.ndim - 2)), 3)
    all_distance = np.linalg.norm(safe_points - expanded_centres, axis=-1)
    safe_nearest = np.where(nearest_valid[..., None], nearest, 0.)
    distance = np.linalg.norm(safe_nearest - safe_centres, axis=-1)
    jpairs = jv[1:] & jv[:-1]
    qpairs = qv[1:] & qv[:-1]
    ppairs = nearest_valid[1:] & nearest_valid[:-1] & qpairs
    joint_delta = np.diff(safe_proposal, axis=0)
    centre_delta = np.diff(safe_centres, axis=0)
    point_delta = np.diff(safe_nearest, axis=0)
    residual = np.linalg.norm(point_delta - centre_delta, axis=-1)
    same_owner = nearest_owner[1:] == nearest_owner[:-1]
    static_pair = same_owner & (nearest_owner[1:] == 0)
    metrics = dict(
        proposal_joint_speed_m_s=_summary(np.linalg.norm(joint_delta, axis=-1) * fps, jpairs),
        proposal_foot_speed_m_s=_summary(np.linalg.norm(joint_delta[:, feet], axis=-1) * fps, jpairs[:, feet]),
        proposal_foot_horizontal_speed_m_s=_summary(np.linalg.norm(joint_delta[:, feet][:, :, (0, 2)], axis=-1) * fps, jpairs[:, feet]),
        query_centre_speed_m_s=_summary(np.linalg.norm(centre_delta, axis=-1) * fps, qpairs),
        all_returned_neighbor_distance_m=_summary(all_distance, valid),
        selected_neighbor_distance_m=_summary(distance, nearest_valid),
        selected_point_displacement_m=_summary(np.linalg.norm(point_delta, axis=-1), ppairs),
        selected_point_speed_m_s=_summary(np.linalg.norm(point_delta, axis=-1) * fps, ppairs),
        selected_point_minus_centre_displacement_m=_summary(residual, ppairs),
        selected_point_minus_centre_displacement_same_owner_m=_summary(residual, ppairs & same_owner),
        selected_point_minus_centre_displacement_static_owner_m=_summary(residual, ppairs & static_pair),
    )
    fractions = dict(
        invalid_neighbor_slots=_fraction(~valid, eligible),
        invalid_selected_neighbor=_fraction(~nearest_valid, qv),
        reliable_normal_among_valid_neighbors=_fraction(reliable, valid),
        reliable_normal_among_valid_selected_neighbors=_fraction(nearest_reliable, nearest_valid),
        selected_owner_changed_adjacent_frames=_fraction(~same_owner, ppairs),
        selected_neighbor_validity_changed_adjacent_frames=_fraction(nearest_valid[1:] != nearest_valid[:-1], qpairs),
    )
    if gt_joints_m is not None:
        gt = _array(gt_joints_m, shape=proposal.shape, name='gt_joints_m', dtype=np.float64)
        _finite_valid(gt, jv, 'gt_joints_m')
        safe_gt = np.where(jv[..., None], gt, 0.)
        error = np.linalg.norm(safe_proposal - safe_gt, axis=-1)
        gt_velocity = np.linalg.norm(np.diff(safe_gt, axis=0), axis=-1) * fps
        low = jpairs[:, feet] & (gt_velocity[:, feet] < threshold)
        metrics.update(proposal_joint_error_m=_summary(error, jv),
                       proposal_foot_error_m=_summary(error[:, feet], jv[:, feet]),
                       proposal_foot_horizontal_speed_gt_low_m_s=_summary(
                           np.linalg.norm(joint_delta[:, feet][:, :, (0, 2)], axis=-1) * fps, low))
    feature_shape = None
    if feature_values is not None:
        features = _array(feature_values, name='feature_values', dtype=np.float64)
        if features.ndim != 3 or features.shape[:2] != qv.shape or not features.shape[-1]:
            raise ValueError('feature_values must be nonempty [T,A,F]')
        mask = qv if feature_valid is None else qv & _array(
            feature_valid, shape=qv.shape, name='feature_valid', dtype=bool)
        _finite_valid(features, mask, 'feature_values')
        safe_features = np.where(mask[..., None], features, 0.)
        feature_delta = np.linalg.norm(np.diff(safe_features, axis=0), axis=-1)
        metrics['feature_adjacent_frame_l2_difference'] = _summary(feature_delta, mask[1:] & mask[:-1])
        metrics['feature_adjacent_frame_rms_difference'] = _summary(
            feature_delta / math.sqrt(features.shape[-1]), mask[1:] & mask[:-1])
        feature_shape = list(features.shape)
    elif feature_valid is not None:
        raise ValueError('feature_valid requires feature_values')
    return dict(schema=SCHEMA, window_id=None if window_id is None else str(window_id),
                query_kind=str(query_kind), timestep=int(timestep), alpha_bar=float(alpha_bar),
                alpha_bar_bin=stage, fps=fps,
                shapes=dict(proposal=list(proposal.shape), query_centres=list(centres.shape),
                            surface_points=original_point_shape, features=feature_shape),
                selected_surface_slot=dict(scale_index=selected_scale,
                                           neighbor_index=None if directly_selected else int(neighbor_index),
                                           supplied_already_selected=directly_selected),
                foot_indices=list(feet), gt_low_speed_threshold_m_s=threshold,
                gt_used_posthoc_only=gt_joints_m is not None,
                counts=dict(valid_frames=int(fv.sum()), valid_joint_frames=int(jv.sum()),
                            valid_query_frames=int(qv.sum()), valid_adjacent_joint_pairs=int(jpairs.sum()),
                            valid_adjacent_query_pairs=int(qpairs.sum()),
                            valid_adjacent_selected_point_pairs=int(ppairs.sum())),
                metrics=metrics, fractions=fractions,
                interpretation=dict(
                    coordinates='metres in one fixed frame, Y up; axes 0/2 are horizontal',
                    query_points='returned surface samples, not tracked material identities',
                    displacement='adjacent motion frames within this DDIM step, not adjacent diffusion steps',
                    compensated='delta selected point minus delta query centre; a geometric descriptor, not a contact-switch detector',
                    reliable_normal='supplied boolean geometry flag, not calibrated continuous query reliability',
                    missing='zero-count metrics are None; invalid padding never contributes',
                    independence='DDIM steps and adjacent frames from one window are correlated; counts are opportunities, not independent trials'))


def aggregate_query_steps(steps: Sequence[Mapping], *, edges=ALPHA_BAR_EDGES):
    """Pool diagnostics by query kind and cumulative-alpha bin.

    Means use metric opportunity counts.  Window and step counts are separate;
    diffusion steps do not multiply the number of independent windows.  Pooled
    quantiles cannot be reconstructed from per-step summaries and are omitted.
    This aggregation is descriptive and does not compute significance tests.
    """
    groups = defaultdict(list)
    for row in steps:
        if row.get('schema') != SCHEMA:
            raise ValueError('Unexpected scene-query diagnostic schema')
        stage = alpha_bar_bin(row['alpha_bar'], edges)
        groups[(row['query_kind'], stage['index'])].append((row, stage))
    output = []
    for (kind, _), rows in sorted(groups.items()):
        pooled_metrics = {}
        for name in sorted({key for row, _ in rows for key in row['metrics']}):
            observed = [row['metrics'][name] for row, _ in rows if name in row['metrics']]
            count = sum(value['count'] for value in observed)
            nonempty = [value for value in observed if value['count']]
            pooled_metrics[name] = dict(count=count,
                mean=None if not count else sum(value['mean'] * value['count'] for value in nonempty) / count,
                min=None if not count else min(value['min'] for value in nonempty),
                max=None if not count else max(value['max'] for value in nonempty),
                contributing_steps=len(nonempty), quantiles='not reconstructed from summaries')
        pooled_fractions = {}
        for name in sorted({key for row, _ in rows for key in row['fractions']}):
            observed = [row['fractions'][name] for row, _ in rows if name in row['fractions']]
            denominator = sum(value['denominator'] for value in observed)
            numerator = sum(value['numerator'] for value in observed)
            pooled_fractions[name] = dict(numerator=numerator, denominator=denominator,
                fraction=None if not denominator else numerator / denominator)
        ids = {row['window_id'] for row, _ in rows if row['window_id'] is not None}
        output.append(dict(query_kind=kind, alpha_bar_bin=rows[0][1], steps=len(rows),
                           unique_named_windows=len(ids), unnamed_steps=sum(row['window_id'] is None for row, _ in rows),
                           metrics=pooled_metrics, fractions=pooled_fractions))
    return dict(schema=SCHEMA, alpha_bar_edges=list(edges), steps=len(steps), groups=output,
                interpretation='Opportunity-weighted descriptive means; DDIM steps are correlated, not independent training/window replicates. Quantiles and confidence intervals are not pooled.')
