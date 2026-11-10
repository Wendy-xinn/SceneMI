"""Sequence-level observed support patches and fit-free trajectory queries.

Extraction owns RANSAC, least squares and convex hulls. Querying owns only
bounded spatial lookup, plane evaluation and temporal candidate selection.
The input is the complete observed static map, never a scan or body labels.
All coordinates are metres in the public, gravity-aligned Y-up world frame.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


VERSION = 'sequence-support-surfaces-v1'
UNKNOWN, UNCERTAIN, RELIABLE = 0, 1, 2


def validate_policy(policy):
    """Require explicit public gravity and bounded extraction/query settings."""
    if policy.get('version') != VERSION:
        raise ValueError('Unsupported support-surface protocol')
    gravity = policy['gravity']
    if (gravity['source'] != 'public_calibrated_world'
            or not np.array_equal(gravity['up_world'], [0., 1., 0.])):
        raise ValueError('Support input must have public calibrated Y-up gravity')
    extraction, query = policy['extraction'], policy['query']
    positive = ('radius_m', 'stride_m', 'voxel_m', 'height_bin_m',
                'minimum_span_m', 'ransac_distance_m', 'maximum_fit_error_m')
    if any(not math.isfinite(extraction[k]) or extraction[k] <= 0 for k in positive):
        raise ValueError('Support extraction distances must be finite and positive')
    if (not extraction['voxel_m'] <= extraction['stride_m'] <= extraction['radius_m']
            or not 0 < extraction['maximum_slope_deg'] <= 25
            or not 0 < extraction['normal_agreement_deg'] < 90):
        raise ValueError('Invalid support coverage or normal policy')
    for key in ('minimum_points', 'ransac_iterations', 'maximum_layers_per_tile'):
        if type(extraction[key]) is not int or extraction[key] < 3:
            raise ValueError('Invalid bounded support extraction budget')
    for key in ('horizontal_radius_m', 'reliable_distance_m', 'uncertain_distance_m',
                'maximum_step_above_anchor_m', 'maximum_height_difference_m',
                'layer_separation_m', 'ambiguity_margin_m'):
        if not math.isfinite(query[key]) or query[key] <= 0:
            raise ValueError('Support query distances must be finite and positive')
    if (not query['reliable_distance_m'] < query['uncertain_distance_m'] <= query['horizontal_radius_m']
            or not 0 < query['minimum_reliable_confidence'] <= 1
            or not 0 < query['minimum_reliable_coverage'] <= 1
            or not math.isfinite(query['temporal_weight']) or query['temporal_weight'] < 0
            or type(query['maximum_candidates']) is not int or query['maximum_candidates'] < 2):
        raise ValueError('Invalid support reliability or candidate policy')
    return policy


def _empty_arrays():
    return dict(coefficients=np.empty((0, 3), np.float64),
                normals=np.empty((0, 3), np.float32),
                centres=np.empty((0, 2), np.float64),
                quality=np.empty((0, 4), np.float32),
                fit_error=np.empty(0, np.float32),
                sample_offsets=np.zeros(1, np.int64),
                sample_ids=np.empty(0, np.int32),
                hull_offsets=np.zeros(1, np.int64),
                hull_equations=np.empty((0, 3), np.float64))


def extract_support_surfaces(index, policy, *, seed=0):
    """Fit bounded local planes once per immutable sequence map.

    Height components keep disconnected stair/seat levels separate. Open3D
    RANSAC rejects outliers; retained observations determine both the final
    least-squares plane and its bounded footprint. Holes remain holes because
    queries additionally check distance to actual inlier samples.
    """
    validate_policy(policy)
    p = policy['extraction']
    minimum_y = math.cos(math.radians(p['maximum_slope_deg']))
    eligible = np.flatnonzero(index.normal_valid & (index.normals[:, 1] >= minimum_y))
    result = _empty_arrays()
    if len(eligible) < p['minimum_points']:
        return result
    # Open3D is needed only by offline preparation, never by query workers.
    import open3d as o3d
    o3d.utility.random.seed(int(seed) % (2**31-1))
    points = np.asarray(index.points, np.float64)
    horizontal = points[eligible][:, [0, 2]]
    tree = cKDTree(horizontal)
    cells = np.unique(np.floor(horizontal / p['stride_m']).astype(np.int64), axis=0)
    centres = (cells + .5) * p['stride_m']
    arrays = {k: [] for k in result if k not in ('sample_offsets', 'hull_offsets')}
    sample_offsets, hull_offsets = [0], [0]
    agreement = math.cos(math.radians(p['normal_agreement_deg']))
    for centre in centres:
        ids = eligible[tree.query_ball_point(centre, p['radius_m'], return_sorted=True)]
        if len(ids) < p['minimum_points']:
            continue
        bins = np.floor(points[ids, 1] / p['height_bin_m']).astype(np.int64)
        levels = np.unique(bins)
        components = np.split(levels, np.flatnonzero(np.diff(levels) > 1)+1)
        for component in components:
            remaining = ids[(bins >= component[0]) & (bins <= component[-1])]
            for _ in range(p['maximum_layers_per_tile']):
                if len(remaining) < p['minimum_points']:
                    break
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining]))
                _, inliers = cloud.segment_plane(distance_threshold=p['ransac_distance_m'],
                    ransac_n=3, num_iterations=p['ransac_iterations'], probability=.999)
                if len(inliers) < p['minimum_points']:
                    break
                chosen = remaining[np.asarray(inliers, np.int64)]
                keep_remaining = np.ones(len(remaining), bool)
                keep_remaining[inliers] = False
                remaining = remaining[keep_remaining]
                patch = points[chosen]
                xz = patch[:, [0, 2]]
                if (np.ptp(xz, axis=0) < p['minimum_span_m']).any():
                    continue
                design = np.column_stack((xz, np.ones(len(patch))))
                coefficients = np.linalg.lstsq(design, patch[:, 1], rcond=None)[0]
                normal = np.array([-coefficients[0], 1., -coefficients[1]])
                normal /= np.linalg.norm(normal)
                good = ((np.abs(design @ coefficients - patch[:, 1]) <= p['ransac_distance_m'])
                        & (index.normals[chosen] @ normal >= agreement))
                if good.sum() < p['minimum_points']:
                    continue
                chosen, patch, xz, design = chosen[good], patch[good], xz[good], design[good]
                coefficients = np.linalg.lstsq(design, patch[:, 1], rcond=None)[0]
                normal = np.array([-coefficients[0], 1., -coefficients[1]])
                normal /= np.linalg.norm(normal)
                error = float(np.sqrt(np.mean((design @ coefficients - patch[:, 1])**2)))
                if (error > p['maximum_fit_error_m'] or normal[1] < minimum_y
                        or (np.ptp(xz, axis=0) < p['minimum_span_m']).any()):
                    continue
                try:
                    hull = ConvexHull(xz)
                except QhullError:
                    continue
                area = hull.volume
                if area < .25 * p['minimum_span_m']**2:
                    continue
                density = min(1., len(chosen) * p['voxel_m']**2 / max(area, 1e-12))
                consistency = float(np.clip(index.normals[chosen] @ normal, 0., 1.).mean())
                residual = math.exp(-(error/p['maximum_fit_error_m'])**2)
                spread = min(1., area/(2*p['minimum_span_m'])**2)
                arrays['coefficients'].append(coefficients)
                arrays['normals'].append(normal)
                arrays['centres'].append(centre)
                arrays['quality'].append([density, consistency, residual, spread])
                arrays['fit_error'].append(error)
                arrays['sample_ids'].append(chosen.astype(np.int32))
                arrays['hull_equations'].append(hull.equations)
                sample_offsets.append(sample_offsets[-1]+len(chosen))
                hull_offsets.append(hull_offsets[-1]+len(hull.equations))
    if not arrays['coefficients']:
        return result
    for name in ('coefficients', 'normals', 'centres', 'quality', 'fit_error'):
        result[name] = np.asarray(arrays[name], dtype=result[name].dtype)
    for name in ('sample_ids', 'hull_equations'):
        result[name] = np.concatenate(arrays[name]).astype(result[name].dtype)
    result['sample_offsets'] = np.asarray(sample_offsets, np.int64)
    result['hull_offsets'] = np.asarray(hull_offsets, np.int64)
    return result


def _trajectory_selection(cost, height, temporal_weight):
    """Viterbi on candidate heights, independently for each body anchor.

    An unknown frame breaks continuity. IDs may change between overlapping
    patches, so continuity is defined by height rather than a patch ID.
    """
    frames, anchors, count = cost.shape
    back = np.zeros((frames, anchors, count), np.int32)
    cumulative = cost[0].copy()
    best = np.zeros((frames, anchors), np.int32)
    best[0] = cumulative.argmin(-1)
    for t in range(1, frames):
        transition = cumulative[:, :, None] + temporal_weight * np.abs(
            height[t-1, :, :, None]-height[t, :, None, :])
        available = np.isfinite(cumulative).any(-1)
        transition[~available] = 0.
        back[t] = transition.argmin(1)
        cumulative = cost[t]+transition.min(1)
        best[t] = cumulative.argmin(-1)
        minimum = cumulative.min(-1, keepdims=True)
        np.subtract(cumulative, minimum, out=cumulative, where=np.isfinite(minimum))
    selection = np.zeros((frames, anchors), np.int32)
    selection[-1] = cumulative.argmin(-1)
    for t in range(frames-1, 0, -1):
        selection[t-1] = back[t, np.arange(anchors), selection[t]]
        # A missing current frame cannot backtrack an earlier valid segment.
        gap = ~np.isfinite(cost[t]).any(-1)
        selection[t-1, gap] = best[t-1, gap]
    return selection


@dataclass
class SupportSurfaceAtlas:
    """Immutable fitted patches; online queries never run geometry fitting."""
    arrays: dict
    points: np.ndarray
    policy: dict

    def __post_init__(self):
        validate_policy(self.policy)
        a = self.arrays
        count = len(a['coefficients'])
        if (a['coefficients'].shape != (count, 3) or a['normals'].shape != (count, 3)
                or a['quality'].shape != (count, 4) or a['centres'].shape != (count, 2)
                or a['sample_offsets'].shape != (count+1,)
                or a['hull_offsets'].shape != (count+1,)
                or a['sample_offsets'][0] != 0 or a['hull_offsets'][0] != 0
                or a['sample_offsets'][-1] != len(a['sample_ids'])
                or a['hull_offsets'][-1] != len(a['hull_equations'])
                or (a['sample_ids'] < 0).any() or (a['sample_ids'] >= len(self.points)).any()
                or any(not np.isfinite(v).all() for v in a.values())):
            raise ValueError('Invalid support patch arrays')
        self.centres = cKDTree(a['centres'])
        self.samples = []
        for lo, hi in zip(a['sample_offsets'][:-1], a['sample_offsets'][1:]):
            if hi <= lo:
                raise ValueError('Support patch has no measured samples')
            self.samples.append(cKDTree(self.points[a['sample_ids'][lo:hi]][:, [0, 2]]))

    def query(self, anchors):
        anchors = np.asarray(anchors, np.float32)
        if anchors.ndim not in (2, 3) or anchors.shape[-1] != 3 or not np.isfinite(anchors).all():
            raise ValueError('Support queries expect finite Q,3 or T,A,3 anchors')
        shape = anchors.shape[:-1]
        flat = anchors.reshape(-1, 3)
        q, a = self.policy['query'], self.arrays
        count, size = q['maximum_candidates'], len(flat)
        costs = np.full((size, count), np.inf)
        heights = np.zeros((size, count), np.float32)
        ids = np.full((size, count), -1, np.int32)
        confidences = np.zeros((size, count), np.float32)
        distances = np.zeros((size, count), np.float32)
        reliable = np.zeros((size, count), bool)
        # Build sparse proposals in patch groups, batching all frames/anchors.
        pools = self.centres.query_ball_point(flat[:, [0, 2]],
            self.policy['extraction']['radius_m']+q['horizontal_radius_m'], workers=1)
        owner = np.repeat(np.arange(size), [len(p) for p in pools])
        patch = np.concatenate(pools).astype(np.int32) if len(owner) else np.empty(0, np.int32)
        proposal_height = np.zeros(len(owner))
        proposal_confidence = np.zeros(len(owner))
        proposal_distance = np.full(len(owner), np.inf)
        proposal_reliable = np.zeros(len(owner), bool)
        proposal_cost = np.full(len(owner), np.inf)
        for patch_id in np.unique(patch):
            at = np.flatnonzero(patch == patch_id)
            p = flat[owner[at]][:, [0, 2]]
            height = np.column_stack((p, np.ones(len(p)))) @ a['coefficients'][patch_id]
            vertical = flat[owner[at], 1]-height
            distance, neighbours = self.samples[patch_id].query(p, k=16, workers=1)
            valid_neighbours = np.isfinite(distance) & (distance <= q['reliable_distance_m'])
            safe = np.minimum(neighbours, len(self.samples[patch_id].data)-1)
            relative = self.samples[patch_id].data[safe]-p[:, None]
            quadrant = (relative >= 0).astype(np.int32) @ np.array([1, 2])
            coverage = np.stack([(valid_neighbours & (quadrant == i)).any(-1)
                                 for i in range(4)], -1).mean(-1)
            nearest = distance[:, 0]
            lo, hi = a['hull_offsets'][patch_id:patch_id+2]
            equations = a['hull_equations'][lo:hi]
            inside = (p @ equations[:, :2].T+equations[:, 2]).max(-1) <= 1e-6
            quality = float(np.prod(a['quality'][patch_id])**.5)
            confidence = quality * np.exp(-nearest/q['uncertain_distance_m']) * (.5+.5*coverage)
            confidence *= np.where(inside, 1., .25)
            observed = ((nearest <= q['uncertain_distance_m'])
                        & (vertical >= -q['maximum_step_above_anchor_m'])
                        & (np.abs(vertical) <= q['maximum_height_difference_m']))
            trusted = (inside & (nearest <= q['reliable_distance_m'])
                       & (coverage >= q['minimum_reliable_coverage'])
                       & (confidence >= q['minimum_reliable_confidence']))
            proposal_height[at] = height
            proposal_confidence[at] = confidence
            proposal_distance[at] = nearest
            proposal_reliable[at] = trusted
            proposal_cost[at] = np.where(observed,
                np.abs(vertical)+.2*nearest+.1*(1-confidence), np.inf)
        available = np.isfinite(proposal_cost)
        for k in range(count):
            at = np.flatnonzero(available)
            if not len(at):
                break
            order = at[np.lexsort((patch[at], proposal_cost[at], owner[at]))]
            _, first = np.unique(owner[order], return_index=True)
            chosen = order[first]
            rows = owner[chosen]
            costs[rows, k] = proposal_cost[chosen]
            heights[rows, k] = proposal_height[chosen]
            ids[rows, k] = patch[chosen]
            confidences[rows, k] = proposal_confidence[chosen]
            distances[rows, k] = proposal_distance[chosen]
            reliable[rows, k] = proposal_reliable[chosen]
            # Overlapping tiles describing one height are one candidate layer.
            selected_height = np.full(size, np.inf)
            selected_height[rows] = proposal_height[chosen]
            available &= np.abs(proposal_height-selected_height[owner]) >= q['layer_separation_m']
        selection = (_trajectory_selection(costs.reshape(*shape, count),
                     heights.reshape(*shape, count), q['temporal_weight']).reshape(-1)
                     if anchors.ndim == 3 and len(anchors) else costs.argmin(-1))
        rows = np.arange(size)
        selected_ids = ids[rows, selection]
        valid = selected_ids >= 0
        selected_height = heights[rows, selection]
        other_layer = np.abs(heights-selected_height[:, None]) >= q['layer_separation_m']
        alternatives = np.where(other_layer, costs, np.inf).min(-1)
        ambiguous = valid & (alternatives <= costs[rows, selection]+q['ambiguity_margin_m'])
        confidence = confidences[rows, selection]*np.where(ambiguous, .25, 1.)
        trusted = reliable[rows, selection] & valid & ~ambiguous
        normal = np.zeros((size, 3), np.float32)
        error = np.zeros(size, np.float32)
        point_count = np.zeros(size, np.int32)
        normal[valid] = a['normals'][selected_ids[valid]]
        error[valid] = a['fit_error'][selected_ids[valid]]
        point_count[valid] = np.diff(a['sample_offsets'])[selected_ids[valid]]
        result = dict(height=selected_height, normal=normal, confidence=confidence.astype(np.float32),
            valid=valid, reliable=trusted, state=np.where(trusted, RELIABLE,
                np.where(valid, UNCERTAIN, UNKNOWN)).astype(np.int8),
            fit_error=error, nearest_horizontal=distances[rows, selection],
            ambiguous=ambiguous, point_count=point_count, patch_id=selected_ids,
            candidate_height=heights, candidate_valid=ids >= 0,
            candidate_confidence=confidences, candidate_reliable=reliable,
            candidate_patch_id=ids)
        return {k: v.reshape(*shape, *v.shape[1:]) for k, v in result.items()}
