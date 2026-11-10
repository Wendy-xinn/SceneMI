"""Full observed-map indices; fixed-size local queries and bounded support planes.

The CPU index stores every voxel. Only a small per-anchor result is copied to
the model. Empty observations remain unknown, never evidence of free space.
Coordinates are metric, gravity is +Y, and no GT motion is consulted.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError


@dataclass(frozen=True)
class SceneView:
    """A shared world map viewed in one camera-canonical window frame."""
    index: object
    origin: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        if np.asarray(self.origin).shape != (3,) or np.asarray(self.rotation).shape != (3, 3):
            raise ValueError('Expected a window origin and world-to-window rotation')
        np.testing.assert_allclose(self.rotation @ self.rotation.T, np.eye(3), atol=2e-5)
        if not np.isfinite(self.origin).all() or np.linalg.det(self.rotation) < .9999:
            raise ValueError('Scene views require a finite proper rigid transform')

    def to_world(self, points):
        return np.asarray(points) @ self.rotation + self.origin

    def to_local(self, points):
        return (np.asarray(points)-self.origin) @ self.rotation.T


class ObservedSceneIndex:
    def __init__(self, points, normals=None, normal_valid=None, observations=None):
        self.points = np.asarray(points, np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError('Map points must be finite metric N,3 coordinates')
        self.normals = np.zeros_like(self.points) if normals is None else np.asarray(normals, np.float32)
        self.normal_valid = np.zeros(len(self.points), bool) if normal_valid is None else np.asarray(normal_valid, bool)
        self.observations = np.ones(len(self.points), np.int32) if observations is None else np.asarray(observations, np.int32)
        if (self.normals.shape != self.points.shape or self.normal_valid.shape != (len(self.points),)
                or self.observations.shape != (len(self.points),) or (self.observations < 1).any()
                or not np.isfinite(self.normals).all()):
            raise ValueError('Map normals, validity and observation counts disagree')
        self.tree = cKDTree(self.points)
        self.horizontal = cKDTree(self.points[:, [0, 2]])

    def nearest(self, anchors):
        points = np.asarray(anchors)
        if not len(self.points):
            return np.full(points.shape[:-1], np.inf)
        return self.tree.query(points.reshape(-1, 3), workers=1)[0].reshape(points.shape[:-1])

    def query(self, anchors, radius, count=16):
        """Exact nearest K within radius; density counts all neighbours, not K."""
        anchors = np.asarray(anchors, np.float32)
        if (anchors.shape[-1] != 3 or not np.isfinite(anchors).all()
                or not np.isfinite(radius) or radius <= 0 or type(count) is not int or count < 1):
            raise ValueError('Expected finite anchors, positive metric radius and K')
        shape = anchors.shape[:-1]; flat = anchors.reshape(-1, 3)
        radius = float(np.float32(radius))
        distance, ids = self.tree.query(flat, k=list(range(1, count+1)),
            distance_upper_bound=np.nextafter(float(radius), np.inf), workers=1)
        valid = np.isfinite(distance) & (distance <= radius)
        density = self.tree.query_ball_point(flat, radius, return_length=True, workers=1)
        nearest = self.nearest(anchors)
        points = np.zeros((*flat.shape[:1], count, 3), np.float32)
        normals = np.zeros_like(points); nv = np.zeros(valid.shape, bool)
        observations = np.zeros(valid.shape, np.int32)
        if len(self.points):
            safe = np.minimum(ids, len(self.points)-1)
            points[valid] = self.points[safe[valid]]
            normals[valid] = self.normals[safe[valid]]
            nv[valid] = self.normal_valid[safe[valid]]
            observations[valid] = self.observations[safe[valid]]
        # Angular occupancy is a coverage descriptor, not a free-space claim.
        relative = points-flat[:, None]
        octants = (relative >= 0).astype(np.int32) @ np.array([1, 2, 4])
        coverage = np.stack([(valid & (octants == i)).any(-1) for i in range(8)], -1).mean(-1)
        return dict(points=points.reshape(*shape, count, 3), normals=normals.reshape(*shape, count, 3),
            normal_valid=nv.reshape(*shape, count), valid=valid.reshape(*shape, count),
            observations=observations.reshape(*shape, count),
            distance=np.where(valid, distance, 0.).astype(np.float32).reshape(*shape, count),
            nearest_distance=nearest.astype(np.float32), neighbour_count=density.reshape(shape),
            angular_coverage=coverage.astype(np.float32).reshape(shape))

    def support_planes(self, anchors, policy):
        """Observed horizontal patches with limited interpolation and ambiguity.

        Each height component is fitted separately, so stair levels cannot
        turn into one invented ramp. A query outside the observed convex hull
        or far from its samples is invalid. Multiple plausible levels lower
        confidence; the output is a support candidate, not a floor class.
        """
        anchors = np.asarray(anchors, np.float32)
        flat = anchors.reshape(-1, 3)
        size = len(flat)
        result = dict(height=np.zeros(size, np.float32), confidence=np.zeros(size, np.float32),
            valid=np.zeros(size, bool), normal=np.zeros((size, 3), np.float32),
            fit_error=np.zeros(size, np.float32), nearest_horizontal=np.zeros(size, np.float32),
            ambiguous=np.zeros(size, bool), point_count=np.zeros(size, np.int32))
        pools = self.horizontal.query_ball_point(flat[:, [0, 2]], policy['search_radius_m'], workers=1)
        for i, (anchor, ids) in enumerate(zip(flat, pools)):
            ids = np.asarray(ids, np.int64)
            ids = ids[self.normal_valid[ids] & (self.normals[ids, 1] >= policy['minimum_normal_y'])]
            if len(ids) < policy['minimum_points']:
                continue
            points = self.points[ids]
            height = points[:, 1]
            allowed = ((height-anchor[1] <= policy['maximum_step_above_anchor_m']) &
                       (np.abs(height-anchor[1]) <= policy['maximum_height_difference_m']))
            points = points[allowed]
            if len(points) < policy['minimum_points']:
                continue
            bins = np.floor(points[:, 1]/policy['height_bin_m']).astype(np.int64)
            levels = np.unique(bins)
            components = np.split(levels, np.flatnonzero(np.diff(levels) > 1)+1)
            candidates = []
            for component in components:
                patch = points[(bins >= component[0]) & (bins <= component[-1])]
                xz = patch[:, [0, 2]]
                if (len(patch) < policy['minimum_points'] or
                        (np.ptp(xz, axis=0) < policy['minimum_horizontal_span_m']).any()):
                    continue
                try:
                    hull = ConvexHull(xz)
                except QhullError:
                    continue
                # A convex envelope alone can bridge a large unobserved hole.
                distance = float(np.linalg.norm(xz-anchor[[0, 2]], axis=-1).min())
                inside = np.max(hull.equations[:, :2] @ anchor[[0, 2]]+hull.equations[:, 2]) <= 1e-6
                if not inside or distance > policy['max_extrapolation_m']:
                    continue
                design = np.column_stack((xz, np.ones(len(patch))))
                coefficients = np.linalg.lstsq(design, patch[:, 1], rcond=None)[0]
                residual = np.abs(design @ coefficients-patch[:, 1])
                # Refit after trimming isolated outliers; retain a measured bound.
                keep = residual <= policy['maximum_fit_error_m']*2
                if keep.sum() < policy['minimum_points']:
                    continue
                coefficients = np.linalg.lstsq(design[keep], patch[keep, 1], rcond=None)[0]
                try:
                    retained_hull = ConvexHull(xz[keep])
                except QhullError:
                    continue
                if (np.max(retained_hull.equations[:, :2] @ anchor[[0, 2]]+
                           retained_hull.equations[:, 2]) > 1e-6 or
                        np.linalg.norm(xz[keep]-anchor[[0, 2]], axis=-1).min() > policy['max_extrapolation_m']):
                    continue
                error = float(np.sqrt(np.mean((design[keep] @ coefficients-patch[keep, 1])**2)))
                normal = np.array([-coefficients[0], 1., -coefficients[1]])
                normal /= np.linalg.norm(normal)
                y = float(np.r_[anchor[[0, 2]], 1.] @ coefficients)
                if error > policy['maximum_fit_error_m'] or normal[1] < policy['minimum_normal_y']:
                    continue
                candidates.append((abs(y-anchor[1]), y, normal, error, distance, int(keep.sum())))
            if not candidates:
                continue
            candidates.sort(key=lambda row: row[0])
            _, y, normal, error, distance, count = candidates[0]
            ambiguous = any(abs(other[1]-y) > policy['ambiguity_height_m'] for other in candidates[1:])
            confidence = (min(1., count/(2*policy['minimum_points'])) *
                np.exp(-error/policy['maximum_fit_error_m']) *
                np.exp(-distance/policy['max_extrapolation_m']) * (.25 if ambiguous else 1.))
            result['height'][i], result['normal'][i] = y, normal
            result['fit_error'][i], result['nearest_horizontal'][i] = error, distance
            result['confidence'][i], result['valid'][i] = confidence, not ambiguous
            result['ambiguous'][i], result['point_count'][i] = ambiguous, count
        return {k: v.reshape(*anchors.shape[:-1], *v.shape[1:]) for k, v in result.items()}


def query_view(view, anchors, policy, *, ground_mask=None, dynamic=None):
    """Query a window view once after B; preserve frame and body-part axes.

    Dynamic geometry is a tuple of current-frame points, owners, normals and
    normal-valid masks. It is never inserted into the static world index.
    """
    anchors = np.asarray(anchors, np.float32)
    if anchors.ndim != 3 or anchors.shape[-1] != 3:
        raise ValueError('Expected T,A,3 skin anchors')
    world = view.to_world(anchors)
    result = {}
    count = policy['query']['neighbours_per_scale']
    for name, radius in (('near', policy['query']['near_radius_m']), ('context', policy['query']['context_radius_m'])):
        query = view.index.query(world, radius, count)
        query['points'] = view.to_local(query['points']).astype(np.float32)
        query['points'][~query['valid']] = 0.
        query['normals'] = (query['normals'] @ view.rotation.T).astype(np.float32)
        query['owner'] = np.zeros_like(query['observations'])
        if dynamic is not None:
            points, owners, normals, normal_valid = dynamic
            if len(points) != len(anchors):
                raise ValueError('Dynamic geometry and skin anchors have different clocks')
            for t in range(len(anchors)):
                active = (owners[t] > 0) & (owners[t] < 100)
                if not active.any():
                    continue
                # The existing rigid-object convention is retained. Target
                # skin and partner bodies do not become scene-map points.
                index = ObservedSceneIndex(points[t, active], normals[t, active], normal_valid[t, active])
                extra = index.query(anchors[t], radius, count)
                _, nearest_ids = index.tree.query(extra['points'].reshape(-1, 3), workers=1)
                extra['owner'] = owners[t, active][nearest_ids].reshape(extra['valid'].shape)
                distance = np.concatenate((np.where(query['valid'][t], query['distance'][t], np.inf),
                                           np.where(extra['valid'], extra['distance'], np.inf)), -1)
                chosen = np.argsort(distance, axis=-1, kind='stable')[:, :count]
                for field in ('points', 'normals', 'normal_valid', 'valid', 'observations', 'distance', 'owner'):
                    combined = np.concatenate((query[field][t], extra[field]), axis=1)
                    ids = np.broadcast_to(chosen[..., None], (*chosen.shape, 3)) if combined.ndim == 3 else chosen
                    query[field][t] = np.take_along_axis(combined, ids, axis=1)
                query['neighbour_count'][t] += extra['neighbour_count']
                query['nearest_distance'][t] = np.minimum(query['nearest_distance'][t], extra['nearest_distance'])
        relative = query['points']-anchors[..., None, :]
        octant = (relative >= 0).astype(np.int32) @ np.array([1, 2, 4])
        query['angular_coverage'] = np.stack([(query['valid'] & (octant == i)).any(-1) for i in range(8)], -1).mean(-1).astype(np.float32)
        result[name] = query
    mask = np.ones(anchors.shape[1], bool) if ground_mask is None else np.asarray(ground_mask, bool)
    if mask.shape != anchors.shape[1:2]:
        raise ValueError('Ground-query mask must describe the anchor axis')
    ground = view.index.support_planes(world[:, mask], policy['ground'])
    complete = {}
    for name, value in ground.items():
        array = np.zeros((*anchors.shape[:2], *value.shape[2:]), dtype=value.dtype)
        array[:, mask] = value
        complete[name] = array
    ground_position = world.copy();ground_position[..., 1] = complete['height']
    complete['offset'] = (view.to_local(ground_position)-anchors).astype(np.float32)
    complete['offset'][~complete['valid']] = 0.
    complete['normal'] = (complete['normal'] @ view.rotation.T).astype(np.float32)
    result['ground'] = complete
    return result
