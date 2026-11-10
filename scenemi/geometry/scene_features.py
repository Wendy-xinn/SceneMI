"""Inference-time body/surface relations from observed geometry only.

An oriented *local plane offset* is not an inside/outside SDF. Free-space
evidence is restricted to measured camera-to-first-hit ray segments. Missing
points, empty occupancy cells, and space behind a first hit remain unknown.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch

from scenemi.geometry.observed_relations import local_planes


FEATURE_NAMES = ('relative_x', 'relative_y', 'relative_z', 'nearest_distance',
                 'normal_x', 'normal_y', 'normal_z', 'plane_offset',
                 'normal_valid', 'neighbour_coverage', 'surface_reliability',
                 'observed_free', 'near_observed_surface', 'unknown')


class ObservedNeighborhoods:
    """Reuse detached KD trees across the denoising steps of one window."""
    def __init__(self, history, history_valid, current, owners):
        history = history.detach().cpu().numpy()
        history_valid = history_valid.detach().cpu().numpy()
        current = current.detach().cpu().numpy()
        owners = owners.detach().cpu().numpy()
        valid = history_valid & np.isfinite(history).all(-1)
        self.static = self.tree(history[valid], np.zeros(int(valid.sum()), np.int64))
        self.current = []
        for points, ids in zip(current, owners):
            valid = (ids >= 0) & (ids < 100) & np.isfinite(points).all(-1)
            self.current.append(self.tree(points[valid], ids[valid]))

    @staticmethod
    def tree(points, owners):
        return (cKDTree(points) if len(points) else None, points, owners)

    def query(self, anchors, k, radius):
        query = anchors.detach().cpu().numpy()

        def lookup(entry, centres):
            tree, points, owners = entry
            shape = centres.shape[:-1] + (k,)
            p = np.zeros(shape+(3,), np.float32)
            ids = np.zeros(shape, np.int64)
            distance = np.full(shape, np.inf)
            if tree is not None:
                count = min(k, len(points))
                d, index = tree.query(centres, k=list(range(1, count+1)), distance_upper_bound=radius)
                good = np.isfinite(d)
                index = np.minimum(index, len(points)-1)
                p[..., :count, :] = np.where(good[..., None], points[index], 0.)
                ids[..., :count] = owners[index]
                distance[..., :count] = d
            return p, ids, distance

        static = lookup(self.static, query)
        current = [lookup(entry, query[t]) for t, entry in enumerate(self.current)]
        combined = [np.concatenate((static[j], np.stack([row[j] for row in current])),
                       axis=-2 if j == 0 else -1) for j in range(3)]
        order = np.argsort(combined[2], axis=-1, kind='stable')[..., :k]
        points = np.take_along_axis(combined[0], order[..., None], axis=2)
        owners = np.take_along_axis(combined[1], order, axis=2)
        valid = np.isfinite(np.take_along_axis(combined[2], order, axis=2))
        return (torch.as_tensor(points, device=anchors.device, dtype=anchors.dtype),
                torch.as_tensor(owners, device=anchors.device),
                torch.as_tensor(valid, device=anchors.device))


def measured_free_space(anchors, camera, points, owners, camera_valid, *, tube_m,
                        endpoint_margin_m, minimum_depth_m, frame_chunk=16):
    """Conservative small tubes BEFORE valid current-frame first-hit points.

No rays are inferred for historical points whose observation origins are not
stored. No interpolation across missing pixels or unmeasured ray directions.
"""
    outputs = []
    for start in range(0, len(anchors), frame_chunk):
        end = start+frame_chunk
        cp, p, ids = camera[start:end], points[start:end], owners[start:end]
        finite = torch.isfinite(p).all(-1) & torch.isfinite(cp).all(-1)[:, None]
        good = (ids >= 0) & (ids < 100) & finite & camera_valid[start:end, None]
        rays = torch.where(good[..., None], p-cp[:, None], 0.)
        length = rays.norm(dim=-1)
        direction = rays/length.clamp_min(1e-6)[..., None]
        delta = anchors[start:end]-cp[:, None]
        along = torch.einsum('tac,tkc->tak', delta, direction)
        perpendicular2 = (delta.square().sum(-1)[..., None]-along.square()).clamp_min(0.)
        free = good[:, None] & (along > minimum_depth_m)
        free &= along < length[:, None]-endpoint_margin_m
        free &= perpendicular2 <= tube_m**2
        outputs.append(free.any(-1))
    return torch.cat(outputs)


@torch.no_grad()
def relation_features(anchors, rotations, batch, policy, neighborhoods=None):
    features, gates, diagnostics = [], [], []
    for i, query in enumerate(anchors):
        cache = neighborhoods[i] if neighborhoods is not None else ObservedNeighborhoods(
            batch['local_history_points'][i], batch['local_history_valid'][i],
            batch['local_frame_points'][i], batch['local_frame_owner'][i])
        points, owners, valid = cache.query(query, policy['neighbours'], policy['query_radius_m'])
        # A local plane cannot bridge two remote surface patches just because
        # both happen to have owner 0. Fit a compact neighbourhood of the hit.
        compact = (points-points[..., :1, :]).norm(dim=-1) <= policy['plane_radius_m']
        plane = local_planes(points, owners, valid & compact,
            minimum_points=policy['minimum_points'], maximum_ratio=policy['maximum_planarity_ratio'],
            minimum_spread_m=policy['minimum_spread_m'])
        cp = batch['camera'][i, :, :3]*2
        normal = plane['normal']
        orientation = ((cp[:, None]-plane['centre'])*normal).sum(-1)
        normal = torch.where((orientation < 0)[..., None], -normal, normal)
        normal = normal*plane['valid'][..., None]
        relative = points[..., 0, :]-query
        hit = valid[..., 0]
        relative = torch.where(hit[..., None], relative, 0.)
        distance = relative.norm(dim=-1)
        coverage = plane['neighbours'].to(query)/policy['neighbours']
        reliability = plane['valid'].to(query)*coverage*torch.exp(-distance/policy['reliability_distance_m'])
        free = measured_free_space(query, cp, batch['local_frame_points'][i],
            batch['local_frame_owner'][i], (batch['camera_available'][i, :, 0] > 0) &
            (batch['camera_confidence'][i, :, 0] > 0),
            tube_m=policy['ray_tube_m'], endpoint_margin_m=policy['ray_endpoint_margin_m'],
            minimum_depth_m=policy['ray_minimum_depth_m'])
        # Nearby observations are not proof that a particular anchor is
        # occupied. "near" only describes measured surface context.
        unknown = ~hit & ~free
        local_relative = torch.einsum('taik,tai->tak', rotations[i], relative)
        local_normal = torch.einsum('taik,tai->tak', rotations[i], normal)
        offset = ((query-plane['centre'])*normal).sum(-1)
        f = torch.cat((local_relative/policy['query_radius_m'],
            (distance/policy['query_radius_m'])[..., None], local_normal,
            (offset/policy['query_radius_m'])[..., None], plane['valid'][..., None].to(query),
            coverage[..., None], reliability[..., None], free[..., None].to(query),
            hit[..., None].to(query), unknown[..., None].to(query)), -1)
        # Valid but nonplanar points retain a weaker point-distance signal;
        # a measured free ray also carries information without a local plane.
        point_reliability = hit.to(query)*valid.float().mean(-1)*torch.exp(-distance/policy['reliability_distance_m'])
        gate = torch.maximum(torch.maximum(reliability, .25*point_reliability), free.to(query))
        features.append(f)
        gates.append(gate)
        diagnostics.append(torch.stack((hit.float().mean(), plane['valid'].float().mean(),
            free.float().mean(), unknown.float().mean(), gate.mean())))
    return torch.stack(features), torch.stack(gates), torch.stack(diagnostics)
