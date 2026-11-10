"""CPU nearest neighbours with bounded CUDA plane/ray tensor operations.

Only execution placement changes. Reuse the established neighbourhood ordering,
plane validity, ray evidence and feature definitions. No geometry is predicted
or cached across changing body queries.
"""
from contextlib import contextmanager
import time

import numpy as np
import torch

from scenemi import scene_adapter, scene_joint_replay
from scenemi.geometry import scene_features
from scenemi.geometry.observed_relations import local_planes as reference_planes


TREE_INPUTS = ('local_history_points', 'local_history_valid',
               'local_frame_points', 'local_frame_owner')
LAST_PROFILE = {}


def bounded_planes(points, owners, valid, *, minimum_points=6, maximum_ratio=.03,
                   minimum_spread_m=.005, chunk=1024):
    if points.device.type == 'cpu':
        return reference_planes(points, owners, valid, minimum_points=minimum_points,
            maximum_ratio=maximum_ratio, minimum_spread_m=minimum_spread_m)
    valid = valid & torch.isfinite(points).all(-1)
    valid = valid & (owners == owners[..., :1])
    count = valid.sum(-1)
    safe = torch.where(valid[..., None], points, 0.)
    centre = safe.sum(-2)/count.clamp_min(1)[..., None]
    delta = torch.where(valid[..., None], points-centre[..., None, :], 0.)
    covariance = delta.transpose(-1, -2)@delta/count.clamp_min(1)[..., None, None]
    flat = covariance.reshape(-1, 3, 3)
    # Bound cuSOLVER workspace without changing the independent 3x3 problems.
    values, vectors = [], []
    for start in range(0, len(flat), chunk):
        v, q = torch.linalg.eigh(flat[start:start+chunk])
        values.append(v); vectors.append(q[..., 0])
    eigenvalues = torch.cat(values).reshape(*count.shape, 3)
    normal = torch.cat(vectors).reshape(*count.shape, 3)
    planar = (count >= minimum_points) & (eigenvalues[..., 1] >= minimum_spread_m**2)
    planar &= eigenvalues[..., 0]/eigenvalues.sum(-1).clamp_min(1e-12) <= maximum_ratio
    return dict(centre=centre.detach(), normal=normal.detach(), valid=planar.detach(),
                neighbours=count.detach(), owner=owners[..., 0].detach())


class PreparedQuery:
    def __init__(self, values, k, radius):
        self.values, self.k, self.radius = values, k, radius

    def query(self, anchors, k, radius):
        if k != self.k or radius != self.radius:
            raise ValueError('A prepared neighbourhood cannot change its query policy')
        return self.values


@contextmanager
def hybrid_geometry(trainer, *, eigh_chunk=8192, ray_frame_chunk=128):
    """Install only inside the new process; restore every hook on exit."""
    previous = (trainer.collate, scene_adapter.relation_features,
                scene_joint_replay.relation_features, scene_features.local_planes,
                scene_features.measured_free_space)
    original_collate = previous[0]
    payload = [None, None, None]
    profile = dict(cpu_tree_seconds=0., cpu_query_seconds=0., calls=0,
                   host_anchor_copies=0, device_neighbour_copies=0)

    def collate(samples, device='cpu'):
        # Retain original CPU observations before their first upload. Avoid
        # downloading the large scene tensors immediately after collate.
        cpu = {key: torch.as_tensor(np.stack([s[key] for s in samples]))
               for key in TREE_INPUTS}
        batch = original_collate(samples, device)
        payload[:] = [batch['local_frame_points'], cpu, None]
        return batch

    @torch.no_grad()
    def compute(anchors, rotations, batch, policy, neighborhoods=None):
        if payload[0] is not batch['local_frame_points']:
            raise ValueError('Geometry requires its matching original CPU observations')
        if payload[2] is None:
            tick = time.monotonic(); cpu = payload[1]
            payload[2] = [scene_features.ObservedNeighborhoods(
                cpu['local_history_points'][i], cpu['local_history_valid'][i],
                cpu['local_frame_points'][i], cpu['local_frame_owner'][i])
                for i in range(len(anchors))]
            profile['cpu_tree_seconds'] += time.monotonic()-tick
        tick = time.monotonic()
        query = anchors.detach().cpu()  # One synchronization for the entire microbatch.
        profile['host_anchor_copies'] += 1
        values = [cache.query(query[i], policy['neighbours'], policy['query_radius_m'])
                  for i, cache in enumerate(payload[2])]
        profile['cpu_query_seconds'] += time.monotonic()-tick
        packed = [torch.stack([row[j] for row in values]).to(anchors.device)
                  for j in range(3)]
        profile['device_neighbour_copies'] += 3
        prepared = [PreparedQuery(tuple(value[i] for value in packed),
                    policy['neighbours'], policy['query_radius_m'])
                    for i in range(len(anchors))]
        profile['calls'] += 1
        return scene_features.relation_features(anchors, rotations, batch, policy, prepared)

    def planes(*args, **kwargs):
        return bounded_planes(*args, **kwargs, chunk=eigh_chunk)

    def free_space(*args, **kwargs):
        kwargs['frame_chunk'] = ray_frame_chunk if args[0].device.type == 'cuda' else 16
        return previous[4](*args, **kwargs)

    trainer.collate = collate
    scene_adapter.relation_features = scene_joint_replay.relation_features = compute
    scene_features.local_planes = planes
    scene_features.measured_free_space = free_space
    try:
        yield profile
    finally:
        (trainer.collate, scene_adapter.relation_features,
         scene_joint_replay.relation_features, scene_features.local_planes,
         scene_features.measured_free_space) = previous
        LAST_PROFILE.clear(); LAST_PROFILE.update(profile)
