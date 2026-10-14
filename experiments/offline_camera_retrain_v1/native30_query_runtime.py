"""Batch the original radius/FPS calculation without changing selected samples.

Sorted KD-tree hits, nearest-first sampling, first-index ties and float32
distance arithmetic follow the reference implementation. No approximate KNN,
point budget change, or radius change is introduced.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.metric_surface_tokens import SCALES


def query_nearest_surfaces(scene, vertices, rotations):
    """Exact first wide sample without enumerating the whole .6m ball.

    The reference ranks squared distances in the probe's NumPy dtype, whereas
    KD trees use doubles. Two-neighbour lookups detect possible rounded ties;
    only those rows enumerate a small tie ball, with the original ID order.
    Radius membership remains the KD tree's double-precision closed ball.
    """
    frames, anchors = vertices.shape[:2]
    if len(scene.dynamic) != frames or rotations.shape != (frames, anchors, 3, 3):
        raise ValueError('Predicted-body query does not match observed scene clock')
    shape = (frames, anchors, 3, 1)
    result = {k: np.zeros((*shape, 3), np.float32) for k in ('points', 'normals', 'velocities')}
    result.update({k: np.zeros(shape, bool) for k in ('valid', 'normal_reliable', 'velocity_valid')})
    result['owners'] = np.zeros(shape, np.int64)
    if not anchors:
        return result
    radius = SCALES[2]
    eps = np.finfo(np.result_type(vertices.dtype, np.float32)).eps * 32
    def nearest(cloud, tree, probes):
        if tree is None:
            return np.zeros(len(probes), np.int64), np.zeros(len(probes), bool)
        distances, ids = tree.query(probes, k=[1, 2], distance_upper_bound=np.nextafter(radius, np.inf))
        valid = distances[:, 0] <= radius
        chosen = np.where(valid, ids[:, 0], 0)
        ambiguous = valid & (distances[:, 1] <= np.minimum(radius, distances[:, 0] * (1 + eps) + np.finfo(float).tiny))
        rows = np.flatnonzero(ambiguous)
        if len(rows):
            hits = tree.query_ball_point(probes[rows], np.minimum(radius, distances[rows, 0] * (1 + eps) + np.finfo(float).tiny), return_sorted=True)
            for row, pool in zip(rows, hits):
                pool = np.asarray(pool, np.int64)
                delta = cloud['points'][pool] - probes[row]
                chosen[row] = pool[(delta ** 2).sum(-1).argmin()]
        return chosen, valid
    flat = vertices.reshape(-1, 3)
    static_ids, static_valid = nearest(scene.static, scene._static_tree, flat)
    static_ids, static_valid = static_ids.reshape(frames, anchors), static_valid.reshape(frames, anchors)
    for t, dynamic in enumerate(scene.dynamic):
        if len(dynamic['points']) and scene._dynamic_trees[t] is None:
            scene._dynamic_trees[t] = cKDTree(dynamic['points'])
        di, dv = nearest(dynamic, scene._dynamic_trees[t], vertices[t])
        sv = static_valid[t]; si = static_ids[t]
        sd = np.full(anchors, np.inf); dd = np.full(anchors, np.inf)
        if sv.any(): sd[sv] = ((scene.static['points'][si[sv]] - vertices[t, sv]) ** 2).sum(-1)
        if dv.any(): dd[dv] = ((dynamic['points'][di[dv]] - vertices[t, dv]) ** 2).sum(-1)
        # Static IDs precede all dynamic IDs, including equal rounded distances.
        use_static = sv & (~dv | (sd <= dd)); use_dynamic = dv & ~use_static
        for mask, cloud, ids in ((use_static, scene.static, si), (use_dynamic, dynamic, di)):
            if not mask.any(): continue
            dest = (t, mask, 2, 0)
            for key in ('points', 'normals', 'owners', 'velocities', 'velocity_valid'):
                result[key][dest] = cloud[key][ids[mask]]
            result['normal_reliable'][dest] = cloud['reliable'][ids[mask]]
            result['valid'][dest] = True
    return result


def select_radius_samples(points, probes, static_hits, dynamic_hits,
                          static_count, neighbors):
    """Select all anchors of one frame together; invalid columns are masked."""
    count = np.asarray([len(s) + len(d) for s, d in zip(static_hits, dynamic_hits)])
    anchors = len(probes)
    selected = np.zeros((anchors, neighbors), np.int64)
    valid = np.arange(neighbors)[None] < count[:, None]
    maximum = int(count.max(initial=0))
    if not maximum:
        return selected, valid
    pool = np.zeros((anchors, maximum), np.int64)
    for a, (static, dynamic) in enumerate(zip(static_hits, dynamic_hits)):
        n = len(static)
        pool[a, :n] = static
        pool[a, n:n + len(dynamic)] = np.asarray(dynamic, np.int64) + static_count
    available = np.arange(maximum)[None] < count[:, None]
    delta = points[pool] - probes[:, None]
    squared = (delta ** 2).sum(-1)
    row = np.arange(anchors)
    chosen = np.zeros_like(selected)
    chosen[:, 0] = np.where(available, squared, np.inf).argmin(-1)
    spread = np.full((anchors, maximum), np.inf)
    spread[~available] = -np.inf
    for i in range(1, min(neighbors, maximum)):
        squared = ((delta - delta[row, chosen[:, i - 1], None]) ** 2).sum(-1)
        spread = np.minimum(spread, squared)
        spread[row, chosen[:, i - 1]] = -1
        chosen[:, i] = spread.argmax(-1)
    selected[:] = pool[row[:, None], chosen]
    return selected, valid


def query_surfaces(scene, vertices, rotations, neighbors=8, scale_indices=(0, 1, 2)):
    frames, anchors = vertices.shape[:2]
    if len(scene.dynamic) != frames or rotations.shape != (frames, anchors, 3, 3):
        raise ValueError('Predicted-body query does not match observed scene clock')
    if neighbors < 1:
        raise ValueError('At least one radius sample is required')
    shape = (frames, anchors, 3, neighbors)
    result = {key: np.zeros((*shape, 3), np.float32)
              for key in ('points', 'normals', 'velocities')}
    result.update({key: np.zeros(shape, bool)
                   for key in ('valid', 'normal_reliable', 'velocity_valid')})
    result['owners'] = np.zeros(shape, np.int64)
    static_count = len(scene.static['points'])
    for t in range(frames):
        dynamic = scene.dynamic[t]
        dynamic_count = len(dynamic['points'])
        if not static_count and not dynamic_count:
            continue
        if dynamic_count and scene._dynamic_trees[t] is None:
            scene._dynamic_trees[t] = cKDTree(dynamic['points'])
        cloud = {key: np.concatenate((scene.static[key], dynamic[key]))
                 for key in scene.static} if dynamic_count else scene.static
        empty = [[] for _ in range(anchors)]
        for si in scale_indices:
            radius = SCALES[si]
            static_hits = scene._static_tree.query_ball_point(
                vertices[t], radius, return_sorted=True) if static_count else empty
            dynamic_hits = scene._dynamic_trees[t].query_ball_point(
                vertices[t], radius, return_sorted=True) if dynamic_count else empty
            ids, valid = select_radius_samples(cloud['points'], vertices[t],
                static_hits, dynamic_hits, static_count, neighbors)
            dest = (t, slice(None), si)
            for key in ('points', 'normals', 'owners', 'velocities', 'velocity_valid'):
                values = cloud[key][ids]
                if values.ndim == 3:
                    values = np.where(valid[..., None], values, 0)
                else:
                    values = np.where(valid, values, 0)
                result[key][dest] = values
            result['normal_reliable'][dest] = cloud['reliable'][ids] & valid
            result['valid'][dest] = valid
    return result


def select_radius_samples_cuda(points, probes, pools, counts, neighbors):
    """Same nearest/FPS order on CUDA; explicit sums keep reference rounding."""
    with torch.no_grad():
        pool = torch.as_tensor(pools, device=points.device)
        query = torch.as_tensor(probes, device=points.device)
        delta = points[pool].to(query.dtype) - query[:, None]
        available = torch.arange(pool.shape[1], device=points.device)[None] < torch.as_tensor(counts, device=points.device)[:, None]
        def squared(x):
            # NumPy's three-element sum accumulates in this order. Keep the
            # multiplies and adds separate; do not use AMP/TF32/fast-math.
            return (x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]) + x[..., 2] * x[..., 2]
        first = torch.where(available, squared(delta), torch.inf).argmin(-1)
        row = torch.arange(len(pool), device=points.device)
        chosen = [first]
        spread = torch.full(pool.shape, torch.inf, dtype=delta.dtype, device=points.device)
        spread.masked_fill_(~available, -torch.inf)
        for _ in range(1, min(neighbors, pool.shape[1])):
            distance = squared(delta - delta[row, chosen[-1], None])
            spread = torch.minimum(spread, distance)
            spread[row, chosen[-1]] = -1
            chosen.append(spread.argmax(-1))
        while len(chosen) < neighbors:
            chosen.append(torch.zeros_like(first))
        ids = pool[row[:, None], torch.stack(chosen, -1)]
        valid = torch.arange(neighbors, device=points.device)[None] < torch.as_tensor(counts, device=points.device)[:, None]
        return ids.cpu().numpy(), valid.cpu().numpy()


def query_surfaces_cuda(scene, vertices, rotations, device, neighbors=8, scale_indices=(0, 1, 2)):
    """Exact CPU radius memberships and CUDA FPS across frame/anchor chunks.

    Membership stays with the original KD trees (including radius boundaries).
    Dynamic samples retain their frame's offset, owner, normal and velocity.
    Chunk size is a temporary-memory limit, independent of model/data settings.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        return query_surfaces(scene, vertices, rotations, neighbors, scale_indices)
    frames, anchors = vertices.shape[:2]
    if len(scene.dynamic) != frames or rotations.shape != (frames, anchors, 3, 3):
        raise ValueError('Predicted-body query does not match observed scene clock')
    if neighbors < 1:
        raise ValueError('At least one radius sample is required')
    shape = (frames, anchors, 3, neighbors)
    result = {key: np.zeros((*shape, 3), np.float32)
              for key in ('points', 'normals', 'velocities')}
    result.update({key: np.zeros(shape, bool)
                   for key in ('valid', 'normal_reliable', 'velocity_valid')})
    result['owners'] = np.zeros(shape, np.int64)
    static_count = len(scene.static['points'])
    counts = np.asarray([len(c['points']) for c in scene.dynamic], np.int64)
    offsets = static_count + np.r_[0, counts.cumsum()[:-1]]
    cloud = {key: np.concatenate([scene.static[key]] + [c[key] for c in scene.dynamic])
             for key in scene.static}
    if not len(cloud['points']):
        return result
    points = torch.as_tensor(cloud['points'], device=device)
    # About 1024 query rows per chunk; all three radii remain unchanged.
    frame_chunk = max(1, 1024 // max(anchors, 1))
    for lo in range(0, frames, frame_chunk):
        hi = min(frames, lo + frame_chunk)
        for si in scale_indices:
            probes = vertices[lo:hi].reshape(-1, 3)
            static_hits = scene._static_tree.query_ball_point(
                probes, SCALES[si], return_sorted=True) if static_count else [[] for _ in probes]
            pools = []; frame_ids = []
            for t in range(lo, hi):
                if counts[t]:
                    if scene._dynamic_trees[t] is None:
                        scene._dynamic_trees[t] = cKDTree(scene.dynamic[t]['points'])
                    dynamic_hits = scene._dynamic_trees[t].query_ball_point(
                        vertices[t], SCALES[si], return_sorted=True)
                else:
                    dynamic_hits = [[] for _ in range(anchors)]
                for a, dynamic in enumerate(dynamic_hits):
                    static = static_hits[(t-lo)*anchors+a]
                    pools.append(list(static) + [int(v) + int(offsets[t]) for v in dynamic])
                    frame_ids.append(t)
            sizes = np.asarray([len(v) for v in pools], np.int64)
            maximum = int(sizes.max(initial=0))
            if not maximum:
                continue
            padded = np.zeros((len(pools), maximum), np.int64)
            for i, pool in enumerate(pools):
                padded[i, :len(pool)] = pool
            ids, valid = select_radius_samples_cuda(points, probes, padded, sizes, neighbors)
            ids = ids.reshape(hi-lo, anchors, neighbors)
            valid = valid.reshape(hi-lo, anchors, neighbors)
            dest = (slice(lo, hi), slice(None), si)
            for key in ('points', 'normals', 'owners', 'velocities', 'velocity_valid'):
                values = cloud[key][ids]
                result[key][dest] = np.where(valid[..., None], values, 0) if values.ndim == 4 else np.where(valid, values, 0)
            result['normal_reliable'][dest] = cloud['reliable'][ids] & valid
            result['valid'][dest] = valid
    return result
