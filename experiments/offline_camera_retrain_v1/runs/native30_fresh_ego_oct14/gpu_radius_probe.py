"""Isolated exact-output GPU radius prototype; not installed in the trainer.

Brute-force double-precision radius membership is bounded by query chunks.
Near-boundary rows fall back to the original KD-tree membership. Original
float32/64 nearest/FPS order and source IDs remain unchanged.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.metric_surface_tokens import SCALES
from experiments.offline_camera_retrain_v1.native30_query_runtime import select_radius_samples_cuda


def query_gpu(scene, vertices, rotations, device='cuda', neighbors=8,
              scale_indices=(0, 1, 2), query_rows=512):
    frames, anchors = vertices.shape[:2]
    if len(scene.dynamic) != frames or rotations.shape != (frames, anchors, 3, 3):
        raise ValueError('Scene clock mismatch')
    if neighbors < 1 or query_rows < 1:
        raise ValueError('Invalid query size')
    shape = (frames, anchors, 3, neighbors)
    result = {k: np.zeros((*shape, 3), np.float32) for k in ('points', 'normals', 'velocities')}
    result.update({k: np.zeros(shape, bool) for k in ('valid', 'normal_reliable', 'velocity_valid')})
    result['owners'] = np.zeros(shape, np.int64)
    cloud = {k: np.concatenate([scene.static[k]] + [c[k] for c in scene.dynamic]) for k in scene.static}
    if not anchors or not len(cloud['points']):
        return result
    static_count = len(scene.static['points'])
    sizes = np.asarray([len(c['points']) for c in scene.dynamic], np.int64)
    offsets = static_count + np.r_[0, sizes.cumsum()[:-1]]
    points = torch.as_tensor(cloud['points'], device=device)
    double_points = points.double()
    flat = vertices.reshape(-1, 3)
    output = {k: v.reshape(len(flat), *v.shape[2:]) for k, v in result.items()}
    dynamic_width = int(sizes.max(initial=0))
    with torch.no_grad():
        for lo in range(0, len(flat), query_rows):
            hi = min(len(flat), lo + query_rows)
            probes = torch.as_tensor(flat[lo:hi], device=device, dtype=torch.float64)
            frame_ids = np.arange(lo, hi) // anchors
            dyn_ids = torch.as_tensor(offsets[frame_ids, None], device=device) + torch.arange(dynamic_width, device=device)[None]
            dyn_valid = torch.arange(dynamic_width, device=device)[None] < torch.as_tensor(sizes[frame_ids, None], device=device)
            dyn_points = double_points[torch.where(dyn_valid, dyn_ids, 0)]
            columns = []
            for xyz in range(3):
                static_d = (double_points[None, :static_count, xyz] - probes[:, None, xyz]).square()
                dynamic_d = (dyn_points[..., xyz] - probes[:, None, xyz]).square()
                columns.append(torch.cat((static_d, dynamic_d), dim=1))
            d2 = (columns[0] + columns[1]) + columns[2]
            del columns, dyn_points
            if dynamic_width:
                d2[:, static_count:].masked_fill_(~dyn_valid, torch.inf)
            for si in scale_indices:
                radius = SCALES[si]
                mask = d2 <= radius ** 2
                border = ((d2 - radius ** 2).abs() <= 128 * np.finfo(np.float64).eps * radius ** 2).any(-1)
                border_rows = border.nonzero().flatten().cpu().numpy()
                for row in border_rows:
                    frame = int(frame_ids[row])
                    mask[row] = False
                    if static_count:
                        ids = scene._static_tree.query_ball_point(flat[lo + row], radius, return_sorted=True)
                        mask[row, torch.as_tensor(ids, device=device, dtype=torch.long)] = True
                    if sizes[frame]:
                        if scene._dynamic_trees[frame] is None:
                            scene._dynamic_trees[frame] = cKDTree(scene.dynamic[frame]['points'])
                        ids = scene._dynamic_trees[frame].query_ball_point(flat[lo + row], radius, return_sorted=True)
                        mask[row, static_count + torch.as_tensor(ids, device=device, dtype=torch.long)] = True
                counts = mask.sum(-1)
                maximum = int(counts.max())
                if not maximum:
                    continue
                row_ids, local_ids = mask.nonzero(as_tuple=True)
                starts = counts.cumsum(0) - counts
                ordinal = torch.arange(len(row_ids), device=device) - starts[row_ids]
                global_ids = torch.where(local_ids < static_count, local_ids,
                    local_ids - static_count + torch.as_tensor(offsets[frame_ids], device=device)[row_ids])
                padded = torch.zeros((hi-lo, maximum), dtype=torch.long, device=device)
                padded[row_ids, ordinal] = global_ids
                ids, valid = select_radius_samples_cuda(points, flat[lo:hi], padded, counts, neighbors)
                dest = (slice(lo, hi), si)
                for key in ('points', 'normals', 'owners', 'velocities', 'velocity_valid'):
                    values = cloud[key][ids]
                    output[key][dest] = np.where(valid[..., None], values, 0) if values.ndim == 3 else np.where(valid, values, 0)
                output['normal_reliable'][dest] = cloud['reliable'][ids] & valid
                output['valid'][dest] = valid
    return result
