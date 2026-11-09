"""Ground coordinates from observed, oriented surfaces, without motion labels."""
import numpy as np


DEFAULT_POLICY = dict(height_bin_m=.02, minimum_normal_y=.95,
    minimum_points=32, minimum_horizontal_span_m=.5)


def observed_ground_height(points, normals, normal_valid, policy=None):
    """Lowest supported horizontal patch; fail explicitly if ground is unknown.

    Contiguous height bins form a candidate surface. Small patches are rejected
    by both point count and horizontal extent. Selection uses no body trajectory.
    """
    p = dict(DEFAULT_POLICY if policy is None else policy)
    points, normals = np.asarray(points), np.asarray(normals)
    valid = np.asarray(normal_valid,bool)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3 or valid.shape != points.shape[:1]:
        raise ValueError('Expected matching oriented surface points')
    keep = valid & np.isfinite(points).all(-1) & np.isfinite(normals).all(-1)
    keep &= normals[:,1] >= p['minimum_normal_y']
    surface = points[keep]
    if not len(surface):
        raise ValueError('No observed horizontal ground evidence')
    bins = np.floor(surface[:,1]/p['height_bin_m']).astype(np.int64)
    levels = np.unique(bins)
    groups = np.split(levels,np.flatnonzero(np.diff(levels)>1)+1)
    candidates = []
    for group in groups:
        patch = surface[(bins>=group[0]) & (bins<=group[-1])]
        span = np.ptp(patch[:,[0,2]],axis=0)
        if len(patch) >= p['minimum_points'] and bool((span>=p['minimum_horizontal_span_m']).all()):
            candidates.append(dict(height_m=float(np.median(patch[:,1])),points=len(patch),
                horizontal_span_m=span.tolist(),height_range_m=[float(patch[:,1].min()),float(patch[:,1].max())]))
    if not candidates:
        raise ValueError('Observed horizontal patches do not establish ground')
    selected = min(candidates,key=lambda x:x['height_m'])
    return selected['height_m'],dict(policy=p,candidates=candidates,selected=selected,
        source='observed surface positions and normals only',motion_or_personal_shape_used=False)


def translate_sample_ground(sample, height_m):
    """Translate camera, target and observed points together; leave rest/angles."""
    result = dict(sample)
    camera = result['camera'] = sample['camera'].copy()
    camera[...,1] -= height_m/2
    motion = result['motion'] = sample['motion'].copy()
    motion[...,1] -= height_m/2
    motion[...,135:].reshape(*motion.shape[:-1],22,3)[...,1] -= height_m/2
    points = result['static_points'] = sample['static_points'].copy()
    points[...,1] -= np.asarray(sample['static_valid'])*height_m
    frame = result['frame_scene_points'] = sample['frame_scene_points'].copy()
    frame[...,1] -= (np.asarray(sample['frame_scene_owner'])>=0)*height_m
    return result
