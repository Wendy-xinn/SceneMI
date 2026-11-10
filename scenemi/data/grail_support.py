"""Additional sole-support checks, independent of the immutable v1 screen.

Only lower foot skin points over horizontal/sloped triangles count as support.
Nearness to a stair riser or a higher part of the foot is not sole contact.
"""
import numpy as np

from .grail_contact import sustained, with_context
from .grail_quality import quantiles
from .grail_visibility import index_ranges


def support_faces(vertices, faces, minimum_normal_y):
    triangles = np.asarray(vertices)[np.asarray(faces)]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    norm = np.linalg.norm(normal, axis=-1)
    return np.asarray(faces)[np.abs(normal[:, 1]) >= minimum_normal_y*np.maximum(norm, 1e-12)]


def lower_skin(points, fraction):
    count = max(1, int(np.ceil(points.shape[-2]*fraction)))
    order = np.argsort(points[..., 1], axis=-1, kind='stable')[..., :count]
    return np.take_along_axis(points, order[..., None], axis=-2)


def sole_measure(vertices, feet, scene_vertices, scene_faces, policy):
    import open3d as o3d
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    faces = support_faces(scene_vertices, scene_faces, policy['support_normal_y'])
    if not len(faces):
        raise ValueError('No terrain support triangles, including the exported ground')
    scene = static_ray_scene(scene_vertices, faces)
    gaps, fractions, speeds = [], [], []
    for ids in feet:
        points = np.asarray(vertices[:, ids], np.float32)
        lower = lower_skin(points, policy['lower_foot_fraction'])
        rays = np.zeros((*lower.shape[:-1], 6), np.float32)
        rays[..., :3] = lower
        rays[..., 1] += policy['ray_origin_lift_m']
        rays[..., 4] = -1
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
        gap = hit-policy['ray_origin_lift_m']
        near = np.isfinite(gap) & (np.abs(gap) <= policy['support_gap_m'])
        fractions.append(near.mean(-1))
        # Inf means no support hit. It must not disappear from a nan quantile.
        gaps.append(np.quantile(gap, .1, axis=-1, method='nearest'))
        velocity = np.gradient(points.mean(1), axis=0)*policy['fps']
        speeds.append(np.linalg.norm(velocity, axis=-1))
    return dict(sole_gap_q10_m=np.stack(gaps, -1),
                sole_supported_fraction=np.stack(fractions, -1),
                foot_speed_mps=np.stack(speeds, -1))


def support_events(arrays, probabilities, category, policy):
    enough = arrays['sole_supported_fraction'] >= policy['minimum_supported_fraction']
    unsupported = ~enough
    n = len(enough)
    events = {}
    if category != 'sitting':
        duration = policy['minimum_event_frames']
        events['sole_both_feet_unsupported'] = sustained(unsupported.all(-1), duration)
        for side in range(2):
            expected_stance = ((probabilities[:, side] > policy['contact_probability']) &
                              (arrays['foot_speed_mps'][:, side] <= policy['stance_max_speed_mps']))
            events[f'sole_estimated_stance_unsupported_{side}'] = sustained(
                expected_stance & unsupported[:, side], duration)
    bad = np.zeros(n, bool)
    for event in events.values():
        bad |= event
    bad = with_context(bad, policy['event_context_frames'])
    record = dict(events={k:index_ranges(v) for k,v in events.items() if v.any()},
        bad_frame_ranges=index_ranges(bad),flagged_frames=int(bad.sum()),
        sole_gap_q10_m=quantiles(arrays['sole_gap_q10_m']),
        sole_supported_fraction=quantiles(arrays['sole_supported_fraction']),
        support_measure='Lowest 15% of foot skin by world height; downward support rays against abs(normal_y)>=0.7 triangles, excluding risers. Vertex coverage is not contact area.',
        source_contact_is_estimated=True,physics_validity_proven=False)
    return record, bad
