"""Check pre-sit chair observations without changing the virtual camera.

Geometry is used only to screen recordings. Scene inputs must still come from
the saved first-hit observations, including their causal recording history.
"""
import numpy as np

from .grail_contact import sustained
from .grail_visibility import index_ranges


def full_frustum(vertices, position, rotation, camera):
    """All mesh vertices in the camera frustum; convex hull is equivalent."""
    from scipy.spatial import ConvexHull
    vertices = np.asarray(vertices)
    hull = vertices[ConvexHull(vertices).vertices]
    local = np.einsum('tvi,tij->tvj', hull[None]-position[:, None], rotation)
    z = local[..., 2]
    hx = np.tan(np.deg2rad(camera['hfov_deg']/2))
    hy = np.tan(np.deg2rad(camera['vfov_deg']/2))
    return ((z >= camera['near_m']) & (z <= camera['far_m']) &
            (np.abs(local[..., 0]) <= z*hx) &
            (np.abs(local[..., 1]) <= z*hy)).all(-1)


def seated_phase(joints, policy):
    velocity = np.gradient(joints[:, 0], axis=0)*policy['fps']
    thigh = joints[:, [4, 5]]-joints[:, [1, 2]]
    pose = ((np.abs(thigh[..., 1])/np.maximum(np.linalg.norm(thigh, axis=-1), 1e-8)
             < policy['seated_thigh_vertical_fraction']).all(-1) &
            (np.abs(velocity[:, 1]) < policy['seated_root_vertical_speed_mps']))
    return sustained(pose, policy['seat_min_stable_frames'])


def precheck(meta, policy):
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    joints = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3)*2
    seated = seated_phase(joints, policy)
    onset = int(np.flatnonzero(seated)[0]) if seated.any() else None
    full = full_frustum(readonly_array(meta, 'scene_object_vertices'),
                        readonly_array(meta, 'camera_position_scenemi_yup'),
                        readonly_array(meta, 'camera_rotation_scenemi_yup'),
                        meta['virtual_camera'])
    object_pixels = np.array([x['object_above_ground'] for x in meta['primary_visibility_by_frame']])
    candidates = full & (object_pixels >= policy['chair_min_object_pixels'])
    candidates[np.arange(len(full)) >= (onset if onset is not None else 0)] = False
    stable = sustained(candidates, policy['chair_full_view_consecutive_frames'])
    return dict(first_stable_sit_frame=onset, full_outline_frame_ranges=index_ranges(full),
                before_sit_outline_ranges=index_ranges(candidates),
                full_outline_before_sit_possible=bool(stable.any())), candidates, seated


def ready_windows(good, onset, policy):
    """Readiness can start only after a run observed strictly before sitting."""
    good = np.asarray(good, bool).copy()
    good[np.arange(len(good)) >= (onset if onset is not None else 0)] = False
    runs = [(a, b) for a, b in index_ranges(good)
            if b-a+1 >= policy['chair_full_view_consecutive_frames']]
    ready_frame = runs[0][0]+policy['chair_full_view_consecutive_frames']-1 if runs else None
    ready = np.arange(len(good)) >= ready_frame if ready_frame is not None else np.zeros(len(good), bool)
    prefix = np.r_[0, ready.astype(int).cumsum()]
    frames = policy['frames']
    windows = (prefix[frames:]-prefix[:-frames] >= policy['chair_ready_frames_per_window']
               if len(good) >= frames else np.zeros(0, bool))
    return ready_frame, windows


def observe_seat(meta, vertices, body, seated, candidates, policy):
    """Find the hip support height, then count actual first-hit seat samples.

    The saved 512 samples/frame include self-occlusion. Counts are sampled
    points, not full-resolution pixels. A geometric seat is never fed as an
    observation to the model.
    """
    import open3d as o3d
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    from .grail_support import support_faces
    v = readonly_array(meta, 'scene_object_vertices')
    f = support_faces(v, readonly_array(meta, 'scene_object_faces'), policy['seat_normal_y'])
    heights = v[f, 1].mean(-1)
    f, heights = f[heights > policy['seat_min_height_m']], heights[heights > policy['seat_min_height_m']]
    n = len(vertices)
    counts = np.zeros(n, int)
    if not len(f) or not seated.any():
        return counts, None
    scene = static_ray_scene(v, f)
    hips = np.asarray(vertices[seated][:, body.seat], np.float32)
    closest = scene.compute_closest_points(o3d.core.Tensor(hips), nthreads=1)['points'].numpy()
    valid = ((np.linalg.norm((hips-closest)[..., [0, 2]], axis=-1) <= .04) &
             (np.abs(hips[..., 1]-closest[..., 1]) <= policy['seat_gap_m']))
    if not valid.any():
        return counts, None
    height = float(np.median(closest[..., 1][valid]))
    seat_faces = f[np.abs(heights-height) <= policy['chair_seat_height_band_m']]
    if not len(seat_faces):
        return counts, height
    seat = static_ray_scene(v, seat_faces)
    points = readonly_array(meta, 'visible_frame_points_scenemi_yup')
    owners = readonly_array(meta, 'visible_frame_owner')
    for t in np.flatnonzero(candidates):
        # Static geometry owner 0; invalid samples have owner -1.
        actual = np.asarray(points[t][owners[t] == 0], np.float32)
        if len(actual):
            distance = seat.compute_distance(o3d.core.Tensor(actual), nthreads=1).numpy()
            counts[t] = int((distance <= policy['chair_support_point_distance_m']).sum())
    return counts, height
