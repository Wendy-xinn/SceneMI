"""Shared metric, time-local mesh visibility (no point-cloud depth approximation).

Owner IDs: 0 static scene; 1..99 tracked objects; 100 wearer (head excluded);
101+ other people (complete meshes). Body hits occlude but are not scene input.
Only the wearer's head skin is removed, never its hands, limbs or torso.
"""
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp


def wearer_faces(faces, weights, model_type="smplx"):
    # SMPL-X joints: head=15, jaw=22, eyes=23/24. Neck stays an occluder.
    weights = np.asarray(weights)
    if model_type not in ("smpl", "smplx"):raise ValueError("Unknown wearer topology")
    head_joints=[15,22,23,24] if model_type=="smplx" else [15]
    head = weights[:, head_joints].sum(axis=1) >= .5
    keep = ~np.all(head[np.asarray(faces)], axis=1)
    if not keep.any() or keep.all():
        raise ValueError('Head-only exclusion failed; check SMPL-X skinning topology')
    return np.asarray(faces)[keep], {'head_vertices': int(head.sum()),
                                    'excluded_head_faces': int((~keep).sum()),
                                    'retained_body_faces': int(keep.sum())}


def interpolate_transforms(times, rotations, positions, queries):
    times, queries = np.asarray(times), np.asarray(queries)
    if np.any(queries < times[0]) or np.any(queries > times[-1]):
        raise ValueError('Object query outside source time range (no clamping)')
    r = Slerp(times, Rotation.from_matrix(rotations))(queries).as_matrix()
    p = np.stack([np.interp(queries, times, np.asarray(positions)[:, k])
                  for k in range(3)], axis=-1)
    return r.astype(np.float32), p.astype(np.float32)


def ray_grid(rotation, width, height, hfov, vfov, intrinsics=None):
    xx, yy = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
    local = np.stack(((2*xx/width-1)*np.tan(np.deg2rad(hfov/2)),
                      (1-2*yy/height)*np.tan(np.deg2rad(vfov/2)),
                      np.ones_like(xx)), axis=-1)
    if intrinsics is not None:
        fx,fy,cx,cy=np.asarray(intrinsics)
        if fx<=0 or fy<=0:raise ValueError("Invalid focal length")
        local=np.stack(((xx-cx)/fx,-(yy-cy)/fy,np.ones_like(xx)),axis=-1)
    return (local @ np.asarray(rotation).T).astype(np.float32)


def static_ray_scene(vertices, faces):
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def render_scene(meshes, position, rotation, width=128, height=96,
                 hfov=66.56, vfov=40.49, near=.05, far=4., static_scene=None, rigid_objects=None, intrinsics=None):
    """Exact first hit across ALL meshes. Rays start on the near clipping plane.

    Depth is optical +Z depth, not Euclidean distance. Invalid depth = inf,
    owner = -1. Normals/back-face winding do not bypass occlusion.
    """
    if not np.allclose(np.asarray(rotation).T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError('Camera rotation not orthogonal')
    ray_scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    labels = []
    for vertices, faces, owner in meshes:
        if not len(faces):
            continue
        if not np.isfinite(vertices).all():
            raise ValueError('Non-finite mesh')
        gid = ray_scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                                     o3d.core.Tensor(np.asarray(faces, np.uint32)))
        assert gid == len(labels)
        labels.append(owner)
    directions = ray_grid(rotation, width, height, hfov, vfov, intrinsics=intrinsics)
    origins = np.asarray(position, np.float32) + near * directions
    rays = np.concatenate((origins, directions), axis=-1)
    if labels:
        hit = ray_scene.cast_rays(o3d.core.Tensor(rays))
        depth = hit['t_hit'].numpy() + near
        gid = hit['geometry_ids'].numpy()
    else:
        depth = np.full((height,width),np.inf,np.float32)
        gid = np.zeros((height,width),np.uint32)
    valid = np.isfinite(depth) & (depth <= far)
    owner = np.full(depth.shape, -1, np.int32)
    owner[valid] = np.asarray(labels, np.int32)[gid[valid]]
    if static_scene is not None:
        static_depth = static_scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()+near
        use_static = np.isfinite(static_depth) & (static_depth<=far) & (static_depth<depth)
        depth[use_static] = static_depth[use_static]
        owner[use_static] = 0
        valid |= use_static
    # Cast rays in each rigid object's local frame against its cached BVH.
    # Rigid transforms preserve ray parameter/depth; no frozen geometry or approximation.
    for cached_scene, object_rotation, object_position, object_owner in (rigid_objects or []):
        object_rotation = np.asarray(object_rotation)
        if not np.allclose(object_rotation.T@object_rotation,np.eye(3),atol=1e-4):
            raise ValueError('Object transform must be rigid')
        local_origins = (origins-np.asarray(object_position))@object_rotation
        local_directions = directions@object_rotation
        local_rays = np.concatenate((local_origins,local_directions),axis=-1).astype(np.float32)
        object_depth = cached_scene.cast_rays(o3d.core.Tensor(local_rays))['t_hit'].numpy()+near
        use_object = np.isfinite(object_depth)&(object_depth<=far)&(object_depth<depth)
        depth[use_object]=object_depth[use_object];owner[use_object]=object_owner;valid|=use_object
    depth[~valid] = np.inf
    points = np.zeros((*depth.shape, 3), np.float32)
    points[valid] = np.asarray(position) + depth[valid, None]*directions[valid]
    return dict(depth=depth, owner=owner, points=points)


def sample_scene_hits(render, count=512):
    ids = np.flatnonzero((render['owner'] >= 0) & (render['owner'] < 100))
    if len(ids) > count:
        ids = ids[np.linspace(0, len(ids)-1, count).astype(int)]
    points = np.zeros((count, 3), np.float32)
    owners = np.full(count, -1, np.int32)
    points[:len(ids)] = render['points'].reshape(-1, 3)[ids]
    owners[:len(ids)] = render['owner'].ravel()[ids]
    return points, owners


def temporal_bps(static_frames, dynamic_frames, camera_position, camera_rotation, anchors, return_valid=False, initial_static=None):
    """Causal static history + CURRENT observed dynamic surfaces, never a trail.

    Unobserved dynamic objects are unknown, not GT-completed. Static surfaces
    observed earlier remain known even when temporarily occluded.
    """
    from scipy.spatial import cKDTree
    history = [] if initial_static is None or not len(initial_static) else [np.asarray(initial_static,np.float32)]
    seen=set()
    if history:
        combined=np.concatenate(history);cells=np.floor(combined/.025).astype(np.int32);_,ix=np.unique(np.ascontiguousarray(cells).view(np.dtype((np.void,12))).reshape(-1),return_index=True);ix.sort();history=[combined[ix]];seen={tuple(c) for c in cells[ix]}
    result = []
    masks = []
    for t, (position, rotation) in enumerate(zip(camera_position, camera_rotation)):
        if len(static_frames[t]):
            incoming=np.asarray(static_frames[t]);cells=np.floor(incoming/.025).astype(np.int32);keep=[]
            for i,c in enumerate(cells):
                key=tuple(c)
                if key not in seen:seen.add(key);keep.append(i)
            if keep:history.append(incoming[keep])
        pieces = history + ([dynamic_frames[t]] if len(dynamic_frames[t]) else [])
        if not pieces:
            result.append(np.zeros_like(anchors, dtype=np.float32))
            masks.append(np.zeros(len(anchors),bool))
            continue
        points = np.concatenate(pieces)
        probes = anchors @ rotation.T + position
        nearest = points[cKDTree(points).query(probes)[1]]
        delta = (nearest-probes) @ rotation
        result.append(delta/np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1)/2)
        masks.append(np.ones(len(anchors),bool))
    result=np.asarray(result,np.float32)
    return (result,np.asarray(masks)) if return_valid else result
