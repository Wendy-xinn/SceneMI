"""GIMO source audit and metric native30 EGO export, without model training.

World = transform_norm(scan) / annotation.scale, as in GIMO's official
evaluation loader. OBJ vertex order is recovered from the preserved face
order, independently for EVERY frame. Released vertices remain the body GT;
mesh-regressed joints are labelled approximations, never kinematic GT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'gimo-native30-ego128-v2'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load_settings(path):
    config = yaml.safe_load(Path(path).read_text())
    if config['version'] != PROTOCOL or config['data']['fps'] != 30:
        raise ValueError('Expected GIMO native30 protocol')
    paths = yaml.safe_load((ROOT / config['paths']).read_text())
    raw = Path(paths['gimo_raw'])
    if not (raw / 'dataset.csv').exists():
        raise FileNotFoundError(raw / 'dataset.csv')
    return config, raw, (ROOT / config['output']).resolve()


def source_rows(raw):
    import csv
    with (Path(raw) / 'dataset.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = []
    for i, row in enumerate(rows):
        if row['training'] not in ('0', '1'):
            raise ValueError('Unknown official split')
        result.append(dict(annotation=i, scene=row['scene'], recording=row['sequence_path'],
            first=int(row['start_frame']), end=int(row['end_frame']),
            transform=row['transformation'], official_split='train' if row['training'] == '1' else 'test'))
    return result


def contiguous_ranges(ids):
    ids = np.asarray(sorted(set(ids)), dtype=np.int64)
    if not len(ids):
        return []
    return [[int(chunk[0]), int(chunk[-1]) + 1]
            for chunk in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)]


def split_rows(rows, config):
    """Freeze official rows plus a separate, deduplicated project overlay.

    Validation comes only from complete train-only recordings. The official
    test stays intact. Train/test overlap is resolved in favour of test;
    repeated frames within one split are owned by the first CSV annotation.
    """
    membership = defaultdict(set)
    for row in rows:
        membership[(row['scene'], row['recording'])].add(row['official_split'])
    candidates = sorted(key for key, value in membership.items() if value == {'train'})
    seed = config['data']['seed']
    candidates.sort(key=lambda x: hashlib.sha256(f'{seed}:{x[0]}/{x[1]}'.encode()).hexdigest())
    fraction = config['data']['validation_fraction_of_train_only_recordings']
    count = min(len(candidates), max(1, round(len(candidates) * fraction))) if candidates and fraction else 0
    validation = set(candidates[:count])
    test_frames = defaultdict(set)
    for row in rows:
        if row['official_split'] == 'test':
            test_frames[(row['scene'], row['recording'])].update(range(row['first'], row['end']))
    used = defaultdict(set)
    result, removed = [], []
    for original in rows:
        row = dict(original)
        key = (row['scene'], row['recording'])
        row['split'] = 'validation' if key in validation else row['official_split']
        full = set(range(row['first'], row['end']))
        overlap = full & test_frames[key] if row['official_split'] == 'train' else set()
        duplicate = (full - overlap) & used[(key, row['split'])]
        keep = full - overlap - duplicate
        used[(key, row['split'])].update(keep)
        row['kept_ranges'] = contiguous_ranges(keep)
        row['retained_frames'] = len(keep)
        row['id'] = f"{row['scene']}__{row['recording']}__a{row['annotation']:03d}"
        result.append(row)
        if overlap or duplicate:
            removed.append(dict(annotation=row['annotation'], test_overlap_ranges=contiguous_ranges(overlap),
                                duplicate_ranges=contiguous_ranges(duplicate)))
    return result, dict(validation_recordings=['/'.join(k) for k in sorted(validation)],
        validation_candidates=len(candidates), validation_fraction=fraction, seed=seed,
        official_test_preserved=True, changes=removed,
        mixed_train_test_recordings=sum(len(v) > 1 for v in membership.values()),
        baseline_exact_membership_verified=False)


def read_pv(path):
    """Keep Windows 100 ns timestamps as integers (float64 loses precision)."""
    lines = Path(path).read_text().splitlines()
    cx, cy, width, height = map(float, lines[0].split(','))
    parts = [line.split(',') for line in lines[1:] if line.strip()]
    times = np.array([int(row[0]) for row in parts], np.int64)
    values = np.array([[float(x) for x in row[1:]] for row in parts], np.float64)
    if values.shape[1] != 18 or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid/nonmonotonic PV timestamps or poses')
    intr = np.c_[values[:, :2], np.full(len(values), cx), np.full(len(values), cy)]
    return times, values[:, 2:].reshape(-1, 4, 4), intr, (int(width), int(height))


def read_obj(path, reference_faces=None, return_sha=False):
    """Fast untextured released SMPL-X OBJ reader with verified permutation."""
    raw = Path(path).read_bytes()
    text = raw.decode()
    first = text.find('\nv ') + 1 if not text.startswith('v ') else 0
    boundary = text.find('\nf ') + 1
    if boundary <= first or '/' in text[boundary:]:
        raise ValueError('Unsupported source body OBJ layout')
    vertices = np.fromstring(text[first:boundary].replace('v ', ''), sep=' ', dtype=np.float32).reshape(-1, 3)
    faces = np.fromstring(text[boundary:].replace('f ', ''), sep=' ', dtype=np.int32).reshape(-1, 3) - 1
    if reference_faces is None:
        return vertices, faces
    value = recover_vertices(vertices, faces, reference_faces)
    return (value, hashlib.sha256(raw).digest()) if return_sha else value


def recover_vertices(vertices, faces, reference_faces):
    if faces.shape != reference_faces.shape:
        raise ValueError('Released mesh does not have the pinned topology')
    mapping = np.full(len(vertices), -1, np.int32)
    mapping[reference_faces.ravel()] = faces.ravel()
    if (np.any(mapping < 0) or len(np.unique(mapping)) != len(vertices)
            or not np.array_equal(mapping[reference_faces], faces)):
        raise ValueError('Cannot recover a bijective standard vertex order')
    return vertices[mapping]


def fit_rigid(template, points):
    """Least-squares proper rotations taking a fixed template into each frame."""
    center = np.mean(template, axis=0)
    target = np.mean(points, axis=-2)
    cov = np.einsum('vi,...vj->...ij', template-center, points-target[..., None, :])
    u, _, vt = np.linalg.svd(cov)
    correction = np.broadcast_to(np.eye(3), cov.shape).copy()
    correction[..., 2, 2] = np.linalg.det(np.swapaxes(vt, -1, -2) @ np.swapaxes(u, -1, -2))
    rotation = np.swapaxes(vt, -1, -2) @ correction @ np.swapaxes(u, -1, -2)
    return rotation, target


def mount_calibration(camera, head):
    from scipy.spatial.transform import Rotation
    relative = np.linalg.inv(camera) @ head
    rotation = Rotation.from_matrix(relative[:, :3, :3]).mean().as_matrix()
    mount = np.eye(4)
    mount[:3, :3] = rotation
    mount[:3, 3] = np.median(relative[:, :3, 3], axis=0)
    predicted = camera @ mount
    position = np.linalg.norm(predicted[:, :3, 3] - head[:, :3, 3], axis=-1)
    angle = Rotation.from_matrix(np.swapaxes(predicted[:, :3, :3], -1, -2) @ head[:, :3, :3]).magnitude()
    return mount, relative, position, np.rad2deg(angle)


def window_starts(frame_ids, valid, frames=128, stride=128):
    frame_ids, valid = np.asarray(frame_ids), np.asarray(valid, bool)
    good = np.flatnonzero(valid)
    if not len(good):
        return []
    breaks = np.flatnonzero((np.diff(good) != 1) | (np.diff(frame_ids[good]) != 1)) + 1
    return [int(part[i]) for part in np.split(good, breaks)
            for i in range(0, len(part)-frames+1, stride)]


def inventory(config, raw, output):
    rows, overlay = split_rows(source_rows(raw), config)
    result = dict(protocol=PROTOCOL, raw=str(raw), csv_sha256=digest(raw/'dataset.csv'),
        csv_rows=len(rows), scenes=len({r['scene'] for r in rows}),
        recordings=len({(r['scene'], r['recording']) for r in rows}),
        official_rows=dict(Counter(r['official_split'] for r in rows)),
        official_frames_including_duplicate_annotations=dict((s, sum(r['end']-r['first'] for r in rows if r['official_split']==s)) for s in ('train','test')),
        project_retained_frames=dict((s, sum(r['retained_frames'] for r in rows if r['split']==s)) for s in ('train','validation','test')),
        overlay=overlay, rows=rows)
    atomic_json(output/'inventory.json', result)
    atomic_json(output/'config.json', config)
    return result


@lru_cache(maxsize=1)
def body_assets():
    import torch
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    torch.set_num_threads(1)
    model = load_model('smplx', 'neutral')
    template = model.v_template.detach().numpy()
    weights = model.lbs_weights.detach().numpy()
    # Back of the skull avoids face, jaw and eye expression deformation.
    head = np.flatnonzero((weights[:, 15] > .98) & (template[:, 2] < 0) & (template[:, 1] > .20))
    if len(head) < 12:
        raise ValueError('Cannot identify a rigid skull surface')
    head = head[np.linspace(0, len(head)-1, min(len(head), 64)).astype(int)]
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import wearer_faces
    occluder, _ = wearer_faces(model.faces, weights)
    return dict(faces=np.asarray(model.faces, np.int32), template=template,
        shapedirs=model.shapedirs.detach().numpy()[:, :, :10],
        regressor=model.J_regressor.detach().numpy()[:22], head=head, occluder=occluder,
        feet=[np.flatnonzero(weights[:, js].sum(1) > .5) for js in ([7, 10], [8, 11])])


@lru_cache(maxsize=1)
def scene_assets(scene_path):
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    path = Path(scene_path)
    v, f = read_scan_obj(path)
    # BVH is built in native scanner space; every annotation uses its own
    # source scale only when expressing the resulting metric world points.
    return v, f, static_ray_scene(v, f), digest(path)


def read_scan_obj(path):
    """Stream scan triangles, including GIMO's six-column XYZ/RGB vertices.

    Bounded text batches avoid retaining millions of Python strings while
    building a scan BVH. Vertex colours and UVs do not change the geometry.
    """
    import re
    vertices, faces, pending = [], [], []
    kind, width, vertex_count = None, None, 0

    def flush():
        if not pending:
            return
        text = re.sub(r'/[^\s]+', '', ' '.join(pending)) if kind == 'f' else ' '.join(pending)
        values = np.fromstring(text, sep=' ', dtype=np.float32 if kind == 'v' else np.int32)
        if values.size != len(pending)*width:
            raise ValueError('Malformed scan OBJ numeric data')
        values = values.reshape(-1, width)
        if kind == 'v':
            if width not in (3, 4, 6) or (width == 4 and not np.all(values[:, 3] == 1)):
                raise ValueError('Unsupported scan vertex layout')
            vertices.append(values[:, :3].copy())
        else:
            if width < 3 or (values == 0).any():
                raise ValueError('Invalid scan face indices')
            values = np.where(values < 0, values+vertex_count, values-1)
            faces.append(np.stack([values[:, [0, j, j+1]] for j in range(1, width-1)], axis=1).reshape(-1, 3))
        pending.clear()

    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith(('v ', 'f ')):
                continue
            fields = line[2:].split('#', 1)[0].split()
            next_kind, next_width = line[0], len(fields)
            if (next_kind, next_width) != (kind, width) or len(pending) >= 16384:
                flush()
                kind, width = next_kind, next_width
            pending.append(' '.join(fields))
            vertex_count += next_kind == 'v'
    flush()
    v = np.concatenate(vertices) if vertices else np.empty((0, 3), np.float32)
    f = np.concatenate(faces).astype(np.int32) if faces else np.empty((0, 3), np.int32)
    if (not len(v) or not len(f)
            or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v)):
        raise ValueError('Invalid scan OBJ geometry')
    return v, f


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def valid_cameras(cameras):
    """Exclude broken source tracking frames without inventing replacements."""
    rotations = cameras[:, :3, :3]
    errors = np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations-np.eye(3)), axis=(1, 2))
    valid = (np.isfinite(cameras).all(axis=(1, 2)) & (errors <= .01)
             & (np.linalg.det(rotations) > .99)
             & np.isclose(cameras[:, 3], [0, 0, 0, 1]).all(-1))
    return valid, errors


def prepare_segment(task):
    config, raw_string, output_string, row = task
    raw, output = Path(raw_string), Path(output_string)
    folder = output/'segments'/row['id']
    record = folder/'metadata.json'
    implementation = digest(__file__)
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['implementation_sha256'] != implementation:
            raise ValueError('Preprocessor changed; use a new output folder')
        if any(digest(folder/name) != sha for name, sha in saved['output_sha256'].items()):
            raise ValueError('Corrupt resumed segment')
        return saved
    if not row['retained_frames']:
        return dict(**row, skipped='duplicate_annotation')
    started = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    source = raw/row['scene']/row['recording']
    holo_path = source/'transform_scene2scene.txt'
    if not holo_path.exists():
        # Keep the original official membership visible. Missing calibration
        # cannot be replaced by an identity matrix or a body-derived camera.
        saved = dict(**row, skipped='missing_released_camera_to_scene_calibration',
            source_folder=str(source), missing_file=str(holo_path),
            available_camera=str(source/f"{row['recording']}_pv.txt"),
            available_motion=str(source/'smplx_local'))
        atomic_json(folder/'unavailable.json', saved)
        return saved
    assets = body_assets()
    frames = np.concatenate([np.arange(a, b) for a, b in row['kept_ranges']])
    times, camera, intr, size = read_pv(source/f"{row['recording']}_pv.txt")
    if frames[-1] >= len(times):
        raise ValueError(f"{row['id']}: motion/PV index out of range")
    info_path = source/row['transform']
    info = json.loads(info_path.read_text())
    scale = float(info['scale'])
    p2s = np.asarray(info['transformation'], np.float64)
    norm_path = source.parent/'scene_obj/transform_norm.txt'
    norm = np.loadtxt(norm_path).reshape(4, 4)
    h2s = np.eye(4)
    h2s[:3] = np.loadtxt(holo_path).reshape(-1, 4)[:3]
    s2w = norm.copy()
    s2w[:3] /= scale
    p2w = s2w @ p2s
    p2w[:3, :3] *= scale
    cameras = s2w @ h2s @ camera[frames]
    # s2w contains 1/scale; camera basis vectors must remain unit vectors.
    cameras[:, :3, :3] *= scale
    from scipy.spatial.transform import Rotation
    camera_valid, errors = valid_cameras(cameras)
    if not camera_valid.any():
        return dict(**row, skipped='all_source_camera_frames_invalid')
    cameras[camera_valid, :3, :3] = Rotation.from_matrix(cameras[camera_valid, :3, :3]).as_matrix()
    cameras[~camera_valid] = np.eye(4)
    intr = intr[frames].copy()
    cfg = config['scene']
    intr *= [cfg['width']/size[0], cfg['height']/size[1], cfg['width']/size[0], cfg['height']/size[1]]
    scene_path = source.parent/'scene_obj/textured_output.obj'
    if not scene_path.exists():
        scene_path = source.parent/'scene_obj/scene.obj'
    scene_v, scene_f, scanner_scene, scene_sha = scene_assets(str(scene_path))
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import render_scene, sample_scene_hits
    n, point_count = len(frames), cfg['points_per_frame']
    vertices = np.lib.format.open_memmap(folder/'vertices_world.npy', mode='w+', dtype=np.float32, shape=(n, 10475, 3))
    points = np.empty((n, point_count, 3), np.float32)
    owners = np.empty((n, point_count), np.int16)
    depth = np.empty((n, cfg['height'], cfg['width']), np.float16)
    hit_owner = np.empty(depth.shape, np.int16)
    joints = np.empty((n, 22, 3), np.float32)
    head_points = np.empty((n, len(assets['head']), 3), np.float32)
    foot_distance = np.empty((n, 2), np.float32)
    params = defaultdict(list)
    frame_hash = hashlib.sha256()
    valid = np.ones(n, bool)
    invalid_frames = []
    parameter_shapes = dict(orient=(3,), trans=(3,), latent=(32,), jaw_pose=(3,),
        beta=(10,), expression=(10,), lhand=(45,), rhand=(45,))
    # Render in scanner coordinates to reuse the exact complete scan BVH.
    for i, frame in enumerate(frames):
        obj = source/'smplx_local'/f'{frame}.obj'
        pkl = obj.with_suffix('.pkl')
        try:
            if not camera_valid[i]:
                raise ValueError('Nonrigid or nonfinite released PV pose; excluded without interpolation')
            original, obj_sha = read_obj(obj, assets['faces'], return_sha=True)
            pkl_bytes = pkl.read_bytes()
            values = pickle.loads(pkl_bytes)
            values = {key:np.asarray(values[key].detach().cpu() if hasattr(values[key],'detach') else values[key],np.float32)
                for key in parameter_shapes}
            if any(v.shape!=parameter_shapes[k] or not np.isfinite(v).all() for k,v in values.items()):
                raise ValueError('Invalid source pose parameters')
        except (UnicodeError, ValueError, OSError, pickle.UnpicklingError, EOFError, RuntimeError, KeyError) as exc:
            valid[i] = False
            invalid_frames.append(dict(frame=int(frame),error=f'{type(exc).__name__}: {exc}'))
            vertices[i]=0;points[i]=0;owners[i]=-1;depth[i]=np.inf;hit_owner[i]=-1
            joints[i]=0;head_points[i]=0;foot_distance[i]=0
            for key,shape in parameter_shapes.items():params[key].append(np.zeros(shape,np.float32))
            for path in (obj,pkl):
                if path.exists():frame_hash.update(bytes.fromhex(digest(path)))
            continue
        world = transform_points(original, p2w).astype(np.float32)
        if not np.isfinite(world).all():
            raise ValueError('Invalid released body mesh')
        vertices[i] = world
        joints[i] = assets['regressor'] @ world
        head_points[i] = world[assets['head']]
        for key, value in values.items():
            params[key].append(value)
        frame_hash.update(obj.name.encode()); frame_hash.update(obj_sha)
        frame_hash.update(hashlib.sha256(pkl_bytes).digest())
        scan_camera = h2s @ camera[frame]
        scan_camera[:3, :3] = Rotation.from_matrix(scan_camera[:3, :3]).as_matrix()
        scan_body = transform_points(original * scale, p2s).astype(np.float32)
        render = render_scene([(scan_body, assets['occluder'], 100)], scan_camera[:3, 3],
            scan_camera[:3, :3] @ np.diag([1., 1., -1.]), width=cfg['width'], height=cfg['height'],
            near=cfg['near_m']*scale, far=cfg['far_m']*scale,
            static_scene=scanner_scene, intrinsics=intr[i])
        sampled, own = sample_scene_hits(render, point_count)
        points[i] = transform_points(sampled, s2w)
        points[i, own < 0] = 0
        owners[i] = own
        depth[i] = render['depth']/scale
        hit_owner[i] = render['owner']
        import open3d as o3d
        for foot, ids in enumerate(assets['feet']):
            sole = scan_body[ids]
            sole = sole[sole[:, 1] <= np.quantile(sole[:, 1], .1)]
            distances = scanner_scene.compute_distance(o3d.core.Tensor(sole.astype(np.float32))).numpy()/scale
            foot_distance[i, foot] = np.median(distances)
    vertices.flush()
    if not valid.any():
        return dict(**row,skipped='all_source_body_frames_invalid',invalid_frames=invalid_frames)
    shaped = assets['template'] + np.einsum('vci,i->vc', assets['shapedirs'], params['beta'][np.flatnonzero(valid)[0]])
    head_rotation, head_center = fit_rigid(shaped[assets['head']], head_points[valid])
    head = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    head[valid, :3, :3], head[valid, :3, 3] = head_rotation, head_center
    mount, relative_good, position_good, angle_good = mount_calibration(cameras[valid], head[valid])
    relative=np.broadcast_to(np.eye(4),(n,4,4)).copy();relative[valid]=relative_good
    position_error=np.zeros(n);angle_error=np.zeros(n)
    position_error[valid]=position_good;angle_error[valid]=angle_good
    # HoloLens head tracking provides a separate sensor check. No GT-derived
    # body correction or camera smoothing is applied.
    hhe_path = source/f"{row['recording']}_head_hand_eye.csv"
    hhe_rows = [line.split(',') for line in hhe_path.read_text().splitlines() if line.strip()]
    hhe_t = np.array([int(x[0]) for x in hhe_rows], np.int64)
    hhe = np.array([[float(x) for x in line[1:17]] for line in hhe_rows]).reshape(-1, 4, 4)
    order = np.argsort(hhe_t)
    hhe_t, hhe = hhe_t[order], hhe[order]
    selected_times = times[frames]
    right = np.clip(np.searchsorted(hhe_t, selected_times), 0, len(hhe_t)-1)
    left = np.maximum(right-1, 0)
    nearest = np.where(np.abs(hhe_t[left]-selected_times) <= np.abs(hhe_t[right]-selected_times), left, right)
    sensor_head = s2w @ h2s @ hhe[nearest]
    sensor_head[:, :3, :3] *= scale
    time_error = np.abs(hhe_t[nearest]-selected_times)/1e7
    arrays = dict(source_frame_ids=frames.astype(np.int32), source_timestamps_100ns=selected_times,
        source_bvh_frame_ids=np.load(source/'pose_index.npy')[frames], camera_world=cameras.astype(np.float32),
        intrinsics=intr.astype(np.float32), head_vertex_world=head.astype(np.float32),
        camera_from_head_vertex=relative.astype(np.float32), camera_from_head_vertex_fixed=mount.astype(np.float32),
        sensor_head_world=sensor_head.astype(np.float32), head_sensor_time_error_s=time_error.astype(np.float32),
        head_vertex_ids=assets['head'], joints_mesh_regressed_world=joints,
        visible_frame_points=points, visible_frame_owner=owners,
        depth=depth, depth_owner=hit_owner, observation_valid=valid,
        foot_scene_surface_distance_m=foot_distance,
        camera_head_position_residual_m=position_error.astype(np.float32), source_camera_valid=camera_valid,
        camera_head_rotation_residual_deg=angle_error.astype(np.float32),
        scanner_to_world=s2w.astype(np.float32), body_to_world=p2w.astype(np.float32))
    good = points[owners == 0]
    if len(good):
        _, ids = np.unique(np.floor(good/cfg['voxel_m']).astype(np.int32), axis=0, return_index=True)
        union = good[np.sort(ids)]
    else:
        union = np.empty((0, 3), np.float32)
    arrays['observed_static_union'] = union
    for name, value in arrays.items():
        with (folder/(name+'.npy')).open('wb') as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    np.savez_compressed(folder/'source_parameters.npz', **{k:np.stack(v) for k,v in params.items()})
    starts = window_starts(frames, valid, config['data']['frames'], config['data']['stride'])
    coverage = np.zeros(n, bool)
    for start in starts:
        coverage[start:start+config['data']['frames']] = True
    dt = np.diff(selected_times)/1e7
    adjacent = valid[1:] & valid[:-1]
    speed = (np.linalg.norm(np.diff(cameras[:, :3, 3], axis=0), axis=1)/np.maximum(dt, 1e-7))[adjacent]
    count = (owners == 0).sum(1)
    quality = dict(camera_rotation_orthogonal_max_error_before_projection=float(errors[camera_valid].max()),
        camera_head_position_residual_m=dict(median=float(np.median(position_good)), p90=float(np.quantile(position_good, .9)), maximum=float(position_good.max())),
        camera_head_rotation_residual_deg=dict(median=float(np.median(angle_good)), p90=float(np.quantile(angle_good, .9)), maximum=float(angle_good.max())),
        frames_with_no_scene_input=int((count == 0).sum()), median_scene_points=float(np.median(count)),
        static_depth_hit_fraction=float((hit_owner == 0).mean()), body_occlusion_fraction=float((hit_owner == 100).mean()),
        head_sensor_time_error_ms_p95=float(np.quantile(time_error, .95)*1000),
        timestamp_dt_s_p01_p99=np.quantile(dt, [.01, .99]).tolist() if len(dt) else [],
        camera_speed_mps_p99=float(np.quantile(speed, .99)) if len(speed) else 0.,
        sole_nearest_scene_distance_m_median=float(np.median(foot_distance[valid])),
        note='Heuristic review candidates; no source body alignment, floor snapping, pose repair, or physical certification')
    bvh_ids = arrays['source_bvh_frame_ids']
    quality['nonmonotonic_source_bvh_steps'] = int((np.diff(bvh_ids) <= 0).sum())
    quality['invalid_source_body_frames'] = len(invalid_frames)
    quality['invalid_source_camera_frames'] = int((~camera_valid).sum())
    quality['native_timestamp_duration_s'] = float((selected_times[-1]-selected_times[0])/1e7)
    quality['review_flags'] = [label for label, yes in (
        ('camera_head_position_p90_over_10cm', quality['camera_head_position_residual_m']['p90'] > .1),
        ('camera_head_rotation_p90_over_15deg', quality['camera_head_rotation_residual_deg']['p90'] > 15),
        ('empty_scene_frames', quality['frames_with_no_scene_input'] > 0),
        ('corrupt_source_body_frames', len(invalid_frames)>0),
        ('camera_speed_p99_over_5mps', quality['camera_speed_mps_p99'] > 5),
        ('nonmonotonic_source_bvh_steps', quality['nonmonotonic_source_bvh_steps'] > 0),
        ('sole_distance_median_over_15cm', quality['sole_nearest_scene_distance_m_median'] > .15)) if yes]
    saved = dict(**row, protocol=PROTOCOL, folder=str(folder), frames=n, fps=30,
        window_starts=starts, windows_128=len(starts), covered_frames=int(coverage.sum()), tail_or_invalid_frames=int((~coverage).sum()),
        scene_mesh=str(scene_path), scene_mesh_sha256=scene_sha, observed_static_union_points=len(union),
        camera_convention='released PV OpenGL: +X right, +Y up, -Z forward; matrices map camera to world',
        world_convention='GIMO canonical Y-up; transform_norm(scanner)/annotation.scale; metres in source SMPL-X scale',
        scene_input='synthetic first-hit scan observations with original body occlusion, wearer head excluded; 512 sampled static hits/frame',
        memory_scope='segment observations; scenes action builds split/recording/world-isolated offline map unions',
        calibration_scope='diagnostic GT fit within annotation; not a model input or a cross-split calibration',
        joints_protocol='SMPL-X regressor applied to released posed skin; approximate display joints, NOT exact FK joints or baseline metric targets',
        pose_protocol='original latent/shape/hands preserved as float32 on valid source frames; exact decoder export is separate',
        implementation_sha256=implementation, config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        invalid_source_frames=invalid_frames,
        inputs_sha256={str(p):digest(p) for p in (info_path, norm_path, holo_path, source/f"{row['recording']}_pv.txt", hhe_path, source/'pose_index.npy')},
        source_body_files_sha256=frame_hash.hexdigest(), quality=quality, elapsed_s=time.monotonic()-started,
        output_sha256={p.name:digest(p) for p in sorted(folder.iterdir()) if p.suffix in ('.npy', '.npz')})
    atomic_json(record, saved)
    return saved


def summarize(config, output, inventory_record, records):
    windows, table = [], []
    for split in ('train', 'validation', 'test'):
        selected = [r for r in records if r['split'] == split and 'skipped' not in r]
        table.append(dict(split=split, annotation_segments=len(selected),
            recordings=len({(r['scene'],r['recording']) for r in selected}), scenes=len({r['scene'] for r in selected}),
            frames=sum(r['frames'] for r in selected), windows_128=sum(r['windows_128'] for r in selected),
            window_frames=sum(r['covered_frames'] for r in selected), tail_or_invalid_frames=sum(r['tail_or_invalid_frames'] for r in selected),
            review_flagged_segments=sum(bool(r['quality']['review_flags']) for r in selected)))
        for row in selected:
            frames = np.load(Path(row['folder'])/'source_frame_ids.npy')
            for start in row['window_starts']:
                windows.append(dict(split=split, segment=row['id'], folder=row['folder'], start=start, frames=128,
                    scene=row['scene'], recording=row['recording'], source_first=int(frames[start]), source_end_exclusive=int(frames[start+127])+1))
    all_positions = [np.load(Path(r['folder'])/'camera_head_position_residual_m.npy')[np.load(Path(r['folder'])/'observation_valid.npy')] for r in records if 'skipped' not in r]
    all_angles = [np.load(Path(r['folder'])/'camera_head_rotation_residual_deg.npy')[np.load(Path(r['folder'])/'observation_valid.npy')] for r in records if 'skipped' not in r]
    positions, angles = np.concatenate(all_positions), np.concatenate(all_angles)
    failures = [r for r in records if r.get('skipped') == 'preprocessing_failed']
    summary = dict(protocol=PROTOCOL, status='prepared_with_failures' if failures else 'prepared',
        complete=len(records)==len(inventory_record['rows']) and not failures,
        table=table, segments=len(records), windows=len(windows), fps=30, frames_per_window=128,
        unavailable_segments=[dict(annotation=r['annotation'],id=r['id'],split=r['split'],frames=r['retained_frames'],reason=r['skipped'])
            for r in records if 'skipped' in r and r['skipped'] != 'duplicate_annotation'],
        window_duration_s=128/30, overlay=inventory_record['overlay'],
        camera_head_position_residual_cm_median_p90_p99=(np.quantile(positions,[.5,.9,.99])*100).tolist(),
        camera_head_rotation_residual_deg_median_p90_p99=np.quantile(angles,[.5,.9,.99]).tolist(),
        baseline_protocol=dict(exact_GIMO_split_membership_published=False, source_csv_preserved=True,
            s2fusion_CIRCLE_70_30_is_not_GIMO_protocol=True, s2fusion_frames=120, project_frames=128,
            s2fusion_inputs='head + both hands; pre-scanned scene crops', project_inputs='released ego camera; rendered visible scan surfaces',
            published_results_directly_comparable=False),
        training_started=False, exact_pose_FK_export_ready=False,
        caveat='Original meshes/camera/scene ready for source review. Exact FK pose export requires the original matching VPoser decoder; mesh-regressed joints must not be used as official benchmark targets.')
    atomic_json(output/'summary.json', summary)
    atomic_json(output/'segments.json', dict(protocol=PROTOCOL, segments=records))
    (output/'windows.jsonl').write_text(''.join(json.dumps(w,ensure_ascii=False)+'\n' for w in windows))
    import csv
    with (output/'statistics.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    return summary


def prepare(config, raw, output, limit=None, workers=None):
    workers = workers or config['runtime']['workers']
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four bounded preprocessing workers')
    inv = inventory(config, raw, output)
    rows = inv['rows'][:limit] if limit else inv['rows']
    records=[]
    tasks=[(config,str(raw),str(output),row) for row in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_runtime) as pool:
        pending={pool.submit(prepare_segment,task):task[-1]['id'] for task in tasks}
        for future in as_completed(pending):
            try:
                row=future.result()
            except Exception as exc:
                original=next(r for r in rows if r['id']==pending[future])
                row=dict(**original,skipped='preprocessing_failed',error=f'{type(exc).__name__}: {exc}')
                atomic_json(output/'failures'/(row['id']+'.json'),row)
            records.append(row)
            print(json.dumps(dict(done=len(records),total=len(rows),id=row['id'],frames=row.get('frames',0),
                windows=row.get('windows_128',0),seconds=round(row.get('elapsed_s',0),1),quality=row.get('quality',{}).get('review_flags',[]))),flush=True)
            atomic_json(output/'progress.json',dict(done=len(records),total=len(rows),last=row['id']))
    records.sort(key=lambda x:x['annotation'])
    return summarize(config,output,inv,records)


def worker_runtime():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def verify(output):
    info=json.loads((output/'segments.json').read_text())
    used=defaultdict(set)
    for row in info['segments']:
        if 'skipped' in row:
            continue
        folder=Path(row['folder'])
        for name, sha in row['output_sha256'].items():
            if digest(folder/name)!=sha:
                raise ValueError(f'Corrupt {folder/name}')
        frames=np.load(folder/'source_frame_ids.npy')
        valid=np.load(folder/'observation_valid.npy')
        for frame in frames:
            key=(row['scene'],row['recording'],int(frame))
            if used[key] and used[key]!={row['split']}:
                raise ValueError('Cross-split source frame leakage')
            used[key].add(row['split'])
        for start in row['window_starts']:
            if (len(frames[start:start+128])!=128 or np.any(np.diff(frames[start:start+128])!=1)
                    or not valid[start:start+128].all()):
                raise ValueError('Invalid 128-frame slice')
    result=dict(passed=True,source_frames=len(used),segments=len(info['segments']),
        array_hashes_verified=True,no_cross_split_source_frame_overlap=True,all_windows_128_contiguous=True)
    if (output/'pose_summary.json').exists():
        pose=json.loads((output/'pose_summary.json').read_text())
        for row in pose['records']:
            folder=output/'segments'/row['id']
            for name, sha in row['output_sha256'].items():
                if digest(folder/name)!=sha:
                    raise ValueError('Corrupt exact pose: '+str(folder/name))
        result['exact_pose_arrays_verified']=True
    if (output/'scene_manifest.json').exists():
        from scenemi.data.gimo_scene import verify as verify_scenes
        result['scenes']=verify_scenes(output)
    atomic_json(output/'verification.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','prepare','verify','scenes','review','serve'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/gimo128.yaml')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--port',type=int,default=8850)
    args=parser.parse_args(argv)
    config,raw,output=load_settings(args.config)
    if args.action=='inventory':
        result=inventory(config,raw,output); result={k:v for k,v in result.items() if k!='rows'}
    elif args.action=='prepare':
        result=prepare(config,raw,output,args.limit,args.workers)
    elif args.action=='verify':
        result=verify(output)
    elif args.action=='scenes':
        from scenemi.data.gimo_scene import prepare as prepare_scenes
        result=prepare_scenes(config, output)
    else:
        from scenemi.gimo_visualization import review,serve
        result=review(output) if args.action=='review' else serve(output,args.port)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
