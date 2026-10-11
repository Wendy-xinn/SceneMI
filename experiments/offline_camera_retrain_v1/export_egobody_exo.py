"""Offline calibrated exo observations from geometry, with no RGB requirement.

Exact triangle first hits; OpenCV rays use the official rational distortion.
Bodies occlude the scan but never become static scene observations. Joint
availability comes from visible skin regions, not rays to internal joints.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.data import HERE, PROJECTS, SOURCE
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from experiments.offline_camera_retrain_v1.export_native20_training_scenes import resample_fits
from experiments.offline_camera_retrain_v1.export_rest_joints import load_model

PROTOCOL = 'offline-egobody-master-exo-nearclip-firsthit-distorted-pixels-v2'


def camera_rays(calibration, transform, width=320, height=180):
    k = np.asarray(calibration['camera_mtx'], np.float64)
    distortion = np.asarray(calibration['k'], np.float64)
    x, y = np.meshgrid((np.arange(width) + .5) * 1920 / width,
                       (np.arange(height) + .5) * 1080 / height)
    pixels = np.stack((x, y), -1)
    # undistortPointsIter avoids the five-iteration default near wide-angle edges.
    normalized = cv2.undistortPointsIter(
        pixels.reshape(-1, 1, 2), k, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10)).reshape(-1, 2)
    directions = np.column_stack((normalized, np.ones(len(normalized)))) @ transform[:3, :3].T
    origins = np.broadcast_to(transform[:3, 3], directions.shape)
    return np.column_stack((origins, directions)).astype(np.float32), pixels.reshape(-1, 2)


def regional_faces(model):
    weights = model.lbs_weights.detach().cpu().numpy()
    regions = weights.argmax(1)
    parents = model.parents.detach().cpu().numpy()
    for j in range(22, len(parents)):
        ancestor = j
        while ancestor >= 22:
            ancestor = int(parents[ancestor])
        regions[regions == j] = ancestor
    # Majority skinning support across the triangle, rather than a bone centre.
    face_weights = weights[model.faces].sum(1)
    for j in range(22, len(parents)):
        ancestor = j
        while ancestor >= 22:
            ancestor = int(parents[ancestor])
        face_weights[:, ancestor] += face_weights[:, j]
    return face_weights[:, :22].argmax(1).astype(np.int32)


def cast_body(vertices, faces, rays):
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(faces, np.uint32)))
    clipped = rays.copy()
    clipped[:, :3] += .1 * clipped[:, 3:]
    result = scene.cast_rays(o3d.core.Tensor(clipped))
    return result['t_hit'].numpy() + .1, result['primitive_ids'].numpy()


def export_one(row, original, information, poses, output, model_cache, scene_cache,
               width=320, height=180, device='cuda'):
    started = time.monotonic()
    folder = output / row['split'] / row['sequence_id']
    metadata_path = folder / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('protocol') != PROTOCOL:
            raise ValueError('Existing exo cache has another protocol')
        for filename, digest in metadata['output_sha256'].items():
            if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Corrupt exo observation: {filename}')
        return metadata
    folder.mkdir(parents=True, exist_ok=True)
    record = row['recording']
    root = PROJECTS / 'egobody'
    calibration_path = root / 'kinect_cam_params/kinect_master/Color.json'
    transform_path = root / 'calibrations' / record / 'cal_trans/kinect12_to_world' / f"{row['scene_id']}.json"
    calibration = json.loads(calibration_path.read_text())
    transform = np.asarray(json.loads(transform_path.read_text())['trans'], np.float64)
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3) or np.linalg.det(transform[:3, :3]) < .999:
        raise ValueError('Invalid external camera rotation')
    rays, pixels = camera_rays(calibration, transform, width, height)
    key = (record, row['scene_id'], width, height)
    if key not in scene_cache:
        mesh_path = Path(original[f"{record}/{row['role']}"]['scene_mesh'])
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        clipped = rays.copy()
        clipped[:, :3] += .1 * clipped[:, 3:]
        depths = scene.cast_rays(o3d.core.Tensor(clipped))['t_hit'].numpy() + .1
        scene_cache.clear()
        scene_cache[key] = (depths, str(mesh_path))
    static_depth, mesh_path = scene_cache[key]
    q = row['source_first_frame'] + np.arange(row['frames_20fps']) * 1.5
    first, last = int(np.floor(q[0])), int(np.ceil(q[-1]))
    roles = [row['role'], 'interactee' if row['role'] == 'camera_wearer' else 'camera_wearer']
    people = []
    source_digest = hashlib.sha256(calibration_path.read_bytes() + transform_path.read_bytes())
    for role in roles:
        fit_dir = Path(original[f'{record}/{role}']['body_fits'])
        idx = int(fit_dir.name.rsplit('_', 1)[1])
        gender = information[record][f'body_idx_{idx}'].split()[-1].lower()
        if gender not in model_cache:
            model_cache[gender] = load_model('smpl', gender).to(device)
        model = model_cache[gender]
        fits = []
        for frame in range(first, last + 1):
            path = fit_dir / f'results/frame_{frame:05d}/000.pkl'
            raw = path.read_bytes()
            source_digest.update(raw)
            fits.append(pickle.loads(raw, encoding='latin1'))
        params = resample_fits(fits, q - first, 'smpl')
        beta = np.asarray(poses[row['sequence_id']]['betas'], np.float32) if role == row['role'] else np.median(params['betas'], axis=0)
        params['betas'] = np.broadcast_to(beta, params['betas'].shape).copy()
        people.append((model, params, regional_faces(model), gender))
    n = len(q)
    joints = np.zeros((n, 22, 3), np.float32)
    availability = np.zeros((n, 22), bool)
    confidence = np.zeros((n, 22), np.float32)
    counts = np.zeros((n, 22), np.int32)
    ever_observed = np.zeros(len(rays), bool)
    snapshots = {}
    near, far = .1, 12.
    for lo in range(0, n, 64):
        hi = min(n, lo + 64)
        bodies = []
        for model, params, labels, gender in people:
            with torch.inference_mode():
                tensors = {k: torch.from_numpy(v[lo:hi].copy()).to(device) for k, v in params.items()}
                fitted = model(**tensors, return_verts=True)
                vertices = fitted.vertices.cpu().numpy() @ transform[:3, :3].T + transform[:3, 3]
                body_joints = fitted.joints[:, :22].cpu().numpy() @ transform[:3, :3].T + transform[:3, 3]
            bodies.append((vertices, body_joints, model.faces, labels))
        for offset, t in enumerate(range(lo, hi)):
            depth = static_depth.copy()
            owner = np.where(np.isfinite(depth) & (depth >= near) & (depth <= far), 0, -1)
            depth[owner < 0] = np.inf
            target_depth = target_part = None
            for person, (vertices, body_joints, faces, labels) in enumerate(bodies):
                bd, primitive = cast_body(vertices[offset], faces, rays)
                hits = np.isfinite(bd) & (bd >= near) & (bd <= far)
                parts = np.full(len(rays), -1, np.int32)
                parts[hits] = labels[primitive[hits]]
                if person == 0:
                    target_depth, target_part = bd, parts
                    joints[t] = body_joints[offset]
                closer = hits & (bd < depth)
                depth[closer] = bd[closer]
                owner[closer] = person + 100
            reference = np.bincount(target_part[target_part >= 0], minlength=22)
            visible = np.bincount(target_part[owner == 100], minlength=22)
            fraction = visible / np.maximum(reference, 1)
            # Pixel support and foreground coverage jointly determine reliability.
            availability[t] = (visible >= 2) & (fraction >= .15)
            confidence[t] = fraction * np.minimum(visible / 12., 1.)
            confidence[t, ~availability[t]] = 0
            counts[t] = visible
            ever_observed |= owner == 0
            if t in (0, n // 2, n - 1):
                snapshots[f'owner_{t}'] = owner.reshape(height, width)
                snapshots[f'parts_{t}'] = np.where(owner == 100, target_part, -1).reshape(height, width)
                snapshots[f'depth_{t}'] = depth.reshape(height, width)
        if hi == n or hi % 256 == 0:
            print(f"exo {row['split']} {row['sequence_id']}: {hi}/{n}", flush=True)
    visible_scene = rays[ever_observed, :3] + static_depth[ever_observed, None] * rays[ever_observed, 3:]
    _, unique = np.unique(np.floor(visible_scene / .025).astype(np.int32), axis=0, return_index=True)
    visible_scene = visible_scene[np.sort(unique)].astype(np.float32)
    # Frozen synthetic conditions contain no hidden joint values or rotations.
    joints[~availability] = 0
    arrays = {'source_frame_ids.npy': q, 'visible_joints_world.npy': joints,
              'position_available.npy': availability, 'position_confidence.npy': confidence,
              'visible_pixel_counts.npy': counts, 'static_visible_points_world.npy': visible_scene,
              'sensor_to_world.npy': transform.astype(np.float32)}
    for name, array in arrays.items():
        np.save(folder / name, array)
    np.savez_compressed(folder / 'visibility_snapshots.npz', **snapshots)
    metadata = {'protocol': PROTOCOL, 'dataset': 'egobody', 'split': row['split'],
                'sequence_id': row['sequence_id'], 'recording': record, 'role': row['role'],
                'frames': n, 'output_fps': 20, 'width': width, 'height': height,
                'near_m': near, 'far_m': far, 'input_kind': 'ideal_geometry_simulated_position_only',
                'time_scope': 'offline_complete_available_segment_union',
                'scene_mesh': mesh_path, 'calibration': str(calibration_path),
                'world_transform': str(transform_path), 'rotation_available': False,
                'visibility': 'exact first-hit skin-region pixel support; complete target/partner meshes and static scan',
                'body_shape_protocol': 'known native shape for controlled simulation; no real-video shape-estimation claim',
                'static_points': len(visible_scene), 'available_joint_fraction': float(availability.mean()),
                'available_head_fraction': float(availability[:, 15].mean()),
                'source_sha256': source_digest.hexdigest(), 'elapsed_s': time.monotonic() - started,
                'output_sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in arrays}}
    temp = folder / 'metadata.tmp'
    temp.write_text(json.dumps(metadata, indent=2))
    temp.replace(metadata_path)
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=HERE / 'data/egobody_exo_offline_v2')
    p.add_argument('--split', choices=['train', 'validation', 'all'], default='all')
    p.add_argument('--sequence')
    p.add_argument('--limit', type=int)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=180)
    args = p.parse_args()
    torch.set_num_threads(2)
    original = {r['sequence_id']: r for r in map(json.loads, (SOURCE / 'sequences.jsonl').read_text().splitlines())}
    information = {r['recording_name']: r for r in csv.DictReader((PROJECTS / 'egobody/data_info_release.csv').open())}
    models, scenes, manifest = {}, {}, []
    splits = ['validation', 'train'] if args.split == 'all' else [args.split]
    args.output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = [dict(r, split=split) for r in map(json.loads, (SOURCE / f'egobody_sequences/{split}.jsonl').read_text().splitlines())]
        if args.sequence:
            rows = [r for r in rows if r['sequence_id'] == args.sequence]
        if args.limit:
            rows = rows[:args.limit]
        poses = {r['sequence_id']: r for r in map(json.loads, (HERE / f'data/poses/{split}.jsonl').read_text().splitlines())}
        for row in rows:
            meta = export_one(row, original, information, poses, args.output, models, scenes,
                              args.width, args.height)
            manifest.append(dict(split=split, sequence_id=row['sequence_id'], recording=row['recording'],
                                 path=str((args.output / split / row['sequence_id']).resolve()),
                                 frames=meta['frames'], static_points=meta['static_points'],
                                 available_joint_fraction=meta['available_joint_fraction']))
            atomic = args.output / 'manifest.tmp'
            atomic.write_text(''.join(json.dumps(r) + '\n' for r in manifest))
            atomic.replace(args.output / 'manifest.jsonl')
    print(json.dumps({'segments': len(manifest), 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
