"""Versioned moving EXO observations, separate from sealed native30 exports.

Reuse source camera motion with explicitly synthetic optics. Recast the static
scene, dynamic objects and complete target body for every frame. This module
does not train a model, filter examples, or populate a future-inclusive map.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import numpy as np

PROTOCOL = 'trumans-exo-moving-synthetic-optics-v1'


def camera_trajectory(released, policy):
    """Apply one declared optical-frame conversion to the recovered poses."""
    if policy['protocol'] != PROTOCOL or policy['trajectory'] != 'paired_smplx_camera_to_world':
        raise ValueError('Unknown moving-camera protocol')
    if policy['benchmark_camera_verified'] is not False:
        raise ValueError('Synthetic optics must not claim release calibration')
    axes = np.asarray(policy['optical_from_release'], float)
    if not np.array_equal(axes, [1, -1, -1]):
        raise ValueError('This protocol declares X right, Y up, -Z forward in the release frame')
    pose = np.asarray(released, np.float64).copy()
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Invalid recovered camera clock or transforms')
    if not np.allclose(pose[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous camera transforms')
    r = pose[:, :3, :3]
    if (not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=2e-4)
            or not np.allclose(np.linalg.det(r), 1, atol=2e-4)):
        raise ValueError('Camera rotations must be proper rigid rotations')
    pose[:, :3, :3] = r @ np.diag(axes)
    w, h = policy['image_size']
    hfov = float(policy['horizontal_fov_deg'])
    if min(w, h) <= 0 or not 1 < hfov < 179 or policy['square_pixels'] is not True:
        raise ValueError('Invalid synthetic pinhole intrinsics')
    if policy['near_m'] != .1 or policy['far_m'] <= .1 or policy['environment_samples'] != 512:
        raise ValueError('Incompatible native first-hit clipping or sample count')
    focal = w / (2 * np.tan(np.deg2rad(hfov / 2)))
    camera = dict(kind='synthetic_moving_observer', view_id='paired_release_motion_synthetic_optics',
        selection='Recovered per-frame pose with one fixed optical conversion; no GT look-at or visibility selection',
        K=np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.]]),
        distortion=np.zeros(8), source_size=[w, h], size=[w, h],
        near=policy['near_m'], far=policy['far_m'], benchmark_camera_verified=False,
        release_intrinsics_verified=False, release_optical_axes_verified=False,
        optical_convention='+X right, +Y down, +Z forward (OpenCV)',
        optical_from_release=axes.tolist(), calibration=None,
        position_source='paired global/camera SMPL-X with native rest-pelvis pivot',
        rotation_source='paired coordinate-frame rotation plus declared synthetic optical conversion')
    return pose.astype(np.float32), camera


def render_frame(camera, pose, static_scene, bodies, face_labels, objects, frame, valid=True):
    """Recompute *static* depth too: it changes when the observer moves."""
    import open3d as o3d
    from experiments.offline_camera_retrain_v1.native30_exo import camera_rays, first_hit
    observer = dict(camera, position=pose[:3, 3], rotation=pose[:3, :3])
    rays = camera_rays(observer)
    clipped = rays.copy()
    clipped[:, :3] += observer['near'] * clipped[:, 3:]
    depth = static_scene.cast_rays(o3d.core.Tensor(clipped))['t_hit'].numpy() + observer['near']
    result = first_hit(observer, rays, depth, bodies, face_labels, objects, frame, valid)
    return result, rays


def projection_error(points, rays, camera, pose, owners):
    """Check world hit -> current camera -> original ray pixel round trips."""
    take = owners >= 0
    if not take.any():
        return 0.
    local = (points[take] - pose[:3, 3]) @ pose[:3, :3]
    if (local[:, 2] <= 0).any():
        raise ValueError('First-hit point behind the current observer')
    k = camera['K']
    uv = local[:, :2] / local[:, 2:] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    w, h = camera['size']
    x, y = np.meshgrid(np.arange(w)+.5, np.arange(h)+.5)
    expected = np.stack((x, y), -1).reshape(-1, 2)[take]
    error = float(np.abs(uv-expected).max())
    if error > .02:
        raise ValueError(f'Moving observer ray/projection mismatch: {error} pixels')
    return error


def verify_paired_body(identity, vertices, joints, released, model):
    """Independently forward the camera-space source fit, including hands."""
    import torch
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from scenemi.native_visualization import sha256
    source = PROJECTS/'TRUMANS/smplx_result_in_cam'/(identity['sequence_id']+'_smplx_results.pkl')
    with source.open('rb') as stream:
        fit = pickle.load(stream)
    start, length = identity['start'], len(joints)
    q = np.asarray(readonly_array(identity, 'source_frame_ids'))[start:start+length].astype(int)
    beta = np.asarray(identity['native_body']['betas'], np.float32)
    jerror, verror = 0., 0.
    for lo in range(0, length, 16):
        hi = min(lo+16, length)
        params = {k: torch.from_numpy(np.asarray(fit[k], np.float32)[q[lo:hi]].copy()) for k in
            ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'transl')}
        params['betas'] = torch.from_numpy(np.broadcast_to(beta, (hi-lo, len(beta))).copy())
        params.update(leye_pose=torch.zeros(hi-lo, 3), reye_pose=torch.zeros(hi-lo, 3))
        with torch.inference_mode():
            body = model(**params)
        r, p = released[lo:hi, :3, :3], released[lo:hi, :3, 3]
        expected_j = np.einsum('tji,tkj->tki', r, joints[lo:hi]-p[:, None])
        expected_v = np.einsum('tji,tkj->tki', r, vertices[lo:hi]-p[:, None])
        jerror = max(jerror, float(np.abs(expected_j-body.joints[:, :22].numpy()).max()))
        verror = max(verror, float(np.abs(expected_v-body.vertices.numpy()).max()))
    if max(jerror, verror) > 2e-5:
        raise ValueError(f'Paired SMPL-X camera transform failed: joints={jerror}, vertices={verror}')
    return dict(camera_fit=str(source), camera_fit_sha256=sha256(source),
        paired_joint_max_error_m=jerror, paired_vertex_max_error_m=verror,
        proves='body-coordinate rigid transform only; not RGB intrinsics or optical-axis calibration')


def prepare(source_manifest, config, output):
    import torch
    import trimesh
    import yaml
    from experiments.offline_camera_retrain_v1.data import PROJECTS
    from experiments.offline_camera_retrain_v1.export_egobody_exo import regional_faces
    from experiments.offline_camera_retrain_v1.export_rest_joints import load_model
    from experiments.offline_camera_retrain_v1.native30_export import sampled_environment
    from experiments.offline_camera_retrain_v1.ops.motion_viewer import atomic_json
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import (
        camera_layers, instant_scene_layers, readonly_array)
    from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
    from scenemi.native_visualization import SCHEMA, camera_motion, observations, sha256
    from scenemi.training import ensure_bounded

    ensure_bounded()
    torch.set_num_threads(1)
    policy = yaml.safe_load(config.read_text())
    source = json.loads(source_manifest.read_text())
    if source['schema'] != SCHEMA or source.get('model_inference') is not False:
        raise ValueError('Expected verified source-only native previews')
    if output.resolve() == source_manifest.parent.resolve():
        raise ValueError('Moving observations must have a separate versioned output')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'PREPARED_CASES.json').exists():
        raise FileExistsError('Completed preview already exists; use a new output or inspect existing records')
    started = time.monotonic()
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    status = dict(state='preparing', protocol=PROTOCOL, started_at_utc=stamp(),
        planned_cases=len(source['cases']), completed_cases=0, training_started=False,
        diagnostic_only=True, full_dataset_reexported=False, test_used=False,
        config=str(config.resolve()), config_sha256=sha256(config))
    from scenemi.native_visualization import ROOT
    dependency_names = [
        'scenemi/data/exo_camera.py', 'scenemi/native_visualization.py',
        'configs/trumans_exo_moving.yaml',
        'experiments/offline_camera_retrain_v1/native30_exo.py',
        'experiments/offline_camera_retrain_v1/native30_export.py',
        'experiments/offline_camera_retrain_v1/export_egobody_exo.py',
        'experiments/offline_camera_retrain_v1/scene_visibility_v2.py',
        'experiments/offline_camera_retrain_v1/ops/viewer_geometry.py',
        'experiments/offline_camera_retrain_v1/ops/viewer_observation.py',
        'experiments/offline_camera_retrain_v1/ops/motion_viewer.py',
    ]
    source_hashes = {name: sha256(ROOT/name) for name in dependency_names}
    atomic_json(output/'PROTOCOL.json', dict(policy=policy,
        dependency_sha256=source_hashes, source_manifest_sha256=sha256(source_manifest),
        body_asset_sha256=sha256(ROOT/'body_models/smplx/SMPLX_MALE.npz'),
        python_version=__import__('sys').version,
        versions={name: __import__(name).__version__ for name in ('numpy', 'torch', 'trimesh', 'open3d')},
        diagnostic_only=True, training_started=False, test_used=False))
    atomic_json(output/'STATUS.json', status)
    model = load_model('smplx', 'male')
    labels = regional_faces(model)
    clips = {r['clip_name']: r for r in map(json.loads,
        (PROJECTS/'TRUMANS/processed/scene_expert_v1/clips.jsonl').read_text().splitlines())}
    cases, audits = [], []
    for case in source['cases']:
        old = case['models'][0]['case']
        if sha256(old['cache']) != old['cache_sha256']:
            raise ValueError('Source preview content changed')
        with np.load(old['cache'], allow_pickle=False) as archive:
            arrays = {k: np.asarray(archive[k]) for k in archive.files}
        meta = json.loads(str(arrays.pop('metadata_json')))
        identity = meta['identity']
        if identity['dataset'] != 'trumans' or identity['split'] != 'validation':
            raise ValueError('Camera acceptance uses the predeclared TRUMANS validation examples only')
        n, start = meta['data_frames'], identity['start']
        released = np.asarray(readonly_array(identity, 'trumans_released_camera_to_world_reference'))[start:start+n]
        pose, camera = camera_trajectory(released, policy)
        pair_check = verify_paired_body(identity, arrays['vertices'], arrays['joints'], released, model)
        valid = np.asarray(readonly_array(identity, 'observation_valid'))[start:start+n].copy()
        clip = clips[identity['sequence_id']]
        mesh_path = PROJECTS/'TRUMANS'/clip['scene']['mesh']
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        static = static_ray_scene(mesh.vertices, mesh.faces)
        objects = []
        for i, note in enumerate(meta['audit']['dynamic_objects']):
            if sha256(note['mesh']) != note['sha256']:
                raise ValueError('Source object mesh changed')
            objects.append((static_ray_scene(arrays[f'object_{i}_vertices'], arrays[f'object_{i}_faces']),
                arrays[f'object_{i}_rotations'], arrays[f'object_{i}_positions'], i+1, arrays[f'object_{i}_valid']))
        k = camera['K'];w, h = camera['size']
        obs = dict(exo_sensor_to_world=pose, exo_intrinsics=np.asarray([k[0,0]/w, k[1,1]/h, .5, .5], np.float32),
            observation_valid=valid, source_frame_ids=np.asarray(readonly_array(identity, 'source_frame_ids'))[start:start+n].copy(),
            trumans_released_camera_to_world_reference=released.copy(),
            exo_trajectory_world=np.zeros((n,22,3), np.float32),
            exo_position_available=np.zeros((n,22), bool), exo_position_confidence=np.zeros((n,22), np.float32),
            exo_frame_points_world=np.zeros((n,512,3), np.float32), exo_frame_owner=np.full((n,512), -1, np.int32))
        obs.update({f'exo_{kind}_counts': np.zeros((n,22), np.int32) for kind in
                    ('reference', 'visible', 'person', 'static', 'rigid')})
        projection_max, debug = 0., {}
        for t in range(n):
            result, rays = render_frame(camera, pose[t], static,
                [(arrays['vertices'][t], arrays['faces'])], labels, objects, t, valid[t])
            points, owner, parts, depth, available, confidence, reference, visible, blocked = result
            projection_max = max(projection_max, projection_error(points, rays, camera, pose[t], owner))
            obs['exo_position_available'][t] = available
            obs['exo_position_confidence'][t] = confidence
            obs['exo_trajectory_world'][t] = np.where(available[:,None], arrays['joints'][t], 0)
            obs['exo_reference_counts'][t], obs['exo_visible_counts'][t] = reference, visible
            for name, counts in blocked.items():
                obs[f'exo_{name}_counts'][t] = counts
            if valid[t]:
                obs['exo_frame_points_world'][t], obs['exo_frame_owner'][t] = sampled_environment(points, owner)
            if t in (0, n//2, n-1):
                debug[f'owner_{t}'], debug[f'depth_{t}'] = owner.reshape(h,w), depth.reshape(h,w)
            if (t+1) % 16 == 0:
                status.update(sequence_id=identity['sequence_id'], source_start=start, current_frames=t+1,
                    frames_per_case=n, updated_at_utc=stamp())
                atomic_json(output/'STATUS.json', status)
                print(f"moving EXO {identity['sequence_id']}: {t+1}/{n}", flush=True)
        folder = output/'observations'/identity['split']/(identity['sequence_id']+f'__{start:06d}')
        folder.mkdir(parents=True, exist_ok=True)
        for name, value in obs.items():
            np.save(folder/(name+'.npy'), value, allow_pickle=False)
        hashes = {name+'.npy': sha256(folder/(name+'.npy')) for name in obs}
        serial = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in camera.items()}
        new_identity = dict(identity, path=str(folder.resolve()), start=0, frames=n,
            exo_protocol=PROTOCOL, exo_camera=serial, output_sha256=hashes,
            source_window=dict(path=identity['path'], start=start, frames=n), diagnostic_only=True)
        atomic_json(folder/'metadata.json', new_identity)
        np.savez_compressed(folder/'FIRST_HIT_SNAPSHOTS.npz', **debug)
        layers, camera_note = camera_layers(new_identity, n, frustum_depth_m=.8)
        current, current_note = instant_scene_layers(new_identity, n)
        arrays.update(layers, **current, **observations(new_identity, n))
        arrays['camera_rotations'] = pose[:, :3, :3]
        arrays['camera_intrinsics'] = np.broadcast_to(obs['exo_intrinsics'], (n, 4)).copy()
        union = np.concatenate([current[f'current_static_{t}'] for t in range(n)])
        if len(union):
            _, keep = np.unique(np.floor(union/.025).astype(np.int64), axis=0, return_index=True)
            union = union[np.sort(keep)]
        arrays['visible_static'] = union.astype(np.float32)
        available = obs['exo_position_available']
        summary = dict(mean_available_joints=float(available.sum(-1).mean()),
            frames_with_no_joint=int((available.sum(-1)==0).sum()), valid_frames=int(valid.sum()))
        summary.update({kind+'_pixel_count': int(obs['exo_'+kind+'_counts'].sum()) for kind in
                        ('reference', 'visible', 'person', 'static', 'rigid')})
        old_summary = meta['audit']['observation_summary']
        audit = dict(meta['audit'], camera_mode='moving', observation_summary=summary,
            fixed_observation_summary=old_summary, moving_camera_motion=camera_motion(pose),
            paired_body_check=pair_check, projection_roundtrip_max_error_px=projection_max,
            source_preview_sha256=old['cache_sha256'], policy_sha256=sha256(config),
            static_mesh_sha256=sha256(mesh_path), observation_arrays_sha256=hashes,
            release_reference_only=False, full_dataset_reexported=False)
        meta.update(identity=new_identity, audit=audit, camera_mode='moving',
            source_window_start=start, camera_protocol=PROTOCOL)
        meta['display_geometry'].update(camera=camera_note, current_scene=current_note)
        arrays['metadata_json'] = np.asarray(json.dumps(meta, ensure_ascii=False))
        cache = output/(identity['sequence_id']+f'_{start:06d}.npz')
        with cache.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        cache.with_suffix('.npz.tmp').replace(cache)
        moved = dict(case, display_label='移动 · '+case['display_label'], models=[dict(case['models'][0],
            case=dict(cache=str(cache.resolve()), cache_sha256=sha256(cache)))])
        fixed = dict(case, display_label='旧固定对照 · '+case['display_label'])
        cases.extend([moved, fixed]);audits.append(audit)
        status.update(completed_cases=len(audits), updated_at_utc=stamp())
        atomic_json(output/'AUDIT.json', dict(protocol=PROTOCOL, cases=audits,
            training_started=False, test_used=False, diagnostic_only=True))
        atomic_json(output/'STATUS.json', status)
        print(json.dumps(dict(sequence_id=identity['sequence_id'], old=old_summary, moving=summary,
                              projection_error_px=projection_max, **pair_check)), flush=True)
    if any(sha256(ROOT/name) != digest for name, digest in source_hashes.items()):
        raise ValueError('A moving camera dependency changed during preprocessing')
    atomic_json(output/'PREPARED_CASES.json', dict(schema=SCHEMA, data_only=True, model_inference=False,
        source_manifest=str(source_manifest.resolve()), source_manifest_sha256=sha256(source_manifest),
        camera_policy=policy, cases=cases, diagnostic_only=True, training_started=False))
    status.update(state='complete', updated_at_utc=stamp(), elapsed_seconds=time.monotonic()-started,
                  viewer_cases=len(cases))
    atomic_json(output/'STATUS.json', status)
    return status
