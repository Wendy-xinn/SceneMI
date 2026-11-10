"""Serve sequence-export audits through the repository's shared MotionViewer."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, '/home/wenxin/projects/VideoMimic')
from experiments.offline_camera_retrain_v1.ops.motion_viewer import MotionViewer, atomic_json
from real2sim.benchmarks.geometry import apply, first_camera_alignment
from scenemi.geometry.body import PARENTS

ROOT = Path('/home/wenxin/projects')
EXPORT_ROOT = ROOT / 'sequence_scene_exports'
OUT = ROOT / 'SceneMI/runs/scene_joint128_conditioning_corrected/sequence_scene_exports_step003000_fk_consistent'


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np.core.numeric._frombuffer
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        raise ValueError(f'unexpected pickle global: {module}.{name}')


def read_pickle(path):
    with Path(path).open('rb') as stream:
        return SafeUnpickler(stream).load()


def tube_topology():
    faces = []
    for bone, _ in enumerate((p for p in PARENTS if p >= 0)):
        base = bone * 8
        for a in range(4):
            b = (a + 1) % 4
            faces.extend(((base+a, base+b, base+4+b), (base+a, base+4+b, base+4+a)))
        faces.extend(((base, base+1, base+2), (base, base+2, base+3),
                      (base+4, base+6, base+5), (base+4, base+7, base+6)))
    return np.asarray(faces, np.int32)


TUBE_FACES = tube_topology()


def tube_vertices(joints, radius=.014):
    joints = np.asarray(joints, np.float32)
    values = []
    for frame in joints:
        vertices = []
        for child, parent in enumerate(PARENTS):
            if parent < 0:
                continue
            start, end = frame[parent], frame[child]
            direction = end - start
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            axis = np.array([1., 0., 0.], np.float32)
            if abs(float(np.dot(axis, direction))) > .85:
                axis = np.array([0., 1., 0.], np.float32)
            side = np.cross(direction, axis)
            side /= max(float(np.linalg.norm(side)), 1e-6)
            other = np.cross(direction, side)
            vertices.extend([start + radius*(side+other), start + radius*(side-other),
                             start + radius*(-side-other), start + radius*(-side+other),
                             end + radius*(side+other), end + radius*(side-other),
                             end + radius*(-side-other), end + radius*(-side+other)])
        values.append(np.asarray(vertices, np.float32))
    return np.asarray(values, np.float32)


def camera_arrays(directory, target_cameras):
    frames = sorted(directory.glob('frame_*'), key=lambda p: int(p.name.split('_')[1]))
    source = np.asarray([read_pickle(f/'cameras.pkl')['cameras']['hsi']['camera_to_world_4x4'] for f in frames], float)
    _, rotation, translation = first_camera_alignment(source, np.asarray(target_cameras), False)
    positions = apply(source[:, :3, 3], 1., rotation, translation).astype(np.float32)
    axes = np.empty((len(source), 3, 2, 3), np.float32)
    for i, matrix in enumerate(source):
        orientation = rotation @ matrix[:3, :3]
        axes[i, :, 0] = positions[i]
        axes[i, 0, 1] = positions[i] + .20 * orientation[:, 0]
        axes[i, 1, 1] = positions[i] + .20 * orientation[:, 1]
        axes[i, 2, 1] = positions[i] + .20 * orientation[:, 2]
    return rotation, translation, positions, axes, np.stack((positions[:-1], positions[1:]), 1)


def scene_arrays(directory, rotation, translation):
    frames = sorted(directory.glob('frame_*'), key=lambda p: int(p.name.split('_')[1]))
    current, union = [], []
    for frame in frames:
        manifest = json.loads((frame/'manifest.json').read_text())
        xyz = np.asarray(trimesh.load(frame/manifest['files']['environment']['hsi']['file'], process=False).vertices, np.float32)
        if len(xyz) > 1200:
            xyz = xyz[np.linspace(0, len(xyz)-1, 1200, dtype=np.int64)]
        xyz = apply(xyz, 1., rotation, translation).astype(np.float32)
        current.append(xyz); union.append(xyz)
    visible = np.concatenate(union, 0) if union else np.empty((0, 3), np.float32)
    if len(visible) > 50000:
        visible = visible[np.linspace(0, len(visible)-1, 50000, dtype=np.int64)]
    return visible, {f'current_static_{i}': value for i, value in enumerate(current)}


def observation_meta(mode, joints, visible):
    observations = np.zeros((len(joints), 22, 9), np.float32)
    meta = np.zeros((len(joints), 22, 5), np.float32)
    if mode == 'exo':
        observations[..., :3] = joints / 2.
        meta[..., 0] = visible; meta[..., 2] = visible; meta[..., 3] = visible
    return observations, meta


def save_model_cache(path, identity, truth, reference, prediction, scene, camera,
                     observations, meta, source_occluded, label):
    truth_vertices, reference_vertices, prediction_vertices = map(
        tube_vertices, (truth, reference, prediction))
    center = (truth.min((0, 1)) + truth.max((0, 1))) * .5
    radius = max(1.2, float(np.linalg.norm(truth[:, 0]-truth[:, 0].mean(0), axis=-1).max())+.8)
    metadata = dict(schema='sequence-export-shared-motion-viewer-v1', identity=identity,
        step=3000, label=label, sources={'sequence_export': identity['sequence_id']},
        generation={'recorded_guidance': 'step3000 scene audit; viewer display only'},
        display_geometry={'camera': {'source_kind': 'HSI export camera; aligned for audit'},
                          'scene': {'status': 'display-only HSI point cloud'}},
        scene_errors=[], metrics={}, regimes=['clean'])
    arrays = dict(truth_joints=truth.astype(np.float32), truth_vertices=truth_vertices,
        faces=TUBE_FACES, clean_reference_joints=reference.astype(np.float32),
        clean_reference_vertices=reference_vertices, clean_joints=prediction.astype(np.float32),
        clean_vertices=prediction_vertices, clean_observations=observations,
        clean_observation_meta=meta, source_occluded=source_occluded.astype(bool),
        visible_static=scene, scene_vertices=np.empty((0, 3), np.float32),
        scene_faces=np.empty((0, 3), np.int32), center=center.astype(np.float32),
        radius=np.asarray(radius, np.float32), camera_positions=camera[0],
        camera_axis_segments=camera[1], camera_path=camera[2],
        camera_frustum_segments=np.empty((len(truth), 0, 2, 3), np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    arrays.update(camera[3]); np.savez_compressed(path, **arrays)


def prepare_cases():
    cache_root = OUT/'shared_motion_viewer_cache'; cache_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for clip_id, directory in enumerate(sorted(EXPORT_ROOT.glob('sequence_*')), 1):
        name = directory.name; gt_root = ROOT/f'egobody_val3_share_20261026/clips/clip{clip_id}/gt'
        cameras = np.load(gt_root/'cameras.npz', allow_pickle=False)
        rotation, translation, positions, axes, path = camera_arrays(directory, cameras['T_S_C'])
        static, current = scene_arrays(directory, rotation, translation)
        camera = (positions, axes, path, current)
        ego = np.load(OUT/name/'ego_predictions.npz', allow_pickle=False)
        ego_gt = np.load(gt_root/'body_camera_wearer_smpl.npz', allow_pickle=False)['joints_scene'][:, :22]
        ego_parent = apply(ego['no_scene_parent_joints_export'], 1., rotation, translation)
        ego_scene = apply(ego['scene_step3000_joints_export'], 1., rotation, translation)
        ego_obs, ego_meta = observation_meta('ego', ego_parent, np.zeros((len(ego_gt), 22), np.float32))
        identity = dict(dataset='egobody', sequence_id=name, recording=name, group='sequence_exports',
                        mode='ego', start=0, data_frames=len(ego_gt), source='sequence_scene_exports')
        models = []
        for key, label, prediction in (('ego_parent','EGO parent（无场景）',ego_parent), ('ego_scene','EGO step3000（有场景）',ego_scene)):
            cache = cache_root/f'{name}_ego_{key}.npz'
            save_model_cache(cache, identity, ego_gt, ego_parent, prediction, static, camera, ego_obs, ego_meta,
                             np.zeros((len(ego_gt), 22), bool), label)
            models.append(dict(key=key, label=label, mode='ego', case={'cache': str(cache)}))
        cases.append(dict(label=f'{name} · EGO', display_label=f'{name} · EGO', dataset='egobody',
                          sequence_id=name, mode='ego', regimes=['clean'], models=models,
                          selection_reason='三段 sequence_scene_exports；step3000 EGO 有/无场景并列。'))
        exo = np.load(OUT/name/'predictions.npz', allow_pickle=False)
        lower = np.load(OUT/name/'exo_lower_only_predictions.npz', allow_pickle=False)
        exo_gt = np.load(gt_root/'body_interactee_smpl.npz', allow_pickle=False)['joints_scene'][:, :22]
        exo_initial = apply(exo['initial_joints_export'], 1., rotation, translation)
        exo_scene = apply(exo['pv_visibility_joints_export'], 1., rotation, translation)
        exo_lower = apply(lower['lower_only_joints_export'], 1., rotation, translation)
        exo_obs, exo_meta = observation_meta('exo', exo_initial, exo['projected_visibility'])
        identity = dict(dataset='egobody', sequence_id=name, recording=name, group='sequence_exports',
                        mode='exo', start=0, data_frames=len(exo_gt), source='sequence_scene_exports')
        models = []
        for key, label, prediction in (('exo_initial','EXO FK初始化',exo_initial), ('exo_scene','EXO step3000场景修复',exo_scene),
                                       ('exo_lower_only','EXO只修下半身',exo_lower)):
            cache = cache_root/f'{name}_exo_{key}.npz'
            save_model_cache(cache, identity, exo_gt, exo_initial, prediction, static, camera, exo_obs, exo_meta,
                             exo['projected_visibility'] < .5, label)
            models.append(dict(key=key, label=label, mode='exo', case={'cache': str(cache)}))
        cases.append(dict(label=f'{name} · EXO', display_label=f'{name} · EXO', dataset='egobody',
                          sequence_id=name, mode='exo', regimes=['clean'], models=models,
                          selection_reason='三段 sequence_scene_exports；输入置信度来自图像 OpenPose 代理。'))
    return cases


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8091); args = parser.parse_args()
    cases = prepare_cases()
    atomic_json(OUT/'SHARED_VIEWER_CASES.json', dict(schema='sequence-export-shared-motion-viewer-v1',
        cases=[dict(label=c['label'], mode=c['mode'], models=[m['key'] for m in c['models']]) for c in cases]))
    viewer = MotionViewer(cases, args.port)
    atomic_json(OUT/'SHARED_VIEWER_SERVER.json', dict(pid=os.getpid(), url=f'http://localhost:{args.port}/',
        bind='127.0.0.1', cases=len(cases), viewer='experiments.offline_camera_retrain_v1.ops.motion_viewer.MotionViewer'))
    print(f'VISER_URL=http://localhost:{args.port}', flush=True); viewer.run()


if __name__ == '__main__': main()
