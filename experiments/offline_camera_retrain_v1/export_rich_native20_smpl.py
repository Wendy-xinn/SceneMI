"""Export RICH's released 30 Hz body fits to canonical SMPL at 20 Hz.

Only static ego-visible scene surfaces come from the audited 5 Hz ray casting
pass. Body poses are resampled directly from the consecutive released fits;
they are never interpolated from that 5 Hz pass. The camera optical +Z must
point away from the face, as verified against the original mesh geometry.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation, Slerp

# Older SMPL model pickles refer to NumPy aliases removed in NumPy 2.
for alias, value in dict(bool=np.bool_, int=int, float=float, complex=complex,
                         object=object, unicode=str, str=str).items():
    if alias not in np.__dict__:
        setattr(np, alias, value)
import smplx

from prepare_rich_pseudo_ego import RICH_GENDER, RICH_TO_SCENEMI, _read_transform
from scene_visibility_v2 import render_scene

HERE = Path(__file__).resolve().parent
RICH = Path('/home/wenxin/projects/RICH/extracted')
SMPL_ROOT = Path('/home/wenxin/projects/ProtoMotions/data/smpl')
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
           13, 14, 16, 17, 18, 19)
FACE_OUT_MOUNT = np.eye(3, dtype=np.float32)


def _slerp_joint_pose(source_ids, poses, query_ids):
    source_local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    source_local = source_local.reshape(len(poses), 22, 3, 3)
    out = np.empty((len(query_ids), 22, 3, 3), np.float32)
    for joint in range(22):
        out[:, joint] = Slerp(source_ids, Rotation.from_matrix(source_local[:, joint]))(
            query_ids).as_matrix()
    return out


def _linear(source_ids, values, query_ids):
    values = np.asarray(values)
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(query_ids, source_ids, flat[:, k])
                    for k in range(flat.shape[1])], axis=-1)
    return out.reshape(len(query_ids), *values.shape[1:]).astype(np.float32)


def _smpl_joints(model, poses, betas, translation, scale, rotation, offset):
    output = []
    with torch.inference_mode():
        for first in range(0, len(poses), 256):
            last = min(first + 256, len(poses))
            body = np.pad(poses[first:last, 1:], ((0, 0), (0, 2), (0, 0)))
            result = model(global_orient=torch.from_numpy(poses[first:last, 0]),
                           body_pose=torch.from_numpy(body.reshape(last-first, -1)),
                           betas=torch.from_numpy(np.broadcast_to(betas, (last-first, 10)).copy()),
                           transl=torch.from_numpy(translation[first:last]),
                           return_verts=False)
            output.append(result.joints[:, :22].numpy())
    joints = np.concatenate(output).astype(np.float32)
    joints = scale * np.einsum('tji,ik->tjk', joints, rotation) + offset
    return joints @ RICH_TO_SCENEMI.T


def _fk(local, rest, translation):
    global_rotation = [local[:, 0]]
    joints = [translation + rest[0]]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        global_rotation.append(global_rotation[parent] @ local[:, joint])
        joints.append(joints[parent] + np.einsum('tij,j->ti', global_rotation[parent],
                                                rest[joint] - rest[parent]))
    return np.stack(joints, axis=1)


def export_sequence(split, scene_bundle, dst, model_cache):
    meta = json.loads((scene_bundle / 'metadata.json').read_text())
    if 'identity face mount' not in meta.get('camera_protocol', ''):
        raise ValueError(f'{scene_bundle}: scene bundle does not use face-out camera')
    # A Viser frustum alone cannot prove the optical sign.  With the full head
    # mesh included, the face normal must point into the optical hemisphere.
    first_vertices = np.load(scene_bundle / 'body_vertices_scenemi_yup.npy', mmap_mode='r')[0]
    faces = np.load(scene_bundle / 'body_faces.npy')
    first_position = np.load(scene_bundle / 'camera_position_scenemi_yup.npy', mmap_mode='r')[0]
    first_rotation = np.load(scene_bundle / 'camera_rotation_scenemi_yup.npy', mmap_mode='r')[0]
    first_mesh = trimesh.Trimesh(vertices=first_vertices, faces=faces, process=False)
    face_normal_dot = float(first_mesh.vertex_normals[9120] @ first_rotation[:, 2])
    if face_normal_dot < .5:
        raise ValueError(f'{scene_bundle}: camera optical +Z enters the face ({face_normal_dot:.3f})')
    first_hits = render_scene([(first_vertices, faces, 100)], first_position,
                              first_rotation, static_scene=None)
    self_hit_fraction = float(np.mean(first_hits['owner'] == 100))
    if self_hit_fraction >= .5:
        raise ValueError(f'{scene_bundle}: >50% camera rays first-hit the full wearer mesh')
    sequence = meta['sequence_id']
    subject = str(meta['subject_id'])
    source = RICH / f'{split}_body' / sequence
    frame_dirs = sorted((p for p in source.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    source_ids = np.asarray([int(p.name) for p in frame_dirs], np.float64)
    if not np.all(np.diff(source_ids) == 1):
        raise ValueError(f'{sequence}: released body fits are not consecutive')
    fits = []
    for folder in frame_dirs:
        with (folder / f'{subject}.pkl').open('rb') as stream:
            fits.append(pickle.load(stream, encoding='latin1'))
    raw_pose = np.asarray([np.concatenate((fit['global_orient'].reshape(1, 3),
                                           fit['body_pose'].reshape(21, 3))) for fit in fits],
                          np.float32)
    raw_transl = np.asarray([fit['transl'].reshape(3) for fit in fits], np.float32)
    betas = np.asarray(fits[0]['betas'], np.float32).reshape(-1)[:10]
    if any(not np.allclose(np.asarray(fit['betas']).reshape(-1)[:10], betas,
                           atol=1e-4) for fit in fits):
        raise ValueError(f'{sequence}: subject shape changes across frames')
    query = source_ids[0] + 1.5 * np.arange(
        int(np.floor((source_ids[-1] - source_ids[0]) / 1.5)) + 1, dtype=np.float64)
    local_camera = _slerp_joint_pose(source_ids, raw_pose, query)
    pose_camera = Rotation.from_matrix(local_camera.reshape(-1, 3, 3)).as_rotvec()
    pose_camera = pose_camera.reshape(len(query), 22, 3).astype(np.float32)
    transl_camera = _linear(source_ids, raw_transl, query)

    scene = meta['scene']
    variant = 'chair' if 'chair' in sequence else 'yoga'
    transform_name = (f'LectureHall_{variant}_multicam2world.json'
                      if scene == 'LectureHall' else f'{scene}_multicam2world.json')
    scale, rotation, offset = _read_transform(RICH / 'multicam2world' / transform_name)
    camera_to_scene = RICH_TO_SCENEMI @ rotation.T
    gender = RICH_GENDER[int(subject)]
    if gender not in model_cache:
        model_cache[gender] = smplx.SMPL(
            str(SMPL_ROOT / f'SMPL_{gender.upper()}.pkl'), num_betas=10, batch_size=1)
    model = model_cache[gender]
    canonical = _smpl_joints(model, pose_camera, betas, transl_camera,
                             scale, rotation, offset)
    source5 = np.load(scene_bundle / 'source_frame_ids.npy').astype(np.float64)
    anchor = np.rint((source5 - query[0]) / 1.5).astype(int)
    if (np.any(anchor < 0) or np.any(anchor >= len(query))
            or np.max(np.abs(query[anchor] - source5)) > 1e-5):
        raise ValueError(f'{sequence}: 5 Hz scene anchors do not align to 20 Hz motion')
    # The original 5 Hz scene bundle contains SMPL-X joints.  Align the SMPL
    # proxy root to those measured positions and interpolate only this small
    # model-to-model translation correction, not the motion itself.
    smplx5 = np.load(scene_bundle / 'joints_scenemi_yup.npy')
    root_delta5 = smplx5[:, 0] - canonical[anchor, 0]
    root_delta = _linear(source5, root_delta5, query)
    canonical += root_delta[:, None]

    with torch.inference_mode():
        shape = torch.from_numpy(betas[None])
        shaped = model.v_template[None] + smplx.lbs.blend_shapes(
            shape, model.shapedirs[:, :, :10])
        rest = smplx.lbs.vertices2joints(model.J_regressor, shaped)[0, :22].numpy()
    rest = (scale * (rest @ rotation) + offset) @ RICH_TO_SCENEMI.T
    rest = (rest - rest[0]).astype(np.float32)
    local_scene = np.einsum('ik,tjkl,lm->tjim', camera_to_scene,
                            local_camera, camera_to_scene.T).astype(np.float32)
    pose_scene = Rotation.from_matrix(local_scene.reshape(-1, 3, 3)).as_rotvec()
    pose_scene = pose_scene.reshape(len(query), 22, 3).astype(np.float32)
    translation = (canonical[:, 0] - rest[0]).astype(np.float32)
    fk = _fk(local_scene, rest, translation)
    fk_error = np.linalg.norm(fk - canonical, axis=-1)
    if float(np.percentile(fk_error, 95)) > .005:
        raise ValueError(f'{sequence}: canonical FK p95 {np.percentile(fk_error, 95):.4f} m')

    global_camera = [local_camera[:, 0]]
    for joint in range(1, 22):
        global_camera.append(global_camera[PARENTS[joint]] @ local_camera[:, joint])
    head_rotation = np.einsum('ij,tjk->tik', camera_to_scene, global_camera[15])
    camera_rotation = head_rotation @ FACE_OUT_MOUNT
    approved_camera_rotation = np.load(scene_bundle / 'camera_rotation_scenemi_yup.npy')
    angle = np.linalg.norm(Rotation.from_matrix(
        np.einsum('tji,tjk->tik', approved_camera_rotation, camera_rotation[anchor])
    ).as_rotvec(), axis=-1)
    if float(np.max(angle)) > np.deg2rad(.1):
        raise ValueError(f'{sequence}: camera orientation differs from approved scene bundle')
    approved_camera_position = np.load(scene_bundle / 'camera_position_scenemi_yup.npy')
    offset5 = np.einsum('tji,tj->ti', camera_rotation[anchor],
                        approved_camera_position - canonical[anchor, 15])
    local_offset = _linear(source5, offset5, query)
    camera_position = canonical[:, 15] + np.einsum('tij,tj->ti',
                                                    camera_rotation, local_offset)
    camera_error = np.linalg.norm(camera_position[anchor] - approved_camera_position, axis=-1)
    if float(np.max(camera_error)) > .002:
        raise ValueError(f'{sequence}: camera position differs from approved scene bundle')

    dst.mkdir(parents=True, exist_ok=True)
    arrays = dict(pose_axis_angle_scenemi_yup=pose_scene,
                  translation_scenemi_yup=translation,
                  joints_scenemi_yup=canonical,
                  rest_joints_scenemi_yup=rest,
                  camera_position_scenemi_yup=camera_position,
                  camera_rotation_scenemi_yup=camera_rotation,
                  source_frame_ids=query)
    for name, array in arrays.items():
        np.save(dst / f'{name}.npy', np.asarray(array, np.float32))
    scene_link = dst / 'visible_static_points_scenemi_yup.npy'
    if not scene_link.exists():
        os.symlink((scene_bundle / scene_link.name).resolve(), scene_link)
    output_meta = dict(meta)
    output_meta.update(output_fps=20.0, frames_20fps=len(query),
                       body_model='SMPL canonical proxy',
                       source_fps=30.0, source_stride_for_motion=1,
                       temporal_protocol='30 Hz released SMPL-X body fits -> 20 Hz SLERP/linear -> canonical SMPL; no 5 Hz body interpolation',
                       scene_temporal_protocol='5 Hz occlusion-aware static visibility union using face-out head camera',
                       scene_bundle=str(scene_bundle.resolve()),
                       face_normal_dot=face_normal_dot,
                       full_wearer_first_hit_fraction=self_hit_fraction,
                       fk_p95_m=float(np.percentile(fk_error, 95)),
                       approved_camera_angle_max_deg=float(np.rad2deg(np.max(angle))),
                       approved_camera_position_max_m=float(np.max(camera_error)))
    (dst / 'metadata.json').write_text(json.dumps(output_meta, indent=2) + '\n')
    return dict(sequence=sequence, frames=len(query), fk_p95_m=output_meta['fk_p95_m'],
                camera_angle_max_deg=output_meta['approved_camera_angle_max_deg'],
                camera_position_max_m=output_meta['approved_camera_position_max_m'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--sequence')
    parser.add_argument('--suffix', default='faceout_oct07')
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene_root = HERE / 'data/scene_visibility_v2_oct05' / f'rich_{args.split}'
    output_root = HERE / 'data/scene_visibility_v2_oct05' / f'rich_{args.split}_smpl_native20_{args.suffix}'
    output_root.mkdir(parents=True, exist_ok=True)
    models = {}
    for scene_bundle in sorted(p for p in scene_root.iterdir() if p.is_dir()):
        if args.sequence and scene_bundle.name != args.sequence:
            continue
        if not (scene_bundle / 'metadata.json').is_file():
            continue
        dst = output_root / scene_bundle.name
        if (dst / 'metadata.json').is_file():
            print('SKIP', scene_bundle.name, flush=True)
            continue
        print(json.dumps(export_sequence(args.split, scene_bundle, dst, models)), flush=True)


if __name__ == '__main__':
    main()
