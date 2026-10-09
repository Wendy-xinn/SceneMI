"""Align 5 Hz RICH SMPL proxy clips with SceneMI's 20 Hz motion protocol.

This is a band-limited proxy: local/body and camera rotations use SLERP;
root and camera translations use linear interpolation. It does not invent
sub-5-Hz motion detail. The static ego-visible map stays the audited 5 Hz map.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
           13, 14, 16, 17, 18, 19)


def linear(values, query):
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(query, np.arange(len(values)), flat[:, i])
                    for i in range(flat.shape[1])], axis=-1)
    return out.reshape(len(query), *values.shape[1:]).astype(np.float32)


def rotations(matrices, query):
    flat = matrices.reshape(len(matrices), -1, 3, 3)
    output = np.empty((len(query), flat.shape[1], 3, 3), np.float32)
    for joint in range(flat.shape[1]):
        output[:, joint] = Slerp(np.arange(len(flat)),
                                 Rotation.from_matrix(flat[:, joint]))(query).as_matrix()
    return output.reshape(len(query), *matrices.shape[1:])


def convert(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    source_pose = np.load(src / 'pose_axis_angle_scenemi_yup.npy')
    count = len(source_pose)
    query = np.arange((count - 1) * 4 + 1, dtype=np.float64) / 4
    local = Rotation.from_rotvec(source_pose.reshape(-1, 3)).as_matrix().reshape(count, 22, 3, 3)
    local = rotations(local, query)
    pose = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(len(query), 22, 3)
    rest = np.load(src / 'rest_joints_scenemi_yup.npy')
    translation = linear(np.load(src / 'translation_scenemi_yup.npy'), query)
    global_rotation = [local[:, 0]]
    joints = [translation + rest[0]]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        global_rotation.append(global_rotation[parent] @ local[:, joint])
        joints.append(joints[parent] + np.einsum('tij,j->ti', global_rotation[parent],
                                                rest[joint] - rest[parent]))
    joints = np.stack(joints, axis=1).astype(np.float32)
    camera = linear(np.load(src / 'camera_position_scenemi_yup.npy'), query)
    camera_rotation = rotations(np.load(src / 'camera_rotation_scenemi_yup.npy'), query)
    frame_ids = linear(np.load(src / 'source_frame_ids.npy').astype(np.float32), query)
    arrays = dict(pose_axis_angle_scenemi_yup=pose, translation_scenemi_yup=translation,
                  joints_scenemi_yup=joints, rest_joints_scenemi_yup=rest,
                  camera_position_scenemi_yup=camera,
                  camera_rotation_scenemi_yup=camera_rotation, source_frame_ids=frame_ids)
    for name, value in arrays.items():
        np.save(dst / f'{name}.npy', np.asarray(value, np.float32))
    os.symlink((src / 'visible_static_points_scenemi_yup.npy').resolve(),
               dst / 'visible_static_points_scenemi_yup.npy')
    meta = json.loads((src / 'metadata.json').read_text())
    meta.update(output_fps=20.0, frames_20fps=len(query),
                temporal_protocol='20 Hz SLERP/linear from 5 Hz SMPL proxy; no high-frequency detail invented',
                source_bundle=str(src.resolve()))
    (dst / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return len(query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.source.iterdir()):
        if not (src / 'metadata.json').is_file():
            continue
        dst = args.output / src.name
        if (dst / 'metadata.json').is_file():
            print('SKIP', src.name, flush=True)
            continue
        print('OK', src.name, convert(src, dst), flush=True)


if __name__ == '__main__':
    main()
