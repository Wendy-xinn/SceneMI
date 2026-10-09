"""Export source SMPL(-X) pose streams aligned with audited 20 Hz sequences.

This preparation is separate from the 22-joint diagnostic experiment. It
provides pose targets required by the SceneMI U-Net and FK supervision.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
PREPARED = PROJECTS / 'diffusion-motion-inbetweening/experiments/offline_sequence_v1/data'
EGOBODY = PROJECTS / 'egobody'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def slerp_rotvec(source, queries):
    from scipy.spatial.transform import Slerp
    frames, joints, _ = source.shape
    times = np.arange(frames, dtype=np.float64)
    rotations = Rotation.from_rotvec(source.reshape(-1, 3))
    # Slerp needs each joint's time series independently.
    output = np.empty((len(queries), joints, 3), np.float32)
    for joint in range(joints):
        series = Rotation.from_rotvec(source[:, joint])
        output[:, joint] = Slerp(times, series)(queries).as_rotvec()
    return output


def linear_sample(source, queries):
    flattened = source.reshape(len(source), -1)
    lower = np.floor(queries).astype(int)
    upper = np.minimum(lower + 1, len(source) - 1)
    weight = (queries - lower).astype(np.float32)[:, None]
    result = flattened[lower] * (1 - weight) + flattened[upper] * weight
    return result.reshape(len(queries), *source.shape[1:]).astype(np.float32)


def write_pose(row, pose, translation, betas, model, gender, source_label):
    folder = Path(row['folder'])
    joints = np.load(folder / 'joints_world.npy', mmap_mode='r')
    if len(pose) != len(joints) or translation.shape != (len(joints), 3):
        raise ValueError(f'{row["sequence_id"]}: pose/position frame mismatch')
    if not np.isfinite(pose).all() or not np.isfinite(translation).all():
        raise ValueError(f'{row["sequence_id"]}: invalid pose values')
    output = HERE / 'data/poses' / row['split'] / row['sequence_id']
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'pose_axis_angle_world.npy', pose.astype(np.float32))
    np.save(output / 'translation_world.npy', translation.astype(np.float32))
    meta = dict(sequence_id=row['sequence_id'], dataset=row['dataset'],
                split=row['split'], frames_20fps=len(joints),
                model=model, gender=gender, betas=np.asarray(betas).tolist(),
                source=source_label, prepared_folder=str(folder.resolve()),
                folder=str(output.resolve()))
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def trumans(row, source):
    with Path(source['smplx_source']).open('rb') as stream:
        fit = pickle.load(stream)
    frames = source['source_frames']
    if len(fit['global_orient']) != frames or len(fit['body_pose']) != frames:
        raise ValueError(f'{row["sequence_id"]}: source fit frame mismatch')
    queries = np.arange(0, frames - 1, 1.5, dtype=np.float64)
    axis_angle = np.concatenate((np.asarray(fit['global_orient'])[:, None],
                                 np.asarray(fit['body_pose']).reshape(frames, 21, 3)), axis=1)
    pose = slerp_rotvec(axis_angle, queries)
    translation = linear_sample(np.asarray(fit['transl']), queries)
    gender_code = str(fit.get('gender', '')).lower().strip()
    gender = {'m': 'male', 'f': 'female', 'n': 'neutral', '': 'neutral'}.get(gender_code, gender_code)
    # The released fit is SMPL-X, but its first 21 body rotations are directly
    # usable by canonical SMPL (the two terminal SMPL hand joints are identity).
    # Export the pose under the canonical SMPL protocol so rest-joint/FK audits
    # and the tracker use one body model across datasets.
    return write_pose(row, pose, translation, np.zeros(10, np.float32),
                      'smpl', gender, source['smplx_source'])


def egobody(row, source, info):
    fit_dir = Path(source['body_fits'])
    body_idx = int(fit_dir.name.split('_')[-1])
    gender = info[row['recording']][f'body_idx_{body_idx}'].split()[-1].lower()
    first = row['source_first_frame']
    last = row['source_last_frame_exclusive']
    poses, translations, shapes = [], [], []
    for frame in range(first, last):
        path = fit_dir / f'results/frame_{frame:05d}/000.pkl'
        with path.open('rb') as stream:
            fit = pickle.load(stream, encoding='latin1')
        poses.append(np.r_[np.asarray(fit['global_orient']).reshape(3),
                           np.asarray(fit['body_pose']).reshape(69)[:63]])
        translations.append(np.asarray(fit['transl']).reshape(3))
        shapes.append(np.asarray(fit['betas']).reshape(10))
    poses = np.asarray(poses).reshape(-1, 22, 3)
    translations = np.asarray(translations)
    queries = np.arange(row['frames_20fps'], dtype=np.float64) * 1.5
    if queries[-1] >= len(poses):
        raise ValueError(f'{row["sequence_id"]}: source fit does not cover prepared run')
    pose = slerp_rotvec(poses, queries)
    translation = linear_sample(translations, queries)
    transform_path = EGOBODY / 'calibrations' / row['recording'] / 'cal_trans/kinect12_to_world' / f'{row["scene_id"]}.json'
    transform = np.asarray(json.loads(transform_path.read_text())['trans'])
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError(f'{row["sequence_id"]}: nonrigid world transform')
    root = Rotation.from_matrix(transform[:3, :3]) * Rotation.from_rotvec(pose[:, 0])
    pose[:, 0] = root.as_rotvec()
    translation = translation @ transform[:3, :3].T + transform[:3, 3]
    betas = np.median(np.asarray(shapes), axis=0)
    return write_pose(row, pose, translation, betas, 'smpl', gender, str(fit_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-group', type=int, default=0,
                        help='0 exports all; a positive value is only for a preparation smoke run')
    args = parser.parse_args()
    source = {r['sequence_id']: r for r in read_jsonl(PREPARED / 'sequences.jsonl')}
    info = {r['recording_name']: r for r in csv.DictReader((EGOBODY / 'data_info_release.csv').open())}
    for split in ('train', 'validation', 'test'):
        manifest = []
        for dataset in ('trumans', 'egobody'):
            rows = read_jsonl(PREPARED / f'{dataset}_sequences/{split}.jsonl')
            if args.limit_per_group:
                rows = rows[:args.limit_per_group]
            for row in rows:
                target = HERE / 'data/poses' / split / row['sequence_id']
                metadata_path = target / 'metadata.json'
                if metadata_path.is_file() and all((target / name).is_file() for name in
                                                   ('pose_axis_angle_world.npy', 'translation_world.npy')):
                    meta = json.loads(metadata_path.read_text())
                    if meta['sequence_id'] != row['sequence_id'] or meta['frames_20fps'] != row['frames_20fps']:
                        raise ValueError(f'{row["sequence_id"]}: stale pose export')
                    if meta['model'] == 'smplx' and meta['gender'] == '':
                        meta['gender'] = 'neutral'
                        metadata_path.write_text(json.dumps(meta, indent=2) + '\n')
                elif dataset == 'trumans':
                    meta = trumans(row, source[row['sequence_id']])
                else:
                    meta = egobody(row, source[f'{row["recording"]}/{row["role"]}'], info)
                manifest.append(meta)
                print(f'{split}: {row["sequence_id"]}: {meta["model"]} {meta["frames_20fps"]}', flush=True)
        if not args.limit_per_group:
            path = HERE / 'data/poses' / f'{split}.jsonl'
            path.write_text(''.join(json.dumps(meta) + '\n' for meta in manifest))


if __name__ == '__main__':
    main()
