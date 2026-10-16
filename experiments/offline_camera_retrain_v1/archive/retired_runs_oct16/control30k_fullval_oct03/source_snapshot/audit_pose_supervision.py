"""Check raw SMPL/SMPL-X forward kinematics against prepared world joints.

This is a gate for the planned SceneMI retraining, not a training script.
It samples all data splits and both EgoBody camera roles without mixing their
coordinate systems. The report records per-case errors and raises if gross
misalignment would poison a joint/FK loss.
"""
import csv
import inspect
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
for name, value in dict(bool=bool, int=int, float=float, complex=complex,
                        object=object, unicode=str, str=str).items():
    if name not in np.__dict__:
        setattr(np, name, value)
import smplx

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[2]
PREPARED = PROJECTS / 'diffusion-motion-inbetweening/experiments/offline_sequence_v1/data'
EGOBODY = PROJECTS / 'egobody'
TRUMANS = PROJECTS / 'TRUMANS'
MODELS = PROJECTS / 'SceneMI/body_models'


def source_rows():
    return {r['sequence_id']: r for r in
            (json.loads(line) for line in (PREPARED / 'sequences.jsonl').read_text().splitlines() if line)}


def sample_rows(dataset, split, role=None):
    manifest = PREPARED / f'{dataset}_sequences/{split}.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if role:
        rows = [r for r in rows if r['role'] == role]
    # Three deterministic clips per split; clip IDs are sorted in each manifest.
    return [rows[i] for i in sorted(set((0, len(rows) // 2, len(rows) - 1)))]


def model_joints(model, pose, translation, betas=None):
    params = dict(global_orient=torch.tensor(pose[:3][None], dtype=torch.float32),
                  body_pose=torch.tensor(pose[3:][None], dtype=torch.float32),
                  transl=torch.tensor(translation[None], dtype=torch.float32),
                  return_verts=False)
    if betas is not None:
        params['betas'] = torch.tensor(betas[None], dtype=torch.float32)
    with torch.no_grad():
        return model(**params).joints[0, :22].numpy()


def trumans_case(row, source, model):
    with Path(source['smplx_source']).open('rb') as stream:
        fit = pickle.load(stream)
    prepared = np.load(Path(row['folder']) / 'joints_world.npy', mmap_mode='r')
    cases = []
    for offset in (0, 2 * (len(prepared) // 4), 2 * ((len(prepared) - 1) // 2)):
        frame = int(offset * 1.5)
        pose = np.r_[fit['global_orient'][frame], fit['body_pose'][frame]]
        joints = model_joints(model, pose, fit['transl'][frame])
        error = np.linalg.norm(joints - prepared[offset], axis=-1)
        cases.append(dict(dataset='trumans', split=row['split'],
                          sequence_id=row['sequence_id'], frame_20fps=offset,
                          mpjpe_m=float(error.mean()), max_joint_error_m=float(error.max())))
    return cases


def egobody_case(row, source, models, info):
    fit_dir = Path(source['body_fits'])
    body_idx = int(fit_dir.name.split('_')[-1])
    rec = info[row['recording']]
    gender = rec[f'body_idx_{body_idx}'].split()[-1].lower()
    if gender not in models:
        model_path = PROJECTS / 'ProtoMotions/data/smpl' / f'SMPL_{gender.upper()}.pkl'
        models[gender] = smplx.SMPL(str(model_path), batch_size=1)
    calibration = EGOBODY / 'calibrations' / row['recording'] / 'cal_trans/kinect12_to_world' / f'{row["scene_id"]}.json'
    transform = np.asarray(json.loads(calibration.read_text())['trans'])
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError(f'{row["sequence_id"]}: nonrigid calibration')
    prepared = np.load(Path(row['folder']) / 'joints_world.npy', mmap_mode='r')
    cases = []
    # Even 20 Hz offsets map exactly onto original 30 Hz fitted frames.
    for offset in (0, 2 * (len(prepared) // 4), 2 * ((len(prepared) - 1) // 2)):
        frame = row['source_first_frame'] + int(offset * 1.5)
        path = fit_dir / f'results/frame_{frame:05d}/000.pkl'
        with path.open('rb') as stream:
            fit = pickle.load(stream, encoding='latin1')
        pose = np.r_[fit['global_orient'].reshape(3), fit['body_pose'].reshape(69)]
        joints = model_joints(models[gender], pose, fit['transl'].reshape(3), fit['betas'].reshape(10))
        joints = joints @ transform[:3, :3].T + transform[:3, 3]
        error = np.linalg.norm(joints - prepared[offset], axis=-1)
        cases.append(dict(dataset='egobody', split=row['split'], role=row['role'],
                          sequence_id=row['sequence_id'], frame_20fps=offset,
                          mpjpe_m=float(error.mean()), max_joint_error_m=float(error.max())))
    return cases


def main():
    sources = source_rows()
    info = {r['recording_name']: r for r in csv.DictReader((EGOBODY / 'data_info_release.csv').open())}
    trumans_model = smplx.create(str(MODELS), model_type='smplx', gender='neutral',
                                num_betas=10, use_pca=False, batch_size=1)
    ego_models = {}
    cases = []
    for split in ('train', 'validation', 'test'):
        for row in sample_rows('trumans', split):
            cases += trumans_case(row, sources[row['sequence_id']], trumans_model)
        for role in ('camera_wearer', 'interactee'):
            for row in sample_rows('egobody', split, role):
                cases += egobody_case(row, sources[f'{row["recording"]}/{role}'], ego_models, info)
    groups = defaultdict(list)
    for case in cases:
        groups[(case['dataset'], case.get('role', 'synthetic_head'))].append(case['mpjpe_m'])
    summary = {f'{dataset}/{role}': dict(cases=len(values), mean_m=float(np.mean(values)),
                                           max_m=float(np.max(values)))
               for (dataset, role), values in groups.items()}
    report = dict(summary=summary, cases=cases,
                  note='EgoBody per-frame betas can differ from the 3s cache prefix-median shape; small residuals are expected.')
    output = HERE / 'pose_supervision_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if any(v['max_m'] > .08 for v in summary.values()):
        raise ValueError('Raw pose and prepared world joints disagree by over 8 cm in a sampled case')


if __name__ == '__main__':
    main()
