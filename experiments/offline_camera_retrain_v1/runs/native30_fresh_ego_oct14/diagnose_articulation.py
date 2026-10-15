"""Read-only articulation/owner audit on saved scheduled official VAL outputs."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.native30_cache_io import verified_array
from experiments.offline_camera_retrain_v1.native30_gt_reliability import quality_weights

ROOT = Path(__file__).resolve().parent


def measures(motion, rest):
    assert motion.shape == (192, 201) and rest.shape == (22, 3)
    assert np.isfinite(motion).all() and np.isfinite(rest).all()
    x = torch.from_numpy(motion.astype(np.float32))
    with torch.no_grad():
        joints = forward_kinematics(x[None], torch.from_numpy(rest)[None])[0].numpy()
        root_rotation = rotation_from_6d(x[:, 3:9]).numpy()
    # Remove both root translation AND root rotation, without aligning to GT.
    local = np.einsum('tji,tkj->tki', root_rotation, joints - joints[:, :1])
    feet = local[:, [10, 11]]
    amplitude = np.sqrt(np.mean(np.sum((feet - feet.mean(0)) ** 2, -1), 0))
    angles = []
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
        u, v = joints[:, hip] - joints[:, knee], joints[:, ankle] - joints[:, knee]
        cosine = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    angles = np.stack(angles, -1)
    velocity = np.diff(joints[:, [10, 11]], axis=0) * 30
    return dict(root_net_displacement_m=float(np.linalg.norm(joints[-1, 0] - joints[0, 0])),
        root_path_m=float(np.linalg.norm(np.diff(joints[:, 0], axis=0), axis=-1).sum()),
        body_local_foot_temporal_RMS_m=amplitude.tolist(),
        knee_angle_95minus5_deg=(np.percentile(angles, 95, axis=0) - np.percentile(angles, 5, axis=0)).tolist(),
        world_foot_speed_mean_m_s=np.linalg.norm(velocity, axis=-1).mean(0).tolist()), local, angles


def main():
    torch.set_num_threads(2)
    folder = ROOT / 'articulation_review_oct15'
    folder.mkdir(exist_ok=True)
    latest = ROOT / 'training/validation_latest.npz'
    with np.load(latest) as a:
        latest_step = int(a['step'])
    archive = folder / f'validation_step{latest_step}.npz'
    if not archive.exists():
        shutil.copyfile(latest, archive)
    sources = [ROOT / 'visual_review_step8000/validation_outputs.npz', archive]
    rows, owner_rows, plot_data = [], [], []
    config = json.loads((HERE / 'native30_fresh_ego_config.json').read_text())
    for source in sources:
        with np.load(source) as a:
            step = int(a['step'])
            panel = next(json.loads(line) for line in (ROOT / 'training/validation.jsonl').read_text().splitlines() if json.loads(line)['step'] == step)
            identities = json.loads(str(a['identities_json']))
            cases = [r for r in panel['cases'] if r['mode'] == 'ego' and r['regime'] == 'clean' and r.get('scene_variant', 'scene_on') == 'scene_on']
            for case in cases:
                i = case['case']
                identity = identities[i]
                assert identity['split'] == 'validation'
                truth, local_gt, knee_gt = measures(a[f'{i}_truth'], a[f'{i}_rest'])
                for regime in ('clean', 'noisy'):
                    prediction, local_pred, knee_pred = measures(a[f'{i}_{regime}_prediction'], a[f'{i}_rest'])
                    row = dict(step=step, case=i, dataset=case['dataset'], panel=case['panel'], regime=regime,
                        sequence_id=case['sequence_id'], truth=truth, prediction=prediction,
                        body_local_foot_position_error_mm=float(np.linalg.norm(local_pred[:, [10, 11]] - local_gt[:, [10, 11]], axis=-1).mean() * 1000),
                        foot_articulation_RMS_ratio=(np.asarray(prediction['body_local_foot_temporal_RMS_m']) / np.maximum(truth['body_local_foot_temporal_RMS_m'], 1e-8)).tolist())
                    rows.append(row)
                    if step == 8000 and regime == 'clean':
                        plot_data.append((row, local_gt, local_pred, knee_gt, knee_pred))
                if step == 8000:
                    path = Path(identity['path'])
                    metadata = json.loads((path / 'metadata.json').read_text())
                    owner = verified_array(path / 'visible_frame_owner.npy', metadata['output_sha256']['visible_frame_owner.npy'], HERE / 'data/native30_verified_read_cache_v1')
                    start = identity['start']
                    values, counts = np.unique(owner[start:start+192], return_counts=True)
                    assert 100 not in values
                    motion = verified_array(path / 'motion_world.npy', metadata['output_sha256']['motion_world.npy'], HERE / 'data/native30_verified_read_cache_v1')
                    rest = verified_array(path / 'rest.npy', metadata['output_sha256']['rest.npy'], HERE / 'data/native30_verified_read_cache_v1')
                    lo, hi = max(0, start-1), min(len(motion), start+193)
                    quality = quality_weights(np.asarray(motion[lo:hi]), np.asarray(rest), config['gt_reliability'])
                    q = quality['gt_position_weight'][start-lo:start-lo+192, [10, 11]]
                    stencils = (q[1:] >= 1) & (q[:-1] >= 1)
                    owner_rows.append(dict(case=i, sequence_id=case['sequence_id'], counts={str(k): int(v) for k, v in zip(values, counts)},
                        GT_foot_position_reliability_mean=float(q.mean()), GT_foot_velocity_stencil_valid_fraction=float(stencils.mean()),
                        meaning={'-1': 'padding/unknown', '0': 'static', '1..99': 'rigid object', '100': 'target (excluded)', '101+': 'other person'}))
    report = dict(observed_at=datetime.datetime.now().astimezone().isoformat(), optimizer_steps=0,
        live_training_modified=False, scope='All scheduled ego windows at 8000 and latest saved VAL. No case selection by prediction. Root translation and rotation removed separately per trajectory. Temporal RMS is articulation amplitude, not gait accuracy; low amplitude can be correct on stationary GT.',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        owner_audit=owner_rows, entries=rows)
    (folder / 'ARTICULATION_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    seconds = np.arange(192) / 30
    for ax, (row, gt, pred, kg, kp) in zip(axs.flat, plot_data):
        for foot, style in ((10, '-'), (11, '--')):
            ax.plot(seconds, gt[:, foot, 2] - gt[:, foot, 2].mean(), style, color='#1559a5', label='GT L/R' if foot == 10 else None)
            ax.plot(seconds, pred[:, foot, 2] - pred[:, foot, 2].mean(), style, color='#be1493', label='Prediction L/R' if foot == 10 else None)
        ax.set_title(f"{row['dataset']} {row['panel']} / step 8000")
        ax.set_ylabel('Body-local foot Z, centered (m)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel('Time (s)')
    fig.suptitle('Foot articulation after removing each trajectory root translation and rotation\nScheduled VAL; trajectories centered only for amplitude display, no alignment for metrics')
    fig.tight_layout()
    fig.savefig(folder / 'FOOT_ARTICULATION_STEP8000.png', dpi=150)
    for r in rows:
        if r['regime'] == 'clean':
            print(json.dumps({k:r[k] for k in ('step','dataset','panel','truth','prediction','foot_articulation_RMS_ratio')}))
    print('owners', json.dumps(owner_rows))


if __name__ == '__main__':
    main()
