"""Paired soft-loss ablations on fixed windows, with imperfect control-track probes."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
from experiments.offline_camera_retrain_v1.data import collate
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample
from experiments.offline_camera_retrain_v1.typed_head_condition import TypedSceneMI as OfflineSceneMI, prepare_observation
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.evaluate_turn_repair import diagnostics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, forward_kinematics
from experiments.offline_camera_retrain_v1.soft_track_robustness import perturb_head_track

BASE = Path(__file__).parent / 'runs'
OUT = BASE / 'typed_head_track_oct10'
REFERENCE = BASE / 'turn_balance_oct09'
METRICS = ['mpjpe_cm', 'pa_mpjpe_mm', 'head_cm', 'head_orientation_mean_deg',
           'pelvis_orientation_mean_deg', 'gt_stance_slide_cm_frame', 'support_floating_m',
           'support_penetration_m', 'both_feet_moving_fraction', 'root_path_length_ratio',
           'head_to_neck_local_rotation_p95_deg', 'neck_to_chest_local_rotation_p95_deg',
           'neck_orientation_mean_deg', 'chest_orientation_mean_deg', 'opposite_turn', 'under_turn',
           'response_head_cm', 'response_pelvis_deg', 'response_head_deg']


def sequence_means(rows, label, metric):
    groups = defaultdict(list)
    for row in rows:
        value = row['variants'][label].get(metric)
        if value is not None:
            groups[row['identity']['sequence_id']].append(value)
    return {key: float(np.mean(values)) for key, values in groups.items()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    (OUT / 'motions').mkdir(exist_ok=True)
    paths = {label: OUT / label / 'last.pt' for label in ('camera_control','joint_clean','joint_noise','joint_reliable')}
    models = {}; configs = {}
    for label, path in paths.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 500
        config = checkpoint['config']
        assert not config.get('hard_head_rotation')
        model = OfflineSceneMI(config['latent_dim'], tuple(config['dim_mults']),
                               body_conditioning=config['body_conditioning'],
                               contact_prediction=config['contact_prediction']).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        models[label] = model; configs[label] = config
        del checkpoint
    assert len({c['source_hash'] for c in configs.values()})==1
    from types import SimpleNamespace
    from experiments.offline_camera_retrain_v1.source_recovery_contract import validate_source_recovery
    from experiments.offline_camera_retrain_v1.train import fingerprint
    audit_path=BASE/'soft_body_turn_robust_oct09/source_recovery_audit.json'
    for c in configs.values():
        validate_source_recovery(audit_path,c['initial_source_hash'],SimpleNamespace(**c))
        assert c['source_recovery_audit_sha256']==fingerprint([audit_path.resolve()])
    config = configs['camera_control']
    dataset = NativeBodyData('validation', seed=20261009, skeleton_profile=config['skeleton_profile'],
                            rich_source=config['rich_source'], trumans_scene_manifest=config['trumans_scene_manifest'],
                            trumans_window_protocol=config['trumans_window_protocol'],
                            contact_root=config['rich_contact_root'], temporal_scene_manifest=config['temporal_scene_manifest'])
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    selected = manifest['selected']
    manifest.update(checkpoints={k: str(v) for k, v in paths.items()},
                    conditions=['clean', 'drift', 'drift_gap', 'position_only', 'low_confidence'],
                    perturbation='head-slot only: 3-5cm x bias, up to 1cm y, 0-2cm z drift; smooth yaw 3-11deg; optional 10-frame missing interval; GT/camera/scene caches unchanged',
                    limitation='ideal anatomical observation drawn from held-out GT for joint variants; simulated confidence from injected error, not calibrated video estimates; clean scene caches; 500-step screening only; clean conditions have explicitly different observation types')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    with (OUT / 'rows.jsonl').open('w') as stream:
        for start in range(0, len(selected), 8):
            members = selected[start:start + 8]
            samples = []; identities = []
            for member in members:
                sample, identity = dataset.sample(member['length'], member['group'],
                    sequence_index=member['sequence_index'], start_index=member['start_index'])
                assert identity['sequence_id'] == member['sequence_id']
                if 'expected_source_start_30fps' in member:
                    assert abs(identity['source_start_30fps'] - member['expected_source_start_30fps']) < 1e-6
                samples.append(sample); identities.append(identity)
            length = members[0]['length']
            assert all(member['length'] == length for member in members)
            batch = {k: v.cuda() for k, v in collate(samples).items()}
            conditions = ['clean', 'drift', 'drift_gap', 'position_only', 'low_confidence']
            # Entire paired batches, first 8 windows of each dataset group:
            # clean/no-control response distinguishes robustness from ignoring inputs.
            if length == 128 and start in (0, 40, 72, 112):
                conditions.append('none')
            for replicate in range(2):
                seed = 20261009 + replicate * 100000 + start
                clean = None
                for condition in conditions:
                    mask = fixed_control_mask(len(members),length,'head',batch['motion'].device)
                    if condition=='none':mask.zero_()
                    altered_by_label={label:prepare_observation(batch,configs[label]['observation_protocol'],evaluation=condition) for label in models}
                    motions={label:ddim_sample(model,altered_by_label[label],length,steps=20,seed=seed,control_mask=mask) for label,model in models.items()}
                    if condition=='clean':clean=motions
                    for i, (member, identity) in enumerate(zip(members, identities)):
                        one = {k: v[i:i+1] for k, v in batch.items()}
                        truth_forward = global_rotations(one['motion'])[0, :, 0, :, 2]
                        valid = float((truth_forward[:, [0, 2]].norm(dim=-1) > .3).float().mean())
                        variants = {}
                        for label, motion in motions.items():
                            value = motion[i:i+1]
                            metrics = diagnostics(value, one['motion'], one)
                            local = rotation_from_6d(value[0, :, 93:99])
                            angle = torch.rad2deg(torch.acos(((local.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))
                            metrics['head_to_neck_local_rotation_p95_deg'] = float(torch.quantile(angle, .95))
                            neck_local = rotation_from_6d(value[0, :, 75:81])
                            neck_angle = torch.rad2deg(torch.acos(((neck_local.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))
                            metrics['neck_to_chest_local_rotation_p95_deg'] = float(torch.quantile(neck_angle, .95))
                            truth_rotation = global_rotations(one['motion'])
                            predicted_rotation = global_rotations(value)
                            for joint, name in ((9, 'chest'), (12, 'neck')):
                                relative = truth_rotation[:, :, joint].transpose(-1, -2) @ predicted_rotation[:, :, joint]
                                metrics[name + '_orientation_mean_deg'] = float(torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))).mean())
                            gt_turn = metrics['pelvis_gt_turn_deg']
                            eligible = abs(gt_turn) >= 30 and valid >= .95
                            metrics['opposite_turn'] = float(metrics['pelvis_turn_deg'] * gt_turn < 0) if eligible else None
                            metrics['under_turn'] = float(abs(metrics['pelvis_turn_deg']) < .5 * abs(gt_turn)) if eligible else None
                            clean_value = clean[label][i:i+1]
                            metrics['response_head_cm'] = float((forward_kinematics(value, one['rest'])[:, :, 15] - forward_kinematics(clean_value, one['rest'])[:, :, 15]).norm(dim=-1).mean() * 100)
                            current_rotation = global_rotations(value)
                            clean_rotation = global_rotations(clean_value)
                            for joint, name in ((0, 'pelvis'), (15, 'head')):
                                relative = clean_rotation[:, :, joint].transpose(-1, -2) @ current_rotation[:, :, joint]
                                metrics['response_' + name + '_deg'] = float(torch.rad2deg(torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))).mean())
                            variants[label] = metrics
                        gt_turn = variants['camera_control']['pelvis_gt_turn_deg']
                        filename = f'motions/{start+i:04d}_{replicate}_{condition}.npz'
                        np.savez_compressed(OUT / filename, truth_motion=one['motion'][0].cpu().numpy(),
                                            rest=one['rest'][0].cpu().numpy(), camera=one['camera'][0].cpu().numpy(),
                                            input_head=altered_by_label['joint_reliable']['trajectory'][i, :, 15].cpu().numpy(),
                                            observation_meta=altered_by_label['joint_reliable']['observation_meta'][i,:,15].cpu().numpy(),
                                            input_head_mask=mask[i, :, 15].cpu().numpy(),
                                            **{k: v[i].cpu().numpy() for k, v in motions.items()})
                        row = dict(index=start+i, replicate=replicate, seed=seed, identity=identity,
                                   group=member['group'], length=length, condition=condition,
                                   turn_bin='unreliable_projection' if valid < .95 else 'large_turn' if abs(gt_turn) >= 60 else 'moderate_turn' if abs(gt_turn) >= 15 else 'small_turn',
                                   variants=variants, motion_file=filename)
                        stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
            (OUT / 'evaluation_status.json').write_text(json.dumps(dict(status='running', completed_windows=start+len(members), total_windows=len(selected), elapsed_s=time.monotonic()-started)))
            print('completed', start+len(members), flush=True)
    summarize(rows)
    (OUT / 'evaluation_status.json').write_text(json.dumps(dict(status='completed', windows=len(selected), evaluated_predictions=len(rows)*len(models), elapsed_s=time.monotonic()-started), indent=2))


def summarize(rows):
    summary = {}; intervals = {}; rng = np.random.default_rng(20261009)
    for condition in sorted({row['condition'] for row in rows}):
        core = [row for row in rows if row['condition'] == condition and row['length'] == 128]
        groups = {'all': core}
        for key in ('group', 'turn_bin'):
            for value in sorted({row[key] for row in core}):
                groups[key + '/' + value] = [row for row in core if row[key] == value]
        for length in (64, 192):
            groups['length/' + str(length)] = [row for row in rows if row['condition'] == condition and row['length'] == length]
        summary[condition] = {}
        for group, members in groups.items():
            if not members: continue
            result = dict(windows=len({row['index'] for row in members}), sequences=len({row['identity']['sequence_id'] for row in members}), variants={})
            for label in members[0]['variants']:
                result['variants'][label] = {metric: float(np.mean(list(values.values()))) if values else None
                    for metric in METRICS for values in [sequence_means(members, label, metric)]}
            summary[condition][group] = result
            if group not in ('all', 'turn_bin/large_turn'): continue
            for label in ('joint_clean','joint_noise','joint_reliable'):
                for metric in METRICS:
                    baseline = sequence_means(members, 'camera_control', metric); candidate = sequence_means(members, label, metric)
                    ids = sorted(set(baseline) & set(candidate))
                    if not ids: continue
                    delta = np.array([candidate[key] - baseline[key] for key in ids])
                    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
                    intervals['/'.join((condition, group, label, metric))] = dict(difference=float(delta.mean()), exploratory_sequence_bootstrap95=np.quantile(boot, [.025, .975]).tolist())
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'paired_intervals.json').write_text(json.dumps(intervals, indent=2))

if __name__ == '__main__':
    main()
