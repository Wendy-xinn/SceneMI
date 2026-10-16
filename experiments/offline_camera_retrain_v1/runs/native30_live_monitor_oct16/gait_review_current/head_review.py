#!/usr/bin/env python3
"""CPU-only audit of saved head evidence; never loads or changes a model.

Reads the immutable validation snapshot and current source/configuration. GT is
used only for scoring. Writes only the requested review JSON (stdout by default).
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
PROJECT = BASE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('PARENTS', 'IDENTITY_6D'):
                    result[target.id] = ast.literal_eval(node.value)
    return result


def fk(motion, rest, parents, identity):
    """NumPy equivalent of supervision.py:13-38, evaluated on CPU."""
    value = motion[:, 3:135].reshape(-1, 22, 6) + np.asarray(identity, dtype=motion.dtype)
    first = value[..., :3]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    second = value[..., 3:] - np.sum(first * value[..., 3:], axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-6)
    local = np.stack((first, second, np.cross(first, second)), axis=-1)
    rotations = [local[:, 0]]
    joints = [motion[:, :3] * 2 + rest[None, 0]]
    for joint in range(1, 22):
        parent = parents[joint]
        joints.append(joints[parent] + np.einsum('tij,j->ti', rotations[parent], rest[joint] - rest[parent]))
        rotations.append(rotations[parent] @ local[:, joint])
    return np.stack(joints, axis=1)


def audit(snapshot, config_path, cases):
    config = json.loads(config_path.read_text())
    mounts_path = BASE / config['mount_defaults']
    mounts = json.loads(mounts_path.read_text())
    skeleton = constants(BASE / 'supervision.py')
    evidence = []
    with np.load(snapshot, allow_pickle=False) as saved:
        step = int(saved['step'])
        identities = json.loads(str(saved['identities_json']))
        for case in cases:
            identity = identities[case]
            mount = identity.get('actual_virtual_mount') or mounts['camera_wearer']
            sigma = float(mount['position_p90_m'])
            truth_motion, rest = saved[f'{case}_truth'], saved[f'{case}_rest']
            truth = fk(truth_motion, rest, **{'parents': skeleton['PARENTS'], 'identity': skeleton['IDENTITY_6D']})
            dataset = 'egobody' if identity['group'] in ('camera_wearer', 'interactee') else identity['group']
            for regime in ('clean', 'noisy'):
                prediction = saved[f'{case}_{regime}_prediction']
                reference = saved[f'{case}_{regime}_reference_motion']
                pred = fk(prediction, rest, skeleton['PARENTS'], skeleton['IDENTITY_6D'])
                ref = fk(reference, rest, skeleton['PARENTS'], skeleton['IDENTITY_6D'])
                observation = saved[f'{case}_{regime}_observations'][:, 15, :3] * 2
                meta = saved[f'{case}_{regime}_observation_meta'][:, 15]
                valid = (meta[:, 0] > 0) & (meta[:, 2] > 0)
                def error(delta):
                    return float(np.linalg.norm(delta[valid], axis=-1).mean() * 1000) if valid.any() else None
                def self_consistency(motion, joints):
                    channels = motion[:, 135:].reshape(-1, 22, 3) * 2
                    return float(np.abs(channels - joints).max() * 1000)
                gain = float(config['refinement']['observation_anchor_gain'])
                evidence.append(dict(case=case, dataset=dataset, sequence_id=identity['sequence_id'],
                    regime=regime, valid_head_frames=int(valid.sum()), total_frames=len(valid),
                    observed_to_GT_head_mean_mm=error(observation - truth[:, 15]),
                    predicted_to_observed_head_mean_mm=error(pred[:, 15] - observation),
                    predicted_to_GT_head_mean_mm=error(pred[:, 15] - truth[:, 15]),
                    reference_to_observed_head_mean_mm=error(ref[:, 15] - observation),
                    max_FK_vs_saved_joint_channel_error_mm=dict(truth=self_consistency(truth_motion, truth),
                        prediction=self_consistency(prediction, pred), reference=self_consistency(reference, ref)),
                    raw_world_first_camera_Y_m=float(identity['origin'][1]),
                    window_camera_first_Y_m=0.,
                    observed_head_first_window_Y_m=float(observation[0, 1]) if valid[0] else None,
                    observed_head_valid_window_Y_range_m=[float(observation[valid, 1].min()), float(observation[valid, 1].max())] if valid.any() else None,
                    mount_position_p90_m=sigma,
                    mount_rotation_p90_rad=float(mount['rotation_p90_rad']),
                    clean_full_confidence_head_translation_gain_at_alpha_1=gain * .05**2 / (.05**2 + sigma**2),
                    clean_head_eligible_for_physics_trusted_guard=sigma < .05,
                    uncertainty_note='Mount floor only; noisy per-frame uncertainty is not stored, so its actual gains are not reconstructed.'))
    legacy_review = BASE / 'archive/historical_trials_oct16/hierarchical_constraint_oct11/ASSESSMENT_zh.md'
    sources = [Path(__file__).resolve(), config_path, mounts_path, BASE / 'supervision.py', BASE / 'native30_data.py',
        BASE / 'native30_observations.py', BASE / 'native30_refinement.py', BASE / 'native30_observation_ik.py',
        BASE / 'native30_physical_guidance.py', BASE / 'hard_head_coordination.py', legacy_review,
        PROJECT / 'docs/local_dependencies/diffusion-motion-inbetweening/experiments/offline_sequence_v1/data_loader.py']
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), step=step,
        snapshot=str(snapshot), snapshot_sha256=digest(snapshot), CPU_only=True,
        model_loaded=False, training_modified=False, GT_used_only_for_scoring=True,
        metrics='Mean Euclidean head distance in millimeters over available/confident observed frames; joints recomputed by NumPy FK.',
        source_sha256={str(path): digest(path) for path in sources},
        refinement=config['refinement'], physical_guidance=config['physical_guidance'],
        coordinate_facts=dict(origin='All three components of first camera world position are subtracted.',
            rotation='Yaw only: gravity Y retained; no pitch/roll normalization.',
            retained='Within-window head vertical changes and relative head-scene height.',
            removed='Raw first-camera world Y; head alone does not encode a local floor height.',
            head_first_sample='Camera first Y is zero, but inverse mount translation means anatomical head first Y need not be zero.'),
        mechanism_facts=dict(reference='Observed-head-aligned kinematic initializer, not an exact final-body projection.',
            anchor='Soft alpha/gain/confidence/uncertainty weighted root translation toward head position.',
            IK='Six damped position-only articulated iterations; no scene or forced head orientation.',
            physics_guard='Hard acceptance inequality limits worsening of trusted measurements to 2mm; it does not set head equal to observation. EgoBody mount sigma exceeds trusted threshold.',
            final_sampling='No call to the legacy exact project_head_pose.'),
        historical_evidence=dict(review=str(legacy_review),
            review_lines='11,20-24', exact_projection=str(BASE / 'hard_head_coordination.py') + ':12-23',
            failure='Legacy exact head position+orientation projection retained incorrect pelvis heading and produced neck twist/foot sliding; 850/906 head-to-neck p95 106.9/141.6 degrees.',
            boundary='This joint intervention does not isolate the effect of exact head position alone.'),
        limitations=['No causal intermediate raw/IK/physics trajectory was logged in this snapshot.',
            'Head and remote floor height do not uniquely specify sitting/crouching, knee flexion or unknown local terrain.',
            'Legacy exact head position+orientation failure does not isolate or disprove every possible exact position constraint.'],
        cases=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=HERE / 'FROZEN_VALIDATION.npz')
    parser.add_argument('--config', type=Path, default=BASE / 'runs/native30_gait_v2_ego55k_oct15/config.json')
    parser.add_argument('--cases', type=int, nargs='+', default=[0, 2, 4])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.snapshot.resolve(), args.config.resolve(), args.cases)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps(dict(output=str(args.output), step=report['step'], cases=len(report['cases'])), ensure_ascii=False))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
