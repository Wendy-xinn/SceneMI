"""Summarize a paired scene/no-scene training and DDIM experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


ERROR_METRICS = (
    'fk_mpjpe_cm',
    'head_error_cm',
    'truth_stance_foot_slide_cm_per_frame',
    'mean_joint_speed_m_per_frame',
    'mean_joint_acceleration_m_per_frame2',
    'contact_coverage',
    'contact_slide_m_per_frame',
    'support_floating_m',
    'support_penetration_m',
)


def last_validation(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return next(row for row in reversed(rows) if 'validation' in row)


def averaged(mapping_list, keys):
    return {key: float(np.mean([mapping[key] for mapping in mapping_list])) for key in keys}


def numpy_state_equal(left, right):
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-run', type=Path, required=True)
    parser.add_argument('--no-scene-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    scene_val = last_validation(args.scene_run / 'training_log.jsonl')
    no_scene_val = last_validation(args.no_scene_run / 'training_log.jsonl')
    scene_checkpoint = torch.load(args.scene_run / 'best.pt', map_location='cpu',
                                  weights_only=False)
    no_scene_checkpoint = torch.load(args.no_scene_run / 'best.pt', map_location='cpu',
                                     weights_only=False)
    rng_state_match = {
        'cpu_torch': torch.equal(scene_checkpoint['torch_rng_state'],
                                 no_scene_checkpoint['torch_rng_state']),
        'cuda': torch.equal(scene_checkpoint['cuda_rng_state'],
                            no_scene_checkpoint['cuda_rng_state']),
        'numpy': numpy_state_equal(scene_checkpoint['numpy_rng_state'],
                                   no_scene_checkpoint['numpy_rng_state']),
        'data': scene_checkpoint['data_rng_state'] == no_scene_checkpoint['data_rng_state'],
        'condition': (scene_checkpoint['condition_rng_state']
                      == no_scene_checkpoint['condition_rng_state']),
        'source_hash': (scene_checkpoint['config']['source_hash']
                        == no_scene_checkpoint['config']['source_hash']),
    }
    if not all(rng_state_match.values()):
        raise ValueError(f'Runs are not strictly paired: {rng_state_match}')
    scene_sample = json.loads((args.scene_run / 'ddim20_validation4.json').read_text())
    no_scene_sample = json.loads((args.no_scene_run / 'ddim20_validation4.json').read_text())
    if (scene_sample['length'], scene_sample['ddim_steps'], scene_sample['samples_per_group']) != (
            no_scene_sample['length'], no_scene_sample['ddim_steps'],
            no_scene_sample['samples_per_group']):
        raise ValueError('Sampling configurations do not match')

    validation_metrics = ('fk_mpjpe_cm', 'head_error_cm', 'foot_sliding_cm_per_frame')
    validation = {}
    for metric in validation_metrics:
        scene_values = [value for key, value in scene_val['validation'].items()
                        if key.endswith('/' + metric)]
        no_scene_values = [value for key, value in no_scene_val['validation'].items()
                           if key.endswith('/' + metric)]
        validation[metric] = {
            'scene': float(np.mean(scene_values)),
            'no_scene': float(np.mean(no_scene_values)),
            'scene_minus_no_scene': float(np.mean(scene_values) - np.mean(no_scene_values)),
        }

    groups = tuple(scene_sample['groups'])
    scene_primary = [scene_sample['groups'][group]['mean_variants']['head_full_scene']
                     for group in groups]
    no_scene_primary = [no_scene_sample['groups'][group]['mean_variants']['head_without_scene']
                        for group in groups]
    truth = [scene_sample['groups'][group]['mean_ground_truth_proxy_metrics']
             for group in groups]
    scene_mean = averaged(scene_primary, ERROR_METRICS)
    no_scene_mean = averaged(no_scene_primary, ERROR_METRICS)
    truth_mean = averaged(truth, tuple(truth[0]))
    sampling = {
        metric: {
            'scene': scene_mean[metric],
            'no_scene': no_scene_mean[metric],
            'scene_minus_no_scene': scene_mean[metric] - no_scene_mean[metric],
        }
        for metric in ERROR_METRICS
    }

    # This within-checkpoint ablation is the cleanest measure of whether the
    # trained network currently reacts to scene features at sampling time.
    scene_disabled = [scene_sample['groups'][group]['mean_variants']['head_without_scene']
                      for group in groups]
    disabled_mean = averaged(scene_disabled, ERROR_METRICS)
    within_scene_checkpoint = {
        metric: {
            'with_scene': scene_mean[metric],
            'scene_disabled': disabled_mean[metric],
            'with_minus_disabled': scene_mean[metric] - disabled_mean[metric],
        }
        for metric in ERROR_METRICS
    }
    per_group = {
        group: {
            metric: {
                'scene': scene_primary[index][metric],
                'no_scene': no_scene_primary[index][metric],
                'scene_minus_no_scene': (scene_primary[index][metric]
                                         - no_scene_primary[index][metric]),
            }
            for metric in ERROR_METRICS
        }
        for index, group in enumerate(groups)
    }
    report = {
        'pairing': 'same seed, sequence order, lengths, diffusion noise and control masks',
        'final_rng_state_match': rng_state_match,
        'steps': scene_val['step'],
        'sampling_config': {
            'length': scene_sample['length'],
            'ddim_steps': scene_sample['ddim_steps'],
            'samples_per_group': scene_sample['samples_per_group'],
            'groups': list(groups),
        },
        'fixed_timestep_validation': validation,
        'ddim_primary_equal_group_mean': sampling,
        'ddim_ground_truth_proxy_equal_group_mean': truth_mean,
        'within_scene_checkpoint_ablation': within_scene_checkpoint,
        'ddim_primary_per_group': per_group,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
