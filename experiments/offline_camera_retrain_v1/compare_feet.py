"""Paired, multi-seed gait evaluation with no inference-time GT correction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.data import GROUPS, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.evaluate_sampling import predicted_motion_metrics
from experiments.offline_camera_retrain_v1.gallery import write_gallery
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, supervised_losses


def metrics(motion, batch):
    joints = forward_kinematics(motion.float(), batch['rest'])
    losses = supervised_losses(motion.float(), batch['motion'], batch)
    gt = batch['joints'] * 2
    head = joints[:, :, 15]
    true_head = gt[:, :, 15]
    result = dict(mpjpe_cm=float(losses['fk_mpjpe_m'] * 100),
                  head_cm=float(losses['head_error_m'] * 100),
                  gt_stance_slide_cm_frame=float(losses['foot_sliding_m_per_frame'] * 100),
                  relative_head_cm=float(((head - head[:, :1]) - (true_head - true_head[:, :1])).norm(dim=-1).mean() * 100),
                  **predicted_motion_metrics(motion, batch, joints))
    # Toe-to-toe excursion is translation invariant and does not reward
    # sliding both feet with the pelvis. Report components, not just a ratio.
    def excursion(x):
        separation = x[:, :, 10, [0, 2]] - x[:, :, 11, [0, 2]]
        return (separation - separation.mean(dim=1, keepdim=True)).square().sum(-1).mean().sqrt()
    result['foot_separation_rms_cm'] = float(excursion(joints) * 100)
    result['gt_foot_separation_rms_cm'] = float(excursion(gt) * 100)
    gt_fk = forward_kinematics(batch['motion'], batch['rest'])
    fk_batch = dict(batch, joints=gt_fk / 2)
    fk_proxies = predicted_motion_metrics(motion, fk_batch, joints)
    for metric in ('contact_coverage', 'support_floating_m', 'support_penetration_m'):
        result['fk_reference_' + metric] = fk_proxies[metric]
    fk_stance = torch.diff(gt_fk[:, :, (10, 11)], dim=1).norm(dim=-1) < .01
    velocity = torch.diff(joints[:, :, (10, 11)], dim=1).norm(dim=-1)
    result['fk_reference_stance_slide_cm_frame'] = float(
        (velocity * fk_stance).sum() / fk_stance.sum().clamp_min(1) * 100)
    gt_path = torch.diff(gt[:, :, 0, [0, 2]], dim=1).norm(dim=-1).sum(dim=1).mean()
    result['gt_root_path_length_m'] = float(gt_path)
    result['root_path_length_ratio'] = result['root_path_length_m'] / max(float(gt_path), 1e-6)
    return result, joints


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='append', required=True, help='label=path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-seed', type=int, default=777)
    parser.add_argument('--sampling-seeds', type=int, nargs='+', default=[777])
    parser.add_argument('--ddim-steps', type=int, nargs='+', default=[20])
    parser.add_argument('--samples-per-group', type=int, default=4)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = OfflineSceneMIData(args.split, seed=args.data_seed)
    cases = []
    for gi, group in enumerate(GROUPS):
        for si in range(args.samples_per_group):
            sample, identity = dataset.sample(args.length, group)
            batch = {k: v.cuda() for k, v in collate([sample]).items()}
            cases.append(dict(group=group, group_index=gi, sample_index=si,
                              identity=identity, batch=batch,
                              camera=(batch['camera'][0, :, :3] * 2).cpu().tolist(),
                              tracks={'GT': (batch['joints'][0] * 2).cpu().numpy().round(4).tolist()},
                              metrics={}))
    rows = []
    for checkpoint_spec in args.checkpoint:
        label, path = checkpoint_spec.split('=', 1)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        if config.get('hard_head_rotation'):
            raise ValueError('Hard-head checkpoint requires calibrated projection; use evaluate_turn_balance.py')
        # Different checkpoint versions may decode different body templates;
        # identical data RNG ensures exactly paired clips, not paired errors.
        versioned_dataset = OfflineSceneMIData(args.split, seed=args.data_seed,
                                               skeleton_profile=config.get('skeleton_profile', 'archived'))
        for case in cases:
            sample, identity = versioned_dataset.sample(args.length, case['group'])
            if identity != case['identity']:
                raise RuntimeError('Skeleton profile changed held-out clip selection')
            case['batch'] = {k: v.cuda() for k, v in collate([sample]).items()}
        del versioned_dataset
        model = OfflineSceneMI(config['latent_dim'], tuple(config['dim_mults'])).cuda().eval()
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        for steps in args.ddim_steps:
            for seed in args.sampling_seeds:
                variant = f'{label}_ddim{steps}_seed{seed}'
                for case in cases:
                    batch = case['batch']
                    motion = ddim_sample(model, batch, args.length, steps=steps,
                                         seed=seed + case['group_index'] * 1000 + case['sample_index'],
                                         control_mask=fixed_control_mask(1, args.length, 'head', 'cuda'))
                    if not torch.isfinite(motion).all():
                        raise RuntimeError('Nonfinite sample')
                    values, joints = metrics(motion, batch)
                    case['tracks'][variant] = joints[0].cpu().numpy().round(4).tolist()
                    case['metrics'][variant] = values
                    rows.append(dict(label=label, variant=variant, ddim_steps=steps, seed=seed,
                                     group=case['group'], identity=case['identity'], metrics=values))
                print('evaluated', variant, flush=True)
        del model
    summaries = {}
    for variant in dict.fromkeys(row['variant'] for row in rows):
        selected = [r for r in rows if r['variant'] == variant]
        def mean(items):
            return {k: float(np.mean([r['metrics'][k] for r in items])) for k in items[0]['metrics']}
        walking = [r for r in selected if r['metrics']['gt_root_path_length_m'] > 1.
                   and r['metrics']['gt_foot_separation_rms_cm'] > 10.]
        summaries[variant] = dict(mean=mean(selected), groups={g: mean([r for r in selected if r['group'] == g]) for g in GROUPS},
                                  walking_n=len(walking), walking_mean=mean(walking) if walking else None)
    report = dict(protocol={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  notes='GT used only for evaluation, never sampling; height is a GT support proxy, not scene collision. Phase mismatches can inflate GT-stance sliding. Slowest-foot speed is phase-independent but alone can reward freezing.',
                  summaries=summaries, rows=rows)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    write_gallery(args.output / 'gallery.html',
                  [{k: v for k, v in case.items() if k not in ('batch', 'group_index', 'sample_index')} for case in cases],
                  'paired foot repair')
    for variant, summary in summaries.items():
        print(variant, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
