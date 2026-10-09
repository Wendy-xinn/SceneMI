"""Small held-out DDIM and condition-ablation evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import GROUPS, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.control import fixed_control_mask
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, ddim_sample
from experiments.offline_camera_retrain_v1.supervision import FEET, forward_kinematics, supervised_losses
from experiments.offline_camera_retrain_v1.gallery import write_gallery
from experiments.offline_camera_retrain_v1.coordination import coordination_losses


def to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def predicted_motion_metrics(motion, batch, joints=None):
    if joints is None:
        joints = forward_kinematics(motion.float(), batch['rest'].float())
    velocity = joints[:, 1:] - joints[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    feet = joints[:, :, FEET]
    foot_velocity = feet[:, 1:] - feet[:, :-1]
    # Evaluation-only proxy: held-out true feet estimate the local support
    # height. This is not an input to the model or a training label.
    true_feet = batch['joints'][:, :, FEET].float() * 2
    support_height = torch.quantile(true_feet[..., 1].flatten(1), .05, dim=1)
    predicted_height = feet[..., 1]
    contact = ((predicted_height[:, 1:] - support_height[:, None, None]).abs() < .05)
    if contact.any():
        contact_slide = foot_velocity.norm(dim=-1)[contact].mean()
    else:
        contact_slide = foot_velocity.norm(dim=-1).mean() * 0.
    lowest = predicted_height.min(dim=-1).values
    floating = (lowest - support_height[:, None] - .05).clamp_min(0).mean()
    penetration = (support_height[:, None] - lowest).clamp_min(0).mean()
    relative = feet - joints[:, :, :1]
    relative_speed = torch.diff(relative[..., [0, 2]], dim=1).norm(dim=-1).mean()
    true_root = batch['joints'][:, :, :1].float() * 2
    true_relative_speed = torch.diff((true_feet - true_root)[..., [0, 2]], dim=1).norm(dim=-1).mean()
    # This phase-independent diagnostic detects foot planting even if a valid
    # generated gait has a different left/right phase from the reference.
    slowest = foot_velocity[..., [0, 2]].norm(dim=-1).min(dim=-1).values
    return {
        **{key: float(value) for key, value in
           coordination_losses(joints, batch['joints'].float() * 2).items()},
        'root_path_length_m': float(velocity[:, :, 0, [0, 2]].norm(dim=-1).sum(dim=1).mean()),
        'foot_horizontal_speed_m_per_frame': float(foot_velocity[..., [0, 2]].norm(dim=-1).mean()),
        'mean_joint_speed_m_per_frame': float(velocity.norm(dim=-1).mean()),
        'mean_joint_acceleration_m_per_frame2': float(acceleration.norm(dim=-1).mean()),
        'contact_coverage': float(contact.float().mean()),
        'contact_slide_m_per_frame': float(contact_slide),
        'support_floating_m': float(floating),
        'support_penetration_m': float(penetration),
        'foot_relative_speed_ratio': float(relative_speed / true_relative_speed.clamp_min(1e-6)),
        'foot_relative_speed_m_per_frame': float(relative_speed),
        'slowest_foot_horizontal_cm_per_frame': float(slowest.mean() * 100),
        'both_feet_moving_fraction': float((slowest >= .01).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=int, default=64, choices=(64, 128, 192))
    parser.add_argument('--ddim-steps', type=int, default=20)
    parser.add_argument('--samples-per-group', type=int, default=4)
    parser.add_argument('--gallery', type=Path)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--history-reset',action='store_true',help='Add paired scene conditioning with static memory reset at target start; preserve current dynamic surfaces')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--data-seed', type=int, default=777)
    parser.add_argument('--sampling-seed', type=int, default=777)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.samples_per_group < 1:
        parser.error('samples-per-group must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = OfflineSceneMI(int(config['latent_dim']), tuple(config['dim_mults']),body_conditioning=config.get('body_conditioning',False),contact_prediction=config.get('contact_prediction',False)).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    data_class=OfflineSceneMIData
    if config.get('body_protocol')=='native_v1':
        from experiments.offline_camera_retrain_v1.native_body_data import NativeBodyData
        data_class=NativeBodyData
    dataset = data_class(args.split, seed=args.data_seed,
                                skeleton_profile=config.get('skeleton_profile', 'archived'),
                                rich_source=config.get('rich_source', 'legacy5interp'), trumans_scene_manifest=config.get('trumans_scene_manifest'),trumans_window_protocol=config.get('trumans_window_protocol','legacy_stable_v1'),**({'contact_root':config.get('rich_contact_root'),'rich_causal_scene':config.get('rich_causal_scene',False),'temporal_scene_manifest':config.get('temporal_scene_manifest')} if config.get('body_protocol')=='native_v1' else {}))
    trained_with_scene = bool(config.get('scene_condition_enabled', True))
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_step': checkpoint['step'],
              'length': args.length, 'ddim_steps': args.ddim_steps,
              'samples_per_group': args.samples_per_group,
              'trained_with_scene': trained_with_scene,
              'primary_variant': ('head_full_scene' if trained_with_scene
                                  else 'head_without_scene'),
              'split': args.split, 'data_seed': args.data_seed,
              'sampling_seed': args.sampling_seed, 'groups': {}}
    gallery_cases = []
    for group_index, group in enumerate(GROUPS):
        cases = []
        for sample_index in range(args.samples_per_group):
            sample, identity = dataset.sample(args.length, group)
            batch = to_cuda(collate([sample]))
            ground_truth_metrics = predicted_motion_metrics(
                batch['motion'], batch, joints=batch['joints'].float() * 2)
            tracks = {'GT': (batch['joints'][0] * 2).cpu().numpy()}
            variants = {}
            masks = {
                'head': fixed_control_mask(1, args.length, 'head', 'cuda'),
                'pelvis': fixed_control_mask(1, args.length, 'pelvis', 'cuda'),
                'wrists': fixed_control_mask(1, args.length, 'wrists', 'cuda'),
            }
            variant_settings=[
                    ('head_full_scene', masks['head'], True, True),
                    ('head_without_scene', masks['head'], False, True),
                    ('without_control', masks['head'], trained_with_scene, False),
                    ('pelvis_trained_scene', masks['pelvis'], trained_with_scene, True),
                    ('wrists_trained_scene', masks['wrists'], trained_with_scene, True)]
            reset_batch=None
            if args.history_reset:
                from experiments.offline_camera_retrain_v1.history_reset_eval import reset_history_batch
                reset_batch=reset_history_batch(batch,identity,args.length)
                variant_settings.append(('head_reset_history',masks['head'],True,True))
            for name, control_mask, use_scene, use_control in variant_settings:
                if args.head_only and name != report['primary_variant']:
                    continue
                generated = ddim_sample(
                    model, reset_batch if name=='head_reset_history' else batch, args.length, steps=args.ddim_steps,
                    seed=args.sampling_seed + group_index * 1000 + sample_index,
                    control_mask=control_mask, use_scene=use_scene, use_control=use_control)
                losses = supervised_losses(generated.float(), batch['motion'], batch)
                generated_joints = forward_kinematics(generated.float(), batch['rest'].float())
                tracks[name] = generated_joints[0].cpu().numpy()
                head = generated_joints[:, :, 15]
                true_head = batch['joints'][:, :, 15].float() * 2
                variants[name] = {
                    'head_path_relative_error_cm': float(
                        ((head - head[:, :1]) - (true_head - true_head[:, :1])).norm(dim=-1).mean() * 100),
                    'head_camera_distance_cm': float(
                        (head - batch['camera'][..., :3] * 2).norm(dim=-1).mean() * 100),
                    'fk_mpjpe_cm': float(losses['fk_mpjpe_m'] * 100),
                    'head_error_cm': float(losses['head_error_m'] * 100),
                    'truth_stance_foot_slide_cm_per_frame': float(
                        losses['foot_sliding_m_per_frame'] * 100),
                    **predicted_motion_metrics(generated, batch, joints=generated_joints),
                }
                from experiments.offline_camera_retrain_v1.evaluate_report_oct07 import wa_error
                variants[name]['wa_mpjpe_cm']=wa_error(generated_joints[0].cpu().numpy(),batch['joints'][0].cpu().numpy()*2)
                from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics
                variants[name].update(standard_motion_metrics(generated_joints[0].cpu().numpy(),batch['joints'][0].cpu().numpy()*2))
                if model.contact_prediction:
                    from experiments.offline_camera_retrain_v1.contact_supervision import contact_metrics
                    variants[name].update({k:float(v) for k,v in contact_metrics(model.contact_logits(generated,reset_batch if name=='head_reset_history' else batch,use_scene=use_scene),batch).items()})
            if args.gallery:
                gallery_cases.append(dict(group=group, identity=identity,
                                          camera=(batch['camera'][0, :, :3] * 2).cpu().numpy().tolist(),
                                          tracks={key: value.round(4).tolist() for key, value in tracks.items()},
                                          metrics=variants))
            cases.append({'identity': identity,
                          'ground_truth_proxy_metrics': ground_truth_metrics,
                          'variants': variants})

        def means(items):
            return {key: float(np.mean([item[key] for item in items])) for key in items[0]}

        report['groups'][group] = {
            'mean_ground_truth_proxy_metrics': means(
                [case['ground_truth_proxy_metrics'] for case in cases]),
            'mean_variants': {
                name: means([case['variants'][name] for case in cases])
                for name in cases[0]['variants']},
            'cases': cases,
        }
    from experiments.offline_camera_retrain_v1.standard_motion_metrics import METRIC_PROTOCOL
    report['standard_metric_protocol'] = METRIC_PROTOCOL
    report['metric_notes'] = {
        'support': 'GT foot-height proxy; not measured scene collision and unreliable on multiple support heights.',
        'contact_slide': 'Zero with zero contact_coverage means no contact, not good foot planting.',
        'head_camera_distance': 'Includes anatomical camera-to-head offset; not an exact constraint error.',
        'ground_truth': ('Native model FK from raw-source20 poses and native shape; archived joints are an audit reference.' if config.get('body_protocol')=='native_v1' else 'Dataset/protocol targets; archived raw joints for legacy TRUMANS/EgoBody.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    if args.gallery:
        write_gallery(args.gallery, gallery_cases, checkpoint['step'])
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
