"""Audit saved scene evidence without fitting or changing existing checkpoints.

Rebuild the registered TRAIN windows, inspect observed-scene/GT correspondences
on a fixed 30 Hz subsample, and measure gradients at the saved denoiser. GT is
used only for audit/supervision; it never enters model_inputs or the queries.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.prepare_native30 import read_config, make_model, write_json
from experiments.offline_camera_retrain_v1.native30_overnight import scoped_data, build_bank, augment
from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
from experiments.offline_camera_retrain_v1.native30_losses import objective
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.native30_scene_support import training_support_targets


def sliced_scene(scene, indices):
    result = copy.copy(scene)
    result.dynamic = [scene.dynamic[i] for i in indices]
    result._dynamic_trees = [None] * len(indices)
    result.cameras = scene.cameras[indices]
    result.dynamic_features = scene.dynamic_features[indices]
    result.dynamic_valid = scene.dynamic_valid[indices]
    return result


def signal_audit(item, identity):
    sample, scene, asset, mode = item
    asset = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in asset.items()}
    indices = np.arange(0, len(sample['motion']), 6)
    motion = torch.from_numpy(sample['motion'][indices])
    points = skin_points(motion, asset)
    rotations = global_rotations(motion[None])[0, :, asset['labels']]
    q = sliced_scene(scene, indices).query(points.numpy(), rotations.numpy())
    valid = q['valid'][:, :, 2, 0]
    nearest = q['points'][:, :, 2, 0]
    distance = np.linalg.norm(points.numpy() - nearest, axis=-1)
    reliable = valid & q['normal_reliable'][:, :, 2, 0]
    annotated = sample['contact_valid'][indices][:, asset['labels']] & (
        sample['contact_target'][indices][:, asset['labels']] > .5)
    static = q['owners'][:, :, 2, 0] == 0
    speed = np.zeros_like(distance)
    speed[1:] = np.linalg.norm(np.diff(points.numpy(), axis=0) * 5, axis=-1)
    feet = np.isin(asset['labels'], [7, 8, 10, 11])[None]
    candidate = feet & reliable & static & (distance < .05) & (speed < .2)
    candidate[0] = False
    delta = points.numpy() - nearest
    signed = (delta * q['normals'][:, :, 2, 0]).sum(-1)
    tangent = np.linalg.norm(delta - signed[..., None] * q['normals'][:, :, 2, 0], axis=-1)
    legacy_contact_mask = reliable & annotated & (tangent < .09) & (distance < .25)
    # Perfect GT need not put every marker of a contacting body REGION onto
    # an observed surface. Report the old loss's nonzero error at perfect GT.
    legacy_contact_loss = (np.maximum(distance - .02, 0)**2)[legacy_contact_mask]
    support = training_support_targets(sample, scene, asset)
    new_labels = support['valid'] & ~sample['contact_valid']
    occ = sample['occupancy']
    p = scene.points
    inside = ((p >= [-6.4, -3.1, -6.4]) & (p < [6.4, 1.7, 6.4])).all(1)
    return dict(group=identity['group'], mode=mode, sequence_id=identity['sequence_id'],
                start=identity['source_start_30fps'], sampled_frames=len(indices),
                original_scene_points=len(p), occupancy_retained_fraction=float(inside.mean()),
                nonempty_voxels=int(occ.sum()), gt_anchor_query_coverage=float(valid.mean()),
                gt_anchor_reliable_fraction=float(reliable.mean()),
                annotated_positive_anchors=int(annotated.sum()),
                annotated_positive_query_covered=int((annotated & valid).sum()),
                annotated_positive_near_5cm=int((annotated & reliable & (distance < .05)).sum()),
                observed_static_foot_support_candidates=int(candidate.sum()),
                old_contact_loss_at_perfect_GT_m2=float(legacy_contact_loss.mean()) if len(legacy_contact_loss) else 0.,
                old_supervised_positive_anchor_count=int(legacy_contact_mask.sum()),
                old_supervised_positive_anchors_beyond_2cm=int((legacy_contact_mask & (distance > .02)).sum()),
                weak_contact_new_valid_joint_frames=int(new_labels.sum()),
                weak_contact_new_positive_joint_frames=int((new_labels & (support['target'] > .5)).sum()),
                note='Fixed every-sixth-frame audit; support candidates are weak targets, not official contact labels')


def grad_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(sum(g.detach().float().square().sum() for g in gradients if g is not None).sqrt())


def gradient_audit(model, item, identity, builder, config, weights):
    observed = augment(item, identity, builder, config, np.random.default_rng(140), noisy=False, yaw=False, head_only=True)
    batch = collate_observations([observed]); known = model_inputs(config, batch)
    target = training_motion(config, batch, known)
    model.eval()
    scene_parameters = tuple(p for name, p in model.named_parameters() if p.requires_grad and
                             any(k in name for k in ('scene_embedding', 'patch_', 'surface_', 'region_condition',
                                                    'contact_', 'correspondence_', 'feedback')))
    rows = []
    for t in (50, 500):
        torch.manual_seed(140 + t)
        timesteps = torch.tensor([t], device=target.device)
        signal = model.alphas[timesteps]
        noisy = signal[:, None, None].sqrt() * target + (1 - signal[:, None, None]).sqrt() * torch.randn_like(target)
        prediction, details = model(noisy, timesteps, known, return_details=True)
        loss, terms = objective(prediction, batch['motion'], batch, signal, details, config, weights)
        geometry_physical = sum(config['geometry']['coefficients'][k] * terms['geometry'][k] for k in
                                ('contact_distance_m2', 'contact_relative_slip_m2_s2', 'observed_plane_proxy_m2'))
        geometry_all = sum(config['geometry']['coefficients'][k] * v for k, v in terms['geometry'].items())
        rows.append(dict(timestep=t, alpha=float(signal), local_feedback_gate=float(signal.square()),
                         base_loss=float(terms['base']['total']), physical_scene_loss=float(geometry_physical),
                         all_geometry_loss=float(geometry_all), total_loss=float(loss),
                         scene_parameter_base_grad_norm=grad_norm(terms['base']['total'], scene_parameters),
                         scene_parameter_physical_grad_norm=grad_norm(geometry_physical, scene_parameters),
                         scene_parameter_all_geometry_grad_norm=grad_norm(geometry_all, scene_parameters),
                         proposal_query_coverage=float(torch.stack([g['closest_valid'].float().mean() for g in details['geometry']]).mean()),
                         contact_gate_mean=float(details['contact_logits'].sigmoid().mean()),
                         correspondence_gate_mean=float(details['correspondence_logits'].sigmoid().mean()),
                         root_feedback_rms_m=float(details['root_delta_m'].square().mean().sqrt()),
                         rotation_feedback_rms_rad=float(details['rotation_delta_rad'].square().mean().sqrt())))
        del prediction, details, loss, terms
    return dict(group=identity['group'], mode=identity['mode'], rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=HERE / 'runs/native30_overnight_ik_oct14/experiment')
    parser.add_argument('--output', type=Path, default=HERE / 'runs/native30_overnight_ik_oct14/SCENE_SIGNAL_AUDIT.json')
    parser.add_argument('--gradients', action='store_true')
    args = parser.parse_args(); began = time.monotonic()
    torch.set_num_threads(4)
    config = json.loads((args.run / 'protocol.json').read_text())['config']
    plan = json.loads((args.run / 'window_plan.json').read_text())['train']
    chosen = []; seen = set()
    for row in plan:
        key = row['group'], row['mode']
        if key not in seen: chosen.append(row); seen.add(key)
    data = scoped_data(config, 'train')
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(),
        json.loads((HERE / config['mount_defaults']).read_text()), config['observation']['input_noise'])
    bank, identities = build_bank(data, builder, chosen)
    report = dict(protocol='scene-signal-input-and-gradient-audit-v1',
                  fitting_steps=0, test_used=False, input_source='saved TRAIN plan; no outcome-based selection',
                  source_checkpoint=str(args.run / 'candidate/last.pt'), signal=[])
    for item, identity in zip(bank, identities):
        result = signal_audit(item, identity); report['signal'].append(result)
        print('SIGNAL', json.dumps(result), flush=True)
    write_json(args.output, report)
    if args.gradients:
        torch.cuda.set_per_process_memory_fraction(.30)
        checkpoint = torch.load(args.run / 'candidate/last.pt', map_location='cpu', weights_only=False)
        model = make_model(config).cuda(); model.load_state_dict(checkpoint['model']); del checkpoint
        weights_path = HERE / config['prepared_output'] / 'TRAIN_gradient_scales.json'
        scale_report = json.loads(weights_path.read_text())
        weights = scale_report.get('coefficients', scale_report.get('observation_weights'))
        if weights is None: raise ValueError(f'Observation weight keys: {list(scale_report)}')
        report['gradients'] = []
        for group, mode in [('camera_wearer', 'ego'), ('camera_wearer', 'exo'), ('rich', 'exo')]:
            i = next(i for i, v in enumerate(identities) if (v['group'], v['mode']) == (group, mode))
            result = gradient_audit(model, bank[i], identities[i], builder, config, weights)
            report['gradients'].append(result); print('GRADIENT', json.dumps(result), flush=True)
            write_json(args.output, report)
    report['elapsed_s'] = time.monotonic() - began; write_json(args.output, report)


if __name__ == '__main__':
    main()
