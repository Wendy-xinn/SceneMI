"""Prepare/evaluate configuration and gradients, with zero optimizer steps."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import collate_observations, observation_derivatives
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_losses import JOINT_WEIGHTS
from experiments.offline_camera_retrain_v1.camera_v2_model import CameraAwareSceneMI, NEW_PREFIXES, camera_features
from experiments.offline_camera_retrain_v1.camera_v2_observations import (
    CameraObservationBuilder, inference_payload_v2, load_frozen_exo_v2)
from experiments.offline_camera_retrain_v1.camera_v2_config import (
    read_config, CONFIG_PATH, effective_sampling_plan, observation_losses,
    optimizer_groups, balanced_weights)
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.contact_supervision import calibrated_contact_loss


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def term_gradients(base, terms, prediction):
    reference = float(torch.autograd.grad(base, prediction, retain_graph=True)[0].norm())
    report = {}
    for term, value in terms.items():
        gradient = torch.autograd.grad(value, prediction, retain_graph=True)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f'Nonfinite {term} output gradient')
        report[term] = dict(loss=float(value.detach()), gradient_norm=float(gradient.norm()))
    return reference, report


def input_preview(items, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 10))
    for i, (group, mode, item) in enumerate(items):
        sample, scene, _, _ = item
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        points = scene.points[::max(1, len(scene.points) // 3000)]
        if len(points):
            ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=.4, alpha=.2, c='gray')
        body = sample['joints'][0] * 2
        ax.scatter(body[:, 0], body[:, 2], body[:, 1], s=12, c='royalblue', label='GT reference')
        available = sample['observation_meta'][0, :, 0] > 0
        positions = sample['trajectory'][0, available, :3] * 2
        ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1], s=26, c='orange', label='provided tracks')
        camera = sample['camera'][0, :3] * 2
        ax.scatter(*camera[[0, 2, 1]], c='red', marker='^', s=45, label='observer camera')
        r0, r1 = sample['camera'][0, 3:6], sample['camera'][0, 6:9]
        forward = np.cross(r0, r1)
        ax.quiver(*camera[[0, 2, 1]], *forward[[0, 2, 1]], length=.5, color='red')
        ax.set(xlabel='X m', ylabel='Z m', zlabel='Y (up) m', title=f'{group}/{mode}: {int(available.sum())} supplied tracks')
        ax.legend(fontsize=7); ax.view_init(elev=18, azim=-65)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def run(config_path=CONFIG_PATH, output=None):
    config = read_config(config_path)
    folder = Path(output or HERE / config['prepared_output']); folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    source = HERE / config['initialization']
    before_sha = sha256(source)
    report = dict(status='running', optimizer_steps=0, source_checkpoint=str(source),
                  source_sha256_before=before_sha, free_disk_before=shutil.disk_usage(HERE).free, checks={})
    cp = torch.load(source, map_location='cpu', weights_only=False); inherited = cp['config']
    data = dataset(inherited, 'train', config['seed'])
    mounts = json.loads((HERE / config['mount_defaults']).read_text())
    builder = CameraObservationBuilder(SkinAnchorCache(), mounts)
    model = CameraAwareSceneMI(inherited['latent_dim'], tuple(inherited['dim_mults']),
                              body_conditioning=True, contact_prediction=True).cuda().eval()
    model.load_legacy_state_dict(cp['model']); model.surface_bps_weight = config['scene']['surface_bps_weight']
    original = UnifiedSceneMI(inherited['latent_dim'], tuple(inherited['dim_mults']),
                              body_conditioning=True, contact_prediction=True).cuda().eval()
    original.load_state_dict(cp['model']); original.surface_bps_weight = model.surface_bps_weight
    del cp
    plan = effective_sampling_plan(data, config)
    groups = optimizer_groups(model, config)
    group_summary = [{key: value for key, value in group.items() if key != 'params'} |
                     dict(parameters=sum(p.numel() for p in group['params'])) for group in groups]
    del groups
    cases = {}; identities = {}; heights = {}
    for group, mode in [('trumans', 'ego'), ('camera_wearer', 'ego'), ('interactee', 'ego'),
                        ('rich', 'ego'), ('camera_wearer', 'exo'), ('interactee', 'exo')]:
        sample, identity = data.sample(128, group, sequence_index=0, start_index=0)
        prepared, scene, asset = builder.build(data, sample, identity, mode,
                                              joints=tuple(range(22)) if mode == 'exo' else (15, 20, 21))
        key = f'{group}/{mode}'
        cases[key] = (prepared, scene, asset, mode); identities[key] = identity
        heights[key] = builder.body_height(identity['native_body'])
        # Scene tensors and geometry remain exactly the old implementation.
        old_sample, old_scene, _ = builder.__class__.__mro__[1].build(
            builder, data, sample, identity, mode,
            joints=tuple(range(22)) if mode == 'exo' else (15, 20, 21))
        for field in ('occupancy', 'bps', 'bps_valid'):
            assert np.array_equal(prepared[field], old_sample[field]), (key, field)
        assert np.array_equal(scene.points, old_scene.points)
    report['checks']['scene_inputs_unchanged_all_six_branches'] = True
    identity = identities['interactee/exo']
    name = identity['sequence_id']; ids = builder.exo[name]['source_frame_ids']
    frozen = load_frozen_exo_v2(builder.exo_root / 'train' / name, identity['native_body'],
                               start=int(np.searchsorted(ids, identity['source_start_30fps'])),
                               skin_cache=builder.skin_cache)
    controlled = inference_payload_v2(collate_observations([cases['interactee/exo']]))
    for key in ('camera', 'trajectory', 'occupancy', 'camera_context', 'camera_intrinsics'):
        assert torch.allclose(frozen[key], controlled[key], atol=1e-6), key
    assert not any(key in frozen for key in ('motion', 'joints', 'contact_target'))
    report['checks']['label_free_frozen_exo_loader_matches_controlled'] = True
    sensor_sample, sensor_identity = data.sample(128, 'camera_wearer', sequence_index=0, start_index=0)
    sensor_observed, _, _ = builder.build(data, sensor_sample, sensor_identity, 'ego',
                                        joints=(15,), noise_rng=np.random.default_rng(0))
    assert np.array_equal(sensor_observed['camera'], sensor_observed['trajectory'][:, 15])
    assert np.array_equal(sensor_observed['camera_available'], sensor_observed['observation_meta'][:, 15, :2])
    assert np.array_equal(sensor_observed['camera_confidence'], sensor_observed['observation_meta'][:, 15, 2:4])
    assert not sensor_observed['camera_available'][..., 0].all()
    report['checks']['self_camera_sensor_augmentation_coupled'] = True
    report['native_heights_m'] = heights
    report['known_shape_policy'] = 'per-subject native shape/rest/scale; unknown shape estimation not implemented'
    items = [cases['camera_wearer/ego'], cases['interactee/exo']]
    batch = collate_observations(items); known = inference_payload_v2(batch)
    noisy = torch.randn(2, 128, 201, device='cuda'); t = torch.tensor([25, 300], device='cuda')
    mask = batch['observation_meta'][..., :2].amax(-1) > 0
    with torch.no_grad():
        baseline = original(noisy, t, known, control_mask=mask)
        reference = model(noisy, t, known, control_mask=mask)
    report['checks']['zero_init_legacy_max_error'] = float((baseline - reference).abs().max())
    assert torch.equal(baseline, reference)
    del original

    # Exercise new routes with temporary nonzero projections; zero-init alone
    # would make masking tests vacuous. No optimizer or trained weights saved.
    saved = {name: p.detach().clone() for name, p in model.named_parameters()
             if name.startswith(('camera_projection.', 'observation_projection.'))}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in saved:
                p.normal_(0, .02)
        active = model(noisy, t, known, control_mask=mask)
        poisoned = dict(batch)
        for key in ('motion', 'joints', 'contact_target', 'body_betas'):
            poisoned[key] = torch.full_like(batch[key], float('nan'))
        assert torch.equal(active, model(noisy, t, inference_payload_v2(poisoned), control_mask=mask))
        hidden = dict(known); trajectory = known['trajectory'].clone()
        trajectory[..., :3] = torch.where(known['observation_meta'][..., 0:1] > 0, trajectory[..., :3], float('nan'))
        trajectory[..., 3:] = torch.where(known['observation_meta'][..., 1:2] > 0, trajectory[..., 3:], float('nan'))
        hidden['trajectory'] = trajectory
        assert torch.equal(active, model(noisy, t, hidden, control_mask=mask))
        gap = dict(known)
        gap['camera_available'] = known['camera_available'].clone(); gap['camera_available'][:, 30:45] = 0
        gap['camera_intrinsics_valid'] = known['camera_intrinsics_valid'].clone(); gap['camera_intrinsics_valid'][:, 30:45] = 0
        gap['mount_valid'] = torch.zeros_like(known['mount_valid'])
        gap_output = model(noisy, t, gap, control_mask=mask)
        poison_gap = dict(gap)
        for key in ('camera', 'camera_confidence', 'camera_intrinsics'):
            poison_gap[key] = gap[key].clone(); poison_gap[key][:, 30:45] = float('nan')
        for key in ('mount_rotation', 'mount_translation_m', 'mount_uncertainty'):
            poison_gap[key] = torch.full_like(gap[key], float('nan'))
        assert torch.equal(gap_output, model(noisy, t, poison_gap, control_mask=mask))
        changed = dict(known, camera=known['camera'].clone())
        changed['camera'][:, :, 0] += .05
        camera_difference = float((model(noisy, t, changed, control_mask=mask) - active).abs().max())
        model.typed_observations_enabled = False
        observation_difference = float((model(noisy, t, known, control_mask=mask) - active).abs().max())
        model.typed_observations_enabled = True
        assert camera_difference > 1e-6 and observation_difference > 1e-6
        bad_relation = dict(known, camera_context=known['camera_context'].clone())
        bad_relation['camera_context'][0, :2] = known['camera_context'].new_tensor([0., 1.])
        try:
            model(noisy, t, bad_relation, control_mask=mask)
        except ValueError:
            pass
        else:
            raise AssertionError('External observer was allowed to be the target head')
    model.zero_grad(set_to_none=True)
    activated = model(noisy, t, known, control_mask=mask)
    activated.square().mean().backward()
    norms = {name: float(p.grad.norm()) for name, p in model.named_parameters()
             if name.startswith(NEW_PREFIXES) and p.grad is not None}
    assert all(np.isfinite(list(norms.values())))
    for prefix in NEW_PREFIXES:
        assert any(value > 0 for name, value in norms.items() if name.startswith(prefix)), prefix
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in saved:
                p.copy_(saved[name])
    model.zero_grad(set_to_none=True)
    report['checks'].update(label_and_hidden_poison_invariant=True, camera_and_mount_masking=True,
                           camera_effect_max=camera_difference, typed_adapter_effect_max=observation_difference,
                           relation_guard=True, activated_parameter_gradient_norms=norms)

    # Scalar derivatives: 1 m/s translation and gap stencil must stay physical.
    position = torch.zeros(1, 24, 1, 9, device='cuda')
    position[0, :, 0, 0] = torch.arange(24, device='cuda') / 40
    position[..., 3:] = position.new_tensor([1, 0, 0, 0, 1, 0])
    meta = position.new_ones(1, 24, 1, 5)
    velocity, acceleration, _, valid = observation_derivatives(position, meta)
    assert torch.allclose(velocity[:, 1:, :, 0], torch.ones_like(velocity[:, 1:, :, 0]), atol=3e-6)
    assert float(acceleration[:, 2:].abs().max()) < 1e-4
    meta[:, 10:12, :, :2] = 0
    _, _, _, gap_valid = observation_derivatives(position, meta)
    assert not gap_valid[:, 10:13, :, 0].any() and not gap_valid[:, 10:14, :, 1].any()
    report['checks']['physical_units_and_no_gap_derivatives'] = True

    rows = []
    for branch, item in cases.items():
        sample_batch = collate_observations([item]); payload = inference_payload_v2(sample_batch)
        for noise_step in config['supervision']['calibration_noise_steps']:
            step = torch.tensor([noise_step], device='cuda'); alpha = cosine_alphas().cuda()[step]
            torch.manual_seed(config['seed'] + noise_step)
            xt = alpha[:, None, None].sqrt() * sample_batch['motion'] + (1 - alpha[:, None, None]).sqrt() * torch.randn_like(sample_batch['motion'])
            with torch.no_grad():
                pred = model(xt, step, payload, control_mask=sample_batch['observation_meta'][..., :2].amax(-1) > 0)
            pred = pred.detach().requires_grad_(True)
            base = supervised_losses(pred, sample_batch['motion'], sample_batch,
                                     profile=config['supervision']['base_profile'], signal_weight=alpha)
            terms = observation_losses(pred, sample_batch['motion'], sample_batch, alpha)
            base_norm, term_report = term_gradients(base['total'], terms, pred)
            buckets = {}
            for label, joint_ids in dict(head=[15], wrists=[20, 21],
                                         lower=[1, 2, 4, 5, 7, 8, 10, 11],
                                         torso_arms=[0, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19]).items():
                subset = dict(sample_batch, observation_meta=sample_batch['observation_meta'].clone())
                select = torch.zeros(22, device='cuda', dtype=torch.bool); select[joint_ids] = True
                subset['observation_meta'][:, :, ~select, :4] = 0
                counts = int(subset['observation_meta'][..., :2].amax(-1).sum())
                if counts:
                    subset_terms = observation_losses(pred, sample_batch['motion'], subset, alpha)
                    _, gradients = term_gradients(base['total'], subset_terms, pred)
                    gradients.pop('body_turn_rate')  # Global root term is not a per-part observation.
                    buckets[label] = dict(valid_entries=counts, terms=gradients)
            rows.append(dict(branch=branch, noise_step=noise_step, identity=identities[branch],
                             body_height_m=heights[branch], base_gradient_norm=base_norm,
                             position_entries=int(sample_batch['observation_meta'][..., 0].sum()),
                             rotation_entries=int(sample_batch['observation_meta'][..., 1].sum()),
                             terms=term_report, body_buckets=buckets))
            print('gradient audit', branch, noise_step, flush=True)
    weights, calibrated = balanced_weights(rows, config['supervision']['target_gradient_ratios'],
                                           config['supervision']['coefficient_caps'])
    # Verify the complete objective and new condition exits, without an update.
    alpha = cosine_alphas().cuda()[t]
    pred = model(noisy, t, known, control_mask=mask)
    base = supervised_losses(pred, batch['motion'], batch, profile='coordination_v1', signal_weight=alpha)
    terms = observation_losses(pred, batch['motion'], batch, alpha)
    contact = calibrated_contact_loss(model.contact_logits(pred, known), batch, alpha)
    total = base['total'] + sum(weights[k] * value for k, value in terms.items()) + config['supervision']['contact_coefficient'] * contact
    assert torch.isfinite(total)
    total.backward()
    train_norms = {name: float(p.grad.norm()) for name, p in model.named_parameters()
                   if name.startswith(('camera_projection.', 'observation_projection.')) and p.grad is not None}
    assert all(np.isfinite(list(train_norms.values()))) and any(v > 0 for v in train_norms.values())
    report['checks']['full_objective_backward_no_step'] = dict(loss=float(total.detach()), new_exit_gradients=train_norms)
    input_preview([(g, m, cases[f'{g}/{m}']) for g, m in
                   [('camera_wearer', 'ego'), ('interactee', 'ego'), ('camera_wearer', 'exo'), ('interactee', 'exo')]], folder / 'input_camera_alignment.png')
    report.update(status='passed', source_sha256_after=sha256(source),
                  free_disk_after=shutil.disk_usage(HERE).free,
                  scope='plumbing, units and initial TRAIN gradients; no trained quality claim')
    assert report['source_sha256_after'] == before_sha
    effective = dict(config=config, inherited_dataset_configuration={key: inherited[key] for key in
                     ('latent_dim', 'dim_mults', 'skeleton_profile', 'rich_source', 'trumans_scene_manifest',
                      'trumans_window_protocol', 'temporal_scene_manifest', 'rich_contact_root')},
                     source_sha256=before_sha, sampling_plan=plan, optimizer_groups=group_summary,
                     mount_defaults_sha256=sha256(HERE / config['mount_defaults']),
                     observation_coefficients=weights, joint_weights=JOINT_WEIGHTS,
                     training_started=False, optimizer_steps=0)
    from experiments.offline_camera_retrain_v1.train_camera_v2 import code_hashes
    effective['code_sha256'] = code_hashes()
    write_json(folder / 'effective_config.json', effective)
    write_json(folder / 'TRAIN_balanced_gradient_audit.json', dict(status='passed', rows=calibrated,
               coefficients=weights, calibration_split='train', optimizer_steps=0,
               scope='six TRAIN windows, six branches and three noise steps; initial calibration, not full-distribution guarantee'))
    write_json(folder / 'input_checks.json', report)
    print(json.dumps(dict(status='passed', optimizer_steps=0, output=str(folder)), indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); run(args.config, args.output)
