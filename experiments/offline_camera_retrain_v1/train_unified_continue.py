"""Fixed128 offline ego/exo continuation from retained original55k."""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import HERE, SOURCE
from experiments.offline_camera_retrain_v1.audit_unified_inputs import dataset
from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.contact_supervision import calibrated_contact_loss
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.training_contract_v2 import covered_frames, domain_probabilities
from experiments.offline_camera_retrain_v1.unified_geometry import SkinAnchorCache
from experiments.offline_camera_retrain_v1.unified_observations import (
    ObservationBuilder, collate_observations, inference_payload)
from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_losses import (
    fit_camera_mounts, track_losses, calibrate_weights, JOINT_WEIGHTS, SCALES)

OUTPUT = HERE / 'runs/ego_exo_surface_continue_oct11'
SEED = 2026101143


def sampling_plan(data, frames=128):
    counts = {}
    for group in ('trumans', 'camera_wearer', 'interactee', 'rich'):
        source = data.rich_members if group == 'rich' else data.base.groups[group]
        counts[group] = sum(covered_frames(starts[frames] if group == 'rich' else
                          [s for s, _ in starts[frames]], frames) for _, starts in source if starts[frames])
    probabilities = domain_probabilities(dict(trumans=counts['trumans'],
                                              egobody=counts['camera_wearer'] + counts['interactee'],
                                              rich=counts['rich']), .5)
    roles = counts['camera_wearer'] + counts['interactee']
    groups = dict(trumans=probabilities['trumans'], rich=probabilities['rich'],
                  camera_wearer=probabilities['egobody'] * counts['camera_wearer'] / roles,
                  interactee=probabilities['egobody'] * counts['interactee'] / roles)
    return dict(valid_frames=counts, group_probabilities=groups, domain_probabilities=probabilities,
                alpha=.5, length=frames, exo_given_egobody=.5)


def export_complete(root):
    path = root / 'manifest.jsonl'
    if not path.exists():
        return False
    rows = list(map(json.loads, path.read_text().splitlines()))
    for split in ('train', 'validation'):
        expected = {r['sequence_id'] for r in map(json.loads, (SOURCE / f'egobody_sequences/{split}.jsonl').read_text().splitlines())}
        observed = {r['sequence_id'] for r in rows if r['split'] == split}
        if observed != expected:
            return False
    return all((Path(row['path']) / 'metadata.json').exists() for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--ego-warmup', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--integration-only', action='store_true')
    args = parser.parse_args(); folder = args.output; folder.mkdir(parents=True, exist_ok=True)
    checks = json.loads((OUTPUT / 'input_checks.json').read_text())
    if checks['status'] != 'passed':
        raise ValueError('Input checks must pass before training')
    if (folder / 'last.pt').exists() and not args.resume:
        raise ValueError('Existing checkpoint: use --resume; never silently restart')
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
    source = HERE / 'runs/native_dynamic_scene20_contact_55k_oct07/last.pt'
    cp = torch.load(source, map_location='cpu', weights_only=False); c = cp['config']
    data = dataset(c, 'train', SEED)
    rng = np.random.default_rng(SEED + 991)
    model = UnifiedSceneMI(c['latent_dim'], tuple(c['dim_mults']), body_conditioning=True, contact_prediction=True).cuda()
    model.load_state_dict(cp['model']); del cp
    cache = SkinAnchorCache(); builder = ObservationBuilder(cache)
    adapter, core = [], []
    for name, parameter in model.named_parameters():
        (adapter if name.startswith(('anchor_embedding.', 'point_embedding.', 'surface_attention.', 'surface_projection.', 'surface_contact.')) else core).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=core, lr=2e-5, multiplier=1.),
                                   dict(params=adapter, lr=1e-4, multiplier=5.)], weight_decay=.01)
    plan = sampling_plan(data)
    started = time.monotonic(); first_step = 0; weights = None; counts = {}
    code_names = ('export_egobody_exo.py', 'unified_geometry.py', 'unified_observations.py',
                  'unified_model.py', 'unified_losses.py', 'train_unified_continue.py',
                  'audit_exo_observations.py', 'unified_metrics.py', 'evaluate_unified_continue.py')
    code_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in code_names}
    protocol = dict(status='training', steps=args.steps, ego_warmup=args.ego_warmup, batch_size=args.batch_size,
                    seed=SEED, initialization=str(source), original_step=55000,
                    optimizer='new AdamW for changed interface; original resumable checkpoint retained',
                    input='offline frozen camera/anatomical position/rotation with separate availability and confidence; no body prefix',
                    shape='known native body calibration; ideal geometry observations, not estimated real video input',
                    exo='EgoBody master Kinect, first-hit visibility; positions only; head/wrists initially',
                    scene='own-view observed surfaces only; offline static union, current dynamic objects; unknown is masked',
                    surface='67 fixed native vertex IDs, generated-pose LBS; 13 local probes; masked cross-attention and predicted-contact feedback',
                    length=128, accepted_inference_lengths=[64, 128, 192], inference='full-window DDIM, not frame autoregression',
                    low_noise_fraction=.25, loss_profile='coordination_v1 retained plus physical observation derivatives',
                    derivative_base_equivalence=dict(velocity_coefficient=10/20**2, acceleration_coefficient=2/20**4,
                                                     explanation='old per-frame base terms retained exactly; physical-unit equivalents'),
                    joint_weights=JOINT_WEIGHTS, scales=SCALES, sampling_plan=plan,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), code_sha256=code_hashes,
                    free_disk_before=shutil.disk_usage(HERE).free)
    mounts_path = folder / 'TRAIN_camera_mounts.json'
    if args.resume:
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        old_protocol = json.loads((folder / 'protocol.json').read_text())
        if old_protocol['code_sha256'] != code_hashes:
            raise ValueError('Resume code changed; register a new continuation explicitly')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        for entry in optimizer.state.values():
            for key, value in entry.items():
                if torch.is_tensor(value): entry[key] = value.cuda()
        torch.set_rng_state(state['torch_rng_state']); torch.cuda.set_rng_state(state['cuda_rng_state'])
        np.random.set_state(state['numpy_rng_state']); data.load_sampler_state(state['data_rng_state'])
        rng.bit_generator.state = state['condition_rng_state']
        first_step = state['step']; weights = state['loss_weights']; counts = state['counts']
        mounts = json.loads(mounts_path.read_text()); del state
        protocol = old_protocol
    else:
        mounts = fit_camera_mounts(data, seed=SEED)
        mounts_path.write_text(json.dumps(mounts, indent=2) + '\n')
    (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    alphas = cosine_alphas().cuda()
    groups = list(plan['group_probabilities']); probabilities = list(plan['group_probabilities'].values())

    def checkpoint(step):
        config = dict(c, unified_protocol='offline_surface_v1', fixed_training_length=128,
                      continuation_steps=args.steps, continuation_seed=SEED, surface_bps_weight=model.surface_bps_weight,
                      loss_weights=weights, sampling_plan=plan)
        save_checkpoint(dict(step=step, original_step=55000, model=model.state_dict(), optimizer=optimizer.state_dict(),
                             config=config, loss_weights=weights, counts=counts,
                             torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
                             numpy_rng_state=np.random.get_state(), data_rng_state=data.sampler_state(),
                             condition_rng_state=rng.bit_generator.state, code_sha256=code_hashes), folder / 'last.pt')

    with (folder / 'trace.jsonl').open('a' if args.resume else 'w') as trace:
        for step in range(first_step + 1, args.steps + 1):
            if step == args.ego_warmup + 1 and not args.integration_only:
                checkpoint(step - 1)
                print('WAITING for complete exo TRAIN/validation export; no partial-dataset sampling', flush=True)
                while not export_complete(builder.exo_root):
                    time.sleep(20)
                from experiments.offline_camera_retrain_v1.audit_exo_observations import audit
                audit(builder.exo_root, folder / 'exo_export_checks.json')
                print('EXO EXPORT COMPLETE; mixed continuation enabled', flush=True)
            group = str(rng.choice(groups, p=probabilities))
            mode = 'exo' if step > args.ego_warmup and group in ('camera_wearer', 'interactee') and rng.random() < .5 else 'ego'
            if args.integration_only: mode = 'ego'
            items, identities = [], []
            for _ in range(args.batch_size):
                sample, identity = data.sample(128, group)
                selected = (15, 20, 21) if mode == 'exo' or rng.random() < .35 else (15,)
                observed, scene, asset = builder.build(data, sample, identity, mode, joints=selected, noise_rng=rng)
                items.append((observed, scene, asset, mode)); identities.append(identity)
            batch = collate_observations(items); known = inference_payload(batch)
            clean = batch['motion']; t = torch.randint(1000, (args.batch_size,), device='cuda')
            t = torch.where(torch.rand(args.batch_size, device='cuda') < .25,
                            torch.randint(100, (args.batch_size,), device='cuda'), t)
            alpha = alphas[t][:, None, None]
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
            scene_on = bool(rng.random() >= .1)
            control = batch['observation_meta'][..., :2].amax(-1) > 0
            model.surface_bps_weight = min(1., max(0., (step - 50) / 950))
            model.train(); optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy, t, known, control_mask=control, use_scene=scene_on)
            base = supervised_losses(prediction, clean, batch, profile='coordination_v1', signal_weight=alpha.flatten())
            observed_losses = track_losses(prediction, clean, batch, alpha.flatten(), mounts, [group] * args.batch_size)
            if weights is None:
                weights, scale_report = calibrate_weights(base['total'], observed_losses, prediction)
                scale_report.update(identities=identities, timesteps=t.tolist(), joint_weights=JOINT_WEIGHTS)
                (folder / 'TRAIN_loss_scale_audit.json').write_text(json.dumps(scale_report, indent=2) + '\n')
            observation_total = sum(weights[key] * value for key, value in observed_losses.items())
            contact = calibrated_contact_loss(model.contact_logits(prediction, known, use_scene=scene_on), batch, alpha.flatten())
            total = base['total'] + observation_total + .1 * contact
            if not torch.isfinite(total): raise FloatingPointError('Nonfinite loss')
            total.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            lr = 2e-5 * min(1., step / 50) * max(.1, .5 * (1 + math.cos(math.pi * step / args.steps)))
            for entry in optimizer.param_groups: entry['lr'] = lr * entry['multiplier']
            optimizer.step(); counts[f'{group}/{mode}'] = counts.get(f'{group}/{mode}', 0) + args.batch_size
            row = dict(step=step, group=group, mode=mode, length=128, identities=identities,
                       t=t.tolist(), scene=scene_on, lr=lr, loss=float(total.detach()), gradient_norm=float(norm),
                       base_loss=float(base['total'].detach()), observation_loss=float(observation_total.detach()),
                       contact_bce=float(contact.detach()), elapsed_s=time.monotonic() - started,
                       surface_bps_weight=model.surface_bps_weight,
                       observed_terms={key: float(value.detach()) for key, value in observed_losses.items()})
            trace.write(json.dumps(row) + '\n'); trace.flush()
            if step == 1 or step % 25 == 0:
                print(step, group, mode, round(row['loss'], 5), 'elapsed', round(row['elapsed_s'], 1), flush=True)
            if step % 100 == 0 or step == args.steps:
                checkpoint(step)
                protocol.update(status='training' if step < args.steps else 'training_completed',
                                completed_steps=step, counts=counts, elapsed_s=time.monotonic() - started,
                                free_disk_after=shutil.disk_usage(HERE).free)
                (folder / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print('TRAINING COMPLETE', args.steps, flush=True)
    if not args.integration_only:
        del model, optimizer, builder, data
        torch.cuda.empty_cache()
        from experiments.offline_camera_retrain_v1.evaluate_unified_continue import evaluate
        evaluate(folder)


if __name__ == '__main__':
    main()
