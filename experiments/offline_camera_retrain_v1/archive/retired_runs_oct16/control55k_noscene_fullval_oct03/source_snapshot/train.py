"""Train the adapted SceneMI AdaGN U-Net from random initialization."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.data import GROUPS, LENGTHS, SOURCE, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.control import MODE_PROBABILITIES, fixed_control_mask, sample_control_masks
from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI, cosine_alphas
from experiments.offline_camera_retrain_v1.supervision import supervised_losses
from experiments.offline_camera_retrain_v1.gallery import write_run_index

HERE = Path(__file__).resolve().parent


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def audited_data_fingerprint():
    ready = json.loads((SOURCE / 'READY.json').read_text())
    if ready.get('status') != 'ready':
        raise ValueError('Original sequence archive is not audited')
    source_paths = [SOURCE / 'sequences.jsonl', SOURCE / 'intervals.jsonl']
    source_paths += [SOURCE / 'trumans_sequences' / f'{s}.jsonl' for s in ('train', 'validation')]
    source_paths += [SOURCE / 'egobody_sequences' / f'{s}.jsonl' for s in ('train', 'validation')]
    if fingerprint(source_paths) != ready['manifest_sha256']:
        raise ValueError('Original sequence manifests changed after audit')
    fk = json.loads((HERE / 'data/poses/fk_audit.json').read_text())
    if any(item['p95_m'] > .08 for split in fk.values() for item in split.values()):
        raise ValueError('FK alignment failed')
    pose_paths = [HERE / 'data/poses' / f'{s}.jsonl' for s in ('train', 'validation', 'test')]
    visible_paths = []
    for dataset, visible_name, source_name in (
            ('TRUMANS', 'visible_trumans_5hz', 'trumans_sequences'),
            ('EgoBody', 'visible_egobody_5hz', 'egobody_sequences')):
        for split in ('train', 'validation', 'test'):
            path = HERE / 'data' / visible_name / f'{split}.jsonl'
            visible_paths.append(path)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            expected = len((SOURCE / source_name / f'{split}.jsonl').read_text().splitlines())
            metadata_valid = all(
                row.get('sample_stride') == 4
                and row.get('candidate_count') == 65536
                and row.get('query_radius_m') == 6.5
                and 'project' in row.get('protocol', '').lower()
                for row in rows)
            if (len(rows) != expected or not metadata_valid
                    or not all(Path(row['path']).is_file() for row in rows)):
                raise ValueError(f'{dataset}/{split}: incomplete or invalid 5 Hz ego-visible scene map')
    return fingerprint(source_paths + pose_paths + visible_paths + [HERE / 'data/poses/fk_audit.json'])


def sample_batch(dataset, length, group, size):
    examples = [dataset.sample(length, group)[0] for _ in range(size)]
    return collate(examples)


def to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'runs/mixed_offline_scenemi_v1')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--validation-samples', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--scene-dropout', type=float, default=.1)
    parser.add_argument('--disable-scene', action='store_true')
    parser.add_argument('--ddim-every', type=int, default=0,
                        help='Run sampling and write a gallery at this checkpoint interval (0 disables)')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init-from', type=Path,
                        help='Warm-start weights only; reset optimizer, schedule and RNG for a paired trial')
    parser.add_argument('--loss-profile', choices=('baseline', 'gait_v1', 'gait_v2', 'coordination_v1'), default='baseline')
    parser.add_argument('--ddim-head-only', action='store_true')
    parser.add_argument('--skeleton-profile', choices=('archived', 'trumans_male_v2'), default='archived')
    parser.add_argument('--precision', choices=('bf16', 'float32'), default='bf16')
    parser.add_argument('--low-noise-fraction', type=float, default=0.,
                        help='Fraction of additional low-noise t=0..99 denoising examples')
    parser.add_argument('--keep-inference-every', type=int, default=5000,
                        help='Keep model-only weights at this sampling interval, plus the final step')
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error('resume and init-from are mutually exclusive')
    if not 0 <= args.low_noise_fraction <= 1:
        parser.error('low-noise-fraction must be in [0, 1]')
    if args.keep_inference_every < 1:
        parser.error('keep-inference-every must be positive')
    if (args.steps < 1 or args.batch_size < 1 or args.save_every < 1
            or args.validation_samples < 1):
        parser.error('steps, batch-size, save-every and validation-samples must be positive')
    if not 0 <= args.scene_dropout <= 1:
        parser.error('scene-dropout must be in [0, 1]')
    if args.ddim_every < 0 or (args.ddim_every and args.ddim_every % args.save_every):
        parser.error('ddim-every must be zero or a positive multiple of save-every')
    if not torch.cuda.is_available():
        raise RuntimeError('SceneMI training requires a CUDA GPU')
    if any((args.output / name).exists() for name in ('last.pt', 'best.pt', 'training_log.jsonl')) and not args.resume:
        raise FileExistsError('Checkpoint exists; use --resume or another output directory')
    data_hash = audited_data_fingerprint()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    train = OfflineSceneMIData('train', seed=args.seed, skeleton_profile=args.skeleton_profile)
    validation = OfflineSceneMIData('validation', seed=777, skeleton_profile=args.skeleton_profile)
    condition_rng = np.random.default_rng(args.seed + 104729)
    for group in GROUPS:
        for length in LENGTHS:
            if not train.summary()[group][str(length)]['sequences']:
                raise ValueError(f'Missing train {group}/{length}')
            if not validation.summary()[group][str(length)]['sequences']:
                raise ValueError(f'Missing validation {group}/{length}')
    train_scenes = {r['scene_family'] for members in train.base.groups.values() for r, _ in members}
    val_scenes = {r['scene_family'] for members in validation.base.groups.values() for r, _ in members}
    if train_scenes & val_scenes:
        raise ValueError('Scene-family leakage')
    model = OfflineSceneMI(args.latent_dim, tuple(args.dim_mults)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    alphas = cosine_alphas().cuda()
    config = dict(**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_hash=data_hash, architecture='SceneMI original AdaGN TemporalUnet + ViT occupancy + BPS MLP',
                  motion_representation='SMPL(-X) root translation, 22 identity-centred residual 6D rotations, 22 global joints',
                  condition=('masked per-joint position/orientation trajectories (head-dominant mixture) '
                             '+ visible static map + camera-centered BPS'),
                  control_mode_probabilities=dict(MODE_PROBABILITIES),
                  scene_condition_enabled=not args.disable_scene,
                  scene_dropout_probability=args.scene_dropout,
                  losses=('x0 + rotation matrix + FK joint/velocity/acceleration + emphasized head path '
                          '+ stance-foot stillness + direct/FK consistency'),
                  train_summary=train.summary(), validation_summary=validation.summary())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.skeleton_profile == 'trumans_male_v2':
        config['skeleton_template_sha256'] = fingerprint([HERE / 'data/body_templates/trumans_male_rest.npy'])
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    first = 0
    best = float('inf')
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location='cpu', weights_only=False)
        if checkpoint['config']['source_hash'] != data_hash:
            raise ValueError('Warm-start source data mismatch')
        model.load_state_dict(checkpoint['model'])
        config['initial_checkpoint_step'] = checkpoint['step']
        config['initial_total_steps'] = (checkpoint['config'].get('initial_total_steps',
                                         checkpoint['config'].get('initial_checkpoint_step', 0))
                                         + checkpoint['step'])
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        del checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key, default in (('loss_profile', 'baseline'), ('precision', 'bf16'), ('low_noise_fraction', 0.)):
            if checkpoint['config'].get(key, default) != getattr(args, key):
                raise ValueError(f'Resume changed {key}; pass stored protocol flags, or use init-from for a new trial')
        if checkpoint['config'].get('skeleton_profile', 'archived') != args.skeleton_profile:
            raise ValueError('Resume cannot change skeleton profile; use init-from for an explicit new trial')
        if checkpoint['config'].get('skeleton_template_sha256') != config.get('skeleton_template_sha256'):
            raise ValueError('Skeleton template changed since checkpoint')
        if checkpoint['config']['source_hash'] != data_hash:
            raise ValueError('Source data changed since checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        first = checkpoint['step']
        best = checkpoint['best_validation']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        train.base.rng.bit_generator.state = checkpoint['data_rng_state']
        if 'condition_rng_state' in checkpoint:
            condition_rng.bit_generator.state = checkpoint['condition_rng_state']
    fixed_validation = {
        (group, length, sample_index): to_cuda(sample_batch(validation, length, group, 1))
        for group in GROUPS for length in (128, 192)
        for sample_index in range(args.validation_samples)}
    log = (args.output / 'training_log.jsonl').open('a', buffering=1)
    start_time = time.time()

    def validation_metrics():
        model.eval()
        buckets = {}
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(777)
            for (group, length, _), batch in fixed_validation.items():
                clean = batch['motion']
                timestep = torch.full((len(clean),), 500, dtype=torch.long, device='cuda')
                alpha = alphas[timestep][:, None, None]
                noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
                head_mask = fixed_control_mask(len(clean), length, 'head', clean.device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    predicted = model(noisy, timestep, batch, control_mask=head_mask,
                                      use_scene=not args.disable_scene)
                losses = supervised_losses(predicted.float(), clean, batch)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    without_scene = model(noisy, timestep, batch, control_mask=head_mask,
                                          use_scene=False)
                    without_control = model(noisy, timestep, batch, control_mask=head_mask,
                                            use_scene=not args.disable_scene, use_control=False)
                values = {
                    'fk_mpjpe_cm': losses['fk_mpjpe_m'] * 100,
                    'head_error_cm': losses['head_error_m'] * 100,
                    'foot_sliding_cm_per_frame': losses['foot_sliding_m_per_frame'] * 100,
                    'x0': losses['x0'],
                    'fk_velocity_mse': losses['fk_velocity_mse'],
                    'fk_acceleration_mse': losses['fk_acceleration_mse'],
                    'scene_effect_l1': (predicted - without_scene).abs().mean(),
                    'control_effect_l1': (predicted - without_control).abs().mean(),
                }
                for metric, value in values.items():
                    buckets.setdefault((group, length, metric), []).append(float(value))
        model.train()
        return {f'{group}/{length}/{metric}': float(np.mean(values))
                for (group, length, metric), values in buckets.items()}

    model.train()
    for step in range(first + 1, args.steps + 1):
        length = int(np.random.choice(LENGTHS, p=[.25, .5, .25]))
        group = GROUPS[int(np.random.randint(len(GROUPS)))]
        batch = to_cuda(sample_batch(train, length, group, args.batch_size))
        clean = batch['motion']
        timestep = torch.randint(len(alphas), (len(clean),), device='cuda')
        if args.low_noise_fraction:
            low_noise = torch.rand(len(clean), device='cuda') < args.low_noise_fraction
            low_t = torch.randint(100, (len(clean),), device='cuda')
            timestep = torch.where(low_noise, low_t, timestep)
        alpha = alphas[timestep][:, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
        control_mask, control_modes = sample_control_masks(
            len(clean), length, condition_rng, clean.device)
        scene_draw = condition_rng.random()
        use_scene = bool(not args.disable_scene and scene_draw >= args.scene_dropout)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            predicted = model(noisy, timestep, batch, control_mask=control_mask,
                              use_scene=use_scene)
        losses = supervised_losses(predicted.float(), clean, batch,
                                   profile=args.loss_profile, signal_weight=alpha.flatten())
        loss = losses['total']
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        warmup = min(1., step / 500)
        decay = max(.1, .5 * (1 + math.cos(math.pi * step / args.steps)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * warmup * decay
        optimizer.step()
        if step == 1 or step % 20 == 0:
            row = dict(step=step, length=length, group=group, elapsed_s=time.time() - start_time,
                       control_modes=control_modes,
                       scene=use_scene,
                       lr=optimizer.param_groups[0]['lr'],
                       gradient_norm=float(gradient_norm),
                       samples_seen=step * args.batch_size,
                       peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                       losses={key: float(value.detach()) for key, value in losses.items()})
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
        if step % args.save_every == 0 or step == args.steps:
            metrics = validation_metrics()
            score = float(np.mean([v for k, v in metrics.items() if k.endswith('/fk_mpjpe_cm')]))
            row = dict(step=step, validation=metrics, validation_mean_fk_mpjpe_cm=score)
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            improved = score < best
            best = min(best, score)
            checkpoint = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                              config=config, best_validation=best,
                              torch_rng_state=torch.get_rng_state(),
                              cuda_rng_state=torch.cuda.get_rng_state(),
                              numpy_rng_state=np.random.get_state(),
                              data_rng_state=train.base.rng.bit_generator.state,
                              condition_rng_state=condition_rng.bit_generator.state)
            temporary = args.output / 'last.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / 'last.pt')
            if improved:
                temporary = args.output / 'best.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(args.output / 'best.pt')
            if args.ddim_every and (step % args.ddim_every == 0 or step == args.steps):
                evaluation_dir = args.output / 'evaluations' / f'step_{step:06d}'
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-u', '-m',
                           'experiments.offline_camera_retrain_v1.evaluate_sampling',
                           '--checkpoint', str((args.output / 'last.pt').resolve()),
                           '--output', str((evaluation_dir / 'metrics.json').resolve()),
                           '--gallery', str((evaluation_dir / 'gallery.html').resolve()),
                           '--length', '128', '--ddim-steps', '20', '--samples-per-group', '4']
                if args.ddim_head_only:
                    command.append('--head-only')
                with (evaluation_dir / 'evaluation.log').open('w') as evaluation_log:
                    result = subprocess.run(command, stdout=evaluation_log,
                                            stderr=subprocess.STDOUT, check=False)
                row = dict(step=step, sampling_exit_code=result.returncode,
                           evaluation_dir=str(evaluation_dir))
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + '\n')
                if result.returncode:
                    raise RuntimeError(f'Sampling failed; inspect {evaluation_dir}/evaluation.log')
                if step % args.keep_inference_every == 0 or step == args.steps:
                    torch.save(dict(step=step, model=model.state_dict(), config=config),
                               evaluation_dir / 'inference.pt')
                write_run_index(args.output)
    log.close()


if __name__ == '__main__':
    main()
