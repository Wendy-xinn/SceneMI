"""Shared CLI implementation for post-sampling EGO scene refinement.

Reuses prepared source data, checkpoint/RNG helpers, bounded execution,
prefetching, metrics and TensorBoard. Formal training is a manual command.
Checkpoints contain the small refiner, optimizer and RNG, and pin the frozen
base and skin assets by SHA256 instead of duplicating their weights.
"""
import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tarfile
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.checkpoint import checkpoint, load_checkpoint, load_trained_model, continuation_identity
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import SOURCE_ORDER, source_cases
from scenemi.data.scene_geometry_data import GeometryEgoData, collate, payload
from scenemi.scene_geometry_training import skin_bases, contract as geometry_contract
from scenemi.scene_geometry_training import load_config as geometry_config
from scenemi.training import deterministic, tensor_digest, now, rng_state, restore_rng
from scenemi.execution import configure_runtime
from scenemi.monitoring import TrainingReporter
from scenemi.prefetch import OrderedPrefetch
from scenemi.evaluate import aggregate
from scenemi.body_diagnostic_metrics import diagnostics
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.refinement.model import SceneRefinementPipeline, LOWER
from scenemi.refinement.losses import objective
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

VERSION = 'ego-scene-refinement-v1'
STOP = False


def merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_config(path):
    path = Path(path).resolve()
    changes = yaml.safe_load(path.read_text())
    if changes.get('version') != VERSION:
        raise ValueError('Expected a scene-refinement configuration')
    c = merge(copy.deepcopy(geometry_config(ROOT/changes['reference_config'])), changes)
    c['_config_path'] = str(path)
    if c['model']['scene_modules_trainable'] is not False:
        raise ValueError('The base scene modules must stay frozen; only the new refiner trains')
    if c['training']['effective_batch_size'] != 128 or c['data']['source_quota_per_update'] != 32:
        raise ValueError('Keep the agreed batch 128 and quota 32/source')
    t, r = c['training'], c['refiner']
    for key in ('microbatch_size', 'planned_updates', 'checkpoint_every', 'quick_validation_every'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f'Invalid training {key}')
    if 128 % t['microbatch_size'] or not 0 <= t['scene_dropout'] < 1:
        raise ValueError('Invalid microbatch or scene dropout')
    if (type(r['width']) is not int or type(r['heads']) is not int or r['width'] < 8
            or r['heads'] < 1 or r['width'] % r['heads']
            or type(r['blocks']) is not int or not 1 <= r['blocks'] <= 6):
        raise ValueError('Invalid spatiotemporal refiner dimensions')
    for key in ('root_yaw_limit_deg', 'torso_limit_deg', 'lower_limit_deg'):
        if not math.isfinite(r[key]) or not 0 < r[key] <= 30:
            raise ValueError(f'Invalid angular bound: {key}')
    if not 2 <= c['sampling']['ddim_steps'] <= 1000:
        raise ValueError('Invalid coarse DDIM step count')
    for key in ('lr', 'clip_norm'):
        if not math.isfinite(c['optimizer'][key]) or c['optimizer'][key] <= 0:
            raise ValueError(f'Invalid optimizer {key}')
    if (not math.isfinite(c['optimizer']['weight_decay']) or c['optimizer']['weight_decay'] < 0
            or type(c['optimizer']['warmup_updates']) is not int or c['optimizer']['warmup_updates'] < 1
            or not 0 < c['optimizer']['lr_floor_fraction'] <= 1):
        raise ValueError('Invalid optimizer warmup/decay policy')
    expected = {'pose_rotation','fk_position','lower_fk','lower_velocity','head_position',
        'head_orientation','contact','residual','scene_nonpenetration','scene_support','scene_sliding'}
    weights = c['refinement_loss']['weights']
    if set(weights) != expected or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Refinement loss weights must be explicit, finite and nonnegative')
    contact = c['refinement_loss']['contact']
    if (not 0 < contact['positive_gap_m'] < contact['negative_gap_m'] < c['geometry']['query_radius_m']
            or not 0 < contact['positive_speed_mps'] < contact['negative_speed_mps']
            or not 0 <= contact['support_normal_y'] <= 1 or contact['positive_class_weight'] <= 0):
        raise ValueError('Contact labels require separate positive/negative and uncertainty bands')
    for key in ('position_scale_m','velocity_scale_mps','head_position_scale_m'):
        if not math.isfinite(c['refinement_loss'][key]) or c['refinement_loss'][key] <= 0:
            raise ValueError(f'Invalid loss scale: {key}')
    if (not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs')
            or c['output'] == 'runs/ego_scene_geometry_22k'):
        raise ValueError('Use a new output directory inside runs')
    if not 0 < c['runtime']['memory_gib'] <= 16 or not 1 <= c['runtime']['cpu_count'] <= 4:
        raise ValueError('Keep bounded CPU/RAM execution')
    if c['runtime'].get('allocator_gib') is not None:
        raise ValueError('GPU allocation remains user-managed')
    if any(type(s) is not int or not 0 < s <= t['planned_updates'] for s in t['full_validation_steps']):
        raise ValueError('Invalid full validation schedule')
    return c


def build_model(c, paths):
    parent = continuation_identity(c)
    base, _, step = load_trained_model(ROOT/parent['checkpoint'], paths)
    if step != 22000:
        raise ValueError('This architecture pins the selected scene-free 22k base')
    base.patch_enabled = base.local_enabled = base.feedback_enabled = False
    return SceneRefinementPipeline(base, skin_bases(c), c['geometry'], c['anchor'], c['refiner'])


def contract(c, paths):
    value = geometry_contract(c, paths)
    value.update(protocol=VERSION, model_kind='scene_pose_refiner_only',
        additional_updates=c['training']['planned_updates'],
        additional_windows=c['training']['planned_updates']*128,
        frozen_base_sha256=c['continuation']['checkpoint_sha256'],
        coarse_training_inputs='complete frozen DDIM samples from public observations; no GT corruption shortcut',
        zero_scene_reference='22k + one temporal B readout', rich_used=False)
    value['source_sha256'][c['reference_config']] = sha256(ROOT/c['reference_config'])
    value.pop('sha256', None)
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def verify_contract(seal):
    for kind in ('source_sha256', 'asset_sha256'):
        for file, expected in seal[kind].items():
            if sha256(ROOT/file) != expected:
                raise ValueError(f'Sealed {kind} dependency changed: {file}')
    parent = seal['continuation']
    if sha256(ROOT/parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('Frozen base checkpoint changed')


def fixed_cases(data, c):
    expected = json.loads((ROOT/c['continuation']['validation_manifest']).read_text())['full']
    actual = source_cases(data)
    if actual != expected:
        raise ValueError('Validation membership differs from the fixed four-source holdout')
    return actual


def restore_selection(out, committed_step):
    """Recover a validation/best publication interrupted between atomic files.

    Only validations at or before last.pt are eligible. A missing selected
    weight can be recovered from last.pt only when its exact step matches.
    """
    selection = out/'selection'
    candidates = []
    for file in sorted(selection.glob('step_*/RESULT.json')):
        step = int(file.parent.name.rsplit('_', 1)[1])
        if step > committed_step:
            continue
        result = json.loads(file.read_text())
        if result.get('state') == 'complete' and all(result['guards'].values()):
            candidates.append(dict(step=step, selection_score_mm=result['selection_score_mm'], guards=result['guards']))
    if not candidates:
        return None
    best = min(candidates, key=lambda item:(item['selection_score_mm'], item['step']))
    destination = selection/'best.pt'
    saved_step = (torch.load(destination, map_location='cpu', weights_only=False, mmap=True)['step']
                  if destination.exists() else None)
    if saved_step != best['step']:
        last = out/'training/last.pt'
        if not last.exists() or torch.load(last, map_location='cpu', weights_only=False, mmap=True)['step'] != best['step']:
            raise ValueError('Selected refiner weights are unavailable; refusing a different selection')
        temporary = destination.with_name(destination.name+'.tmp')
        shutil.copyfile(last, temporary);temporary.replace(destination)
    atomic_json(selection/'SELECTION.json', best)
    return best


def optimizer_for(model, c):
    if any(p.requires_grad for p in model.base.parameters()) or any(p.requires_grad for p in model.skin.parameters()):
        raise AssertionError('Frozen coarse generator or skin has trainable parameters')
    return torch.optim.AdamW(model.refiner.parameters(), lr=c['optimizer']['lr'],
                              weight_decay=c['optimizer']['weight_decay'])


def update(model, optimizer, pairs, c, device, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    factor = min(1., step/c['optimizer']['warmup_updates'])
    fraction = min(1., step/c['training']['planned_updates'])
    factor *= c['optimizer']['lr_floor_fraction']+(1-c['optimizer']['lr_floor_fraction'])*.5*(1+math.cos(math.pi*fraction))
    for group in optimizer.param_groups:
        group['lr'] = c['optimizer']['lr']*factor
    sums, stats = defaultdict(float), defaultdict(float)
    n = len(pairs)
    for first in range(0, n, c['training']['microbatch_size']):
        chunk = pairs[first:first+c['training']['microbatch_size']]
        batch = collate([p[0] for p in chunk], device, pin=c['prepared_data']['pin_memory'])
        public = payload(batch)
        raw, details = model.coarse(public, steps=c['sampling']['ddim_steps'],
            seed=c['seed']+step*c['sampling']['training_seed_offset']+first)
        keep = torch.rand(len(chunk), device=device) >= c['training']['scene_dropout']
        prediction, details = model.refine(raw, details, public, use_scene=keep)
        terms, metrics = objective(model, prediction, details, batch, c['refinement_loss'])
        if not all(torch.isfinite(v).all() for v in terms.values()):
            raise FloatingPointError('Nonfinite refinement loss; optimizer not advanced')
        if terms['total'].requires_grad:
            (terms['total']*(len(chunk)/n)).backward()
        for key, value in terms.items():
            sums[key] += float(value.detach())*(len(chunk)/n)
        for key, value in metrics.items():
            stats[key] += float(value.detach().float().mean())*(len(chunk)/n)
    gradients = {name:float(parameter.grad.norm()) for name, parameter in model.refiner.named_parameters()
                 if parameter.grad is not None and name in ('rotation.weight','contact.2.weight','geometry.0.weight')}
    norm = torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), c['optimizer']['clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(step=step, total=sums['total'], terms=dict(sums), gradient_norm=float(norm),
        lr=optimizer.param_groups[0]['lr'], gradient_groups=gradients, statistics=dict(stats),
        identities=[p[1] for p in pairs], coarse_ddim_steps=c['sampling']['ddim_steps'])


@torch.inference_mode()
def evaluate(model, data, cases, c, device, output=None):
    from tqdm.auto import tqdm
    model.eval()
    refined_rows, base_rows = [], []
    for first in tqdm(range(0, len(cases), c['validation']['eval_batch_size']), desc='Scene refiner validation', mininterval=2):
        if STOP:
            raise InterruptedError('Validation stopped; no partial result is eligible for selection')
        chunk = cases[first:first+c['validation']['eval_batch_size']]
        pairs = [data.sample_case(case) for case in chunk]
        batch = collate([p[0] for p in pairs], device)
        public = payload(batch)
        raw, details = model.coarse(public, steps=c['sampling']['ddim_steps'], seed=c['validation']['seed'])
        baseline = model.locate(raw, details)
        refined, extra = model.refine(raw, details, public)
        target = forward_kinematics(batch['motion'], batch['target_rest'])
        standard = model.base.shape.rest_template[None].expand(len(chunk), -1, -1)
        for motion, rows in ((refined, refined_rows), (baseline, base_rows)):
            positions = forward_kinematics(motion, details['rest'])
            metrics = diagnostics(positions, motion, details['rest'], target, batch['motion'], batch['target_rest'], standard)
            _, geometry = geometry_terms(model, motion, details, batch, torch.ones(len(chunk), device=device), c['geometry'])
            pr, gt = positions[:, :, LOWER]-positions[:, :, :1], target[:, :, LOWER]-target[:, :, :1]
            amplitude = lambda x:(x-x.mean(1, keepdim=True)).square().sum(-1).mean((1, 2)).sqrt()*1000
            pa, ga = amplitude(pr), amplitude(gt)
            for i, item in enumerate(metrics):
                values = {k:v for k,v in item['metrics'].items() if 'support_proxy' not in k}
                values.update({k:float(v[i]) for k,v in geometry.items()})
                values.update(lower_pred_amplitude_mm=float(pa[i]), lower_gt_amplitude_mm=float(ga[i]),
                              lower_amplitude_ratio=float(pa[i]/ga[i].clamp_min(1e-6)))
                rows.append(dict(case=chunk[i], identity=pairs[i][1], metrics=values))
    result, base = aggregate(refined_rows), aggregate(base_rows)
    m, b = result['dataset_macro_mean'], base['dataset_macro_mean']
    result.update(state='complete', baseline=base, test_used=False, rich_used=False,
        completed_at_utc=now(), coarse_ddim_steps=c['sampling']['ddim_steps'],
        guards=dict(head=m['head_error_mm']<=b['head_error_mm']+c['validation']['head_regression_tolerance_mm'],
                    jerk=m['jerk_m_s3']<=b['jerk_m_s3']*c['validation']['jerk_ratio_max']))
    if output is not None:
        output = Path(output);output.mkdir(parents=True, exist_ok=True)
        for name, rows in (('metrics', refined_rows), ('baseline_metrics', base_rows)):
            (output/(name+'.jsonl')).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
        atomic_json(output/'RESULT.json', result)
    return result


def validation(model, data, cases, c, device, output, rng):
    state = rng_state(rng, device)
    try:
        return evaluate(model, data, cases, c, device, output)
    finally:
        restore_rng(state, rng, device)


def preflight(c, paths, device):
    configure_runtime(c, device);deterministic(c['seed'], device)
    model = build_model(c, paths).to(device)
    data = GeometryEgoData(c, paths, 'train')
    validation_data = GeometryEgoData(c, paths, 'validation')
    cases = fixed_cases(validation_data, c)
    seal = contract(c, paths)
    before = tensor_digest(model.base.state_dict())
    rng = np.random.default_rng(c['seed'])
    pairs = (data.pairs(rng) if device == 'cuda' else [data.sample(source, rng) for source in SOURCE_ORDER])
    examples = [next(p for p in pairs if p[1]['dataset']==source) for source in SOURCE_ORDER]
    probe = collate([p[0] for p in examples], device)
    raw, details = model.coarse(payload(probe), steps=c['sampling']['ddim_steps'], seed=c['seed'])
    with torch.no_grad():
        initial, _ = model.refine(raw, details, payload(probe))
        baseline = model.locate(raw, details)
    if not torch.equal(initial, baseline):
        raise AssertionError('Zero-initialized refiner changed the B-aligned base')
    optimizer = optimizer_for(model, c)
    folder = ROOT/c['output']/'preflight'/device;folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    first = update(model, optimizer, pairs, c, device, 1)
    checkpoint(model.refiner, optimizer, rng, device, 1, seal, folder/'resume.pt')
    sample_pairs = lambda:(data.pairs(rng) if device == 'cuda' else [data.sample(source, rng) for source in SOURCE_ORDER])
    second_pairs = sample_pairs()
    second = update(model, optimizer, second_pairs, c, device, 2)
    expected = tensor_digest(model.refiner.state_dict())
    expected_optimizer = tensor_digest(optimizer.state_dict())
    assert load_checkpoint(model.refiner, optimizer, rng, device, seal, folder/'resume.pt') == 1
    replay_pairs = sample_pairs()
    if [p[1] for p in second_pairs] != [p[1] for p in replay_pairs]:
        raise AssertionError('Checkpoint resume changed the next sampled windows')
    replay = update(model, optimizer, replay_pairs, c, device, 2)
    update_seconds = (time.monotonic()-started)/3
    if (tensor_digest(model.refiner.state_dict()) != expected or replay['terms'] != second['terms']
            or tensor_digest(optimizer.state_dict()) != expected_optimizer):
        raise AssertionError('Refiner optimizer/RNG resume is not exact')
    if tensor_digest(model.base.state_dict()) != before:
        raise AssertionError('Refinement changed the frozen base')
    with torch.no_grad():
        fallback, _ = model.refine(raw, details, payload(probe), use_scene=False)
        if not torch.equal(fallback, baseline):
            raise AssertionError('Trained refiner changed its no-scene fallback')
    quick = [next(case for case in cases if case['dataset']==source) for source in SOURCE_ORDER]
    smoke = evaluate(model, validation_data, quick, c, device, folder/'evaluation')
    if first['gradient_groups'].get('rotation.weight', 0.) <= 0:
        raise AssertionError('Actual scene/body loss did not reach the rotation head')
    verify_contract(seal)
    result = dict(state='passed', device=device, contract=seal, diagnostic_only=True,
        formal_training_started=False, real_batch_size=len(pairs), configured_batch_size=128,
        coarse_ddim_steps=c['sampling']['ddim_steps'], frozen_base_unchanged=True,
        zero_initialization_exact=True, no_scene_fallback_exact=True, resume_bitwise_exact=True,
        next_sample_order_exact=True, optimizer_state_exact=True,
        validation_windows=len(cases), first_update=first, smoke=smoke,
        trainable_parameters=sum(p.numel() for p in model.refiner.parameters()),
        seconds_per_update=update_seconds, completed_at_utc=now())
    atomic_json(folder/'RESULT.json', result)
    (folder/'resume.pt').unlink()
    return result


def train(c, paths, device, resume=False):
    global STOP
    configure_runtime(c, device);deterministic(c['seed'], device)
    out = ROOT/c['output'];out.mkdir(parents=True, exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = build_model(c, paths).to(device)
        data = GeometryEgoData(c, paths, 'train')
        heldout = GeometryEgoData(c, paths, 'validation');cases=fixed_cases(heldout, c)
        seal_file = out/'CONTRACT.json'
        seal = json.loads(seal_file.read_text()) if seal_file.exists() else contract(c, paths)
        if seal['config'] != c:
            raise ValueError('Existing experiment configuration differs')
        verify_contract(seal)
        for checked in dict.fromkeys(('cpu', device)):
            file = out/'preflight'/checked/'RESULT.json'
            result = json.loads(file.read_text())
            if result['state'] != 'passed' or result['contract'] != seal:
                raise RuntimeError(f'Fresh {checked} acceptance is required before manual training')
        optimizer = optimizer_for(model, c);rng=np.random.default_rng(c['seed'])
        last = out/'training/last.pt';last.parent.mkdir(exist_ok=True)
        if last.exists() and not resume:
            raise FileExistsError('Training checkpoint exists; use --resume')
        if resume and not last.exists():
            raise FileNotFoundError('No training checkpoint to resume')
        step = load_checkpoint(model.refiner, optimizer, rng, device, seal, last) if resume else 0
        if not seal_file.exists():
            atomic_json(seal_file, seal);atomic_json(out/'CONFIG.json', c)
        if not (out/'SOURCE_SNAPSHOT.tar.gz').exists():
            with tarfile.open(out/'SOURCE_SNAPSHOT.tar.gz', 'w:gz') as archive:
                for file in seal['source_sha256']:
                    path = ROOT/file
                    name = file if not Path(file).is_absolute() else 'external/'+str(path).lstrip('/')
                    archive.add(path, arcname=name)
        # Reuse the shared truncation routine so logs never extend past the
        # committed checkpoint following a crash.
        from scenemi.train import trim_update_log
        log_file = out/'training/updates.jsonl';trim_update_log(log_file, step)
        reporter = TrainingReporter(out/'training', c, step)
        reporter.progress.set_description('Frozen EGO + scene pose refiner')
        selection_file = out/'selection/SELECTION.json';selection_file.parent.mkdir(exist_ok=True)
        best = restore_selection(out, step)
        atomic_json(out/'STATUS.json', dict(state='running', pid=os.getpid(), step=step, at=now()))
        try:
            if best is None:
                if step != 0:
                    raise ValueError('Missing initial heldout validation; refusing selection from an advanced checkpoint')
                checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
                metric = validation(model, heldout, cases, c, device, out/'selection/step_000000', rng)
                checkpoint(model.refiner, optimizer, rng, device, step, seal, out/'selection/best.pt')
                best = dict(step=step, selection_score_mm=metric['selection_score_mm'], guards=metric['guards'])
                atomic_json(selection_file, best)
                checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            pending = out/f'selection/step_{step:06d}/RESULT.json'
            if step > 0 and (step in c['training']['full_validation_steps'] or step == c['training']['planned_updates']) and not pending.exists():
                # Last.pt is saved before validation. Resume that validation
                # before advancing, preserving checkpoint selection semantics.
                validation(model, heldout, cases, c, device, pending.parent, rng)
                best = restore_selection(out, step)
            with log_file.open('a', buffering=1) as log, OrderedPrefetch(data.pairs, rng, c['prepared_data']['prefetch_batches']) as batches:
                while step < c['training']['planned_updates'] and not STOP:
                    start = time.monotonic();pairs, timing=batches.next();step+=1
                    record = update(model, optimizer, pairs, c, device, step)
                    elapsed = time.monotonic()-start
                    record.update(timing, seconds=elapsed, at=now())
                    log.write(json.dumps(record, allow_nan=False)+'\n');reporter.update(record, elapsed)
                    if step % c['training']['checkpoint_every'] == 0 or step in c['training']['full_validation_steps']:
                        checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
                    if step in c['training']['full_validation_steps'] or step == c['training']['planned_updates']:
                        checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
                        metric = validation(model, heldout, cases, c, device, out/f'selection/step_{step:06d}', rng)
                        reporter.validation(metric, step, 'full')
                        if all(metric['guards'].values()) and metric['selection_score_mm'] < best['selection_score_mm']:
                            checkpoint(model.refiner, optimizer, rng, device, step, seal, out/'selection/best.pt')
                            best = dict(step=step, selection_score_mm=metric['selection_score_mm'], guards=metric['guards'])
                            atomic_json(selection_file, best)
                    elif step % c['training']['quick_validation_every'] == 0:
                        quick = [case for source in SOURCE_ORDER for case in [v for v in cases if v['dataset']==source][:c['validation']['quick_windows_per_source']]]
                        metric = validation(model, heldout, quick, c, device, out/f'diagnostics/step_{step:06d}', rng)
                        reporter.validation(metric, step, 'quick_diagnostic')
                    atomic_json(out/'STATUS.json', dict(state='running', step=step, planned_updates=c['training']['planned_updates'],
                        latest_loss=record['total'], seconds_per_update=elapsed, best=best, at=now()))
            checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            state = 'stopped_at_safe_boundary' if STOP else 'complete'
            atomic_json(out/'STATUS.json', dict(state=state, step=step, best=best, at=now()))
            return dict(state=state, step=step, best=best)
        except InterruptedError:
            checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            atomic_json(out/'STATUS.json', dict(state='stopped_at_safe_boundary', step=step, best=best, at=now()))
            return dict(state='stopped_at_safe_boundary', step=step, best=best)
        except BaseException as error:
            atomic_json(out/'STATUS.json', dict(state='failed', step=step, error=f'{type(error).__name__}: {error}', at=now()))
            raise
        finally:
            reporter.close()


def load_refined_model(checkpoint_path, paths):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    seal = state['contract']
    if seal.get('model_kind') != 'scene_pose_refiner_only':
        raise ValueError('Not a scene-refiner checkpoint')
    model = build_model(seal['config'], paths)
    model.refiner.load_state_dict(state['model'], strict=True)
    return model.eval(), seal['config'], state['step']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight','train','evaluate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop);signal.signal(signal.SIGINT, stop)
    if args.resume and args.action != 'train':
        parser.error('--resume applies only to train')
    c, paths = load_config(args.config), load_paths(args.paths)
    if args.action == 'preflight':
        result = preflight(c, paths, args.device)
    elif args.action == 'train':
        result = train(c, paths, args.device, args.resume)
    else:
        if args.checkpoint is None or args.output is None:
            parser.error('evaluate requires --checkpoint and --output')
        if args.output.exists():
            parser.error('Evaluation output exists; choose a new directory')
        configure_runtime(c, args.device)
        model, saved, step = load_refined_model(args.checkpoint, paths)
        if saved != c:
            raise ValueError('Evaluation configuration differs from the saved model')
        data = GeometryEgoData(c, paths, 'validation')
        result = evaluate(model.to(args.device), data, fixed_cases(data, c), c, args.device, args.output)
        result.update(checkpoint_step=step, checkpoint_sha256=sha256(args.checkpoint))
        atomic_json(args.output/'RESULT.json', result)
    print(json.dumps(dict(state=result['state'], output=c['output']), indent=2), flush=True)
    return result
