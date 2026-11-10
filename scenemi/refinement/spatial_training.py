"""Versioned multiscale Refiner training on a frozen selected motion prior.

Uses the established sampler, DDIM, FK/B, checkpoint/RNG, ordered prefetch,
metrics and terminal/TensorBoard pipeline. Historical trainer and checkpoint
formats remain intact. Preparation/preflight never launch formal training.
"""
import argparse
from collections import defaultdict
import copy
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import tarfile
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.checkpoint import checkpoint, load_checkpoint
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import SOURCE_ORDER
from scenemi.data.spatial_scene_data import IndexedGeometryEgoData, collate, payload
from scenemi.data.video_scene import load_policy
from scenemi.execution import configure_runtime
from scenemi.refinement.training import (load_config as reference_config, merge,
    fixed_cases, optimizer_for, restore_selection, verify_contract)
from scenemi.refinement.spatial_model import build_spatial_model
from scenemi.refinement.spatial_losses import objective, geometry_supervision
from scenemi.refinement.model import LOWER
from scenemi.training import deterministic, tensor_digest, rng_state, restore_rng, now
from scenemi.monitoring import TrainingReporter
from scenemi.prefetch import OrderedPrefetch
from scenemi.body_diagnostic_metrics import diagnostics
from scenemi.evaluate import aggregate
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

VERSION = 'ego-spatial-scene-refinement-v1'
STOP = False
SOURCES = ('scenemi/refinement/spatial_training.py', 'scenemi/refinement/spatial_losses.py',
    'scenemi/refinement/spatial_model.py', 'scenemi/geometry/spatial_scene.py',
    'scenemi/data/spatial_scene_data.py', 'scenemi/data/video_scene.py',
    'scenemi/geometry/spatial_execution.py', 'scenemi/data/numeric_mmap.py')


def load_config(file):
    file = Path(file).resolve(); changes = yaml.safe_load(file.read_text())
    if changes.get('version') != VERSION:
        raise ValueError('Expected the versioned spatial Refiner configuration')
    c = merge(reference_config(ROOT/changes['reference_config']), changes)
    c['_config_path'] = str(file)
    c.pop('refinement_loss')  # Historical reference coefficients are inactive.
    t, loss = c['training'], c['spatial_loss']
    if (t['effective_batch_size'] != 128 or c['data']['source_quota_per_update'] != 32
            or type(t['microbatch_size']) is not int or t['microbatch_size'] <= 0 or 128 % t['microbatch_size']
            or not 0 <= t['scene_dropout'] < 1):
        raise ValueError('Keep batch 128, quota 32/source and valid microbatch/dropout')
    for k in ('planned_updates', 'checkpoint_every', 'quick_validation_every'):
        if type(t[k]) is not int or t[k] <= 0:
            raise ValueError(f'Invalid training {k}')
    if (t['full_validation_steps'] != sorted(set(t['full_validation_steps']))
            or any(type(s) is not int or not 0 < s <= t['planned_updates'] for s in t['full_validation_steps'])):
        raise ValueError('Invalid full-validation selection schedule')
    for k, expected in dict(weights={'motion', 'contact', 'geometry', 'residual'},
            motion={'rotation', 'position', 'lower_velocity', 'head_position', 'head_orientation'},
            geometry={'penetration', 'support', 'sliding'},
            residual={'magnitude', 'first_difference', 'second_difference'}).items():
        if set(loss[k]) != expected or any(not math.isfinite(v) or v < 0 for v in loss[k].values()):
            raise ValueError(f'Invalid {k} loss coefficients')
    for k in ('position_scale_m', 'velocity_scale_mps', 'head_position_scale_m'):
        if not math.isfinite(loss[k]) or loss[k] <= 0:
            raise ValueError(f'Invalid loss scale {k}')
    cp = loss['contact']; near = load_policy(ROOT/c['spatial']['policy'])['query']['near_radius_m']
    if (not 0 < cp['positive_gap_m'] < cp['negative_gap_m'] < near
            or not 0 < cp['positive_speed_mps'] < cp['negative_speed_mps']
            or not 0 < cp['support_normal_y'] <= 1 or cp['positive_class_weight'] <= 0):
        raise ValueError('Invalid reliable contact-label bands')
    if (not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs')
            or (ROOT/c['output']).resolve() == (ROOT/'runs/ego_scene_refinement').resolve()):
        raise ValueError('Use a new run directory; preserve the historical Refiner')
    if (c['model']['scene_modules_trainable'] is not False or c['training']['precision'] != 'fp32'
            or c['data']['datasets'] != list(SOURCE_ORDER) or c['data']['excluded_datasets'] != ['rich']
            or c['validation']['test_used'] is not False or c['sampling']['ddim_steps'] != 20):
        raise ValueError('Retain the frozen prior, full DDIM20 and fixed four-corpus protocol')
    if (not 0 < c['runtime']['memory_gib'] <= 16 or not 1 <= c['runtime']['cpu_count'] <= 4
            or c['runtime'].get('allocator_gib') is not None):
        raise ValueError('Keep bounded CPU/RAM and user-managed GPU allocation')
    execution = c.get('spatial_execution', {})
    if execution.get('array_backend', 'numpy') not in ('numpy', 'numeric_header_v1'):
        raise ValueError('Unsupported numeric input backend')
    workers = execution.get('processes', 0)
    if type(workers) is not int or not 0 <= workers <= c['runtime']['cpu_count']:
        raise ValueError('Spatial processes must fit the CPU cgroup')
    if workers and (not (ROOT/execution['index_cache']).resolve().is_relative_to(ROOT/'runs')
            or str((ROOT/execution['index_cache']).resolve()).startswith('/mnt/')):
        raise ValueError('Use a local run directory for complete-map CPU caches')
    for key in ('lr', 'clip_norm'):
        if not math.isfinite(c['optimizer'][key]) or c['optimizer'][key] <= 0:
            raise ValueError(f'Invalid optimizer {key}')
    return c


def data_for(c, paths, split):
    return IndexedGeometryEgoData(c, paths, split,
        spatial_policy=load_policy(ROOT/c['spatial']['policy']),
        video_manifest=ROOT/c['spatial'][split+'_maps'],
        index_cache=ROOT/c['spatial_execution']['index_cache'] if c.get('spatial_execution', {}).get('processes') else None)


def model_for(c, paths):
    model = build_spatial_model(c, paths, load_policy(ROOT/c['spatial']['policy']), ROOT/c['spatial']['prior'])
    if c.get('spatial_execution', {}).get('processes'):
        from scenemi.geometry.spatial_execution import SpatialQueryExecutor
        model.query_executor = SpatialQueryExecutor(c['spatial_execution']['processes'], ROOT/c['spatial_execution']['index_cache'],
            array_backend=c['spatial_execution'].get('array_backend', 'numpy'))
    return model


def contract(c):
    # Explicit inherited import closure, plus all new entry points/configs.
    # The inherited source seal is verified rather than silently repinned.
    policy = load_policy(ROOT/c['spatial']['policy'])
    inherited = torch.load(ROOT/policy['audit']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)['contract']
    verify_contract(inherited)
    seal = copy.deepcopy(inherited)
    files = [Path(c['_config_path']), ROOT/c['reference_config']]
    files += [ROOT/c['spatial'][key] for key in ('policy', 'prior', 'train_maps', 'validation_maps')]
    files += [ROOT/file for file in SOURCES]
    seal['source_sha256'].update({str(p.relative_to(ROOT)): sha256(p) for p in files})
    prior = yaml.safe_load((ROOT/c['spatial']['prior']).read_text())
    for key in ('checkpoint', 'selection'):
        if sha256(ROOT/prior[key]) != prior[key+'_sha256']:
            raise ValueError(f'Selected frozen prior {key} changed')
        seal['asset_sha256'][prior[key]] = prior[key+'_sha256']
    for split in ('train', 'validation'):
        manifest = json.loads((ROOT/c['spatial'][split+'_maps']).read_text())
        if manifest['state'] != 'complete' or manifest['split'] != split or manifest['test_used']:
            raise ValueError('Training/validation require separate completed observed-video maps')
        for row in manifest['maps']:
            seal['asset_sha256'][row['path']] = row['sha256']
    seal['asset_sha256'].update(seal['skin_basis']['identity']['source_sha256'])
    if c.get('execution_continuation'):
        parent = c['execution_continuation']; file = ROOT/parent['manifest']
        if sha256(file) != parent['manifest_sha256']:
            raise ValueError('Execution predecessor manifest changed')
        manifest = json.loads(file.read_text())
        if manifest['version'] != 'spatial-execution-continuation-v1':
            raise ValueError('Unsupported execution predecessor')
        seal['asset_sha256'].update(manifest['files'])
        seal['asset_sha256'][parent['manifest']] = parent['manifest_sha256']
        seal['execution_continuation'] = dict(parent, step=manifest['step'],
            semantics='same optimizer, RNG, data, losses and absolute update budget; numeric I/O execution change only')
    seal.update(protocol=VERSION, config=c, model_kind='spatial_scene_pose_refiner_only',
        frozen_prior=prior, frozen_base_sha256=prior['checkpoint_sha256'],
        additional_updates=c['training']['planned_updates'], additional_windows=c['training']['planned_updates']*128,
        zero_scene_reference='selected Phase B motion+shape prior + temporal B',
        input_scene='complete original-video same-observer voxel index; current rigid-object observations only',
        geometry_semantics='measured local patches + bounded observed support candidates; unknown is not free space',
        loss_groups=list(c['spatial_loss']['weights']), rich_used=False, test_used=False)
    seal.pop('sha256', None)
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    return seal


def update(model, optimizer, pairs, c, device, step):
    model.train();optimizer.zero_grad(set_to_none=True)
    opt = c['optimizer']; fraction = min(1., step/c['training']['planned_updates'])
    factor = min(1., step/opt['warmup_updates'])*(opt['lr_floor_fraction']+
        (1-opt['lr_floor_fraction'])*.5*(1+math.cos(math.pi*fraction)))
    for group in optimizer.param_groups:
        group['lr'] = opt['lr']*factor
    sums, stats = defaultdict(float), defaultdict(float)
    timings = defaultdict(float); n = len(pairs)
    for first in range(0, n, c['training']['microbatch_size']):
        chunk = pairs[first:first+c['training']['microbatch_size']]
        batch = collate([p[0] for p in chunk], device, pin=c['prepared_data']['pin_memory']); public = payload(batch)
        mark = time.monotonic()
        raw, details = model.coarse(public, steps=c['sampling']['ddim_steps'], seed=c['seed']+step*c['sampling']['training_seed_offset']+first)
        timings['coarse_seconds'] += time.monotonic()-mark; mark = time.monotonic()
        keep = torch.rand(len(chunk), device=device) >= c['training']['scene_dropout']
        prediction, details = model.refine(raw, details, public, use_scene=keep)
        timings['refine_seconds'] += time.monotonic()-mark; mark = time.monotonic()
        terms, metrics = objective(model, prediction, details, batch, c['spatial_loss'])
        if not all(torch.isfinite(v).all() for v in terms.values()):
            raise FloatingPointError('Nonfinite loss; optimizer not advanced')
        (terms['total']*(len(chunk)/n)).backward()
        timings['loss_backward_seconds'] += time.monotonic()-mark
        for key, value in terms.items(): sums[key] += float(value.detach())*len(chunk)/n
        for key, value in metrics.items(): stats[key] += float(value.detach().float().mean())*len(chunk)/n
    gradients = {name: float(p.grad.norm()) for name, p in model.refiner.named_parameters()
        if p.grad is not None and name in ('rotation.weight', 'point_encoder.0.weight', 'contact.2.weight', 'geometry.0.weight')}
    norm = torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), opt['clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(step=step, total=sums['total'], terms=dict(sums), statistics=dict(stats),
        gradient_norm=float(norm), gradient_groups=gradients, lr=optimizer.param_groups[0]['lr'],
        identities=[p[1] for p in pairs], coarse_ddim_steps=c['sampling']['ddim_steps'], **timings)


@torch.inference_mode()
def evaluate(model, data, cases, c, device, output):
    from tqdm.auto import tqdm
    model.eval(); refined_rows, base_rows = [], []
    for first in tqdm(range(0, len(cases), c['validation']['eval_batch_size']), desc='Spatial Refiner validation', mininterval=2):
        if STOP: raise InterruptedError('Validation stopped; partial metrics cannot select a checkpoint')
        chunk = cases[first:first+c['validation']['eval_batch_size']]
        pairs = [data.sample_case(case) for case in chunk]
        batch = collate([p[0] for p in pairs], device); public = payload(batch)
        raw, details = model.coarse(public, steps=c['sampling']['ddim_steps'], seed=c['validation']['seed'])
        baseline = model.locate(raw, details); refined, extra = model.refine(raw, details, public)
        target = forward_kinematics(batch['motion'], batch['target_rest'])
        standard = model.base.shape.rest_template[None].expand(len(chunk), -1, -1)
        for motion, rows in ((refined, refined_rows), (baseline, base_rows)):
            positions = forward_kinematics(motion, details['rest'])
            metrics = diagnostics(positions, motion, details['rest'], target, batch['motion'], batch['target_rest'], standard)
            _, geometry, _ = geometry_supervision(model, motion, dict(extra, scene_active=torch.ones(len(chunk), device=device)), batch, labels=False)
            pr, gt = positions[:, :, LOWER]-positions[:, :, :1], target[:, :, LOWER]-target[:, :, :1]
            amplitude = lambda x: (x-x.mean(1, keepdim=True)).square().sum(-1).mean((1, 2)).sqrt()*1000
            pa, ga = amplitude(pr), amplitude(gt)
            for i, item in enumerate(metrics):
                values = {k: v for k, v in item['metrics'].items() if 'support_proxy' not in k}
                values.update({k: float(v[i]) for k, v in geometry.items()})
                values.update(lower_pred_amplitude_mm=float(pa[i]), lower_gt_amplitude_mm=float(ga[i]),
                    lower_amplitude_ratio=float(pa[i]/ga[i].clamp_min(1e-6)),
                    near_hit_fraction=float(extra['anchor_near_hit'][i].float().mean()),
                    scene_evidence_fraction=float(extra['anchor_hit'][i].float().mean()),
                    ground_valid_fraction=float(extra['ground_valid'][i].float().mean()))
                rows.append(dict(case=chunk[i], identity=pairs[i][1], metrics=values))
    result, baseline = aggregate(refined_rows), aggregate(base_rows)
    m, b = result['dataset_macro_mean'], baseline['dataset_macro_mean']
    result.update(state='complete', baseline=baseline, test_used=False, rich_used=False,
        completed_at_utc=now(), coarse_ddim_steps=c['sampling']['ddim_steps'],
        guards=dict(head=m['head_error_mm'] <= b['head_error_mm']+c['validation']['head_regression_tolerance_mm'],
            jerk=m['jerk_m_s3'] <= b['jerk_m_s3']*c['validation']['jerk_ratio_max']))
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('metrics', refined_rows), ('baseline_metrics', base_rows)):
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    atomic_json(output/'RESULT.json', result)
    return result


def validate(model, data, cases, c, device, output, rng):
    state = rng_state(rng, device)
    try: return evaluate(model, data, cases, c, device, output)
    finally: restore_rng(state, rng, device)


def preflight(c, paths, device):
    configure_runtime(c, device);deterministic(c['seed'], device)
    print(f'{device} acceptance: loading frozen Phase B and real train/validation data', flush=True)
    model = model_for(c, paths).to(device)
    try:
        return _preflight(c, paths, device, model)
    finally:
        model.close_queries()


def _preflight(c, paths, device, model):
    data = data_for(c, paths, 'train'); heldout = data_for(c, paths, 'validation')
    cases = fixed_cases(heldout, c);seal = contract(c);verify_contract(seal)
    rng = np.random.default_rng(c['seed']);before = tensor_digest(model.base.state_dict())
    before_tensors = {k: tensor_digest(v) for k, v in model.base.state_dict().items()}
    # Four CPU examples can all lose scene evidence under the actual dropout
    # stream. Exercise a broader real batch without altering training masks.
    sample = lambda: data.pairs(rng) if device == 'cuda' else [data.sample(s, rng) for s in SOURCE_ORDER for _ in range(4)]
    mark = time.monotonic();pairs = sample();first_data_seconds = time.monotonic()-mark
    print(f'{device} acceptance: first real batch prepared in {first_data_seconds:.2f}s', flush=True)
    examples = [next(p for p in pairs if p[1]['dataset'] == source) for source in SOURCE_ORDER]
    probe = collate([p[0] for p in examples], device); public = payload(probe)
    raw, details = model.coarse(public, steps=c['sampling']['ddim_steps'], seed=c['seed']); baseline = model.locate(raw, details)
    with torch.no_grad(): initial, _ = model.refine(raw, details, public)
    if not torch.equal(initial, baseline): raise AssertionError('Zero initialization changed the prior+B result')
    optimizer = optimizer_for(model, c); folder = ROOT/c['output']/'preflight'/device;folder.mkdir(parents=True, exist_ok=True)
    print(f'{device} acceptance: {len(pairs)} examples, full DDIM20, first optimizer update', flush=True)
    initial_state = copy.deepcopy(model.refiner.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict());initial_rng = rng_state(rng, device)
    started = time.monotonic(); first = update(model, optimizer, pairs, c, device, 1)
    first_seconds = time.monotonic()-started
    print(f'{device} first update: {first_seconds:.2f}s, loss {first["total"]:.6f}', flush=True)
    execution_comparison = None
    if model.query_executor is not None:
        expected_state = tensor_digest(model.refiner.state_dict())
        expected_opt = tensor_digest(optimizer.state_dict())
        model.refiner.load_state_dict(initial_state);optimizer.load_state_dict(initial_optimizer)
        restore_rng(initial_rng, rng, device)
        executor, model.query_executor = model.query_executor, None
        try:
            print(f'{device} acceptance: comparing the same optimizer update against serial execution', flush=True)
            mark = time.monotonic();serial = update(model, optimizer, pairs, c, device, 1)
            serial_seconds = time.monotonic()-mark
        finally:
            model.query_executor = executor
        if (serial['terms'] != first['terms'] or serial['gradient_groups'] != first['gradient_groups']
                or tensor_digest(model.refiner.state_dict()) != expected_state
                or tensor_digest(optimizer.state_dict()) != expected_opt):
            raise AssertionError('Process and serial losses, gradients or optimizer updates differ')
        execution_comparison = dict(serial_seconds=serial_seconds, parallel_seconds=first_seconds,
            loss_gradient_weights_optimizer_bitwise_exact=True)
        print(f'{device} execution migration passed: serial {serial_seconds:.2f}s / parallel {first_seconds:.2f}s', flush=True)
    del initial_state, initial_optimizer, initial_rng
    checkpoint(model.refiner, optimizer, rng, device, 1, seal, folder/'resume.pt')
    mark = time.monotonic();second_pairs = sample();second_data_seconds = time.monotonic()-mark
    started = time.monotonic()
    second = update(model, optimizer, second_pairs, c, device, 2)
    second_seconds = time.monotonic()-started
    print(f'{device} acceptance: replaying update 2 from checkpoint', flush=True)
    expected, expected_optimizer = tensor_digest(model.refiner.state_dict()), tensor_digest(optimizer.state_dict())
    if load_checkpoint(model.refiner, optimizer, rng, device, seal, folder/'resume.pt') != 1:
        raise AssertionError('Resume step differs')
    replay_pairs = sample();replay = update(model, optimizer, replay_pairs, c, device, 2)
    if ([p[1] for p in second_pairs] != [p[1] for p in replay_pairs]
            or tensor_digest(model.refiner.state_dict()) != expected or replay['terms'] != second['terms']
            or tensor_digest(optimizer.state_dict()) != expected_optimizer):
        raise AssertionError('Optimizer, RNG, sample order or update replay is not bitwise exact')
    after_tensors = {k: tensor_digest(v) for k, v in model.base.state_dict().items()}
    changed = [k for k in before_tensors if before_tensors[k] != after_tensors[k]]
    gradients = [k for k, p in model.base.named_parameters() if p.grad is not None]
    digest_after = tensor_digest(model.base.state_dict())
    atomic_json(folder/'FROZEN_CHECK.json', dict(before=before, after=digest_after,
        tensor_hashes_before=before_tensors, tensor_hashes_after=after_tensors,
        changed_tensors=changed, gradient_parameters=gradients))
    if digest_after != before or changed or gradients:
        raise AssertionError(f'Frozen prior check failed: changed tensors {changed}, gradients {gradients}; see FROZEN_CHECK.json')
    with torch.no_grad(): fallback, _ = model.refine(raw, details, public, use_scene=False)
    if not torch.equal(fallback, baseline): raise AssertionError('Scene-off result changed after actual optimizer steps')
    if first['gradient_groups'].get('rotation.weight', 0) <= 0 or second['gradient_groups'].get('point_encoder.0.weight', 0) <= 0:
        raise AssertionError('Motion/geometry objectives did not reach the rotation head and point encoder')
    quick = [next(case for case in cases if case['dataset'] == s) for s in SOURCE_ORDER]
    print(f'{device} acceptance: exact resume passed; four-source validation smoke check', flush=True)
    mark = time.monotonic();smoke = evaluate(model, heldout, quick, c, device, folder/'evaluation')
    validation_seconds = time.monotonic()-mark;verify_contract(seal)
    result = dict(state='passed', device=device, contract=seal, diagnostic_only=True,
        formal_training_started=False, real_batch_size=len(pairs), configured_batch_size=128,
        source_quotas={s: sum(p[1]['dataset'] == s for p in pairs) for s in SOURCE_ORDER},
        zero_initialization_exact=True, no_scene_fallback_exact=True, frozen_motion_and_shape_unchanged=True,
        resume_bitwise_exact=True, next_sample_order_exact=True, optimizer_state_exact=True,
        validation_windows=len(cases), first_update=first, second_update=second, smoke=smoke,
        seconds_per_update=second_seconds, first_update_seconds=first_seconds, smoke_validation_seconds=validation_seconds,
        first_data_prepare_seconds=first_data_seconds, second_data_prepare_seconds=second_data_seconds,
        execution_comparison=execution_comparison,
        trainable_parameters=sum(p.numel() for p in model.refiner.parameters()), completed_at_utc=now())
    atomic_json(folder/'RESULT.json', result);(folder/'resume.pt').unlink()
    return result


def scientific_record(record):
    return {k: v for k, v in record.items() if k not in ('at', 'seconds') and not k.endswith('_seconds')}


def execution_config_matches(old, new):
    """Only provenance, numeric I/O, prefetch placement and save cadence differ."""
    old, new = copy.deepcopy(old), copy.deepcopy(new)
    for value in (old, new):
        for key in ('output', '_config_path', 'execution_continuation'): value.pop(key, None)
        value['prepared_data'].pop('prefetch_batches')
        value['spatial_execution'].pop('array_backend', None)
        value['training'].pop('checkpoint_every')
    return old == new


def migrate_execution(c, paths, device):
    """Accept an execution-only continuation; preserve its complete predecessor."""
    configure_runtime(c, device);deterministic(c['seed'], device)
    if (device != 'cuda' or c['prepared_data']['prefetch_batches'] != 0
            or c['spatial_execution'].get('array_backend') != 'numeric_header_v1'):
        raise ValueError('This migration requires synchronous numeric I/O on CUDA')
    out = ROOT/c['output'];out.mkdir(parents=True, exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'training/last.pt').exists(): raise ValueError('Migration was already published; use train --resume')
        manifest = json.loads((ROOT/c['execution_continuation']['manifest']).read_text())
        source = ROOT/manifest['source'];seal = contract(c);verify_contract(seal)
        for checked in ('cpu', device):
            acceptance = json.loads((out/f'preflight/{checked}/RESULT.json').read_text())
            if acceptance['state'] != 'passed' or acceptance['contract'] != seal:
                raise ValueError('Fresh CPU/CUDA preflights are required for execution migration')
        parent = torch.load(source/'training/last.pt', map_location='cpu', weights_only=False)
        parent_payload = tensor_digest({k: parent[k] for k in ('model', 'optimizer', 'rng', 'step', 'device')})
        old = parent['contract'];start = parent['step']
        if (start != manifest['step'] or parent['device'] != device
                or json.loads((source/'CONTRACT.json').read_text()) != old
                or not execution_config_matches(old['config'], c)):
            raise ValueError('Execution migration changed the checkpoint or scientific configuration')
        # Verify the predecessor against its archived sources, without repinning
        # the historical source contract to the new implementation.
        with tarfile.open(source/'SOURCE_SNAPSHOT.tar.gz', 'r:gz') as archive:
            for file, expected in old['source_sha256'].items():
                name = file if not Path(file).is_absolute() else 'external/'+file.lstrip('/')
                if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                    raise ValueError('Predecessor source archive differs: '+file)
        changed = {f for f, h in old['source_sha256'].items() if sha256(ROOT/f) != h}
        if not changed.issubset({'scenemi/refinement/spatial_training.py', 'scenemi/geometry/spatial_execution.py'}):
            raise ValueError('Migration changed unrelated historical dependencies: '+str(changed))
        legacy = [json.loads(s) for s in (source/'training/updates.jsonl').read_text().splitlines()]
        reference = next(r for r in legacy if r['step'] == start+1)
        model = model_for(c, paths).to(device);data = data_for(c, paths, 'train')
        optimizer = optimizer_for(model, c);rng = np.random.default_rng(c['seed'])
        work = out/'migration';work.mkdir(exist_ok=True)
        atomic_json(work/'STATUS.json', dict(state='checking_actual_continuation', step=start, at=now()))
        def restore():
            model.refiner.load_state_dict(parent['model'], strict=True)
            optimizer.load_state_dict(copy.deepcopy(parent['optimizer']));restore_rng(parent['rng'], rng, device)
        def digests():
            return dict(model=tensor_digest(model.refiner.state_dict()), optimizer=tensor_digest(optimizer.state_dict()),
                rng=tensor_digest(rng_state(rng, device)))
        try:
            restore();frozen = tensor_digest(model.base.state_dict())
            print(f'Migration: comparing actual update {start+1} with its original recorded update', flush=True)
            with OrderedPrefetch(data.pairs, rng, 0) as batches:
                first = update(model, optimizer, batches.next()[0], c, device, start+1)
            if scientific_record(first) != scientific_record(reference):
                atomic_json(work/'MISMATCH.json', dict(expected=scientific_record(reference), actual=scientific_record(first)))
                raise AssertionError('Actual continuation differs from the original samples/losses/gradients')
            expected_first = dict(step=start+1, record=scientific_record(first), digests=digests())
            temporary = work/'roundtrip.pt'
            checkpoint(model.refiner, optimizer, rng, device, start+1, seal, temporary)
            second = update(model, optimizer, data.pairs(rng), c, device, start+2);expected_second = digests()
            load_checkpoint(model.refiner, optimizer, rng, device, seal, temporary)
            replay = update(model, optimizer, data.pairs(rng), c, device, start+2)
            if scientific_record(replay) != scientific_record(second) or digests() != expected_second:
                raise AssertionError('Execution continuation is not exactly resumable')
            if frozen != tensor_digest(model.base.state_dict()): raise AssertionError('Migration modified the frozen prior')
            if parent_payload != tensor_digest({k: parent[k] for k in ('model', 'optimizer', 'rng', 'step', 'device')}):
                raise AssertionError('Migration mutated its source optimizer/model/RNG state')
            temporary.unlink()
            atomic_json(work/'FIRST_UPDATE_EXPECTED.json', expected_first)
            (out/'training').mkdir(exist_ok=True);(out/'selection').mkdir(exist_ok=True)
            # Reuse completed, numerically identical validations. Never publish
            # partial evaluations or rerun an already completed full comparison.
            reused = []
            for folder in sorted((source/'selection').glob('step_*')):
                if int(folder.name.split('_')[-1]) <= start and (folder/'RESULT.json').exists():
                    shutil.copytree(folder, out/'selection'/folder.name, dirs_exist_ok=True);reused.append(folder.name)
            best = torch.load(source/'selection/best.pt', map_location='cpu', weights_only=False)
            best['contract'] = seal
            torch.save(best, out/'selection/best.pt.tmp');(out/'selection/best.pt.tmp').replace(out/'selection/best.pt')
            shutil.copy2(source/'selection/SELECTION.json', out/'selection/SELECTION.json')
            (out/'training/updates.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in legacy if r['step'] <= start))
            parent['contract'] = seal
            torch.save(parent, out/'training/last.pt.tmp')
            proof = dict(state='passed', resumed_step=start, original_update_samples_losses_gradients_exact=True,
                model_optimizer_rng_resume_bitwise_exact=True, frozen_prior_unchanged=True,
                benchmark_updates_discarded=True, reused_completed_validations=reused,
                source_checkpoint_sha256=sha256(source/'training/last.pt'), contract_sha256=seal['sha256'], at=now())
            atomic_json(work/'MIGRATION.json', proof)
            (out/'training/last.pt.tmp').replace(out/'training/last.pt')
            atomic_json(out/'STATUS.json', dict(state='ready_to_resume', step=start, at=now()))
            return proof
        finally:
            model.close_queries()


def train(c, paths, device, resume=False):
    configure_runtime(c, device);deterministic(c['seed'], device)
    out = ROOT/c['output'];out.mkdir(parents=True, exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = model_for(c, paths).to(device); data = data_for(c, paths, 'train');heldout = data_for(c, paths, 'validation')
        cases = fixed_cases(heldout, c);seal = contract(c);verify_contract(seal)
        seal_file = out/'CONTRACT.json'
        if seal_file.exists() and json.loads(seal_file.read_text()) != seal:
            raise ValueError('Existing source/configuration/data contract differs')
        for checked in dict.fromkeys(('cpu', device)):
            result = json.loads((out/'preflight'/checked/'RESULT.json').read_text())
            if result['state'] != 'passed' or result['contract'] != seal:
                raise RuntimeError(f'Fresh {checked} acceptance is required before manual training')
        optimizer = optimizer_for(model, c); rng = np.random.default_rng(c['seed'])
        last = out/'training/last.pt';last.parent.mkdir(exist_ok=True)
        if last.exists() != resume: raise ValueError('Use --resume exactly when training/last.pt exists')
        step = load_checkpoint(model.refiner, optimizer, rng, device, seal, last) if resume else 0
        expected_continuation = None
        if c.get('execution_continuation'):
            proof = json.loads((out/'migration/MIGRATION.json').read_text())
            if proof['state'] != 'passed' or proof['contract_sha256'] != seal['sha256']:
                raise ValueError('Accepted execution migration is required')
            expected_continuation = json.loads((out/'migration/FIRST_UPDATE_EXPECTED.json').read_text())
        atomic_json(seal_file, seal);atomic_json(out/'CONFIG.json', c)
        if not (out/'SOURCE_SNAPSHOT.tar.gz').exists():
            temporary = out/'SOURCE_SNAPSHOT.tar.gz.tmp'
            with tarfile.open(temporary, 'w:gz') as archive:
                for file in seal['source_sha256']:
                    path = ROOT/file
                    archive.add(path, arcname=file if not Path(file).is_absolute() else 'external/'+str(path).lstrip('/'))
            temporary.replace(out/'SOURCE_SNAPSHOT.tar.gz')
        from scenemi.train import trim_update_log
        log_file = out/'training/updates.jsonl';trim_update_log(log_file, step)
        reporter = TrainingReporter(out/'training', c, step);reporter.progress.set_description('Frozen Phase B + spatial Refiner')
        selection_file = out/'selection/SELECTION.json';selection_file.parent.mkdir(exist_ok=True)
        best = restore_selection(out, step)
        atomic_json(out/'STATUS.json', dict(state='running', pid=os.getpid(), step=step, at=now()))
        def publish(metric):
            nonlocal best
            reporter.validation(metric, step, 'full')
            if all(metric['guards'].values()) and (best is None or metric['selection_score_mm'] < best['selection_score_mm']):
                checkpoint(model.refiner, optimizer, rng, device, step, seal, out/'selection/best.pt')
                best = dict(step=step, selection_score_mm=metric['selection_score_mm'], guards=metric['guards'])
                atomic_json(selection_file, best)
        try:
            if best is None:
                if step != 0: raise ValueError('Initial full validation is missing')
                checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
                publish(validate(model, heldout, cases, c, device, out/'selection/step_000000', rng))
            pending = out/f'selection/step_{step:06d}/RESULT.json'
            if step > 0 and (step in c['training']['full_validation_steps'] or step == c['training']['planned_updates']) and not pending.exists():
                publish(validate(model, heldout, cases, c, device, pending.parent, rng))
            with log_file.open('a', buffering=1) as log, OrderedPrefetch(data.pairs, rng, c['prepared_data']['prefetch_batches']) as batches:
                while step < c['training']['planned_updates'] and not STOP:
                    started = time.monotonic();pairs, timing = batches.next();next_step = step+1
                    record = update(model, optimizer, pairs, c, device, next_step);step = next_step
                    if expected_continuation is not None and step == expected_continuation['step']:
                        actual = dict(model=tensor_digest(model.refiner.state_dict()), optimizer=tensor_digest(optimizer.state_dict()),
                            rng=tensor_digest(rng_state(rng, device)))
                        if scientific_record(record) != expected_continuation['record'] or actual != expected_continuation['digests']:
                            raise AssertionError('Independent first resumed update differs from migration acceptance')
                        atomic_json(out/'migration/CROSS_PROCESS_CONTINUATION.json', dict(state='passed', step=step,
                            sample_loss_gradient_model_optimizer_rng_exact=True, at=now()))
                    seconds = time.monotonic()-started;record.update(timing, seconds=seconds, at=now())
                    log.write(json.dumps(record, allow_nan=False)+'\n');reporter.update(record, seconds)
                    if reporter.writer is not None:
                        for key, value in record['statistics'].items(): reporter.writer.add_scalar('train/statistics/'+key, value, step)
                    full = step in c['training']['full_validation_steps'] or step == c['training']['planned_updates']
                    if step % c['training']['checkpoint_every'] == 0 or full:
                        checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
                    if full:
                        publish(validate(model, heldout, cases, c, device, out/f'selection/step_{step:06d}', rng))
                    elif step % c['training']['quick_validation_every'] == 0:
                        quick = [case for s in SOURCE_ORDER for case in [v for v in cases if v['dataset'] == s][:c['validation']['quick_windows_per_source']]]
                        metric = validate(model, heldout, quick, c, device, out/f'diagnostics/step_{step:06d}', rng)
                        reporter.validation(metric, step, 'quick_diagnostic')
                    atomic_json(out/'STATUS.json', dict(state='running', pid=os.getpid(), step=step,
                        planned_updates=c['training']['planned_updates'], latest_loss=record['total'], seconds_per_update=seconds, best=best, at=now()))
            checkpoint(model.refiner, optimizer, rng, device, step, seal, last)
            state = 'stopped_at_safe_boundary' if STOP else 'complete'
        except InterruptedError:
            checkpoint(model.refiner, optimizer, rng, device, step, seal, last);state = 'stopped_at_safe_boundary'
        except BaseException as error:
            atomic_json(out/'STATUS.json', dict(state='failed', step=step, error=f'{type(error).__name__}: {error}', at=now()))
            raise
        finally:
            reporter.close()
            model.close_queries()
        result = dict(state=state, step=step, best=best, at=now());atomic_json(out/'STATUS.json', result)
        return result


def run_action(args, c, paths, parser):
    if args.action == 'preflight':
        folder = ROOT/c['output']/'preflight'/args.device;folder.mkdir(parents=True, exist_ok=True)
        with (folder/'RUN.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(folder/'STATUS.json', dict(state='running', pid=os.getpid(), at=now()))
            try:
                result = preflight(c, paths, args.device)
                atomic_json(folder/'STATUS.json', dict(state='passed', at=now()))
            except BaseException as error:
                atomic_json(folder/'STATUS.json', dict(state='failed', error=f'{type(error).__name__}: {error}', at=now()))
                raise
    elif args.action == 'train': result = train(c, paths, args.device, args.resume)
    elif args.action == 'migrate': result = migrate_execution(c, paths, args.device)
    else:
        if args.checkpoint is None or args.output is None: parser.error('evaluate needs --checkpoint and --output')
        if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'runs'):
            parser.error('Evaluation needs a new output directory under runs')
        configure_runtime(c, args.device)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if state['contract'] != contract(c) or state['contract']['model_kind'] != 'spatial_scene_pose_refiner_only':
            raise ValueError('Checkpoint does not match the spatial Refiner contract')
        verify_contract(state['contract']);model = model_for(c, paths).to(args.device)
        model.refiner.load_state_dict(state['model'], strict=True)
        data = data_for(c, paths, 'validation')
        try:
            result = evaluate(model, data, fixed_cases(data, c), c, args.device, args.output)
        finally:
            model.close_queries()
        result.update(checkpoint_step=state['step'], checkpoint_sha256=sha256(args.checkpoint))
        atomic_json(args.output/'RESULT.json', result)
    print(json.dumps(dict(state=result['state'], output=c['output']), indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train', 'evaluate', 'migrate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop);signal.signal(signal.SIGINT, stop)
    c, paths = load_config(args.config), load_paths(args.paths)
    if args.resume and args.action != 'train': parser.error('--resume applies only to train')
    if c.get('spatial_execution', {}).get('array_backend') == 'numeric_header_v1':
        from scenemi.data.numeric_mmap import numeric_mmap_backend
        out = ROOT/c['output'];out.mkdir(parents=True, exist_ok=True)
        backend = numeric_mmap_backend(out/'CURRENT_ARRAY_READ.json')
    else:
        backend = nullcontext()
    with backend:
        run_action(args, c, paths, parser)


if __name__ == '__main__': main()
