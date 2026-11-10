"""AMASS EXO-only adapter training, exact recovery and full-val selection.

Use ``python -m scenemi.exo_runtime run -- ... -m scenemi.exo_training``.
The sealed EGO checkpoint is a required immutable dependency of compact best
and last checkpoints. Validation is synchronous: no candidate can be skipped
because an asynchronous watcher was busy. Test samples are never opened.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_corruption import corrupt_exo
from scenemi.evaluation import summarize, verify_training_seal
from scenemi.exo_model import ExoAdaptedPrior
from scenemi.models import AmassMotionPrior
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import validation_manifest, pin_checkpoint, save_jsonl
from scenemi.training import (deterministic, ensure_bounded, now, tensor_digest,
                              rng_state, restore_rng, optimizer_for, contract)
from scenemi.losses import motion_loss
from scenemi.metrics import motion_metrics
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

CONFIG = ROOT/'configs/amass_exo128.yaml'
INDEX = ROOT/'runs/amass128/index/complete.json'


def settings(path):
    raw = yaml.safe_load(Path(path).read_text())
    reference = yaml.safe_load((ROOT/raw['reference_config']).read_text())
    if (raw['version'] != 'amass-exo-frozen-prior-v1' or raw['validation']['split'] != 'val'
            or raw['validation']['test_used'] is not False or raw['base']['step'] != 55000
            or raw['runtime']['allocator_gib'] != 5.5 or reference['scene']['enabled']):
        raise ValueError('Unexpected EXO adaptation protocol')
    if raw['sampling'] != dict(family='uniform_with_replacement',
            reflection='uniform_original_or_reflected', window='uniform_all_valid_starts',
            batch_distinct_families=False, unseen_priority=False):
        raise ValueError('Sampling differs from the registered historical distribution')
    if raw['validation']['conditions'] != ['clean_masked', 'noisy_gap']:
        raise ValueError('Both predeclared EXO conditions must be fully evaluated')
    effective = copy.deepcopy(reference)
    effective['training'].update(raw['training'])
    effective['optimizer'] = raw['optimizer']
    effective['observation']['mode_probabilities'] = dict(ego=0., exo=1.)
    effective['observation']['input_noise'] = raw['corruption']
    effective['loss']['coefficients'].update(raw['loss_overrides'])
    return raw, reference, effective


def make_model(config, reference, paths, device):
    checkpoint = ROOT/config['base']['checkpoint']
    if sha256(checkpoint) != config['base']['sha256']:
        raise ValueError('Fixed EGO-only 55k initializer changed')
    seal = json.loads((checkpoint.parent/'CONTRACT.json').read_text())
    verify_initialization_seal(seal, reference, paths)
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if state['step'] != config['base']['step'] or state['contract'] != seal:
        raise ValueError('Initializer step/contract differs')
    base = AmassMotionPrior(paths['smplh_models']/'neutral/model.npz')
    base.load_state_dict(state['model'], strict=True)
    del state
    return ExoAdaptedPrior(base, config['adapter']['hidden']).to(device)


def cli_bridge(seal):
    """Prove the only historical CLI change is a new unrelated dispatch.

    The EXO exporter previously added three lines to the stable CLI while the
    old EGO checkpoint sealed that file. All model/data/loss files still match.
    This entry never imports that CLI. Accept exactly this addition, record both
    identities, and reject any further edit instead of weakening source checks.
    """
    name = 'scenemi/__main__.py'
    path = ROOT/name
    original = seal['sources'][name]
    current = sha256(path)
    addition = ("    if len(sys.argv) > 1 and sys.argv[1] == 'exo-camera':\n"
                "        from .data.exo_export import main as exo_main\n"
                "        return exo_main(sys.argv[2:])\n")
    text = path.read_text()
    reconstructed = hashlib.sha256(text.replace(addition, '', 1).encode()).hexdigest()
    if current != original and (text.count(addition) != 1 or reconstructed != original):
        raise ValueError('Unexplained change in historical CLI source')
    return dict(path=name, original_sha256=original, current_sha256=current,
                original_reconstructed_sha256=reconstructed,
                change='only unrelated exo-camera CLI dispatch; not imported by EXO trainer')


def verify_initialization_seal(seal, reference, paths):
    original = {key:value for key,value in seal.items() if key != 'sha256'}
    if hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() != seal['sha256']:
        raise ValueError('Historical training contract is corrupt')
    bridge = cli_bridge(seal)
    checked = copy.deepcopy(seal)
    checked['sources'][bridge['path']] = bridge['current_sha256']
    checked.pop('sha256')
    checked['sha256'] = hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest()
    verify_training_seal(checked, reference, paths, INDEX)


def seal_protocol(config, paths, cases, config_path):
    seal = contract(config, paths, INDEX)
    for path in (Path(config_path).resolve(), ROOT/'scenemi/exo_runtime.py'):
        seal['sources'][str(path.relative_to(ROOT))] = sha256(path)
    old = json.loads((ROOT/config['base']['checkpoint']).with_name('CONTRACT.json').read_text())
    seal['sources'].update(old['sources'])
    bridge = cli_bridge(old)
    seal['sources'][bridge['path']] = bridge['current_sha256']
    seal.update(protocol='amass-exo-frozen-prior-v1', base=config['base'],
                validation_cases=cases, test_used=False, ground_truth_shape_input=False,
                initializer_cli_compatibility=bridge, initializer_contract_sha256=old['sha256'],
                sharing_authorization_sha256=sha256(ROOT/config['output']/'SHARING_AUTHORIZATION.json'))
    seal.pop('sha256')
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    return seal


def verify_sources(seal):
    for name, digest in seal['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Sealed source changed: '+name)
    if sha256(INDEX) != seal['index_sha256'] or sha256(ROOT/seal['base']['checkpoint']) != seal['base']['sha256']:
        raise ValueError('Required immutable AMASS/base identity changed')
    if sha256(ROOT/seal['config']['output']/'SHARING_AUTHORIZATION.json') != seal['sharing_authorization_sha256']:
        raise ValueError('Stage sharing authorization changed')


def example(data, rng, config, effective, *, case=None, condition=None, mode='exo'):
    dropout = copy.deepcopy(effective['observation']['temporal_dropout'])
    if mode == 'ego' or condition == 'clean_masked':
        dropout = None
    elif condition == 'noisy_gap':
        dropout['probabilities']['exo'] = 1.
    options = dict(temporal_dropout=dropout)
    if case is not None:
        options.update(member=case['member'], start=case['start'], reflected=False, yaw=0.)
    sample, identity = data.sample(rng, mode, **options)
    if case and identity['variant_sha256'] != case['variant_sha256']:
        raise ValueError('Validation source changed')
    if mode == 'exo':
        if condition == 'clean_masked':
            recipe = dict(kind='clean_masked', local_sigma_m=0., root_drift_sigma_m=0.)
        else:
            recipe = corrupt_exo(sample, rng, config['corruption'], force=condition == 'noisy_gap')
        identity['corruption'] = recipe
        # Re-anchor from AVAILABLE noisy tracks, including after prefix outages.
        # Targets follow the same observable transform; hidden joints do not
        # define this coordinate frame. Vertical floor coordinates are retained.
        present = sample['observation_meta'][..., 0] > 0
        frame = int(np.flatnonzero(present.any(-1))[0])
        origin = sample['trajectory'][frame, present[frame], :3].mean(0)*2
        origin[1] = 0
        sample['trajectory'][..., :3] -= present[..., None]*origin/2
        sample['motion'][:, :3] -= origin/2
        sample['motion'][:, 135:] -= np.tile(origin/2, 22)
        identity['noisy_observed_anchor_shift_m'] = origin.tolist()
    return sample, identity


def training_batch(data, rng, config, effective, device):
    rows = [example(data, rng, config, effective) for _ in range(config['training']['batch_size'])]
    return collate([row[0] for row in rows], device), [row[1] for row in rows]


def update(model, optimizer, data, rng, config, effective, device, step):
    batch, identities = training_batch(data, rng, config, effective, device)
    target = batch['motion']
    timesteps = torch.randint(0, 1000, (len(target),), device=device)
    alpha = model.alphas[timesteps]
    xt = alpha[:, None, None].sqrt()*target+(1-alpha[:, None, None]).sqrt()*torch.randn_like(target)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, details = model(xt, timesteps, inference_payload(batch), return_details=True)
    terms = motion_loss(prediction, target, batch, alpha.sqrt(), details=details, config=effective)
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite objective; update refused')
    terms['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                config['optimizer']['clip_norm'], error_if_nonfinite=True)
    shape_gradient = sum(float(p.grad.square().sum()) for p in model.exo_shape_encoder.parameters() if p.grad is not None)**.5
    projections = [float(p.weight.grad.norm()) if p.weight.grad is not None else None for p in model.adapter.projections]
    fraction = step/config['training']['planned_updates']
    schedule = min(1., step/config['optimizer']['warmup_updates'])*max(
        config['optimizer']['lr_floor_fraction'], .5*(1+math.cos(math.pi*fraction)))
    for group in optimizer.param_groups:
        group['lr'] = config['optimizer']['lr']*schedule
    optimizer.step()
    return dict(step=step, identities=identities, terms={k:float(v.detach()) for k,v in terms.items()},
        grad_norm=float(norm), shape_grad_norm=shape_gradient, projection_gradients=projections,
        lr=optimizer.param_groups[0]['lr'], timesteps=timesteps.tolist())


def save(model, optimizer, rng, device, step, seal, path, **extra):
    save_checkpoint(dict(protocol='amass-exo-compact-resumable-v1', adapter=model.adapter_state(),
        optimizer=optimizer.state_dict(), rng=rng_state(rng, device), device=device, step=step,
        contract=seal, required_base=seal['base'], **extra), path)


def restore(model, optimizer, rng, device, seal, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != seal or state['device'] != device:
        raise ValueError('Checkpoint code/config/device identity differs')
    model.load_adapter_state(state['adapter'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng, device)
    return state


@torch.inference_mode()
def ego_sentinel(model, validation, cases, config, effective, device):
    chosen = cases[::max(1, len(cases)//8)][:8]
    rows = [example(validation, np.random.default_rng(777+c['case_id']), config,
                    effective, case=c, mode='ego')[0] for c in chosen]
    batch = collate(rows, device)
    payload = inference_payload(batch)
    noisy = torch.zeros_like(batch['motion'])
    times = torch.full((len(rows),), 600, device=device, dtype=torch.long)
    before, bd = model.base(noisy, times, payload, return_details=True)
    after, ad = model(noisy, times, payload, return_details=True)
    exact = torch.equal(before, after) and torch.equal(bd['predicted_betas'], ad['predicted_betas'])
    if not exact or any(p.requires_grad or p.grad is not None for p in model.base.parameters()):
        raise AssertionError('Frozen EGO preservation check failed')
    return dict(exact_original_ego=True, cases=len(chosen), source='val_only',
                max_output_difference=float((before-after).abs().max()))


def masked_error(error, mask):
    return float(error[mask].mean()*1000) if mask.any() else None


@torch.inference_mode()
def evaluate(model, validation, cases, config, effective, device, out, step, mode='exo'):
    out.mkdir(parents=True, exist_ok=True)
    conditions = config['validation']['conditions'] if mode == 'exo' else ['ego_camera']
    policy = config['validation']
    all_rows, aggregates = [], {}
    saved = rng_state(np.random.default_rng(), device)
    was_training = model.training
    started = time.monotonic()
    model.eval()
    # Fixed batch assignment is part of the sampling protocol. EGO uses the
    # old selector's batch=4, so its deterministic DDIM draw grouping matches.
    size = policy['batch_size'] if mode == 'exo' else 4
    try:
        for condition in conditions:
            rows = []
            with (out/f'{condition}.jsonl').open('w') as stream:
                for offset in range(0, len(cases), size):
                    chunk = cases[offset:offset+size]
                    pairs = [example(validation, np.random.default_rng(policy['seed']+c['case_id']),
                        config, effective, case=c, condition=condition, mode=mode) for c in chunk]
                    batch = collate([p[0] for p in pairs], device)
                    payload = inference_payload(batch)
                    prediction = sample_native30(model, payload, steps=policy['ddim_steps'],
                                                 seed=policy['seed'], use_scene=False)
                    _, details = model(prediction, torch.zeros(len(chunk), device=device, dtype=torch.long),
                                       payload, return_details=True)
                    joints = forward_kinematics(prediction, details['rest'])
                    truth = forward_kinematics(batch['motion'], batch['target_rest'])
                    for i, case in enumerate(chunk):
                        metrics = motion_metrics(joints[i], truth[i])
                        error = (joints[i]-truth[i]).norm(dim=-1)
                        observed = batch['observation_meta'][i, ..., 0] > 0
                        diagnostic = dict(visible_mpjpe_mm=masked_error(error, observed),
                            hidden_mpjpe_mm=masked_error(error, ~observed),
                            observed_frames=int(observed.any(-1).sum()),
                            shape_beta_mse=float((details['predicted_betas'][i]-batch['target_betas'][i]).square().mean()))
                        row = dict(case_id=case['case_id'], condition=condition, identity=pairs[i][1],
                                   metrics=metrics, diagnostics=diagnostic)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        rows.append(row)
                    stream.flush()
                    if offset == 0 or len(rows)//64 != offset//64 or len(rows) == len(cases):
                        status = dict(state='evaluating', mode=mode, step=step, condition=condition,
                            completed=len(rows), total=len(cases), conditions=conditions,
                            elapsed_seconds=time.monotonic()-started, updated_at_utc=now(), pid=os.getpid())
                        atomic_json(out.parent/'STATUS.json', status)
                        print(json.dumps(status), flush=True)
            aggregates[condition] = summarize(rows)
            aggregates[condition]['per_corpus'] = {name:summarize([r for r in rows if r['identity']['corpus']==name])
                for name in sorted({r['identity']['corpus'] for r in rows})}
            all_rows.extend(rows)
    finally:
        model.train(was_training)
        restore_rng(saved, np.random.default_rng(), device)
    result = dict(state='complete', step=step, mode=mode, conditions=aggregates, test_used=False,
        selection_score_mm=float(np.mean([a['family_macro_mean']['mpjpe_mm'] for a in aggregates.values()])),
        validation_windows_per_condition=len(cases), validation_families=len(validation.members),
        total_scored_windows=len(all_rows), elapsed_seconds=time.monotonic()-started, finished_at_utc=now())
    atomic_json(out/'RESULT.json', result)
    return result


def preflight(model, data, validation, cases, config, effective, device, out, seal):
    folder = out/'preflight'
    folder.mkdir(parents=True, exist_ok=True)
    file = folder/f'{device}.json'
    result = dict(state='running', device=device, contract_sha256=seal['sha256'], started_at_utc=now())
    atomic_json(file, result)
    optimizer = optimizer_for(model, effective)
    rng = np.random.default_rng(config['seed']+991)
    original = tensor_digest(model.base.state_dict())
    temporary = folder/f'.{device}.pt'
    started = time.monotonic()
    try:
        ego_sentinel(model, validation, cases, config, effective, device)
        update_started = time.monotonic()
        first = update(model, optimizer, data, rng, config, effective, device, 1)
        if device == 'cuda':
            torch.cuda.synchronize()
        first_update_seconds = time.monotonic()-update_started
        if first['shape_grad_norm'] <= 0 or not any(g and g > 0 for g in first['projection_gradients']):
            raise AssertionError('Shape/internal EXO adapters have no gradient')
        save(model, optimizer, rng, device, 1, seal, temporary)
        expected = update(model, optimizer, data, rng, config, effective, device, 2)
        digest = tensor_digest((model.adapter_state(), optimizer.state_dict(), rng_state(rng, device)))
        restore(model, optimizer, rng, device, seal, temporary)
        actual = update(model, optimizer, data, rng, config, effective, device, 2)
        if expected != actual or digest != tensor_digest((model.adapter_state(), optimizer.state_dict(), rng_state(rng, device))):
            raise AssertionError('EXO next update differs after exact recovery')
        if tensor_digest(model.base.state_dict()) != original:
            raise AssertionError('Frozen EGO parameters/buffers changed')
        preservation = ego_sentinel(model, validation, cases, config, effective, device)
        batch, _ = training_batch(data, rng, config, effective, device)
        with torch.no_grad():
            model.eval()
            payload = inference_payload(batch)
            noisy, timesteps = torch.zeros_like(batch['motion']), torch.full((len(batch['motion']),), 500, device=device, dtype=torch.long)
            expected, details = model(noisy, timesteps, payload, return_details=True)
            poisoned = dict(payload, motion=torch.full_like(batch['motion'], float('nan')),
                target_betas=torch.full_like(batch['target_betas'], float('nan')),
                target_rest=torch.full_like(batch['target_rest'], float('nan')),
                rest=torch.full_like(batch['target_rest'], float('nan')))
            poisoned['trajectory'] = payload['trajectory'].clone()
            absent = payload['observation_meta'][..., 0] == 0
            poisoned['trajectory'][absent] = float('nan')
            if not torch.equal(expected, model(noisy, timesteps, poisoned)):
                raise AssertionError('Hidden joints or private labels leaked into inference')
            torch.testing.assert_close(forward_kinematics(expected, details['rest']),
                expected[..., 135:].reshape(len(expected), 128, 22, 3)*2, atol=2e-5, rtol=0)
        metrics = evaluate(model, validation, cases[:8], config, effective, device,
                           folder/f'{device}_ddim', 2)
        verify_sources(seal)
        result.update(state='passed', finished_at_utc=now(), resume_bitwise_exact=True,
            frozen_ego_unchanged=True, private_and_missing_values_isolated=True,
            actual_batch_size=config['training']['batch_size'], preservation=preservation,
            first_update=first, ddim_checks=metrics, elapsed_seconds=time.monotonic()-started,
            first_update_seconds=first_update_seconds,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        atomic_json(file, result)
        return result
    except BaseException as error:
        result.update(state='failed', error=repr(error), updated_at_utc=now())
        atomic_json(file, result)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def train(model, data, validation, cases, config, effective, device, out, seal, resume=False):
    if device != 'cuda':
        raise ValueError('Formal training requires validated CUDA')
    for name in ('cpu', 'cuda'):
        passed = json.loads((out/'preflight'/f'{name}.json').read_text())
        if passed['state'] != 'passed' or passed['contract_sha256'] != seal['sha256']:
            raise ValueError('Missing matching CPU/CUDA preflight: '+name)
    folder, selection = out/'training', out/'selection'
    folder.mkdir(exist_ok=True)
    selection.mkdir(exist_ok=True)
    lock = (folder/'train.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last = folder/'last.pt'
    if last.exists() != resume:
        raise ValueError('Existing last checkpoint requires --resume')
    optimizer = optimizer_for(model, effective)
    rng = np.random.default_rng(config['seed']+991)
    start, log_bytes = 0, 0
    if resume:
        state = restore(model, optimizer, rng, device, seal, last)
        start, log_bytes = state['step'], state['log_bytes']
        with (folder/'updates.jsonl').open('r+b') as stream:
            if stream.seek(0, 2) < log_bytes:
                raise ValueError('Committed training log is missing')
            stream.truncate(log_bytes)
    elif (folder/'updates.jsonl').exists() and (folder/'updates.jsonl').stat().st_size:
        raise ValueError('Refuse to overwrite an uncommitted training history')
    atomic_json(folder/'CONTRACT.json', seal)
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.append(True))
    status = dict(state='training', pid=os.getpid(), started_at_utc=now(), completed_updates=start,
        planned_updates=config['training']['planned_updates'], base=seal['base'],
        training_mode='exo_only_adapters_frozen_ego', contract_sha256=seal['sha256'],
        selection=str(selection/'STATUS.json'), test_used=False)
    if resume:
        with (folder/'RESUMES.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=start, pid=os.getpid(), resumed_at_utc=now()))+'\n')
    else:
        atomic_json(folder/'INITIALIZATION.json', dict(base=seal['base'], optimizer='fresh',
            inference_shape_input=False, sampling=config['sampling'], trainable_parameters=
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
    started, step = time.monotonic(), start
    def report(phase, **extra):
        status.update(state=phase, completed_updates=step, updated_at_utc=now(),
                      elapsed_seconds=time.monotonic()-started, **extra)
        atomic_json(folder/'STATUS.json', status)
    def select():
        candidate = selection/f'step_{step:06d}'
        result_file = candidate/'RESULT.json'
        report('full_exo_validation')
        if result_file.exists() and 'checkpoint_sha256' in json.loads(result_file.read_text()):
            result = json.loads(result_file.read_text())
            if result['contract_sha256'] != seal['sha256'] or result['checkpoint_sha256'] != sha256(last):
                raise ValueError('Stored selection candidate differs from current checkpoint')
        else:
            result = evaluate(model, validation, cases, config, effective, device, candidate, step)
            result.update(checkpoint_sha256=sha256(last), contract_sha256=seal['sha256'])
            atomic_json(result_file, result)
        best_file = selection/'BEST.json'
        best = json.loads(best_file.read_text()) if best_file.exists() else None
        if best is None or result['selection_score_mm'] < best['selection_score_mm']:
            temporary = selection/'.best.pt'
            pin_checkpoint(last, temporary)
            temporary.replace(selection/'best.pt')
            best = dict(step=step, checkpoint_sha256=result['checkpoint_sha256'],
                selection_score_mm=result['selection_score_mm'], metrics=str(result_file),
                checkpoint=str(selection/'best.pt'), full_validation=True, test_used=False,
                contract_sha256=seal['sha256'], updated_at_utc=now())
            atomic_json(best_file, best)
        atomic_json(selection/'STATUS.json', dict(state='complete_candidate', last_evaluated_step=step,
                                                best=best, updated_at_utc=now()))
        report('training', best=best)
    try:
        with (folder/'updates.jsonl').open('a', buffering=1) as log:
            def commit():
                log.flush()
                os.fsync(log.fileno())
                save(model, optimizer, rng, device, step, seal, last, log_bytes=log.tell())
            if not resume:
                commit()
            report('starting')
            # Original EGO model is retained byte-for-byte. Measure full EGO
            # baseline before optimization; repeat at final and every best
            # candidate receives structural + fixed-output preservation checks.
            initial = out/'ego_monitor/step_000000/RESULT.json'
            if not initial.exists():
                report('initial_ego_full_validation')
                evaluate(model, validation, cases, config, effective, device, initial.parent, 0, 'ego')
            if start and start % config['validation']['every_updates'] == 0:
                select()
            report('training')
            for step in range(start+1, config['training']['planned_updates']+1):
                record = update(model, optimizer, data, rng, config, effective, device, step)
                record['updated_at_utc'] = now()
                log.write(json.dumps(record, allow_nan=False)+'\n')
                if step % config['validation']['ego_sentinel_every_updates'] == 0:
                    record_ego = ego_sentinel(model, validation, cases, config, effective, device)
                    with (out/'ego_monitor/preservation.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(step=step, **record_ego, checked_at_utc=now()))+'\n')
                final = step == config['training']['planned_updates']
                if step % config['training']['checkpoint_every'] == 0 or final or stopping:
                    commit()
                if step % config['validation']['every_updates'] == 0 or final:
                    verify_sources(seal)
                    select()
                if step % config['training']['log_every'] == 0 or stopping or final:
                    report('training', last_terms=record['terms'], lr=record['lr'],
                        peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    print(json.dumps(status), flush=True)
                if stopping:
                    report('interrupted')
                    return status
            report('final_ego_full_validation')
            final_metrics = evaluate(model, validation, cases, config, effective, device,
                out/f'ego_monitor/step_{step:06d}', step, 'ego')
            baseline_metrics = json.loads(initial.read_text())
            if final_metrics['conditions'] != baseline_metrics['conditions']:
                raise AssertionError('Full EGO metrics changed despite frozen path')
            verify_sources(seal)
            report('complete', ego_full_metrics_unchanged=True)
            atomic_json(selection/'STATUS.json', dict(state='complete', best=status['best'],
                last_evaluated_step=step, updated_at_utc=now()))
            return status
    except BaseException as error:
        report('failed', error=repr(error), last_durable_checkpoint=str(last))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'train'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ensure_bounded()
    config, reference, effective = settings(args.config)
    if args.device == 'cuda':
        if float(os.environ['SCENEMI_ALLOCATOR_GIB']) != config['runtime']['allocator_gib']:
            raise ValueError('Runtime allocator budget differs from configuration')
        configure_cuda()
    deterministic(config['seed'], args.device)
    paths = load_paths(args.paths)
    out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    data = AmassDataset(INDEX, paths, 'train')
    validation = AmassDataset(INDEX, paths, 'val')
    cases = validation_manifest(validation.members)
    if len(cases) != 2222 or len(validation.members) != 337:
        raise ValueError('Full validation membership changed')
    model = make_model(config, reference, paths, args.device)
    seal = seal_protocol(config, paths, cases, args.config)
    protocol_path = out/'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != seal:
        raise ValueError('Run protocol already sealed differently; do not overwrite it')
    atomic_json(protocol_path, seal)
    manifest = out/'VALIDATION_CASES.jsonl'
    if manifest.exists():
        if [json.loads(line) for line in manifest.read_text().splitlines()] != cases:
            raise ValueError('Frozen full-validation manifest changed')
    else:
        save_jsonl(manifest, cases)
    verify_sources(seal)
    if args.action == 'preflight':
        result = preflight(model, data, validation, cases, config, effective, args.device, out, seal)
    else:
        result = train(model, data, validation, cases, config, effective, args.device, out, seal, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
