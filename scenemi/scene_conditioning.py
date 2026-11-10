"""Matched frozen-prior scene-conditioning experiments.

Commands: preflight --device cpu|cuda, or run [--resume]. All five variants use
the pinned no-scene 30k base, same train windows/RNG, batch, updates, losses and
evaluation seeds. Only compact adapters are saved. Final step is predeclared;
quick results do not select weights or determine which variants are run.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.local_scene import LocalSceneDataset, inference_payload
from scenemi.geometry.scene_features import FEATURE_NAMES
from scenemi import scene_repair as repair, scene_training as engine
from scenemi.metrics import motion_metrics
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_adapter import GeometryConditionedPrior, VARIANTS
from scenemi.scene_evaluation import check_parity
from scenemi.scene_model import native_motion_loss
from scenemi.scene_selection import aggregate_scene_rows
from scenemi.training import deterministic, ensure_bounded, now, restore_rng, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def settings(path):
    raw = yaml.safe_load(Path(path).read_text())
    reference = repair.settings(ROOT/raw['reference_config'])
    if (raw['version'] != 'observed-geometry-denoising-adapters-v1' or
            tuple(raw['variants']) != VARIANTS or raw['training']['planned_updates'] != 3000 or
            raw['training']['batch_size'] != 8 or raw['validation']['test_used'] is not False or
            raw['validation']['selection'] != 'fixed_final_step_no_checkpoint_search'):
        raise ValueError('Unexpected matched adapter experiment protocol')
    config = dict(reference)
    config.update(raw)
    return config, reference


def expanded_cases(data, original):
    cases = [dict(c) for c in original]
    present = {c['member'] for c in cases}
    for member, record in enumerate(data.records):
        if member in present:
            continue
        starts = record['starts']
        category = data.metadata(record)['category'] if record['dataset'] == 'grail' else record['dataset']
        cases.append(dict(member=member, start=int(starts[len(starts)//2]),
            dataset=record['dataset'], family=record['family'], category=category))
    if len({(c['member'],c['start']) for c in cases}) != len(cases):
        raise ValueError('Duplicate expanded validation window')
    if {c['member'] for c in cases} != set(range(len(data.records))):
        raise ValueError('Expanded validation misses eligible source records')
    return cases


def payload(batch, variant):
    values = inference_payload(batch)
    if variant == 'motion_only':
        # Legacy sampler needs an occupancy slot to scope its optional cache;
        # this constant contains no scene information and is not encoded.
        values = {k:v for k,v in values.items() if k not in (
            'occupancy','frame_scene_points','frame_scene_owner','local_history_points',
            'local_history_valid','local_frame_points','local_frame_owner')}
        values['occupancy'] = torch.zeros_like(batch['occupancy'])
        values['frame_scene_points'] = torch.zeros_like(batch['frame_scene_points'])
        values['frame_scene_owner'] = torch.full_like(batch['frame_scene_owner'], -1)
    return values


def make_model(reference, paths, scene_paths, config, variant, device):
    deterministic(config['seed'], device)
    base = repair.make_model(reference, paths, scene_paths)
    model = GeometryConditionedPrior(base, config['adapter'], variant)
    return model.to(device)


def protocol(config, reference, paths, scene_paths, cases, config_path):
    # Imported dependency closure plus explicit dynamic entry points. Preserve
    # the complete base source closure so old checkpoints remain reproducible.
    seal = engine.contract(config, scene_paths, paths, cases)
    base = repair.verify_base(reference, scene_paths)
    seal['sources'].update(base['sources'])
    for path in (Path(__file__), ROOT/'scenemi/adapter_runtime.py', config_path):
        seal['sources'][str(path.resolve().relative_to(ROOT))] = sha256(path)
    seal.update(base_sha256=reference['base']['sha256'], base_step=30000,
        base_validation_sha256=sha256(ROOT/reference['base']['comprehensive_reference']),
        adapter_feature_names=list(FEATURE_NAMES), selection='fixed_final_3000',
        test_used=False, target_shape_input=False)
    seal.pop('sha256')
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    return seal


def checkpoint(model, optimizer, rng, device, step, seal, variant, file):
    save_checkpoint(dict(adapter=model.adapter_state(), optimizer=optimizer.state_dict(),
        rng=rng_state(rng, device), step=step, contract=seal, variant=variant,
        base_sha256=seal['base_sha256']), file)


def restore(model, optimizer, rng, device, seal, variant, file):
    saved = torch.load(file, map_location='cpu', weights_only=False)
    if saved['contract'] != seal or saved['variant'] != variant or saved['base_sha256'] != seal['base_sha256']:
        raise ValueError('Adapter recovery identity changed')
    model.load_adapter_state(saved['adapter'])
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'], rng, device)
    return saved['step']


def update(model, optimizer, data, rng, config, device, step, forced=None):
    examples = [data.sample(rng, member=forced[i%len(forced)] if forced else None)
                for i in range(config['training']['batch_size'])]
    batch = collate([sample for sample,_ in examples], device)
    times = torch.randint(0, 1000, (len(examples),), device=device)
    alpha = model.alphas[times]
    noisy = alpha[:, None, None].sqrt()*batch['motion']+(1-alpha[:, None, None]).sqrt()*torch.randn_like(batch['motion'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, details = model(noisy, times, payload(batch, model.variant), return_details=True)
    terms = native_motion_loss(prediction, batch['motion'], batch, alpha.sqrt(), details=details, config=config)
    joints = forward_kinematics(prediction, details['rest'])
    baseline = forward_kinematics(details['baseline_motion'], details['rest']).detach()
    terms['preserve_head'] = (joints[:, :, 15]-baseline[:, :, 15]).square().mean()
    terms['preserve_root'] = (joints[:, :, 0]-baseline[:, :, 0]).square().mean()
    terms['total'] = (terms['total']+config['adapter']['head_consistency_weight']*terms['preserve_head']+
                      config['adapter']['root_consistency_weight']*terms['preserve_root'])
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite adapter objective')
    terms['total'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), config['optimizer']['clip_norm'], error_if_nonfinite=True)
    fraction = step/config['training']['planned_updates']
    lr = config['optimizer']['lr']*min(1., step/config['optimizer']['warmup_updates'])*(.1+.9*(1+math.cos(math.pi*fraction))/2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    gradients = [float(layer.weight.grad.norm()) if layer.weight.grad is not None else None
                 for layer in model.adapter.projections]
    optimizer.step()
    return dict(step=step, losses={k:float(v.detach()) for k,v in terms.items()},
        gradient_norm=float(norm), lr=lr, projection_gradients=gradients,
        geometry_diagnostics=details['geometry_diagnostics'].detach().cpu().tolist(),
        identities=[identity for _,identity in examples])


@torch.inference_mode()
def evaluate(model, data, cases, config, device, output, progress=None, initial_rows=None,
             *, use_scene=True):
    """Per-case journal permits evaluation recovery without rerunning training."""
    output.mkdir(parents=True, exist_ok=True)
    ledger = output/'cases.jsonl'
    records = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    if not records and initial_rows:
        records = [dict(row) for row in initial_rows]
        ledger.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
    for i,row in enumerate(records):
        c = cases[i]
        if (row['identity']['member'],row['identity']['start']) != (c['member'],c['start']):
            raise ValueError('Resumed evaluation window identities changed')
    saved = rng_state(np.random.default_rng(0), device)
    was_training = model.training
    model.eval()
    started = time.monotonic()
    try:
        with ledger.open('a') as stream:
            for case in cases[len(records):]:
                sample, identity = data.sample(np.random.default_rng(config['validation']['seed']),
                    member=case['member'], start=case['start'], yaw=0.)
                if identity['synthetic_chair'] or identity['dataset'] == 'rich':
                    raise ValueError('Invalid validation observation')
                identity['category'] = case['category']
                batch = collate([sample], device)
                inputs = payload(batch, model.variant)
                prediction = sample_native30(model, inputs, steps=config['validation']['ddim_steps'],
                    seed=config['validation']['seed'], use_scene=use_scene)
                # Native shape depends only on observations, not the scene.
                _, _, shape, _ = model.base.conditions(inputs, True, None)
                metrics = motion_metrics(forward_kinematics(prediction, shape['rest'])[0],
                                         forward_kinematics(batch['motion'], batch['target_rest'])[0])
                row = dict(identity=identity, metrics=metrics)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                records.append(row)
                if progress and (len(records)%10 == 0 or len(records) == len(cases)):
                    progress(len(records), len(cases), time.monotonic()-started)
    finally:
        model.train(was_training)
        restore_rng(saved, np.random.default_rng(0), device)
    result = aggregate_scene_rows(records)
    result.update(windows=len(records), test_used=False, elapsed_seconds=time.monotonic()-started,
                  completed_at_utc=now())
    atomic_json(output/'METRICS.json', result)
    return result


def preflight(model, data, validation, config, reference, device, out, seal):
    folder = out/'preflight'
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(state='running', device=device, contract_sha256=seal['sha256'], formal_training_started=False)
    atomic_json(folder/f'{device}.json', record)
    members = engine.forced_members(data)
    rng = np.random.default_rng(config['seed']+31)
    optimizer = engine.optimizer_for(model, config)
    original = tensor_digest(model.base.state_dict())
    temporary = folder/f'.{device}.pt'
    try:
        examples = [data.sample(rng, member=m, yaw=0.) for m in members]
        batch = collate([s for s,_ in examples], device)
        inputs = payload(batch, model.variant)
        times = torch.full((len(examples),), 100, device=device, dtype=torch.long)
        noisy = torch.randn_like(batch['motion'])
        with torch.no_grad():
            zero, details = model(noisy, times, inputs, return_details=True)
            plain = model(noisy, times, inputs, use_scene=False)
            independent = model.base(noisy, times, inputs, use_scene=False)
            if not torch.equal(plain, independent):
                raise AssertionError('Wrapper changed the independently trained base forward')
            if not torch.equal(zero, plain):
                raise AssertionError('Zero adapter does not exactly reproduce the original prior')
            model.train()
            if model.base.training or model.base.core.training:
                raise AssertionError('Frozen backbone enabled dropout')
            poisoned = dict(inputs, target_rest=torch.full_like(batch['target_rest'], float('nan')),
                target_native_betas=torch.full_like(batch['target_native_betas'], float('nan')),
                target_template=torch.full_like(batch['target_template'], 999),
                motion=torch.full_like(batch['motion'], float('nan')))
            if not torch.equal(zero, model(noisy, times, poisoned)):
                raise AssertionError('Private shape labels leaked into inference')
            # Every declared variant starts at the very same prior, including
            # the geometry-free control and the late-injection ablation.
            for variant in VARIANTS:
                model.variant = variant
                if not torch.equal(plain, model(noisy, times, payload(batch, variant))):
                    raise AssertionError('Variant initialization changed the prior: '+variant)
            model.variant = 'full'
        ddim_parity = None
        if device == 'cuda':
            reference_rows = json.loads((ROOT/reference['base']['comprehensive_reference']).read_text())['rows']
            cases = seal['validation_cases']['comprehensive']
            model.eval()
            with torch.no_grad():
                for dataset in ('egobody','trumans','grail'):
                    case = next(c for c in cases if c['dataset'] == dataset)
                    sample, identity = validation.sample(np.random.default_rng(config['validation']['seed']),
                        member=case['member'], start=case['start'], yaw=0.)
                    b = collate([sample], device)
                    ddim_inputs = payload(b, 'full')
                    prediction = sample_native30(model, ddim_inputs, steps=20, seed=config['validation']['seed'], use_scene=False)
                    _, _, shape, _ = model.base.conditions(ddim_inputs, True, None)
                    measured = motion_metrics(forward_kinematics(prediction,shape['rest'])[0],
                                               forward_kinematics(b['motion'],b['target_rest'])[0])
                    expected_row = next(r for r in reference_rows if r['identity']['member'] == case['member']
                                        and r['identity']['start'] == case['start'])
                    check_parity(measured, expected_row['metrics'])
            ddim_parity = True
        first = update(model, optimizer, data, rng, config, device, 1, members)
        if not any(g is not None and g > 0 for g in first['projection_gradients']):
            raise AssertionError('Internal residuals received no motion gradient')
        checkpoint(model, optimizer, rng, device, 1, seal, model.variant, temporary)
        expected = update(model, optimizer, data, rng, config, device, 2, members)
        digest = tensor_digest((model.adapter_state(), optimizer.state_dict(), rng_state(rng, device)))
        restore(model, optimizer, rng, device, seal, model.variant, temporary)
        actual = update(model, optimizer, data, rng, config, device, 2, members)
        if expected != actual or digest != tensor_digest((model.adapter_state(), optimizer.state_dict(), rng_state(rng, device))):
            raise AssertionError('Adapter resume is not bitwise exact')
        if original != tensor_digest(model.base.state_dict()) or any(p.grad is not None for p in model.base.parameters()):
            raise AssertionError('Frozen prior was updated')
        model.eval()
        with torch.no_grad():
            empty = dict(inputs, local_history_valid=torch.zeros_like(inputs['local_history_valid']),
                local_frame_owner=torch.full_like(inputs['local_frame_owner'], -1),
                local_history_points=torch.zeros_like(inputs['local_history_points']),
                local_frame_points=torch.zeros_like(inputs['local_frame_points']))
            if not torch.equal(plain, model(noisy, times, empty)):
                raise AssertionError('An unknown scene changed the trained full model')
            predicted, details = model(noisy, times, inputs, return_details=True)
            fk = forward_kinematics(predicted, details['rest'])
            if not torch.allclose(fk, predicted[..., 135:].reshape(*predicted.shape[:2], 22, 3)*2, atol=2e-5, rtol=0):
                raise AssertionError('Native output FK mismatch')
        counts = {}
        for variant in VARIANTS:
            counts[variant] = sum(p.numel() for p in model.adapter.parameters())
        record.update(state='passed', zero_adapter_exact=True, private_labels_isolated=True,
            frozen_prior_unchanged=True, exact_resume=True, empty_scene_exact_baseline=True,
            all_variants_zero_exact=True, original_ddim_3_corpus_parity=ddim_parity,
            all_variant_trainable_parameters=counts, first_update=first,
            feature_names=list(FEATURE_NAMES), completed_at_utc=now(),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)
        atomic_json(folder/f'{device}.json', record)
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        atomic_json(folder/f'{device}.json', record)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return record


def train_variant(model, data, validation, config, device, out, seal, resume):
    variant = model.variant
    folder = out/variant
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'CONTRACT.json', seal)
    last = folder/'last.pt'
    optimizer = engine.optimizer_for(model, config)
    rng = np.random.default_rng(config['seed'])
    step = restore(model, optimizer, rng, device, seal, variant, last) if last.exists() and resume else 0
    if last.exists() and not resume:
        raise ValueError('Use --resume; an existing adapter must not be overwritten')
    repair.reconcile_journal(folder, step)
    started, initial = time.monotonic(), step
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(True))

    def status(state, **extra):
        value = dict(state=state, variant=variant, completed_updates=step,
            planned_updates=config['training']['planned_updates'], pid=os.getpid(),
            updates_per_second=(step-initial)/max(time.monotonic()-started, 1e-6),
            contract_sha256=seal['sha256'], updated_at_utc=now(), **extra)
        atomic_json(folder/'STATUS.json', value)
        atomic_json(out/'STATUS.json', value)
        return value
    try:
        with (folder/'updates.jsonl').open('a') as stream:
            while step < config['training']['planned_updates'] and not stop:
                step += 1
                record = update(model, optimizer, data, rng, config, device, step)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                if step%20 == 0:
                    status('training', latest_loss=record['losses']['total'], lr=record['lr'])
                if step%config['training']['checkpoint_every'] == 0:
                    checkpoint(model, optimizer, rng, device, step, seal, variant, last)
                if step%config['training']['validation_every'] == 0:
                    status('validation_quick')
                    def quick_progress(done, total, elapsed):
                        status('validation_quick', completed_windows=done,
                               total_windows=total, elapsed_seconds=elapsed)
                    evaluate(model, validation, seal['validation_cases']['quick'], config, device,
                             folder/'evaluation'/f'quick_{step:06d}', quick_progress)
        checkpoint(model, optimizer, rng, device, step, seal, variant, last)
        if stop:
            status('stopped')
            raise InterruptedError('Adapter pipeline stopped; resume explicitly')
        original = tensor_digest(model.base.state_dict())
        digest = sha256(last)
        def progress(done, total, elapsed):
            status('validation_final', completed_windows=done, total_windows=total, elapsed_seconds=elapsed)
        historical = evaluate(model, validation, seal['validation_cases']['comprehensive'], config, device,
            folder/'evaluation/historical_1416', progress)
        expanded = evaluate(model, validation, seal['validation_cases']['expanded'], config, device,
            folder/'evaluation/all_sequences', progress, historical['rows'])
        if original != tensor_digest(model.base.state_dict()):
            raise AssertionError('Evaluation modified the frozen prior')
        result = dict(state='complete', variant=variant, fixed_step=step, test_used=False,
            checkpoint_sha256=digest, base_sha256=seal['base_sha256'],
            trainable_parameters=sum(p.numel() for p in model.adapter.parameters()),
            historical_score_mm=historical['selection_score_mm'], expanded_score_mm=expanded['selection_score_mm'],
            completed_at_utc=now())
        atomic_json(folder/'RESULT.json', result)
        status('complete', **{k:v for k,v in result.items() if k not in ('state','variant')})
        return result
    except BaseException as error:
        status('stopped' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run'))
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/scene_conditioning.yaml')
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--scene-paths', type=Path, default=ROOT/'configs/scene_paths.local.yaml')
    args = parser.parse_args()
    ensure_bounded()
    if args.device == 'cuda':
        configure_cuda()
    config, reference = settings(args.config)
    paths = load_paths(args.paths)
    model = make_model(reference, paths, args.scene_paths, config, 'full', args.device)
    common = model.base.shape.rest_template[:1].cpu().numpy()
    data = LocalSceneDataset(args.scene_paths, common, config=config)
    validation = LocalSceneDataset(args.scene_paths, common, split='validation', config=config)
    cases = json.loads((ROOT/reference['base']['contract']).read_text())['validation_cases']
    cases = dict(cases, expanded=expanded_cases(validation, cases['comprehensive']))
    seal = protocol(config, reference, paths, args.scene_paths, cases, args.config)
    out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    prepared = out/'PREPARED_CONTRACT.json'
    if prepared.exists() and json.loads(prepared.read_text()) != seal:
        raise ValueError('Prepared experiment dependency set changed')
    atomic_json(prepared, seal)
    atomic_json(out/'VALIDATION_MANIFESTS.json', cases)
    if args.action == 'preflight':
        result = preflight(model, data, validation, config, reference, args.device, out, seal)
        print(json.dumps({k:v for k,v in result.items() if k != 'first_update'}, indent=2), flush=True)
        return
    for device in ('cpu','cuda'):
        checked = json.loads((out/'preflight'/f'{device}.json').read_text())
        if checked['state'] != 'passed' or checked['contract_sha256'] != seal['sha256']:
            raise ValueError('Matching CPU and CUDA acceptance are required')
    # Extend the pinned base's existing 1416 results only on the 699 additional
    # windows. Fresh CUDA parity was checked against the original evaluator.
    parent_rows = json.loads((ROOT/reference['base']['comprehensive_reference']).read_text())['rows']
    def base_progress(done, total, elapsed):
        atomic_json(out/'STATUS.json', dict(state='validation_base_expansion',
            completed_windows=done, total_windows=total, existing_windows=len(parent_rows),
            elapsed_seconds=elapsed, pid=os.getpid(), contract_sha256=seal['sha256'],
            updated_at_utc=now()))
    base_progress(len(parent_rows), len(cases['expanded']), 0.)
    evaluate(model, validation, cases['expanded'], config, args.device,
        out/'base_30k/evaluation/all_sequences', base_progress, initial_rows=parent_rows, use_scene=False)
    del model
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    results = []
    for variant in config['variants']:
        result_file = out/variant/'RESULT.json'
        if result_file.exists() and args.resume:
            result = json.loads(result_file.read_text())
            if result['state'] != 'complete' or result['base_sha256'] != seal['base_sha256']:
                raise ValueError('Invalid completed variant')
            results.append(result)
            continue
        # Refresh admission before each new branch, without waiting behind an
        # unrelated workload or modifying any other process.
        from scenemi import parallel_resources
        parallel_resources.MODULES |= {'scenemi.scene_conditioning'}
        admitted = parallel_resources.admission(args.device, config['runtime']['memory_gib'],
                                               config['runtime']['allocator_gib'])
        if not admitted['admitted']:
            atomic_json(out/'STATUS.json', dict(state='admission_refused', next_variant=variant,
                resource_report=admitted, updated_at_utc=now()))
            raise RuntimeError('Resource admission refused before next adapter branch')
        model = make_model(reference, paths, args.scene_paths, config, variant, args.device)
        results.append(train_variant(model, data, validation, config, args.device, out, seal, args.resume))
        del model
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    atomic_json(out/'RESULT.json', dict(state='complete', variants=results, test_used=False, completed_at_utc=now()))
    atomic_json(out/'STATUS.json', dict(state='complete', variants=results, updated_at_utc=now()))


if __name__ == '__main__':
    main()
