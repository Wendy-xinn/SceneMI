"""Paired short EXO continuation with TRAIN-only reference nondegradation.

The sealed generator, sampler, observation builder and original objective are
reused unchanged. Both arms reset AdamW and replay identical inputs and noise.
The only experimental difference is one calibrated raw-FK hidden-position loss.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from . import adapt, exo_longrun, reference_temporal, window_sampling
from .dependencies import dependency_closure, sha256

PROTOCOL = 'native30-paired-reference-nondegradation-trial-v1'
ARMS = ('control', 'nondegradation')
DEFAULT_RUN = adapt.BASE / 'runs/native30_exo_refinement150_v1'
DEFAULT_PARENT = adapt.BASE / 'runs/native30_exo_natural10k_v1/training/checkpoints/step_010000.pt'


def write(path, value):
    adapt.write(Path(path), value)


def status(run, state, **fields):
    write(Path(run) / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        updated_at=adapt.now(), pid=os.getpid(), TEST_used=False,
        automatic_extension=False, **fields))


def hidden_reference_nondegradation(raw, truth, reference, rest, observation_meta,
                                   quality=None, *, tolerance_m=.005, scale_m=.05):
    """TRAIN-only scalar; inference receives no target or target-quality mask.

    First normalize within each sample by its number of hidden frame/joints.
    Source-fit reliability attenuates the numerator, never its denominator.
    """
    import torch
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    if tolerance_m < 0 or scale_m <= 0:
        raise ValueError('Nonnegative tolerance and positive metre scale required')
    prediction = forward_kinematics(raw, rest)
    with torch.no_grad():
        target = forward_kinematics(truth, rest)
        baseline = forward_kinematics(reference, rest)
        reference_error = (baseline - target).norm(dim=-1)
    hidden = observation_meta[..., 0] <= 0
    error = (prediction - target).norm(dim=-1)
    excess = ((error - reference_error - tolerance_m).clamp_min(0) / scale_m).square()
    weight = hidden.to(raw)
    if quality is not None:
        if quality.shape != hidden.shape:
            raise ValueError('TRAIN position quality shape differs')
        weight = weight * quality.to(raw)
    count = hidden.flatten(1).sum(1)
    per_sample = (excess * weight).flatten(1).sum(1) / count.clamp_min(1)
    return per_sample.mean(), dict(hidden_points=int(hidden.sum()),
        active_hidden_points=int(((error.detach() > reference_error + tolerance_m) & hidden).sum()),
        valid_samples=int((count > 0).sum()), hidden_reference_mean_m=
        float(reference_error[hidden].mean()) if bool(hidden.any()) else None)


def prepare(run, parent=DEFAULT_PARENT, *, updates=150, seed=2026102081, lr=1e-5):
    run, parent = Path(run).resolve(), Path(parent).resolve()
    if (run / 'PLAN.json').exists():
        raise ValueError('Existing trial plan is immutable')
    if updates != 150 or not 0 < lr <= 1e-4:
        raise ValueError('This authorized comparison requires 150 updates and bounded LR')
    effective = adapt.read(parent.parents[2] / 'prepared/effective_config.json')
    config = copy.deepcopy(effective['config'])
    sources = dict(effective['code_sha256'])
    for source, digest in sources.items():
        if sha256(source) != digest:
            raise ValueError('Parent sealed source changed: ' + source)
    sources.update(dependency_closure([Path(__file__)]))
    plan = dict(protocol=PROTOCOL, created_at=adapt.now(), parent_checkpoint=str(parent),
        parent_checkpoint_sha256=sha256(parent), expected_parent_step=10000,
        expected_base_sha256=effective['adaptation_parent']['base_model_sha256'],
        config=config, inherited_effective_config=effective, source_sha256=sources,
        arms=list(ARMS), updates=updates, batch_size=8, seed=seed,
        optimizer=dict(type='AdamW', lr=lr, weight_decay=config['optimizer']['weight_decay'],
                       clip_norm=config['optimizer']['clip_norm'], reset=True, schedule='constant'),
        calibration=dict(train_batches=10, gradient_ratio=.2, coefficient_maximum=1.,
                         optimizer_updates=0, sampler_and_all_RNG_rolled_back=True),
        nondegradation=dict(tolerance_m=.005, scale_m=.05, target='raw_motion_FK',
                           scope='hidden positions; per-sample hidden count normalization'),
        data_policy='TRAIN only; inherited natural-window population; no VAL/TEST for optimization/calibration',
        reference_policy=reference_temporal.POLICY, analytical_physics_enabled=False,
        checkpoint_policy='one atomic full last.pt per arm, every25/final/interruption',
        paired='same initialization, fresh optimizer, sampler, input noise, DDIM training timestep/noise and scene dropout')
    run.mkdir(parents=True, exist_ok=True)
    write(run / 'PLAN.json', plan)
    write(run / 'config.json', config)
    status(run, 'prepared', planned_updates_per_arm=updates, arms=list(ARMS))
    return plan


def verify(run):
    plan = adapt.read(Path(run) / 'PLAN.json')
    if plan['protocol'] != PROTOCOL or sha256(plan['parent_checkpoint']) != plan['parent_checkpoint_sha256']:
        raise ValueError('Parent/trial identity changed')
    for source, digest in plan['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Trial sealed dependency changed: ' + source)
    return plan


def load_model(path, device='cpu', *, expected_sha256=None):
    """Self-contained loader shared by the separate paired VAL evaluator."""
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from .adapter_model import install_adapters, base_state_digest, adapter_state_digest
    path = Path(path).resolve()
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError('Candidate checkpoint changed')
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if saved.get('protocol') != PROTOCOL or saved.get('step') != saved.get('planned_steps'):
        raise ValueError('Require a completed registered paired trial checkpoint')
    config = saved['effective_config']['config']
    with torch.random.fork_rng(devices=[]):
        model = make_model(config)
        install_adapters(model, saved['adapter_spec'])
    model.load_state_dict(saved['model'], strict=True)
    if (base_state_digest(model) != saved['frozen_base_sha256'] or
            adapter_state_digest(model) != saved['adapter_sha256']):
        raise ValueError('Candidate tensor digest differs')
    model.to(device).eval()
    return model, dict(config=config, checkpoint=str(path), checkpoint_sha256=sha256(path),
        stage_step=saved['step'], arm=saved['arm'], parent=saved['parent'],
        frozen_base_sha256=saved['frozen_base_sha256'], reference_policy=reference_temporal.POLICY,
        guidance_enabled=False, trial_protocol=PROTOCOL)


def _small_identity(row):
    keys = ('dataset', 'group', 'sequence_id', 'path', 'split', 'start',
            'source_start_30fps', 'yaw_augmentation_rad', 'mode', 'sampling')
    return {key: row.get(key) for key in keys}


def _fingerprint(known, noisy, times, scene_on, identities):
    h = hashlib.sha256(json.dumps(dict(scene=scene_on, identities=identities),
                                sort_keys=True).encode())
    for key, value in [('noisy', noisy), ('timestep', times)] + [
            (key, known[key]) for key in ('trajectory', 'observation_meta',
            'observation_uncertainty', 'reference_motion', 'reference_confidence',
            'camera', 'occupancy', 'rest')]:
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _training_input(data, builder, config, sampling_plan, rng):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import sample_item
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    pairs = [sample_item(data, builder, config, sampling_plan, rng) for _ in range(8)]
    items, identities = zip(*pairs)
    if any(row['split'] != 'train' or row['mode'] != 'exo' for row in identities):
        raise AssertionError('Optimization/calibration must read TRAIN EXO only')
    batch = collate_observations(items)
    known = model_inputs(config, batch)
    t = torch.randint(1000, (8,), device='cuda')
    low = torch.rand(8, device='cuda') < config['noise']['extra_low_noise_probability']
    t = torch.where(low, torch.randint(config['noise']['low_noise_steps'], t.shape, device='cuda'), t)
    target = training_motion(config, batch, known)
    signal = torch.as_tensor(__import__('experiments.offline_camera_retrain_v1.scene_model',
        fromlist=['cosine_alphas']).cosine_alphas(), device='cuda')[t]
    noisy = signal[:, None, None].sqrt() * target + (1 - signal[:, None, None]).sqrt() * torch.randn_like(target)
    scene_on = bool(rng.random() >= config['scene']['dropout_probability'])
    small = [_small_identity(row) for row in identities]
    return batch, known, t, noisy, signal, scene_on, small, _fingerprint(known, noisy, t, scene_on, small)


def _losses(model, sample, config, coefficients):
    from experiments.offline_camera_retrain_v1.native30_losses import objective
    batch, known, t, noisy, signal, scene_on, identities, digest = sample
    prediction, details = model(noisy, t, known, use_scene=scene_on, return_details=True)
    base, terms = objective(prediction, batch['motion'], batch, signal, details, config, coefficients)
    nd, diagnostics = hidden_reference_nondegradation(details['raw_motion'], batch['motion'],
        known['reference_motion'], batch['rest'], batch['observation_meta'],
        batch.get('gt_position_weight'))
    return base, nd, diagnostics, terms


def _gradient_norm(gradients):
    import torch
    values = [g.detach().square().sum() for g in gradients if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def run_trial(run, *, only_arm=None, resume=False):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
    from experiments.offline_camera_retrain_v1.train_native30 import StopAtUpdateBoundary
    from .adapter_model import adapter_parameters, adapter_spec, adapter_state_digest, base_state_digest
    from .adapter_trial import ensure_frozen, truncate_uncommitted_trace
    run = Path(run).resolve()
    plan = verify(run)
    config, effective = plan['config'], plan['inherited_effective_config']
    adapt.gpu_setup(config)
    adapt.install_io(config, effective, run)
    data, unused_val = adapt.datasets(config)
    del unused_val
    sampling_plan = effective['sampling_plan']
    data.training_sampler = window_sampling.NaturalWindowSampler(data, sampling_plan,
        config['sampler'], plan['seed'] + 371)
    if data.training_sampler.fingerprint != effective['sampling_fingerprint']:
        raise ValueError('Inherited natural TRAIN population differs')
    builder = adapt.observation_builder(config)
    rng = np.random.default_rng(plan['seed'] + 991)
    torch.manual_seed(plan['seed']); np.random.seed(plan['seed']); random.seed(plan['seed'])
    initial_rng = exo_longrun.capture_rng(data, rng)
    model, parent_identity = exo_longrun.load_inference_model(plan['parent_checkpoint'], 'cuda',
        expected_sha256=plan['parent_checkpoint_sha256'])
    if parent_identity['stage_step'] != 10000:
        raise ValueError('Expected exact natural10k parent')
    parameters = list(adapter_parameters(model))
    coefficients = effective['observation_coefficients']
    calibration_file = run / 'CALIBRATION.json'
    if calibration_file.exists():
        calibration = adapt.read(calibration_file)
        if calibration['plan_sha256'] != sha256(run / 'PLAN.json'):
            raise ValueError('Calibration plan differs')
    else:
        status(run, 'calibrating_train_only', planned_batches=10, optimizer_updates=0)
        records = []
        for i in range(10):
            tick = time.monotonic()
            model.train()
            sample = _training_input(data, builder, config, sampling_plan, rng)
            base, nd, diagnostics, terms = _losses(model, sample, config, coefficients)
            bg = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
            ng = torch.autograd.grad(nd, parameters, allow_unused=True)
            records.append(dict(batch=i, base_gradient_norm=_gradient_norm(bg),
                nd_gradient_norm=_gradient_norm(ng), base_loss=float(base.detach()), nd_loss=float(nd.detach()),
                diagnostics=diagnostics, input_sha256=sample[-1], identities=sample[-2],
                elapsed_s=time.monotonic()-tick))
            print('calibration', i+1, records[-1]['base_gradient_norm'], records[-1]['nd_gradient_norm'], flush=True)
            del sample, base, nd, bg, ng, terms
        base_rms = math.sqrt(sum(row['base_gradient_norm']**2 for row in records)/len(records))
        nd_rms = math.sqrt(sum(row['nd_gradient_norm']**2 for row in records)/len(records))
        if not math.isfinite(nd_rms) or nd_rms <= 1e-12:
            raise ValueError('No finite TRAIN nondegradation gradient for calibration')
        weight = min(plan['calibration']['coefficient_maximum'], .2*base_rms/nd_rms)
        calibration = dict(protocol=PROTOCOL, plan_sha256=sha256(run/'PLAN.json'),
            created_at=adapt.now(), coefficient=weight, base_gradient_rms=base_rms,
            nd_gradient_rms=nd_rms, achieved_gradient_ratio=weight*nd_rms/base_rms,
            records=records, TRAIN_only=True, optimizer_updates=0)
        write(calibration_file, calibration)
    del model
    gc.collect(); torch.cuda.empty_cache()
    arms = (only_arm,) if only_arm else ARMS
    for arm in arms:
        folder = run / arm
        folder.mkdir(exist_ok=True)
        path = folder / 'last.pt'
        if path.exists() and not resume:
            raise ValueError('Existing arm checkpoint requires --resume')
        model, _ = exo_longrun.load_inference_model(plan['parent_checkpoint'], 'cuda',
            expected_sha256=plan['parent_checkpoint_sha256'])
        parameters = list(adapter_parameters(model))
        optimizer = torch.optim.AdamW(parameters, lr=plan['optimizer']['lr'],
            weight_decay=plan['optimizer']['weight_decay'])
        exo_longrun.restore_rng(initial_rng, data, rng)
        start, previous_elapsed = 0, 0.
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['plan_sha256'] != sha256(run/'PLAN.json') or saved['calibration_sha256'] != sha256(calibration_file):
                raise ValueError('Resume plan/calibration differs')
            model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value): state[key] = value.cuda()
            exo_longrun.restore_rng(saved, data, rng)
            start, previous_elapsed = saved['step'], saved['elapsed_training_s']
            truncate_uncommitted_trace(folder/'trace.jsonl', start)
            del saved
        if start == plan['updates']:
            del model, optimizer; gc.collect(); torch.cuda.empty_cache(); continue
        controls = {}
        if arm == 'nondegradation':
            controls = {row['step']:row for row in (json.loads(line) for line in
                (run/'control/trace.jsonl').read_text().splitlines())}
            if len(controls) != plan['updates']:
                raise ValueError('Finish the complete control before the treatment')
        started = time.monotonic()
        def commit(step):
            ensure_frozen(model, plan['expected_base_sha256'])
            payload = dict(protocol=PROTOCOL, arm=arm, step=step, planned_steps=plan['updates'],
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                effective_config=effective, adapter_spec=adapter_spec(model),
                adapter_sha256=adapter_state_digest(model), frozen_base_sha256=base_state_digest(model),
                guidance_enabled=False, reference_policy=reference_temporal.POLICY,
                plan_sha256=sha256(run/'PLAN.json'), calibration_sha256=sha256(calibration_file),
                parent=parent_identity, coefficient=calibration['coefficient'] if arm=='nondegradation' else 0.,
                elapsed_training_s=previous_elapsed+time.monotonic()-started,
                **exo_longrun.capture_rng(data, rng))
            save_checkpoint(payload, path)
        completed = start
        with StopAtUpdateBoundary() as stopping, (folder/'trace.jsonl').open('a' if start else 'w') as trace:
            for step in range(start+1, plan['updates']+1):
                tick = time.monotonic()
                model.train(); optimizer.zero_grad(set_to_none=True)
                sample = _training_input(data, builder, config, sampling_plan, rng)
                if controls and sample[-1] != controls[step]['input_sha256']:
                    raise AssertionError('Paired TRAIN sample/noise/scene mismatch at step ' + str(step))
                base, nd, diagnostics, terms = _losses(model, sample, config, coefficients)
                weight = calibration['coefficient'] if arm == 'nondegradation' else 0.
                loss = base + weight*nd
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite paired-trial objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, plan['optimizer']['clip_norm'], error_if_nonfinite=True)
                ensure_frozen(model)
                optimizer.step(); torch.cuda.synchronize()
                row = dict(step=step, arm=arm, loss=float(loss.detach()), base_loss=float(base.detach()),
                    nondegradation_loss=float(nd.detach()), nondegradation_weight=weight,
                    weighted_nondegradation=float((weight*nd).detach()), gradient_norm=float(norm),
                    diagnostics=diagnostics, scene=sample[5], timesteps=sample[2].tolist(),
                    identities=sample[-2], input_sha256=sample[-1], lr=plan['optimizer']['lr'],
                    update_s=time.monotonic()-tick, elapsed_s=previous_elapsed+time.monotonic()-started,
                    terms={group:{key:float(value.detach()) for key,value in items.items()} for group,items in terms.items()})
                trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                completed = step
                if step == 1 or step % 10 == 0:
                    print(arm, step, row['loss'], row['nondegradation_loss'], flush=True)
                    status(run, 'training', arm=arm, completed_updates=step,
                        planned_updates=plan['updates'], loss=row['loss'], elapsed_s=row['elapsed_s'])
                if step % 25 == 0 or step == plan['updates'] or stopping.requested:
                    commit(step)
                del sample, base, nd, loss, terms
                if stopping.requested: break
        write(folder/'RESULT.json', dict(protocol=PROTOCOL, arm=arm, status='completed' if completed==plan['updates'] else 'stopped',
            updates=completed, checkpoint=str(path), checkpoint_sha256=sha256(path),
            base_sha256=base_state_digest(model), adapter_sha256=adapter_state_digest(model),
            elapsed_training_s=previous_elapsed+time.monotonic()-started,
            plan_sha256=sha256(run/'PLAN.json'), calibration_sha256=sha256(calibration_file)))
        del model, optimizer, parameters
        gc.collect(); torch.cuda.empty_cache()
        if completed != plan['updates']:
            status(run, 'stopped', arm=arm, completed_updates=completed); return
    results = {arm:adapt.read(run/arm/'RESULT.json') for arm in ARMS if (run/arm/'RESULT.json').exists()}
    complete = len(results)==2 and all(row['status']=='completed' for row in results.values())
    status(run, 'completed_pending_paired_validation' if complete else 'arm_completed', results=results,
           paired_inputs_exact=True if complete else None, frozen_base_sha256=plan['expected_base_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','train','status'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.command=='status': print(json.dumps(adapt.read(run/'STATUS.json'), ensure_ascii=False, indent=2)); return
    if args.command=='prepare': prepare(run,args.parent); return
    if not (run/'PLAN.json').exists(): prepare(run,args.parent)
    try:
        with adapt.gpu_lock(run), window_sampling.install_natural_window_sampler(), reference_temporal.installed():
            run_trial(run, only_arm=args.arm, resume=args.resume)
    except BaseException as exc:
        status(run,'failed',error=repr(exc)); raise


if __name__=='__main__': main()
