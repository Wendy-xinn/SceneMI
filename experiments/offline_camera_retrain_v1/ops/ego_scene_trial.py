"""Resumable matched EGO trials with an isolated scene-fusion parameter group.

The official 41k checkpoint is strictly loaded before optional fusion attachment.
Both arms train the original EGO/head-only objective with physical guidance OFF.
The sampling, noise and dropout traces are paired; stopping never restarts the
fixed learning-rate clock.  This module does not select/promote a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import weakref

from . import scene_attention_trial as shared
from .dependencies import dependency_closure, sha256

common = shared.common
BASE, MAIN = shared.BASE, shared.MAIN
read, write, require = common.read, common.write, common.require
digest, append, trim_trace = shared.digest, shared.append, shared.trim_trace
data_for, recordings, install_io = shared.data_for, shared.recordings, shared.install_io
DEFAULT_OUT = BASE / 'runs/native30_ego_scene_fusion_v2'
MODULE = 'experiments.offline_camera_retrain_v1.ops.ego_scene_trial'
PROTOCOL = 'matched-ego-scene-fusion-training-v2'
ARMS = dict(control='original', fusion='scene_fusion_v2')
PREFIX = 'scene_fusion_v2.'
PARENT_SHA = 'f8e2d22b155240c2fda8cb7d75aca451bb4152ede2441d592f3952f52068b4d0'
PARENT_MODEL_SHA = '4d789e1e63f1681885c000b1d1795ba444b5a3eee3fd6af225fc9b7b0da6008f'
DEFAULT_WEIGHTS = dict(rich=.5, trumans=.25, egobody=.25)


def status(out, state, **values):
    write(out / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=common.now(), pid=os.getpid(), TEST_used=False,
        automatic_checkpoint_promotion=False, **values))


def dataset(group):
    return 'egobody' if group in ('camera_wearer', 'interactee') else group


def weighted_sampling_plan(original, weights):
    """Reweight datasets while retaining their original within-dataset ratios."""
    require(set(weights) == set(DEFAULT_WEIGHTS), 'Supply rich/trumans/egobody weights')
    require(all(math.isfinite(v) and v > 0 for v in weights.values()), 'Dataset weights must be finite/positive')
    require(original and all(r['mode'] == 'ego' and r['probability'] > 0 for r in original), 'Expected positive EGO-only sampling plan')
    totals = {key: sum(r['probability'] for r in original if dataset(r['group']) == key)
        for key in weights}
    require(all(value > 0 for value in totals.values()), 'Every weighted dataset needs eligible TRAIN windows')
    normal = sum(weights.values())
    result = copy.deepcopy(original)
    for row in result:
        name = dataset(row['group'])
        require(name in weights, 'Unknown sampling group: ' + row['group'])
        row['probability'] *= weights[name] / (normal * totals[name])
    require(math.isclose(sum(r['probability'] for r in result), 1., abs_tol=1e-12), 'Invalid reweighted probability sum')
    return result


def schedule_factor(plan, step):
    require(isinstance(step, int) and 0 <= step <= plan['horizon'], 'LR step outside fixed horizon')
    opts = plan['optimizer']
    return min(1., step / opts['warmup_steps']) * max(opts['floor_fraction'],
        .5 * (1 + math.cos(math.pi * step / plan['horizon'])))


def group_learning_rates(plan, step, fusion=False):
    factor = schedule_factor(plan, step)
    result = dict(backbone=plan['optimizer']['backbone_lr'] * factor)
    if fusion: result['fusion'] = plan['optimizer']['fusion_lr'] * factor
    return result


def parameter_groups(model, plan):
    """Named, disjoint groups; all original and new parameters stay trainable."""
    groups, specification = [], []
    named = list(model.named_parameters())
    require(named and all(p.requires_grad for _, p in named), 'All parameters must be trainable')
    require(len({id(p) for _, p in named}) == len(named), 'Aliased named parameters are unsupported')
    for label, is_new in (('backbone', False), ('fusion', True)):
        entries = [(n, p) for n, p in named if n.startswith(PREFIX) == is_new]
        if not entries: continue
        names, params = zip(*entries)
        lr = plan['optimizer'][label + '_lr']
        groups.append(dict(params=list(params), name=label, parameter_names=list(names), lr=lr))
        specification.append(dict(name=label, parameter_names=list(names), tensors=len(params),
            parameters=sum(p.numel() for p in params), initial_lr=lr))
    require(sum(s['tensors'] for s in specification) == len(named), 'Parameter grouping lost tensors')
    return groups, specification


def closure():
    from . import scene_fusion
    result = common.closure()
    result.update(dependency_closure([Path(__file__), Path(scene_fusion.__file__), shared.IO_ADAPTER]))
    return result


def model_for(plan, arm, *, device='cpu'):
    import torch
    from .scene_fusion import attach_scene_fusion, scene_fusion_metadata
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    require(arm in ARMS, 'Unknown training arm')
    parent = torch.load(plan['parent_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    model = make_model(plan['config'])
    model.load_state_dict(parent['model'], strict=True)
    require(model_digest(model) == plan['parent_model_sha256'], 'Strict-loaded parent tensors differ')
    torch.manual_seed(plan['seed'])
    if arm == 'fusion':
        before = torch.get_rng_state().clone()
        attach_scene_fusion(model, **plan['fusion_options'])
        require(torch.equal(before, torch.get_rng_state()), 'Fusion attachment consumed common CPU RNG')
        require(scene_fusion_metadata(model) == plan['fusion_spec'], 'Fusion specification differs')
    model.guidance_enabled = False
    require(all(p.requires_grad for p in model.parameters()), 'Frozen parameters would change trial scope')
    return model.to(device)


def prepare(out, *, seed=2026101941, backbone_lr=2e-6, fusion_lr=1e-4,
            horizon=3000, warmup=100, floor_fraction=.2, weights=None,
            token_dim=64, heads=4, fusion_dim=128, checkpoint_interval=100):
    import torch
    from .scene_fusion import attach_scene_fusion, scene_fusion_metadata
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, training_plan
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config, model_digest
    common.cpu_setup()
    require(not (out / 'PLAN.json').exists(), 'Immutable stage plan already exists; use a distinct trial directory')
    require(0 < warmup <= horizon and horizon >= 1000, 'Invalid fixed horizon/warmup')
    require(all(math.isfinite(v) and v > 0 for v in (backbone_lr, fusion_lr)), 'Learning rates must be finite/positive')
    require(0 < floor_fraction <= 1 and checkpoint_interval > 0, 'Invalid LR floor/checkpoint interval')
    weights = copy.deepcopy(DEFAULT_WEIGHTS if weights is None else weights)
    parent_path = (MAIN / 'training/best.pt').resolve()
    require(sha256(parent_path) == PARENT_SHA, 'Parent must be the official retained 41k checkpoint')
    parent = torch.load(parent_path, map_location='cpu', mmap=True, weights_only=False)
    config = checkpoint_config(parent)
    require(parent['step'] == 41000 and config['training_mode'] == 'ego'
        and config['ego_head_only_probability'] == 1 and config['batch_size'] == 2
        and config['frames'] == 192, 'Expected 41k EGO/head-only/L192/B2 parent')
    model = make_model(config); model.load_state_dict(parent['model'], strict=True)
    require(model_digest(model) == PARENT_MODEL_SHA, 'Official parent tensor identity differs')
    original_names = [name for name, _ in model.named_parameters()]
    original_parameters = sum(p.numel() for p in model.parameters())
    fusion_options = dict(token_dim=token_dim, heads=heads, fusion_dim=fusion_dim)
    torch.manual_seed(seed); before = torch.get_rng_state().clone()
    attach_scene_fusion(model, **fusion_options)
    require(torch.equal(before, torch.get_rng_state()), 'Fusion attachment must preserve common RNG')
    fusion_spec = scene_fusion_metadata(model)
    state = model.state_dict()
    require(all(bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite initial model')
    zero_keys = fusion_spec['zero_output_state_keys']
    require(zero_keys and all(k.startswith(PREFIX) and bool((state[k] == 0).all()) for k in zero_keys), 'Fusion output must initialize exactly zero')
    new_names = [n for n, _ in model.named_parameters() if n.startswith(PREFIX)]
    require(new_names and [n for n, _ in model.named_parameters() if not n.startswith(PREFIX)] == original_names,
        'Fusion attachment changed original parameter membership')
    new_parameters = sum(p.numel() for n, p in model.named_parameters() if n.startswith(PREFIX))
    del state, model
    train, val = data_for(config, 'train', seed), data_for(config, 'validation', seed + 1)
    overlap = recordings(train) & recordings(val)
    require(not overlap, 'TRAIN/VAL recording overlap: ' + repr(sorted(overlap)))
    training_config = copy.deepcopy(config); training_config['seed'] = seed
    training_config['stage_dataset_weights'] = weights
    original_sampling = training_plan(train, training_config)
    sampling = weighted_sampling_plan(original_sampling, weights)
    sources = closure()
    sources.update({str(path.resolve()): sha256(path) for path in (
        MAIN / 'config.json', MAIN / 'prepared/effective_config.json', BASE / config['mount_defaults'])})
    metadata = common.metadata_identity(val, config)
    metadata.update({str(path.resolve()): sha256(path) for path in [train.manifest, *train.window_index_files]})
    for meta in train.scene_memory.entries:
        path = Path(meta['path']) / 'metadata.json'; metadata[str(path.resolve())] = sha256(path)
    for kind, gender in {(m['native_body']['model'], m['native_body']['gender']) for m in train.scene_memory.entries}:
        if kind == 'smpl':
            path = common.ROOT.parent / 'ProtoMotions/data/smpl' / ('SMPL_' + gender.upper() + '.pkl')
        elif kind == 'smplx':
            path = common.ROOT / 'body_models/smplx' / ('SMPLX_' + gender.upper() + '.npz')
        else: raise ValueError('Unsupported TRAIN body model: ' + kind)
        metadata[str(path.resolve())] = sha256(path)
    sources.update(metadata)
    out.mkdir(parents=True, exist_ok=True)
    pinned = out / 'parent.pt'
    if not pinned.exists(): os.link(parent_path, pinned)
    require(sha256(pinned) == PARENT_SHA, 'Pinned parent differs')
    effective = parent.get('effective_config') or read(MAIN / 'prepared/effective_config.json')
    plan = dict(protocol=PROTOCOL, prepared_at=common.now(), seed=seed, config=config,
        training_config=training_config, arms=ARMS, parent_checkpoint=str(pinned.resolve()),
        parent_source_checkpoint=str(parent_path), parent_checkpoint_sha256=PARENT_SHA,
        parent_model_sha256=PARENT_MODEL_SHA, parent_step=41000, source_sha256=sources,
        parent_effective_data_sha256=effective['data_sha256'], fusion_options=fusion_options,
        fusion_spec=fusion_spec, original_parameter_names=original_names, new_parameter_names=new_names,
        initial_original_parameters=original_parameters, fusion_parameters=new_parameters,
        dataset_weights=weights, original_sampling_plan=original_sampling, sampling_plan=sampling,
        optimizer=dict(type='AdamW', backbone_lr=backbone_lr, fusion_lr=fusion_lr,
            warmup_steps=warmup, floor_fraction=floor_fraction,
            weight_decay=config['optimizer']['weight_decay'], clip_norm=config['optimizer']['clip_norm']),
        horizon=horizon, gate_stop=1000, checkpoint_interval=checkpoint_interval,
        batch_size=2, frames=192, guidance_enabled=False, precision='FP32', reference_policy='original',
        optimizer_reset=True, parent_optimizer_reused=False, all_parameters_trainable=True,
        train_recordings=len(recordings(train)), validation_recordings=len(recordings(val)),
        train_validation_recording_overlap=[], TEST_used=False, automatic_checkpoint_promotion=False,
        checkpoint_policy='One atomic resumable last.pt per arm; parent is hard-linked; no best/numbered copies.',
        interpretation='1000 is a validation gate, not convergence. Continue only after matched fixed-VAL review; schedule stays fixed.',
        validation_policy='Independent evaluator seals development/confirmation recording panels. Training loss cannot select/promote a model.')
    write(out / 'PLAN.json', plan)
    status(out, 'prepared_cpu_only', optimizer_updates=0, horizon=horizon, gate_stop=1000,
        free_disk_before=shutil.disk_usage(BASE).free, parent_step=41000)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    return plan


def verify(out):
    plan = read(out / 'PLAN.json')
    require(plan['protocol'] == PROTOCOL and plan['arms'] == ARMS, 'Stage protocol changed')
    require(plan['guidance_enabled'] is False and plan['reference_policy'] == 'original', 'Runtime recipe changed')
    for path, expected in plan['source_sha256'].items():
        require(sha256(path) == expected, 'Sealed dependency changed: ' + path)
    require(sha256(plan['parent_checkpoint']) == plan['parent_checkpoint_sha256'] == PARENT_SHA, 'Parent checkpoint changed')
    return plan


def configure_cuda(memory_fraction):
    import torch
    require(0 < memory_fraction <= .8, 'GPU memory fraction must be in (0,.8]')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def rng_state(data, rng):
    import numpy as np
    import torch
    return dict(data_rng=data.sampler_state(), condition_rng=copy.deepcopy(rng.bit_generator.state),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state())


def restore_rng(state, data=None, rng=None):
    import numpy as np
    import torch
    if data is not None: data.load_sampler_state(state['data_rng'])
    if rng is not None: rng.bit_generator.state = state['condition_rng']
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    np.random.set_state(state['numpy_rng'])


def geometry_hasher():
    import numpy as np
    cache = weakref.WeakKeyDictionary()
    def identity(scene):
        if scene in cache: return cache[scene]
        if hasattr(scene, 'base'):
            value = dict(base=identity(scene.base), attrs={k: v for k, v in vars(scene).items() if isinstance(v, np.ndarray)})
        else:
            value = {k: getattr(scene, k) for k in ('static', 'dynamic', 'cameras', 'dynamic_owner_ids', 'dynamic_features', 'dynamic_valid') if hasattr(scene, k)}
        cache[scene] = digest(value)
        return cache[scene]
    return identity


def prepare_batch(plan, data, builder, rng, alphas, step, geometry):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import sample_item
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    config = plan['config']
    pairs = [sample_item(data, builder, plan['training_config'], plan['sampling_plan'], rng) for _ in range(plan['batch_size'])]
    items, identities = zip(*pairs)
    require(all(i['mode'] == 'ego' and i['split'] == 'train' for i in identities), 'Non-TRAIN/EGO sample')
    batch = collate_observations(items); known = model_inputs(config, batch)
    times = torch.randint(1000, (plan['batch_size'],), device='cuda')
    low = torch.rand(plan['batch_size'], device='cuda') < config['noise']['extra_low_noise_probability']
    times = torch.where(low, torch.randint(config['noise']['low_noise_steps'], times.shape, device='cuda'), times)
    target = training_motion(config, batch, known); signal = alphas[times]; noise = torch.randn_like(target)
    noisy = signal[:, None, None].sqrt() * target + (1 - signal[:, None, None]).sqrt() * noise
    scene = bool(rng.random() >= config['scene']['dropout_probability'])
    proof = dict(step=step, identities=identities, scene=scene, timesteps=times.tolist(),
        schedule_factor=schedule_factor(plan, step),
        known_sha256=common.tensor_fingerprint(known), target_sha256=digest(target), noise_sha256=digest(noise),
        noisy_sha256=digest(noisy), supervision_sha256=digest(batch),
        geometry_sha256=digest([geometry(s) for _, s, _, _ in items]),
        torch_rng_before_forward_sha256=digest(torch.get_rng_state()),
        cuda_rng_before_forward_sha256=digest(torch.cuda.get_rng_state()))
    return batch, known, noisy, times, signal, proof


def coefficients_for(plan):
    import torch
    parent = torch.load(plan['parent_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    coefficients = parent.get('effective_config', {}).get('observation_coefficients')
    if coefficients is None: coefficients = read(MAIN / 'prepared/effective_config.json')['observation_coefficients']
    return coefficients


def gradient_report(model):
    """Pre-clipping group scale; zero-init upstream gradients may start at zero."""
    import torch
    result = {}
    for label, is_new in (('backbone', False), ('fusion', True)):
        rows = [(n, p) for n, p in model.named_parameters() if n.startswith(PREFIX) == is_new]
        if not rows: continue
        gradients = [p.grad for _, p in rows if p.grad is not None]
        require(all(bool(torch.isfinite(g).all()) for g in gradients), 'Nonfinite ' + label + ' gradient')
        square = sum(float(g.detach().double().square().sum()) for g in gradients)
        result[label] = dict(parameter_tensors=len(rows), gradient_tensors=len(gradients),
            nonzero_gradient_tensors=sum(bool(g.count_nonzero()) for g in gradients), l2=math.sqrt(square))
    output = dict(model.named_parameters()).get(PREFIX + 'output_projection.weight')
    if output is not None:
        result['zero_output_projection'] = dict(has_gradient=output.grad is not None,
            gradient_l2=0. if output.grad is None else float(output.grad.detach().double().norm()))
    return result


def preflight(out, memory_fraction=.65):
    """Actual TRAIN zero-output equality and discardable backward/Adam checks."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    plan = verify(out)
    require(not any((out / arm / 'training/last.pt').exists() for arm in ARMS), 'Preflight precedes formal training')
    with common.gpu_lock():
        configure_cuda(memory_fraction); install_io(plan, out)
        data = data_for(plan['config'], 'train', plan['seed'])
        builder = common.observation_builder(plan['config'], 'cuda')
        rng = np.random.default_rng(plan['seed'] + 991)
        torch.manual_seed(plan['seed']); np.random.seed(plan['seed'])
        alphas = trainer.cosine_alphas().cuda()
        batch, known, noisy, times, signal, proof = prepare_batch(plan, data, builder, rng, alphas, 1, geometry_hasher())
        # Preflight needs an active scene even if the first paired dropout draw is OFF.
        known_sha = common.tensor_fingerprint(known); random = rng_state(data, rng)
        expected, equivalence, gradients, timings = {}, [], {}, {}
        coefficients = coefficients_for(plan)
        for arm in ARMS:
            status(out, 'GPU_preflight', arm=arm, optimizer_updates=0)
            model = model_for(plan, arm, device='cuda').eval()
            before = trainer.model_digest(model)
            for scene in (True, False):
                restore_rng(random)
                with torch.no_grad(): prediction = model(noisy, times, known, use_scene=scene).detach().cpu()
                require(bool(torch.isfinite(prediction).all()), 'Nonfinite preflight prediction')
                if arm == 'control': expected[scene] = prediction
                else:
                    maximum = float((prediction - expected[scene]).abs().max())
                    require(torch.equal(prediction, expected[scene]), 'Zero-output fusion changed parent output')
                    equivalence.append(dict(use_scene=scene, exact=torch.equal(prediction, expected[scene]), maximum_absolute_feature_delta=maximum))
            require(trainer.model_digest(model) == before, 'Forward preflight mutated model state')
            require(common.tensor_fingerprint(known) == known_sha, 'Preflight mutated known inputs')
            groups, spec = parameter_groups(model, plan)
            optimizer = torch.optim.AdamW(groups, weight_decay=plan['optimizer']['weight_decay'])
            require(not optimizer.state, 'Preflight Adam must start fresh')
            restore_rng(random); model.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started = time.monotonic()
            prediction, details = model(noisy, times, known, use_scene=True, return_details=True)
            loss, _ = trainer.objective(prediction, batch['motion'], batch, signal, details, plan['config'], coefficients)
            require(bool(torch.isfinite(loss)), 'Nonfinite actual TRAIN preflight objective')
            loss.backward(); gradients[arm] = gradient_report(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['optimizer']['clip_norm'], error_if_nonfinite=True)
            rates = group_learning_rates(plan, 1, arm == 'fusion')
            for group in optimizer.param_groups: group['lr'] = rates[group['name']]
            optimizer.step(); torch.cuda.synchronize()
            timings[arm] = dict(compute_s=time.monotonic() - started, loss=float(loss.detach()),
                gradient_norm=float(norm), group_lrs=rates, parameter_groups=spec,
                optimizer_state_entries=len(optimizer.state),
                cuda_rng_after_update_sha256=digest(torch.cuda.get_rng_state()),
                torch_rng_after_update_sha256=digest(torch.get_rng_state()))
            require(trainer.model_digest(model) != before, 'Discardable optimizer update changed no weights')
            if arm == 'fusion':
                require(gradients[arm]['zero_output_projection']['gradient_l2'] > 0, 'New output projection received no gradient')
                require(bool(model.scene_fusion_v2.output_projection.weight.count_nonzero()), 'New output projection did not update')
            del model, optimizer, groups, prediction, details, loss
            gc.collect(); torch.cuda.empty_cache()
        for key in ('cuda_rng_after_update_sha256', 'torch_rng_after_update_sha256'):
            require(timings['control'][key] == timings['fusion'][key], 'Fusion changed paired forward RNG: ' + key)
        require(common.tensor_fingerprint(known) == known_sha, 'Backward preflight mutated known inputs')
        verify(out)
        report = dict(status='passed_actual_TRAIN_GPU', checked_at=common.now(), plan_sha256=sha256(out / 'PLAN.json'),
            formal_optimizer_updates=0, throwaway_updates_per_arm=1, physics_OFF=True,
            real_train_identities=proof['identities'], paired_input=proof, use_scene_for_backward=True,
            zero_initialization_equivalence=equivalence, initial_gradient_scales=gradients, updates=timings,
            no_model_or_optimizer_saved=True,
            interpretation='Gradient magnitudes verify connectivity only; they do not predict validation quality. Zero-projection upstream gradients can be zero on the first update.',
            next_required_check='Train both arms to two matched updates, audit, then extend to the fixed VAL gate.')
        write(out / 'PREFLIGHT.json', report)
        status(out, 'preflight_passed', optimizer_updates=0, gate_stop=plan['gate_stop'], horizon=plan['horizon'])
        return report


def train(out, arm, stop_step=1000, resume=False, memory_fraction=.65):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
    plan = verify(out); check = read(out / 'PREFLIGHT.json')
    require(check['status'] == 'passed_actual_TRAIN_GPU' and check['plan_sha256'] == sha256(out / 'PLAN.json'), 'Actual-input GPU preflight required')
    require(arm in ARMS and 0 < stop_step <= plan['horizon'], 'Invalid arm/stop boundary')
    folder = out / arm / 'training'; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'last.pt'
    require(checkpoint.exists() == bool(resume), 'Checkpoint presence requires matching --resume')
    with (out / arm / 'job.lock').open('a') as lock, common.gpu_lock():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(out); configure_cuda(memory_fraction); install_io(plan, out / arm)
        config = plan['config']; data = data_for(config, 'train', plan['seed'])
        builder = common.observation_builder(config, 'cuda'); model = model_for(plan, arm, device='cuda')
        groups, group_spec = parameter_groups(model, plan)
        optimizer = torch.optim.AdamW(groups, weight_decay=plan['optimizer']['weight_decay'])
        coefficients = coefficients_for(plan)
        rng = np.random.default_rng(plan['seed'] + 991); counts = {}; start = 0
        torch.manual_seed(plan['seed']); np.random.seed(plan['seed'])
        if resume:
            state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            require(state['stage_plan_sha256'] == sha256(out / 'PLAN.json') and state['architecture'] == ARMS[arm]
                and state['config'] == config and state['training_config'] == plan['training_config']
                and state['planned_steps'] == plan['horizon'] and state['optimizer_group_spec'] == group_spec
                and state['guidance_enabled'] is False, 'Resume checkpoint differs from fixed stage')
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
            # Non-capturable AdamW keeps scalar step counters on CPU.
            for entry in optimizer.state.values():
                for key, value in entry.items():
                    if torch.is_tensor(value) and key != 'step': entry[key] = value.cuda()
            restore_rng(state, data, rng); start = state['step']; counts = state['counts']; del state
        require(start <= stop_step, 'Checkpoint exceeds requested stop boundary')
        for path in (folder / 'trace.jsonl', out / arm / 'input_trace.jsonl'): trim_trace(path, start)
        alphas = trainer.cosine_alphas().cuda(); began = time.monotonic(); current = start
        geometry = geometry_hasher()
        def commit(step):
            value = dict(config=config, effective_config=dict(config=config), training_config=plan['training_config'],
                step=step, planned_steps=plan['horizon'], parent_step=41000, architecture=ARMS[arm],
                fusion_spec=plan['fusion_spec'] if arm == 'fusion' else None,
                fusion_options=plan['fusion_options'] if arm == 'fusion' else None,
                stage_plan_sha256=sha256(out / 'PLAN.json'), source_sha256=plan['source_sha256'],
                model=model.state_dict(), optimizer=optimizer.state_dict(), optimizer_group_spec=group_spec,
                counts=counts, guidance_enabled=False, optimizer_reset_on_initial_start=True,
                precision='FP32', TEST_used=False, **rng_state(data, rng))
            save_checkpoint(value, checkpoint)
        if not resume:
            write(out / arm / 'INITIAL_STATE.json', dict(parent_model_sha256=plan['parent_model_sha256'],
                architecture=ARMS[arm], base_strict_load=True, optimizer_entries=len(optimizer.state),
                parameter_groups=group_spec, trainable_parameters=sum(p.numel() for p in model.parameters()),
                common_rng_sha256=digest(rng_state(data, rng))))
            commit(0)
        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(out / arm / 'tensorboard'), purge_step=start + 1, flush_secs=30)
            write(out / arm / 'TENSORBOARD.json', dict(enabled=True,
                log_dir=str((out / arm / 'tensorboard').resolve()), purge_step=start + 1,
                authoritative_update_trace=str((folder / 'trace.jsonl').resolve())))
        except ImportError as exc:
            write(out / arm / 'TENSORBOARD.json', dict(enabled=False, reason=repr(exc),
                authoritative_update_trace=str((folder / 'trace.jsonl').resolve())))
        status(out / arm, 'training', step=start, stop_step=stop_step, horizon=plan['horizon'])
        try:
            with trainer.StopAtUpdateBoundary() as stopping, (folder / 'trace.jsonl').open('a') as trace, (out / arm / 'input_trace.jsonl').open('a') as inputs:
                for step in range(start + 1, stop_step + 1):
                    started = time.monotonic()
                    batch, known, noisy, times, signal, proof = prepare_batch(plan, data, builder, rng, alphas, step, geometry)
                    torch.cuda.synchronize(); preparation_s = time.monotonic() - started
                    compute_started = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
                    prediction, details = model(noisy, times, known, use_scene=proof['scene'], return_details=True)
                    loss, terms = trainer.objective(prediction, batch['motion'], batch, signal, details, config, coefficients)
                    require(bool(torch.isfinite(loss)), 'Nonfinite stage objective')
                    loss.backward()
                    gradients = gradient_report(model) if step <= 2 or step % plan['checkpoint_interval'] == 0 else None
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['optimizer']['clip_norm'], error_if_nonfinite=True)
                    rates = group_learning_rates(plan, step, arm == 'fusion')
                    for group in optimizer.param_groups: group['lr'] = rates[group['name']]
                    optimizer.step(); torch.cuda.synchronize(); compute_s = time.monotonic() - compute_started
                    for identity in proof['identities']:
                        key = identity['group'] + '/ego'; counts[key] = counts.get(key, 0) + 1
                    proof['cuda_rng_after_update_sha256'] = digest(torch.cuda.get_rng_state())
                    proof['torch_rng_after_update_sha256'] = digest(torch.get_rng_state())
                    append(inputs, proof)
                    row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                        group_lrs=rates, group_gradients_before_clip=gradients, scene=proof['scene'],
                        preparation_s=preparation_s, compute_s=compute_s, update_s=time.monotonic() - started,
                        elapsed_s=time.monotonic() - began,
                        terms={k: {n: float(v.detach()) for n, v in group.items()} for k, group in terms.items()},
                        support_cache_hits=getattr(getattr(data, 'support_cache', None), 'hits', 0),
                        support_cache_misses=getattr(getattr(data, 'support_cache', None), 'misses', 0))
                    append(trace, row)
                    if writer is not None:
                        writer.add_scalar('loss/total', row['loss'], step)
                        writer.add_scalar('gradient/norm_before_clip', row['gradient_norm'], step)
                        for name, rate in rates.items(): writer.add_scalar('lr/' + name, rate, step)
                        for label, values in row['terms'].items():
                            for name, value in values.items(): writer.add_scalar('loss/' + label + '/' + name, value, step)
                        writer.add_scalar('time/preparation_s', preparation_s, step)
                        writer.add_scalar('time/compute_s', compute_s, step)
                        writer.add_scalar('progress/session_elapsed_wall_s', row['elapsed_s'], step)
                        writer.add_scalar('progress/parent_plus_stage_updates', plan['parent_step'] + step, step)
                    current = step
                    if step % plan['checkpoint_interval'] == 0 or step == stop_step or stopping.requested: commit(step)
                    status(out / arm, 'training', step=step, stop_step=stop_step, horizon=plan['horizon'],
                        last_loss=float(loss.detach()), last_group_lrs=rates, last_preparation_s=preparation_s,
                        last_compute_s=compute_s, last_update_s=time.monotonic() - started)
                    if stopping.requested: break
            verify(out)
            status(out / arm, 'completed_boundary' if current == stop_step else 'stopped', step=current,
                stop_step=stop_step, horizon=plan['horizon'], independent_validation_required=True,
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                free_disk_after=shutil.disk_usage(BASE).free)
        except BaseException as exc:
            status(out / arm, 'failed', step=current, error=repr(exc), last_checkpoint_retained=checkpoint.exists())
            raise
        finally:
            if writer is not None: writer.close()


def tensor_updates(saved, initial, names):
    import torch
    rows = []
    for name in names:
        old, new = initial[name], saved[name]
        require(old.shape == new.shape and old.dtype == new.dtype, 'Parameter layout changed: ' + name)
        difference = (new - old).double()
        rows.append(dict(name=name, parameters=new.numel(), finite=bool(torch.isfinite(new).all()),
            changed=not torch.equal(old, new), l2_delta=float(difference.norm()),
            maximum_absolute_delta=float(difference.abs().max())))
    return dict(tensors=len(rows), changed_tensors=sum(r['changed'] for r in rows),
        l2_delta=math.sqrt(sum(r['l2_delta'] ** 2 for r in rows)),
        maximum_absolute_delta=max((r['maximum_absolute_delta'] for r in rows), default=0.), details=rows)


def audit_optimizer(state, plan, arm, trace):
    import torch
    options = state['optimizer']; specs = state['optimizer_group_spec']
    groups = options['param_groups']; entries = options['state']
    require(len(specs) == len(groups) == (2 if arm == 'fusion' else 1), 'Optimizer group count differs')
    names = []; identifiers = []; result = []
    for group, spec in zip(groups, specs):
        require(group['name'] == spec['name'] and group['parameter_names'] == spec['parameter_names']
            and len(group['params']) == spec['tensors'], 'Optimizer parameter names/layout differ')
        require(math.isclose(group['weight_decay'], plan['optimizer']['weight_decay']), 'Weight decay differs')
        expected_rate = (plan['optimizer'][spec['name'] + '_lr'] if state['step'] == 0 else
            group_learning_rates(plan, state['step'], arm == 'fusion')[spec['name']])
        require(math.isclose(group['lr'], expected_rate, rel_tol=1e-12), 'Saved optimizer LR clock differs')
        names.extend(spec['parameter_names']); identifiers.extend(group['params'])
        steps = []; parameter_steps = {}
        for identifier, name in zip(group['params'], spec['parameter_names']):
            entry = entries.get(identifier)
            if entry is None: continue
            require(all(bool(torch.isfinite(v).all()) for v in entry.values() if torch.is_tensor(v)), 'Nonfinite optimizer state')
            require(entry['exp_avg'].shape == state['model'][name].shape and entry['exp_avg_sq'].shape == state['model'][name].shape, 'Optimizer moment shape differs')
            counter = float(entry['step']); require(counter.is_integer() and 0 < counter <= state['step'], 'Invalid Adam step')
            steps.append(int(counter)); parameter_steps[name] = int(counter)
        if spec['name'] == 'fusion':
            active = sum(row['scene'] for row in trace)
            require(len(steps) == (spec['tensors'] if active else 0)
                and all(step == active for step in steps), 'Fusion Adam counters do not match scene-active updates')
        result.append(dict(name=spec['name'], tensors=spec['tensors'], parameters=spec['parameters'],
            state_entries=len(steps), minimum_adam_step=min(steps, default=0), maximum_adam_step=max(steps, default=0),
            distinct_adam_steps=sorted(set(steps)), parameter_adam_steps=parameter_steps, saved_lr=group['lr']))
    require(len(names) == len(set(names)) and len(identifiers) == len(set(identifiers)), 'Overlapping optimizer groups')
    expected = plan['original_parameter_names'] + (plan['new_parameter_names'] if arm == 'fusion' else [])
    require(set(names) == set(expected) and set(entries).issubset(set(identifiers)), 'Optimizer omitted/added parameters')
    require(sum(r['parameters'] for r in result) == plan['initial_original_parameters'] + (plan['fusion_parameters'] if arm == 'fusion' else 0), 'Optimizer parameter count differs')
    for row in trace:
        expected_rates = group_learning_rates(plan, row['step'], arm == 'fusion')
        require(row['group_lrs'] == expected_rates, 'Trace LR clock differs')
    return result


def audit(out):
    import torch
    common.cpu_setup(); plan = verify(out); states, inputs, traces = {}, {}, {}
    parent = torch.load(plan['parent_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    initialized = model_for(plan, 'fusion').state_dict()
    update_report = {}
    for arm in ARMS:
        state = torch.load(out / arm / 'training/last.pt', map_location='cpu', mmap=True, weights_only=False)
        require(state['architecture'] == ARMS[arm] and state['stage_plan_sha256'] == sha256(out / 'PLAN.json')
            and state['parent_step'] == 41000 and state['guidance_enabled'] is False
            and state['config'] == plan['config'] and state['training_config'] == plan['training_config']
            and state['planned_steps'] == plan['horizon'], 'Saved checkpoint identity differs')
        inputs[arm] = [json.loads(x) for x in (out / arm / 'input_trace.jsonl').read_text().splitlines() if x]
        trace = [json.loads(x) for x in (out / arm / 'training/trace.jsonl').read_text().splitlines() if x]
        traces[arm] = trace
        require([r['step'] for r in inputs[arm]] == list(range(1, state['step'] + 1))
            and [r['step'] for r in trace] == list(range(1, state['step'] + 1)), 'Incomplete/uncommitted trace')
        require(all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace), 'Nonfinite training trace')
        require(all(bool(torch.isfinite(v).all()) for v in state['model'].values()), 'Nonfinite saved model')
        require(set(state['model']) == set(parent['model']) | (set(initialized) - set(parent['model']) if arm == 'fusion' else set()), 'State tensor membership differs')
        if arm == 'fusion': require(state['fusion_spec'] == plan['fusion_spec'], 'Saved fusion metadata differs')
        original = tensor_updates(state['model'], parent['model'], plan['original_parameter_names'])
        fresh = tensor_updates(state['model'], initialized, plan['new_parameter_names']) if arm == 'fusion' else None
        if state['step'] > 0: require(original['changed_tensors'] > 0, 'Original model did not update')
        scene_updates = sum(row['scene'] for row in inputs[arm])
        if arm == 'fusion' and scene_updates:
            require(any(row['changed'] for row in fresh['details'] if row['name'] in plan['fusion_spec']['zero_output_state_keys']), 'New output projection never updated')
        optim = audit_optimizer(state, plan, arm, trace)
        update_report[arm] = dict(step=state['step'], scene_on_updates=scene_updates, original=original,
            fusion=fresh, optimizer_groups=optim, all_model_and_optimizer_tensors_finite=True,
            timing=dict(preparation_s=sum(r['preparation_s'] for r in trace), compute_s=sum(r['compute_s'] for r in trace),
                interpretation='Preparation includes data/cache/fingerprinting; compute includes forward/backward/Adam plus periodic gradient audit. Cache order can affect preparation time.'))
        states[arm] = state
    require(states['control']['step'] == states['fusion']['step'], 'Matched update budgets differ')
    require(inputs['control'] == inputs['fusion'], 'Actual sample/reference/supervision/noise/dropout/geometry/RNG differs')
    initials = [read(out / arm / 'INITIAL_STATE.json') for arm in ARMS]
    require(initials[0]['common_rng_sha256'] == initials[1]['common_rng_sha256'], 'Initial RNG differs')
    require(all(row['optimizer_entries'] == 0 and row['base_strict_load'] for row in initials), 'Optimizer was not fresh/parent not strict-loaded')
    for key in ('counts', 'data_rng', 'condition_rng', 'torch_rng', 'cuda_rng', 'numpy_rng'):
        require(digest(states['control'][key]) == digest(states['fusion'][key]), 'Final paired state differs: ' + key)
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    report = dict(status='passed', checked_at=common.now(), updates_per_arm=states['control']['step'],
        parent_step=41000, plan_sha256=sha256(out / 'PLAN.json'), horizon=plan['horizon'],
        physics_OFF_both=True, actual_samples_reference_supervision_noise_timesteps_dropout_geometry_RNG_exact=True,
        common_initial_and_final_rng_exact=True, optimizer_reset_both=True, all_parameters_trainable=True,
        independent_validation_required=True, TEST_used=False, CUDA_initialized=False,
        checkpoint_sha256={arm: sha256(out / arm / 'training/last.pt') for arm in ARMS},
        parameter_updates_and_optimizer=update_report)
    write(out / 'MATCHED_AUDIT.json', report)
    write(out / ('UPDATE_SANITY_' + str(states['control']['step']) + '.json'), report)
    return report


def snapshot(out):
    plan = read(out / 'PLAN.json')
    return dict(protocol=PROTOCOL, horizon=plan['horizon'], gate_stop=plan['gate_stop'],
        plan_sha256=sha256(out / 'PLAN.json'), free_disk=shutil.disk_usage(BASE).free,
        stage=read(out / 'STATUS.json'), arms={arm: read(out / arm / 'STATUS.json')
            if (out / arm / 'STATUS.json').exists() else dict(status='not_started') for arm in ARMS})


def checkpoint_steps(out):
    import torch
    result = {}
    for arm in ARMS:
        path = out / arm / 'training/last.pt'
        if path.exists():
            state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            require(state['stage_plan_sha256'] == sha256(out / 'PLAN.json') and state['architecture'] == ARMS[arm],
                'Pipeline checkpoint identity differs')
            result[arm] = state['step']
            del state
        else: result[arm] = 0
    return result


def run(out, stop_step=1000, memory_fraction=.65):
    """Sequential child jobs, one explicit boundary; safe to launch detached."""
    plan = verify(out)
    require(2 <= stop_step <= plan['horizon'], 'Run boundary must be between 2 and fixed horizon')
    with (out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_steps = checkpoint_steps(out)
        require(max(initial_steps.values()) <= stop_step, 'Existing checkpoint exceeds explicit run boundary')
        record = dict(protocol=PROTOCOL, status='running', started_at=common.now(), pid=os.getpid(),
            plan_sha256=sha256(out / 'PLAN.json'), stop_step=stop_step, initial_steps=initial_steps,
            independent_children_release_CUDA=True, events=[], TEST_used=False,
            automatic_evaluation=False, automatic_checkpoint_promotion=False)
        def child(command, *arguments):
            argv = [sys.executable, '-u', '-m', MODULE, command, '--out', str(out), *map(str, arguments)]
            event = dict(command=argv, started_at=common.now(), status='running')
            record['events'].append(event); write(out / 'PIPELINE.json', record)
            environment = dict(os.environ, PYTHONUNBUFFERED='1')
            with (out / 'pipeline_console.log').open('a') as stream:
                process = subprocess.Popen(argv, cwd=common.ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT)
                event['pid'] = process.pid; write(out / 'PIPELINE.json', record)
                code = process.wait()
            event.update(status='completed' if code == 0 else 'failed', returncode=code, ended_at=common.now())
            write(out / 'PIPELINE.json', record)
            require(code == 0, 'Child failed; inspect pipeline_console.log: ' + command)
        def advance(arm, boundary):
            existing = checkpoint_steps(out)[arm]
            if existing >= boundary: return
            options = ['--arm', arm, '--stop-step', boundary, '--memory-fraction', memory_fraction]
            if (out / arm / 'training/last.pt').exists(): options.append('--resume')
            child('train', *options)
        try:
            if (out / 'PREFLIGHT.json').exists():
                proof = read(out / 'PREFLIGHT.json')
                require(proof['status'] == 'passed_actual_TRAIN_GPU' and proof['plan_sha256'] == record['plan_sha256'], 'Existing preflight differs')
            else:
                require(not any((out / arm / 'training/last.pt').exists() for arm in ARMS), 'Existing training has no preflight')
                child('preflight', '--memory-fraction', memory_fraction)
            if max(initial_steps.values()) <= 2:
                for arm in ARMS: advance(arm, 2)
                child('audit')
            else:
                proof = read(out / 'UPDATE_SANITY_2.json')
                require(proof['status'] == 'passed' and proof['updates_per_arm'] == 2
                    and proof['plan_sha256'] == record['plan_sha256'], 'Resume above 2 requires historical paired two-update gate')
            if stop_step > 2:
                for arm in ARMS: advance(arm, stop_step)
                child('audit')
            final = checkpoint_steps(out)
            require(all(step == stop_step for step in final.values()), 'Pipeline ended at unequal budgets')
            record.update(status='completed', ended_at=common.now(), final_steps=final,
                validation_required_before_further_extension=True)
            write(out / 'PIPELINE.json', record)
            return record
        except BaseException as exc:
            record.update(status='failed', ended_at=common.now(), error=repr(exc), final_steps=checkpoint_steps(out))
            write(out / 'PIPELINE.json', record)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'train', 'audit', 'status', 'run'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--stop-step', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026101941)
    parser.add_argument('--backbone-lr', type=float, default=2e-6)
    parser.add_argument('--fusion-lr', type=float, default=1e-4)
    parser.add_argument('--horizon', type=int, default=3000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--floor-fraction', type=float, default=.2)
    parser.add_argument('--dataset-weights', type=json.loads, default=DEFAULT_WEIGHTS,
        help='JSON object, e.g. {"rich":0.5,"trumans":0.25,"egobody":0.25}')
    parser.add_argument('--token-dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--fusion-dim', type=int, default=128)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--memory-fraction', type=float, default=.65)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if args.command == 'prepare':
        report = prepare(out, seed=args.seed, backbone_lr=args.backbone_lr, fusion_lr=args.fusion_lr,
            horizon=args.horizon, warmup=args.warmup, floor_fraction=args.floor_fraction,
            weights=args.dataset_weights, token_dim=args.token_dim, heads=args.heads,
            fusion_dim=args.fusion_dim, checkpoint_interval=args.checkpoint_interval)
        print(json.dumps(dict(status='prepared_cpu_only', plan_sha256=sha256(out / 'PLAN.json'),
            horizon=report['horizon'], dataset_weights=report['dataset_weights']), indent=2))
    elif args.command == 'preflight': print(json.dumps(preflight(out, args.memory_fraction), indent=2))
    elif args.command == 'audit':
        report = audit(out)
        print(json.dumps({k: v for k, v in report.items() if k != 'parameter_updates_and_optimizer'}, indent=2))
    elif args.command == 'status': print(json.dumps(snapshot(out), indent=2))
    elif args.command == 'run': print(json.dumps(run(out, args.stop_step, args.memory_fraction), indent=2))
    else:
        require(args.arm is not None, 'Training requires an explicit arm')
        train(out, args.arm, args.stop_step, args.resume, args.memory_fraction)


if __name__ == '__main__':
    main()
