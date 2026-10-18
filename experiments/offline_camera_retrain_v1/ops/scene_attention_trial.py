"""Matched EGO-only warm-start trials of scene attention, without source edits.

Both arms strictly inherit the retained 41k model, reset AdamW, train every
parameter with identical samples/noise/dropout, and disable physical guidance.
The attention arm adds a zero-output scene adapter after the strict parent load.
The stage clock is fixed at 1500; stopping at 500 is a pilot, not convergence.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import weakref

from . import validation_review as common
from .dependencies import dependency_closure, sha256

BASE = common.BASE
MAIN = common.MAIN
DEFAULT_OUT = BASE / 'runs/native30_ego_scene_attention_trial'
ARMS = dict(control='original', attention='scene_cross_attention')
PROTOCOL = 'matched-ego-scene-attention-training-v1'
IO_ADAPTER = BASE / 'runs/native30_final_review_oct16/io_recovery.py'
read, write, require = common.read, common.write, common.require


def digest(value):
    """Content identity of nested arrays/state; scene geometry is sealed separately."""
    import numpy as np
    import torch
    result = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            add(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            result.update(str(item.dtype).encode()); result.update(str(item.shape).encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                add(key); add(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for child in item: add(child)
        elif isinstance(item, (str, int, float, bool, type(None), np.generic)):
            result.update(type(item).__name__.encode()); result.update(str(item).encode())
        else:
            result.update((type(item).__module__ + '.' + type(item).__name__).encode())
    add(value)
    return result.hexdigest()


def status(out, state, **values):
    write(out / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=common.now(), pid=os.getpid(), TEST_used=False, **values))


def data_for(config, split, seed):
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.prepare_native30 import window_options
    return Native30Data(BASE / config['train_manifest'], split, seed,
        frames=config['frames'], memory_protocol=config['scene']['memory_protocol'],
        quality_config=config['gt_reliability'], exo_max_empty_frames=config.get('exo_max_empty_frames'),
        **window_options(config))


def recordings(data):
    return {(('egobody' if group in ('camera_wearer', 'interactee') else group),
        meta.get('recording') or meta['sequence_id']) for group, members in data.members.items()
        for meta, _ in members}


def closure():
    from . import scene_attention
    sources = common.closure()
    sources.update(dependency_closure([Path(__file__), Path(scene_attention.__file__), IO_ADAPTER]))
    return sources


def prepare(out, seed=2026101841):
    import torch
    from .scene_attention import attach_scene_attention, scene_attention_metadata
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, training_plan
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config, model_digest
    common.cpu_setup()
    require(not (out / 'PLAN.json').exists(), 'An immutable stage plan already exists')
    parent_path = (MAIN / 'training/best.pt').resolve()
    parent_sha = sha256(parent_path)
    require(parent_sha == 'f8e2d22b155240c2fda8cb7d75aca451bb4152ede2441d592f3952f52068b4d0',
        'Parent must be the retained original 41k checkpoint')
    parent = torch.load(parent_path, map_location='cpu', mmap=True, weights_only=False)
    config = checkpoint_config(parent)
    require(parent['step'] == 41000 and config['training_mode'] == 'ego'
        and config['ego_head_only_probability'] == 1 and config['batch_size'] == 2,
        'Expected 41k EGO-only/head-only/B2 parent')
    model = make_model(config); model.load_state_dict(parent['model'], strict=True)
    base_digest = model_digest(model)
    require(base_digest == '4d789e1e63f1681885c000b1d1795ba444b5a3eee3fd6af225fc9b7b0da6008f',
        'Strict-loaded parent model identity differs')
    base_parameters = sum(p.numel() for p in model.parameters())
    torch.manual_seed(seed)
    attach_scene_attention(model)
    attention_spec = scene_attention_metadata(model)
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()), 'Nonfinite initial weights')
    require(bool((model.scene_cross_attention.output_projection.weight == 0).all())
        and bool((model.scene_cross_attention.output_projection.bias == 0).all()), 'New output must start exactly zero')
    train, val = data_for(config, 'train', seed), data_for(config, 'validation', seed + 1)
    overlap = recordings(train) & recordings(val)
    require(not overlap, 'TRAIN/VAL recordings overlap: ' + repr(sorted(overlap)))
    training_config = copy.deepcopy(config); training_config['seed'] = seed
    sampling = training_plan(train, training_config)
    require(sampling and all(row['mode'] == 'ego' for row in sampling), 'Non-EGO sampling plan')
    sources = closure()
    sources.update({str(path.resolve()): sha256(path) for path in (
        MAIN / 'config.json', MAIN / 'prepared/effective_config.json', BASE / config['mount_defaults'])})
    # The original index/metadata identities and lazy array checks are retained.
    metadata = common.metadata_identity(val, config)
    metadata.update({str(path.resolve()): sha256(path) for path in [train.manifest, *train.window_index_files]})
    for meta in train.scene_memory.entries:
        path = Path(meta['path']) / 'metadata.json'; metadata[str(path.resolve())] = sha256(path)
    for model_type, gender in {(meta['native_body']['model'], meta['native_body']['gender'])
            for meta in train.scene_memory.entries}:
        if model_type == 'smpl':
            path = common.ROOT.parent / 'ProtoMotions/data/smpl' / ('SMPL_' + gender.upper() + '.pkl')
        elif model_type == 'smplx':
            path = common.ROOT / 'body_models/smplx' / ('SMPLX_' + gender.upper() + '.npz')
        else: raise ValueError('Unsupported TRAIN body model: ' + model_type)
        metadata[str(path.resolve())] = sha256(path)
    sources.update(metadata)
    out.mkdir(parents=True, exist_ok=True)
    parent_target = out / 'parent.pt'
    if not parent_target.exists(): os.link(parent_path, parent_target)
    require(sha256(parent_target) == parent_sha, 'Pinned parent differs')
    effective = parent.get('effective_config') or read(MAIN / 'prepared/effective_config.json')
    plan = dict(protocol=PROTOCOL, prepared_at=common.now(), config=config, training_config=training_config,
        seed=seed, parent_checkpoint=str(parent_target.resolve()), parent_source_checkpoint=str(parent_path),
        parent_checkpoint_sha256=parent_sha, parent_step=41000, parent_model_sha256=base_digest,
        arms=ARMS, attention_spec=attention_spec, initial_original_parameters=base_parameters,
        attention_parameters=attention_spec['parameters'], source_sha256=sources,
        parent_effective_data_sha256=effective['data_sha256'], sampling_plan=sampling,
        optimizer=dict(type='AdamW', lr=2e-5, warmup_steps=25, floor_fraction=.2,
            weight_decay=config['optimizer']['weight_decay'], clip_norm=config['optimizer']['clip_norm']),
        horizon=1500, pilot_stop=500, checkpoint_interval=100, batch_size=2,
        guidance_enabled=False, all_parameters_trainable=True, precision='FP32',
        reference_policy='original', optimizer_reset=True, parent_optimizer_reused=False,
        train_recordings=len(recordings(train)), validation_recordings=len(recordings(val)),
        train_validation_recording_overlap=[], TEST_used=False, automatic_checkpoint_promotion=False,
        checkpoint_policy='One atomic resumable last.pt per arm; pinned parent is a hard link.',
        interpretation='500 matched updates are a pilot trend, not convergence or final architecture acceptance.',
        validation_policy='Separate evaluator seals development and confirmation recording panels; no training-time checkpoint selection.')
    write(out / 'PLAN.json', plan)
    status(out, 'prepared_cpu_only', optimizer_updates=0, free_disk_before=shutil.disk_usage(BASE).free,
        parent_step=41000, pilot_stop=500, horizon=1500)
    require(not torch.cuda.is_initialized(), 'Preparation initialized CUDA')
    return plan


def verify(out):
    plan = read(out / 'PLAN.json')
    require(plan['protocol'] == PROTOCOL and plan['arms'] == ARMS, 'Stage protocol changed')
    for path, expected in plan['source_sha256'].items():
        require(sha256(path) == expected, 'Sealed dependency changed: ' + path)
    require(sha256(plan['parent_checkpoint']) == plan['parent_checkpoint_sha256'], 'Parent checkpoint changed')
    return plan


def lr_at(plan, step):
    options = plan['optimizer']
    return options['lr'] * min(1., step / options['warmup_steps']) * max(options['floor_fraction'],
        .5 * (1 + math.cos(math.pi * step / plan['horizon'])))


def trim_trace(path, committed_step):
    if not path.exists(): return
    raw = path.read_bytes(); kept, dropped = [], []
    for line in raw.splitlines(keepends=True):
        try: keep = line.endswith(b'\n') and json.loads(line)['step'] <= committed_step
        except (ValueError, KeyError): keep = False
        (kept if keep else dropped).append(line)
    if dropped:
        tail = b''.join(dropped)
        path.with_name(path.name + '.uncommitted.' + hashlib.sha256(tail).hexdigest()).write_bytes(tail)
        temporary = path.with_suffix('.trim.tmp'); temporary.write_bytes(b''.join(kept)); temporary.replace(path)
    parsed = [json.loads(line) for line in kept]
    require([r['step'] for r in parsed] == list(range(1, committed_step + 1)), 'Committed trace is incomplete')


def append(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())


def install_io(plan, out):
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    spec = importlib.util.spec_from_file_location('scene_attention_verified_io', IO_ADAPTER)
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    adapter.install(SimpleNamespace(trainer=trainer), dict(data_sha256=plan['parent_effective_data_sha256']),
        out / 'IO_RECOVERY.jsonl')


def preflight(out, memory_fraction=.65):
    """Real TRAIN-input zero-update GPU equivalence before either arm starts."""
    import numpy as np
    import torch
    from .scene_attention import attach_scene_attention, scene_attention_metadata
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, sample_item
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    plan = verify(out)
    require(0 < memory_fraction <= .8, 'Invalid GPU memory fraction')
    require(not any((out/arm/'training/last.pt').exists() for arm in ARMS), 'Preflight must precede training')
    with common.gpu_lock():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        install_io(plan,out)
        config=plan['config']; data=data_for(config,'train',plan['seed'])
        builder=common.observation_builder(config,'cuda'); rng=np.random.default_rng(plan['seed']+991)
        items, identities=zip(*(sample_item(data,builder,plan['training_config'],plan['sampling_plan'],rng) for _ in range(2)))
        require(all(i['split']=='train' and i['mode']=='ego' for i in identities), 'Non-TRAIN/EGO preflight')
        batch=collate_observations(items); known=model_inputs(config,batch)
        before=common.tensor_fingerprint(known)
        generator=torch.Generator(device='cuda').manual_seed(plan['seed'])
        noisy=torch.randn(known['reference_motion'].shape,device='cuda',generator=generator)
        times=torch.tensor([50,500],device='cuda',dtype=torch.long)
        parent=torch.load(plan['parent_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        expected={}; proof=[]
        for arm in ARMS:
            status(out,'zero_update_preflight',arm=arm,optimizer_updates=0)
            model=make_model(config);model.load_state_dict(parent['model'],strict=True)
            require(model_digest(model)==plan['parent_model_sha256'],'Preflight parent differs')
            torch.manual_seed(plan['seed'])
            if arm=='attention':
                attach_scene_attention(model)
                require(scene_attention_metadata(model)==plan['attention_spec'],'Preflight attention specification differs')
            model=model.cuda().eval();model.guidance_enabled=False
            initial=model_digest(model)
            for scene in (True,False):
                torch.manual_seed(plan['seed'])
                with torch.no_grad(): prediction=model(noisy,times,known,use_scene=scene).detach().cpu()
                require(bool(torch.isfinite(prediction).all()),'Nonfinite preflight output')
                require(before==common.tensor_fingerprint(known),'Preflight mutated inference inputs')
                if arm=='control': expected[scene]=prediction
                else:
                    delta=float((prediction-expected[scene]).abs().max())
                    require(torch.allclose(prediction,expected[scene],atol=1e-6,rtol=1e-6),'Zero-output attention changed parent output')
                    proof.append(dict(use_scene=scene,exact=torch.equal(prediction,expected[scene]),maximum_absolute_feature_delta=delta))
            require(model_digest(model)==initial,'Preflight mutated model tensors')
            del model;gc.collect();torch.cuda.empty_cache()
        verify(out)
        report=dict(status='passed_zero_update_actual_TRAIN_GPU',checked_at=common.now(),
            plan_sha256=sha256(out/'PLAN.json'),optimizer_updates=0,physics_OFF=True,
            real_train_identities=identities,common_input_sha256=before,zero_initialization_equivalence=proof,
            next_required_check='Both arms train --stop-step 2, then audit, before extending matched training.')
        write(out/'PREFLIGHT.json',report)
        status(out,'zero_update_preflight_passed',optimizer_updates=0,pilot_stop=500)
        return report


def train(out, arm, stop_step=500, resume=False, memory_fraction=.65):
    import numpy as np
    import torch
    from .scene_attention import attach_scene_attention, scene_attention_metadata
    from experiments.offline_camera_retrain_v1 import train_native30 as trainer
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, sample_item
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, training_motion
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
    plan = verify(out)
    check=read(out/'PREFLIGHT.json')
    require(check['status']=='passed_zero_update_actual_TRAIN_GPU'
        and check['plan_sha256']==sha256(out/'PLAN.json'),'Actual-input zero-update preflight is required')
    require(arm in ARMS and 0 < stop_step <= plan['horizon'], 'Invalid arm or stop boundary')
    require(0 < memory_fraction <= .8, 'Invalid GPU memory fraction')
    folder = out / arm / 'training'; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'last.pt'
    require(checkpoint.exists() == bool(resume), 'Checkpoint presence requires matching --resume')
    with (out / arm / 'job.lock').open('a') as lock, common.gpu_lock():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(out)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        install_io(plan, out / arm)
        config = plan['config']; options = plan['training_config']
        data = data_for(config, 'train', plan['seed']); builder = common.observation_builder(config, 'cuda')
        model = make_model(config)
        initial = torch.load(plan['parent_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(initial['model'], strict=True)
        require(trainer.model_digest(model) == plan['parent_model_sha256'], 'Strict parent tensor identity differs')
        coefficients = initial.get('effective_config', {}).get('observation_coefficients')
        if coefficients is None: coefficients = read(MAIN / 'prepared/effective_config.json')['observation_coefficients']
        del initial
        torch.manual_seed(plan['seed'])
        if arm == 'attention':
            attach_scene_attention(model)
            require(scene_attention_metadata(model) == plan['attention_spec'], 'Attention specification differs')
        model.guidance_enabled = False; model = model.cuda()
        require(all(p.requires_grad for p in model.parameters()), 'Every model parameter must be trainable')
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['optimizer']['lr'],
            weight_decay=plan['optimizer']['weight_decay'])
        rng = np.random.default_rng(plan['seed'] + 991); counts = {}; start = 0
        # Erase constructor/module-install RNG consumption in both arms.
        torch.manual_seed(plan['seed']); np.random.seed(plan['seed'])
        if resume:
            state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            require(state['stage_plan_sha256'] == sha256(out / 'PLAN.json')
                and state['architecture'] == ARMS[arm] and state['config'] == config
                and state['planned_steps'] == plan['horizon'], 'Resume checkpoint differs from stage')
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
            for entry in optimizer.state.values():
                for key, value in entry.items():
                    if torch.is_tensor(value): entry[key] = value.cuda()
            data.load_sampler_state(state['data_rng']); rng.bit_generator.state = state['condition_rng']
            torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
            np.random.set_state(state['numpy_rng']); start = state['step']; counts = state['counts']; del state
        require(start <= stop_step, 'Checkpoint is beyond requested stop boundary')
        for path in (folder / 'trace.jsonl', out / arm / 'input_trace.jsonl'): trim_trace(path, start)
        alphas = trainer.cosine_alphas().cuda(); began = time.monotonic(); current = start
        geometry_cache = weakref.WeakKeyDictionary()
        def geometry_digest(scene):
            if scene in geometry_cache: return geometry_cache[scene]
            if hasattr(scene, 'base'):
                value = dict(base=geometry_digest(scene.base), attrs={k:v for k,v in vars(scene).items() if isinstance(v,np.ndarray)})
            else:
                value = {k:getattr(scene,k) for k in ('static','dynamic','cameras','dynamic_owner_ids','dynamic_features','dynamic_valid') if hasattr(scene,k)}
            result = digest(value); geometry_cache[scene] = result; return result
        def commit(step):
            save_checkpoint(dict(config=config, effective_config=dict(config=config), step=step,
                planned_steps=plan['horizon'], parent_step=41000, architecture=ARMS[arm],
                attention_spec=plan['attention_spec'] if arm == 'attention' else None,
                stage_plan_sha256=sha256(out / 'PLAN.json'), source_sha256=plan['source_sha256'],
                model=model.state_dict(), optimizer=optimizer.state_dict(), counts=counts,
                data_rng=data.sampler_state(), condition_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state(),
                guidance_enabled=False, optimizer_reset_on_initial_start=True, TEST_used=False), checkpoint)
        if not resume:
            write(out / arm / 'INITIAL_STATE.json', dict(parent_model_sha256=plan['parent_model_sha256'],
                architecture=ARMS[arm], base_strict_load=True, optimizer_entries=len(optimizer.state),
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                common_rng_sha256=digest(dict(data=data.sampler_state(),condition=rng.bit_generator.state,
                    torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state()))))
            commit(0)
        status(out / arm, 'training', step=start, stop_step=stop_step, horizon=plan['horizon'])
        try:
            with trainer.StopAtUpdateBoundary() as stopping, (folder / 'trace.jsonl').open('a') as trace, (out / arm / 'input_trace.jsonl').open('a') as inputs:
                for step in range(start + 1, stop_step + 1):
                    started = time.monotonic()
                    pairs = [sample_item(data, builder, options, plan['sampling_plan'], rng) for _ in range(2)]
                    items, identities = zip(*pairs)
                    require(all(i['mode'] == 'ego' and i['split'] == 'train' for i in identities), 'Non-TRAIN/EGO sample')
                    batch = collate_observations(items); known = model_inputs(config, batch)
                    times = torch.randint(1000, (2,), device='cuda')
                    low = torch.rand(2, device='cuda') < config['noise']['extra_low_noise_probability']
                    times = torch.where(low, torch.randint(config['noise']['low_noise_steps'], times.shape, device='cuda'), times)
                    target = training_motion(config, batch, known); signal = alphas[times]
                    noise = torch.randn_like(target)
                    noisy = signal[:,None,None].sqrt() * target + (1-signal[:,None,None]).sqrt() * noise
                    scene_on = bool(rng.random() >= config['scene']['dropout_probability'])
                    audit = dict(step=step, identities=identities, scene=scene_on, timesteps=times.tolist(), lr=lr_at(plan,step),
                        known_sha256=common.tensor_fingerprint(known), target_sha256=digest(target), noise_sha256=digest(noise),
                        noisy_sha256=digest(noisy), supervision_sha256=digest(batch),
                        geometry_sha256=digest([geometry_digest(scene) for _,scene,_,_ in items]),
                        torch_rng_before_forward_sha256=digest(torch.get_rng_state()), cuda_rng_before_forward_sha256=digest(torch.cuda.get_rng_state()))
                    torch.cuda.synchronize(); preparation_s=time.monotonic()-started
                    compute_started=time.monotonic()
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    prediction, details = model(noisy, times, known, use_scene=scene_on, return_details=True)
                    loss, terms = trainer.objective(prediction, batch['motion'], batch, signal, details, config, coefficients)
                    require(bool(torch.isfinite(loss)), 'Nonfinite stage objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['optimizer']['clip_norm'], error_if_nonfinite=True)
                    for group in optimizer.param_groups: group['lr'] = audit['lr']
                    optimizer.step(); torch.cuda.synchronize()
                    compute_s=time.monotonic()-compute_started
                    for identity in identities:
                        key = identity['group'] + '/ego'; counts[key] = counts.get(key,0) + 1
                    audit['cuda_rng_after_update_sha256'] = digest(torch.cuda.get_rng_state())
                    audit['torch_rng_after_update_sha256'] = digest(torch.get_rng_state())
                    append(inputs, audit)
                    append(trace, dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), lr=audit['lr'],
                        scene=scene_on, preparation_s=preparation_s, compute_s=compute_s,
                        update_s=time.monotonic()-started, elapsed_s=time.monotonic()-began,
                        terms={k:{n:float(v.detach()) for n,v in group.items()} for k,group in terms.items()},
                        support_cache_hits=getattr(getattr(data,'support_cache',None),'hits',0),
                        support_cache_misses=getattr(getattr(data,'support_cache',None),'misses',0)))
                    current = step
                    if step % plan['checkpoint_interval'] == 0 or step == stop_step or stopping.requested: commit(step)
                    status(out / arm, 'training', step=step, stop_step=stop_step, horizon=plan['horizon'],
                        last_loss=float(loss.detach()), last_update_s=time.monotonic()-started)
                    if stopping.requested: break
            verify(out)
            status(out / arm, 'completed_pilot' if current == stop_step else 'stopped', step=current,
                stop_step=stop_step, horizon=plan['horizon'], checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=sha256(checkpoint), free_disk_after=shutil.disk_usage(BASE).free)
        except BaseException as exc:
            status(out / arm, 'failed', step=current, error=repr(exc), last_checkpoint_retained=checkpoint.exists())
            raise


def audit(out):
    import torch
    common.cpu_setup(); plan = verify(out); states, inputs = {}, {}
    for arm in ARMS:
        state = torch.load(out / arm / 'training/last.pt', map_location='cpu', mmap=True, weights_only=False)
        require(state['architecture'] == ARMS[arm] and state['stage_plan_sha256'] == sha256(out / 'PLAN.json')
            and state['parent_step'] == 41000 and state['guidance_enabled'] is False, 'Stage checkpoint identity differs')
        inputs[arm] = [json.loads(line) for line in (out / arm / 'input_trace.jsonl').read_text().splitlines() if line]
        require([r['step'] for r in inputs[arm]] == list(range(1,state['step']+1)), 'Uncommitted or incomplete input trace')
        trace = [json.loads(line) for line in (out / arm / 'training/trace.jsonl').read_text().splitlines() if line]
        require(len(trace) == state['step'] and all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in trace), 'Invalid training trace')
        require(all(math.isclose(r['lr'],lr_at(plan,r['step']),rel_tol=1e-12) for r in trace), 'Stage LR clock differs')
        require(all(bool(torch.isfinite(v).all()) for v in state['model'].values()), 'Nonfinite saved model')
        states[arm] = state
    require(states['control']['step'] == states['attention']['step'], 'Matched arm budgets differ')
    require(inputs['control'] == inputs['attention'], 'Actual samples/noise/dropout/geometry/RNG differ between arms')
    initials = [read(out / arm / 'INITIAL_STATE.json') for arm in ARMS]
    require(initials[0]['common_rng_sha256'] == initials[1]['common_rng_sha256'], 'Initial sampling RNG differs')
    for name in ('counts','data_rng','condition_rng','torch_rng','cuda_rng','numpy_rng'):
        require(digest(states['control'][name]) == digest(states['attention'][name]), 'Final paired state differs: '+name)
    require(not torch.cuda.is_initialized(), 'CPU pair audit initialized CUDA')
    report = dict(status='passed', checked_at=common.now(), updates_per_arm=states['control']['step'],
        parent_step=41000, plan_sha256=sha256(out/'PLAN.json'), physics_OFF_both=True,
        actual_samples_reference_supervision_noise_timesteps_dropout_geometry_RNG_exact=True,
        common_initial_and_final_rng_exact=True, optimizer_reset_both=True, all_parameters_trainable=True,
        horizon=1500, independent_validation_required=True, TEST_used=False, CUDA_initialized=False,
        checkpoint_sha256={arm:sha256(out/arm/'training/last.pt') for arm in ARMS})
    write(out/'MATCHED_AUDIT.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','preflight','train','audit'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--stop-step', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2026101841)
    parser.add_argument('--memory-fraction', type=float, default=.65)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if args.command == 'prepare': prepare(out,args.seed)
    elif args.command == 'preflight': print(json.dumps(preflight(out,args.memory_fraction),indent=2))
    elif args.command == 'audit': print(json.dumps(audit(out),indent=2))
    else:
        require(args.arm is not None, 'Training requires an explicit arm')
        train(out,args.arm,args.stop_step,args.resume,args.memory_fraction)


if __name__ == '__main__':
    main()
