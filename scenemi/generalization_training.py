"""Resumable, measured-budget A/B/C fine-tuning of the existing EGO model.

Run through ``python -m scenemi resources run`` for CPU/RAM isolation. This
module adds orchestration and supervision, not a new denoiser or sampler.
"""
import argparse
from collections import defaultdict
import copy
import csv
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tarfile
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.checkpoint import checkpoint, load_checkpoint, load_trained_model
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_geometry_cache import PreparedAmass
from scenemi.data.ego import FourEgoData, SOURCE_ORDER, collate, payload, source_cases
from scenemi.native_joint_data import SceneFreeEgoDataset
from scenemi.data.checked_scene_dataset import verify_mapped_file
from scenemi.pipeline import native_motion_loss
from scenemi.observations import public_observations
from scenemi.prefetch import OrderedPrefetch
from scenemi.training import deterministic, ensure_bounded, now, rng_state, restore_rng, tensor_digest
from scenemi.generalization_losses import shape_objective, focused_objective
from scenemi.generalization_evaluation import evaluate
from scenemi.monitoring import TrainingReporter
from scenemi.heldout_evaluation import trained_state, data_for as rich_data_for
from scenemi.gpu_policy import configure_cuda

CONFIG = ROOT/'configs/ego_generalization.yaml'
STAGES = ('A', 'B', 'C')
NAMES = dict(A='shape_ft', B='control_ft', C='focused_ft')
STOP = False


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def acceptance_signature(c):
    files = ('scenemi/generalization_training.py', 'scenemi/generalization_losses.py',
             'scenemi/generalization_evaluation.py')
    return dict(config_sha256=json_hash(c), sources={name:sha256(ROOT/name) for name in files})


def load_config(file):
    c = yaml.safe_load(Path(file).read_text())
    if c['version'] != 'ego-generalization-abc-v1':
        raise ValueError('Unsupported experiment protocol')
    if c['training']['effective_batch_size'] != 128 or c['training']['source_quota'] != 32:
        raise ValueError('This experiment requires exactly 32 samples per source, batch 128')
    if sum((c['budget']['phase_a_fraction'], 2*c['budget']['phase_bc_fraction_each'], c['budget']['overhead_fraction'])) > 1.000001:
        raise ValueError('Budget fractions exceed the declared wall time')
    return c


class VerifiedCameraDataset(SceneFreeEgoDataset):
    """Read the existing verified Linux mirror without resolving DrvFS paths.

    The retained reader already prefers these content-addressed bytes. Avoid
    its repeated source Path.resolve() across the Windows mount, retaining
    exactly the same digest check and bounded mapping cache.
    """
    def array(self, meta, name):
        key = meta['path'], name
        if key not in self.array_cache:
            filename = name+'.npy'
            source = str(meta.get('array_sources', {}).get(filename, str(Path(meta['path'])/filename)))
            expected = meta['output_sha256'][filename]
            mirror = Path(self.paths['verified_cache'])/(expected+'.npy')
            if source.startswith('/mnt/') and mirror.exists():
                stamp = mirror.stat()
                verify_mapped_file(str(mirror), stamp.st_size, stamp.st_mtime_ns, stamp.st_ino, expected)
                self.array_cache[key] = np.load(mirror, mmap_mode='r', allow_pickle=False)
                while len(self.array_cache) > 64:
                    self.array_cache.popitem(last=False)
            else:
                return super().array(meta, name)
        self.array_cache.move_to_end(key)
        return self.array_cache[key]


class CameraData(FourEgoData):
    """Original sampling and membership, with the verified AMASS Linux cache.

    Native sources use CheckedSceneDataset's existing verified array reader;
    scene overlays, meshes, normals and occupancy are never loaded.
    """
    def __init__(self, c, paths, split, common):
        reference = yaml.safe_load((ROOT/c['reference_data_config']).read_text())
        self.split = split
        original = AmassDataset(ROOT/reference['amass_index'], paths, 'train' if split == 'train' else 'val')
        self.amass = PreparedAmass(original, ROOT/c['amass_cache'])
        self.native = VerifiedCameraDataset(ROOT/reference['scene_paths'], common,
            split=split, config=reference, selection=ROOT/c['selection'])
        self.datasets = dict(amass=self.amass, egobody=self.native, trumans=self.native, grail=self.native)

    def pairs(self, rng, quota=32):
        return [self.sample(source, rng) for source in SOURCE_ORDER for _ in range(quota)]


def configure_stage(model, stage, c):
    model.requires_grad_(False)
    prefixes = ('shape_encoder', 'native_body.coefficients', 'native_body.templates') if stage == 'A' else tuple(c['motion_prefixes'])
    selected = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix+'.') for prefix in prefixes):
            parameter.requires_grad_(True)
            selected.append(name)
    if not selected or any(not any(n == p or n.startswith(p+'.') for n in selected) for p in prefixes):
        raise ValueError(f'Trainable prefix was not found: {prefixes}')
    return dict(prefixes=list(prefixes), names=selected,
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def training_mode(model, stage):
    if stage == 'A':
        model.eval()
        model.shape_encoder.train()
        model.native_body.coefficients.train()
        model.native_body.templates.train()
    else:
        # Preserve the existing EGO trainer's conditioning dropout semantics.
        model.train()
        model.shape_encoder.eval()
        model.native_body.eval()


def optimizer_for(model, c, stage):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=c['optimizer']['shape_lr' if stage == 'A' else 'motion_lr'], weight_decay=c['optimizer']['weight_decay'])


def update(model, optimizer, pairs, c, loss_config, device, stage, step, planned):
    training_mode(model, stage)
    optimizer.zero_grad(set_to_none=True)
    total, terms_sum = 0., defaultdict(float)
    stream = hashlib.sha256()
    stream.update(json.dumps([x[1] for x in pairs], sort_keys=True).encode())
    micro = c['training']['microbatch_size']
    for first in range(0, len(pairs), micro):
        part = pairs[first:first+micro]
        batch = collate([x[0] for x in part], device)
        fraction = len(part)/len(pairs)
        if stage == 'A':
            details = model.predicted_body(public_observations(payload(batch)), batch['body_format'])
            terms = shape_objective(details, batch, c['shape_loss'])
        else:
            timestep = torch.randint(0, 1000, (len(part),), device=device)
            alpha = model.alphas[timestep]
            noise = torch.randn_like(batch['motion'])
            stream.update(timestep.detach().cpu().numpy().tobytes())
            stream.update(noise.detach().cpu().numpy().tobytes())
            noisy = alpha[:, None, None].sqrt()*batch['motion']+(1-alpha[:, None, None]).sqrt()*noise
            prediction, details = model(noisy, timestep, payload(batch), use_scene=False, return_details=True)
            terms = native_motion_loss(prediction, batch['motion'], batch, alpha.sqrt(), details=details, config=loss_config)
            if stage == 'C':
                focused = focused_objective(prediction, batch, details, alpha, timestep, c['focused_loss'],
                                            loss_config['loss']['coefficients']['root_orientation'])
                terms['total'] = terms['total']+focused['focused_increment']
                terms.update(focused)
        if not torch.isfinite(terms['total']):
            raise FloatingPointError(f'Nonfinite {stage} loss at {step}')
        (terms['total']*fraction).backward()
        for key, value in terms.items():
            terms_sum[key] += float(value.detach())*fraction
        total += float(terms['total'].detach())*fraction
    trainable = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, c['optimizer']['clip_norm'], error_if_nonfinite=True)
    if not float(norm) > 0:
        raise FloatingPointError(f'No trainable gradient in stage {stage}')
    factor = min(1., step/max(1, min(c['optimizer']['warmup_updates'], planned//10)))
    floor = c['optimizer']['lr_floor_fraction']
    factor *= floor+(1-floor)*.5*(1+math.cos(math.pi*min(1., step/planned)))
    lr = c['optimizer']['shape_lr' if stage == 'A' else 'motion_lr']*factor
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return dict(step=step, total=total, terms=dict(terms_sum), gradient_norm=float(norm), lr=lr,
                data_noise_sha256=stream.hexdigest(), source_counts={s:sum(x[1]['dataset'] == s for x in pairs) for s in SOURCE_ORDER})


def load_model(file, paths, template=None):
    if template is None:
        return load_trained_model(file, paths)[0]
    # 26k has extra scene/skin buffers. Every base-model tensor must exist and
    # have the same shape. Extra names are recorded, not silently loaded.
    state = torch.load(file, map_location='cpu', mmap=True, weights_only=False)
    keys = template.state_dict().keys()
    missing = set(keys)-set(state['model'])
    if missing:
        raise ValueError(f'Comparison checkpoint lacks base tensors: {sorted(missing)}')
    template.load_state_dict({k:state['model'][k] for k in keys}, strict=True)
    return template.requires_grad_(False).eval()


def verify_inputs(c):
    if sha256(ROOT/c['initialization']['checkpoint']) != c['initialization']['sha256']:
        raise ValueError('22k initialization changed')
    check = json.loads((ROOT/c['selection_check']).read_text())
    if check['state'] != 'passed' or check['selection_sha256'] != sha256(ROOT/c['selection']):
        raise ValueError('Current four-source selection was not accepted')
    for file, digest in check['dependencies_sha256'].items():
        if sha256(file) != digest:
            raise ValueError(f'Selection dependency changed: {file}')
    protocol = yaml.safe_load((ROOT/c['evaluation']['rich_protocol']).read_text())
    evidence = {}
    for key in ('reference', 'candidate'):
        state = trained_state(protocol, key)
        if key == 'reference' and any(c['model'][k] != state['contract']['config']['model'][k] for k in ('latent_dim', 'dim_mults')):
            raise ValueError('Experiment model dimensions differ from the retained 22k model')
        evidence[key] = dict(checkpoint=protocol[key]['checkpoint'], sha256=sha256(ROOT/protocol[key]['checkpoint']),
                             source_migrations=state.get('_inference_source_migrations', {}))
        del state
    return evidence


def contract(c, config_file, evidence):
    # Actual imported first-party closure, supplemented by declared dynamic
    # evaluation/config/data entries. No blanket repository glob.
    files = {Path(__file__).resolve(), Path(config_file).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and name.endswith('.py') and Path(name).is_absolute():
            file = Path(name).resolve()
            if file.is_file() and (file.is_relative_to(ROOT) or file.is_relative_to(ROOT.parent/'diffusion-motion-inbetweening')) and not file.is_relative_to(ROOT/'tests'):
                files.add(file)
    reference = yaml.safe_load((ROOT/c['reference_data_config']).read_text())
    for name in ('reference_data_config', 'reference_loss_config', 'selection', 'selection_check', 'validation_manifest'):
        files.add(ROOT/c[name])
    files.update(ROOT/reference[k] for k in ('scene_paths', 'amass_index'))
    files.add(ROOT/'configs/paths.local.yaml')
    files.add(ROOT/'configs/runtime_policy.yaml')
    files.add(ROOT/c['evaluation']['rich_protocol'])
    check = json.loads((ROOT/c['selection_check']).read_text())
    files.update(Path(p) for p in check['dependencies_sha256'])
    value = dict(version=c['version'], config=c, parent_audit=evidence, test_used=False,
        source_sha256={str(p):sha256(p) for p in sorted(files)},
        environment={k:importlib.metadata.version(k) for k in ('torch', 'numpy', 'scipy', 'einops')})
    value['sha256'] = json_hash(value)
    return value


def verify_contract(seal, c):
    if seal['config'] != c:
        raise ValueError('Cannot resume with a changed experiment configuration')
    for name, digest in seal['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError(f'Sealed dependency changed: {name}')
    if seal['sha256'] != json_hash({k:v for k,v in seal.items() if k != 'sha256'}):
        raise ValueError('Invalid experiment contract')


def fixed_cases(c, validation):
    existing = json.loads((ROOT/c['validation_manifest']).read_text())['full']
    current = source_cases(validation)
    if existing != current:
        raise ValueError('Current four-source validation membership differs from the retained manifest')
    return existing


def preserve_source(out, seal):
    with tarfile.open(out/'SOURCE_SNAPSHOT.tar.gz', 'w:gz') as archive:
        for name in seal['source_sha256']:
            p = Path(name)
            if p.suffix in ('.py', '.yaml', '.json') and p.stat().st_size < 8*2**20:
                archive.add(p, arcname=str(p).lstrip('/'), recursive=False)


def preflight(c, paths, device, config_file=CONFIG):
    """Real four-source gradients + exact next-update resume in all stages."""
    ensure_bounded()
    if device == 'cuda':
        configure_cuda()
    deterministic(c['seed'], device)
    out = ROOT/c['output']/'preflight'/device
    out.mkdir(parents=True, exist_ok=True)
    evidence = verify_inputs(c)
    loss_config = yaml.safe_load((ROOT/c['reference_loss_config']).read_text())
    model = load_model(ROOT/c['initialization']['checkpoint'], paths).to(device)
    common = model.shape.rest_template[:1].detach().cpu().numpy()
    data = CameraData(c, paths, 'train', common)
    validation = CameraData(c, paths, 'validation', common)
    cases = fixed_cases(c, validation)
    parity = {}
    # Independently replay each native source through the retained reader.
    # Labels, public inputs and identities must be bitwise identical.
    for source in SOURCE_ORDER[1:]:
        rng = np.random.default_rng(c['seed'])
        rng0 = copy.deepcopy(rng.bit_generator.state)
        native = data.native
        original_array = native.array
        expected = data.sample(source, rng)
        native.array = lambda meta, name:SceneFreeEgoDataset.array(native, meta, name)
        native.array_cache.clear()
        rng.bit_generator.state = rng0
        try:
            actual = data.sample(source, rng)
        finally:
            native.array = original_array
            native.array_cache.clear()
        if expected[1] != actual[1] or any(not np.array_equal(expected[0][k], actual[0][k]) for k in expected[0]):
            raise AssertionError(f'Linux mirror adapter changed {source} sampling')
        parity[source] = True
    print(json.dumps(dict(event='preflight_data_ready', device=device, windows=len(cases))), flush=True)
    results = {}
    state0 = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for stage in STAGES:
        model.load_state_dict(state0)
        deterministic(c['seed'], device)
        selection = configure_stage(model, stage, c)
        optimizer = optimizer_for(model, c, stage)
        rng = np.random.default_rng(c['seed'])
        quota = 1 if device == 'cpu' else c['training']['source_quota']
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        start = time.monotonic()
        with OrderedPrefetch(lambda r:data.pairs(r, quota), rng, depth=2) as batches:
            first = update(model, optimizer, batches.next()[0], c, loss_config, device, stage, 1, 1000)
            seal = dict(preflight=True, config=c, stage=stage)
            checkpoint(model, optimizer, rng, device, 1, seal, out/'resume.pt')
            expected = update(model, optimizer, batches.next()[0], c, loss_config, device, stage, 2, 1000)
            digest = tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device)))
        load_checkpoint(model, optimizer, rng, device, seal, out/'resume.pt')
        with OrderedPrefetch(lambda r:data.pairs(r, quota), rng, depth=2) as batches:
            actual = update(model, optimizer, batches.next()[0], c, loss_config, device, stage, 2, 1000)
        if actual != expected or tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device))) != digest:
            raise AssertionError(f'Exact prefetched optimizer/RNG resume failed in {stage}')
        if any(not torch.equal(p, frozen[n]) for n,p in model.named_parameters() if not p.requires_grad):
            raise AssertionError(f'Frozen weights changed in {stage}')
        results[stage] = dict(trainable=selection, first_update=first, exact_resume=True,
                              frozen_unchanged=True, batch_size=4*quota, seconds=time.monotonic()-start)
        print(json.dumps(dict(event='preflight_stage_passed', stage=stage, device=device,
                              seconds=results[stage]['seconds'], parameters=selection['parameters'])), flush=True)
        del frozen, optimizer
        gc.collect()
        if stage == 'C':
            # Selected checkpoints remain compatible with the ordinary public
            # inference loader, including its strict tensor namespace check.
            reloaded, _, saved_step = load_trained_model(out/'resume.pt', paths)
            saved = torch.load(out/'resume.pt', map_location='cpu', mmap=True, weights_only=False)
            if saved_step != 1 or tensor_digest(reloaded.state_dict()) != tensor_digest(saved['model']):
                raise AssertionError('New checkpoint cannot be restored by the existing inference loader')
            del saved, reloaded
        (out/'resume.pt').unlink()
    if results['B']['first_update']['data_noise_sha256'] != results['C']['first_update']['data_noise_sha256']:
        raise AssertionError('Control/focused preflight data or diffusion noise differ')
    model.load_state_dict(state0)
    # One fixed window per source checks shape decoding, sampling, metrics and
    # the same camera readout used by all final comparison models.
    small = [next(x for x in cases if x['dataset'] == source) for source in SOURCE_ORDER]
    eval_result = evaluate(model, validation, small, c, device, out/'evaluation')
    seal = contract(c, config_file, evidence)
    verify_contract(seal, c)
    result = dict(state='passed', device=device, at=now(), stages=results,
                  evaluation=eval_result, validation_windows=len(cases), rich_arrays_opened=False,
                  acceptance_signature=acceptance_signature(c), existing_inference_loader_passed=True,
                  bc_first_batch_data_noise_equal=True, native_mirror_bitwise_parity=parity,
                  dependency_contract_passed=True, dependency_contract_sha256=seal['sha256'])
    atomic_json(out/'RESULT.json', result)
    return result


def benchmark(c, paths, data, out, device):
    timings = {}
    for stage in STAGES:
        deterministic(c['seed'], device)
        model = load_model(ROOT/c['initialization']['checkpoint'], paths).to(device)
        trainable = configure_stage(model, stage, c)
        optimizer = optimizer_for(model, c, stage)
        rng = np.random.default_rng(c['seed'])
        loss_config = yaml.safe_load((ROOT/c['reference_loss_config']).read_text())
        durations, waits = [], []
        count = c['budget']['warmup_updates']+c['budget']['measured_updates']
        with OrderedPrefetch(data.pairs, rng, c['training']['prefetch_depth']) as batches:
            for step in range(1, count+1):
                start = time.monotonic()
                pairs, timing = batches.next()
                result = update(model, optimizer, pairs, c, loss_config, device, stage, step, 1000)
                if device == 'cuda':
                    torch.cuda.synchronize()
                if step > c['budget']['warmup_updates']:
                    durations.append(time.monotonic()-start)
                    waits.append(timing['data_wait_seconds'])
        timings[stage] = dict(mean_seconds=float(np.mean(durations)), median_seconds=float(np.median(durations)),
                              max_seconds=max(durations), mean_data_wait_seconds=float(np.mean(waits)), trainable=trainable,
                              real_batch_size=128, final_update=result)
        print(json.dumps(dict(event='measured_speed', stage=stage, seconds=timings[stage]['mean_seconds'])), flush=True)
        del model, optimizer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    seconds = c['budget']['hours']*3600
    safety = c['budget']['timing_safety_factor']
    a = max(1, int(seconds*c['budget']['phase_a_fraction']/(timings['A']['mean_seconds']*safety)))
    bc = max(1, int(seconds*c['budget']['phase_bc_fraction_each']/(max(timings[s]['mean_seconds'] for s in ('B','C'))*safety)))
    plan = dict(state='fixed_before_training', measured_at=now(), hours=c['budget']['hours'], timings=timings,
                steps=dict(A=a, B=bc, C=bc), benchmark_updates_discarded=True,
                reserved_overhead_seconds=seconds*c['budget']['overhead_fraction'],
                estimated_training_seconds=a*timings['A']['mean_seconds']+bc*sum(timings[s]['mean_seconds'] for s in ('B','C')))
    atomic_json(out/'BUDGET.json', plan)
    return plan


def validation_preserving_rng(model, data, cases, c, device, output, rng, shape_only=False):
    state = rng_state(rng, device)
    try:
        return evaluate(model, data, cases, c, device, output, shape_only=shape_only)
    finally:
        restore_rng(state, rng, device)


def trim_log(file, step):
    if file.exists():
        lines = file.read_text().splitlines()
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise
                # A SIGKILL can leave an uncommitted final log line. Model,
                # optimizer and RNG are restored from the atomic checkpoint.
                break
            rows.append(row)
        file.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows if r['step'] <= step))


def restore_selection(folder, name):
    metadata, best = folder/'SELECTION.json', folder/(name+'.pt')
    if not best.exists():
        if metadata.exists():
            raise ValueError('Selection metadata exists without its checkpoint')
        return None
    state = torch.load(best, map_location='cpu', weights_only=False, mmap=True)
    step = state['step']
    metric_file = folder/f'validation_step_{step:06d}'/'RESULT.json'
    metric = json.loads(metric_file.read_text())
    selection = dict(step=step, selection_score_mm=metric['selection_score_mm'], metric=str(metric_file))
    del state
    # Repair a crash between the atomic selected-weight and metadata writes.
    atomic_json(metadata, selection)
    return selection


def run_stage(stage, parent, c, paths, data, validation, cases, plan, seal, out, device, resume):
    global STOP
    folder = out/NAMES[stage]
    folder.mkdir(parents=True, exist_ok=True)
    done = folder/'COMPLETE.json'
    if done.exists():
        record = json.loads(done.read_text())
        if sha256(folder/(NAMES[stage]+'.pt')) != record['selected_sha256'] or record['parent_sha256'] != sha256(parent):
            raise ValueError(f'Completed stage {stage} identity differs')
        return folder/(NAMES[stage]+'.pt')
    deterministic(c['seed']+(0 if stage == 'A' else 1), device)
    model = load_model(ROOT/c['initialization']['checkpoint'], paths)
    model = load_model(parent, paths, template=model).to(device)
    selected = configure_stage(model, stage, c)
    optimizer = optimizer_for(model, c, stage)
    rng = np.random.default_rng(c['seed']+(0 if stage == 'A' else 1))
    steps = plan['steps'][stage]
    stage_seal = dict(seal, stage=stage, parent_sha256=sha256(parent), planned_steps=steps, trainable=selected)
    atomic_json(folder/'CONTRACT.json', stage_seal)
    last, best = folder/'last.pt', folder/(NAMES[stage]+'.pt')
    initial = load_checkpoint(model, optimizer, rng, device, stage_seal, last) if last.exists() and resume else 0
    if last.exists() and not resume:
        raise FileExistsError(f'{last} exists; use --resume')
    log = folder/'updates.jsonl'
    trim_log(log, initial)
    metadata = folder/'SELECTION.json'
    selection = restore_selection(folder, NAMES[stage])
    loss_config = yaml.safe_load((ROOT/c['reference_loss_config']).read_text())
    reporting = copy.deepcopy(c)
    reporting['training']['planned_updates'] = steps
    reporter = TrainingReporter(folder, reporting, initial)
    reporter.progress.mininterval = 2
    checks = sorted({max(1, round(steps*i/c['training']['validation_checks'])) for i in range(1, c['training']['validation_checks']+1)})
    last_save = time.monotonic()
    started = time.monotonic()
    print(json.dumps(dict(event='stage_started', stage=stage, initial_step=initial, planned_steps=steps,
                          parent_sha256=stage_seal['parent_sha256'], trainable_parameters=selected['parameters'], at=now())), flush=True)
    try:
        with log.open('a', buffering=1) as output, OrderedPrefetch(data.pairs, rng, c['training']['prefetch_depth']) as batches:
            # Include A's exact initialization as a selection candidate. This
            # prevents shape-only overfitting from being forcibly propagated.
            if selection is None:
                metric = validation_preserving_rng(model, validation, cases, c, device, folder/'validation_step_000000', rng, stage == 'A')
                checkpoint(model, optimizer, rng, device, initial, stage_seal, best)
                selection = dict(step=initial, selection_score_mm=metric['selection_score_mm'], metric=str(folder/'validation_step_000000/RESULT.json'))
                atomic_json(metadata, selection)
            for step in range(initial+1, steps+1):
                if STOP:
                    checkpoint(model, optimizer, rng, device, step-1, stage_seal, last)
                    raise InterruptedError('Stop requested; checkpoint committed, dependents not started')
                start = time.monotonic()
                pairs, timing = batches.next()
                record = update(model, optimizer, pairs, c, loss_config, device, stage, step, steps)
                elapsed = time.monotonic()-start
                record.update(timing, seconds=elapsed, at=now())
                output.write(json.dumps(record, allow_nan=False)+'\n')
                reporter.update(record, elapsed)
                if step == initial+1 or step % 100 == 0:
                    atomic_json(folder/'PROGRESS.json', dict(stage=stage, step=step, planned_steps=steps, loss=record['total'],
                        seconds_per_step=elapsed, at=now(), source_counts=record['source_counts']))
                if time.monotonic()-last_save >= c['training']['checkpoint_minutes']*60 or step in checks or step == initial+1:
                    checkpoint(model, optimizer, rng, device, step, stage_seal, last)
                    last_save = time.monotonic()
                if step in checks:
                    metric_path = folder/f'validation_step_{step:06d}'
                    metric = validation_preserving_rng(model, validation, cases, c, device, metric_path, rng, stage == 'A')
                    reporter.validation(metric, step, 'shape' if stage == 'A' else 'heldout_B')
                    if metric['selection_score_mm'] < selection['selection_score_mm']:
                        checkpoint(model, optimizer, rng, device, step, stage_seal, best)
                        selection = dict(step=step, selection_score_mm=metric['selection_score_mm'], metric=str(metric_path/'RESULT.json'))
                        atomic_json(metadata, selection)
        atomic_json(done, dict(state='complete', stage=stage, final_step=steps, selected=selection,
            selected_sha256=sha256(best), parent_sha256=stage_seal['parent_sha256'], elapsed_seconds=time.monotonic()-started, at=now()))
    finally:
        reporter.close()
        del model, optimizer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    return best


def fairness(out):
    b = out/NAMES['B']
    c = out/NAMES['C']
    bc = [json.loads((f/'CONTRACT.json').read_text()) for f in (b,c)]
    if any(bc[0][k] != bc[1][k] for k in ('parent_sha256','planned_steps','trainable')):
        raise AssertionError('B/C initialization, budget or trainable layers differ')
    count = 0
    with (b/'updates.jsonl').open() as fb, (c/'updates.jsonl').open() as fc:
        for left, right in zip(fb, fc, strict=True):
            l, r = json.loads(left), json.loads(right)
            if any(l[k] != r[k] for k in ('step','data_noise_sha256','source_counts','lr')):
                raise AssertionError(f'B/C random streams or optimizer schedule differ at {l["step"]}')
            count += 1
    return dict(state='passed', identical_a_checkpoint=True, data_and_noise_steps_verified=count,
                same_trainable_layers=True, same_optimizer_schedule=True)


def final_evaluations(c, paths, validation, cases, selected, out, device):
    lock_file = out/'FINAL_SELECTION_LOCK.json'
    locked = dict(config_sha256=json_hash(c), selected={k:dict(path=str(v), sha256=sha256(v)) for k,v in selected.items()},
                  rich_not_used_for_selection=True)
    if lock_file.exists() and json.loads(lock_file.read_text()) != locked:
        raise ValueError('Final selection lock differs')
    atomic_json(lock_file, locked)
    files = {**c['comparisons'], **{NAMES[k]:str(v) for k,v in selected.items()}}
    results = {}
    for label, file in files.items():
        destination = out/'comparison'/str(label)
        if (destination/'RESULT.json').exists():
            results[str(label)] = json.loads((destination/'RESULT.json').read_text())
            continue
        model = load_model(ROOT/c['initialization']['checkpoint'], paths)
        model = load_model(ROOT/file, paths, template=model).to(device)
        results[str(label)] = evaluate(model, validation, cases, c, device, destination)
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    audit = fairness(out)
    atomic_json(out/'FAIRNESS.json', audit)
    # Configuration and checkpoint selection are now immutable. RICH is opened
    # only after all five four-source evaluations are complete.
    rich_results = {}
    if c['evaluation']['rich_after_selection']:
        protocol = yaml.safe_load((ROOT/c['evaluation']['rich_protocol']).read_text())
        data, _ = rich_data_for(protocol)
        rich_cases = data.native.cases
        for label, file in files.items():
            destination = out/'rich_diagnostic'/str(label)
            if (destination/'RESULT.json').exists():
                rich_results[str(label)] = json.loads((destination/'RESULT.json').read_text())
                continue
            model = load_model(ROOT/c['initialization']['checkpoint'], paths)
            model = load_model(ROOT/file, paths, template=model).to(device)
            rich_results[str(label)] = evaluate(model, data, rich_cases, c, device, destination, rich=True)
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    b, f = [results[NAMES[s]]['dataset_macro_mean'] for s in ('B','C')]
    assessment = dict(focused_beats_control=f['mpjpe_mm'] < b['mpjpe_mm'],
        mpjpe_delta_c_minus_b_mm=f['mpjpe_mm']-b['mpjpe_mm'],
        head_guard_passed=f['head_error_mm'] <= b['head_error_mm']+c['evaluation']['head_regression_tolerance_mm'],
        jerk_guard_passed=f['jerk_m_s3'] <= b['jerk_m_s3']*c['evaluation']['jerk_ratio_max'])
    assessment['success'] = all(assessment[k] for k in ('focused_beats_control','head_guard_passed','jerk_guard_passed'))
    report = dict(state='complete', at=now(), assessment=assessment, fairness=audit,
                  validation=results, rich_diagnostic=rich_results, selection=locked)
    atomic_json(out/'REPORT.json', report)
    fields = list(next(iter(results.values()))['dataset_macro_mean'])
    with (out/'COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model']+fields)
        writer.writeheader()
        for label, result in results.items():
            writer.writerow(dict(model=label, **result['dataset_macro_mean']))
    lines = ['# Camera-only generalization A/B/C', '',
        'All selection uses equal-source/equal-family four-dataset validation. All motion results use the unchanged DDIM sampler and temporal B anchoring.', '',
        '| Model | MPJPE mm | PA-MPJPE mm | Root-relative mm | Root deg | Head mm | Lower mm | Jerk m/s³ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, result in results.items():
        m = result['dataset_macro_mean']
        keys = ('mpjpe_mm','pa_mpjpe_mm','root_relative_mpjpe_mm','root_orientation_deg','head_error_mm','lower_body_mpjpe_mm','jerk_m_s3')
        lines.append('| '+label+' | '+' | '.join(f'{m[k]:.3f}' for k in keys)+' |')
    lines += ['', 'Assessment: '+json.dumps(assessment), '',
        'Shape height is head-to-mean-foot rest-skeleton height, not mesh height. Phase C adds GT velocity matching, not pure smoothing. No GT shape or orientation enters inference.',
        'RICH is a separate post-selection diagnostic. No automatic hyperparameter feedback is implemented. Full metrics and per-window records accompany this report.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def run(c, paths, config_file, device, resume):
    ensure_bounded()
    if device != 'cuda':
        raise ValueError('Formal one-day training requires CUDA; use preflight for CPU')
    configure_cuda()
    out = ROOT/c['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal_file = out/'CONTRACT.json'
        if seal_file.exists() and not resume:
            raise FileExistsError('Experiment already prepared; use --resume')
        for mode in ('cpu','cuda'):
            result = json.loads((out/'preflight'/mode/'RESULT.json').read_text())
            if result['state'] != 'passed':
                raise RuntimeError(f'{mode} preflight did not pass')
            if result.get('acceptance_signature') != acceptance_signature(c):
                raise RuntimeError(f'{mode} preflight source/configuration is stale')
        evidence = verify_inputs(c)
        deterministic(c['seed'], device)
        model = load_model(ROOT/c['initialization']['checkpoint'], paths)
        common = model.shape.rest_template[:1].detach().cpu().numpy().copy()
        del model
        data = CameraData(c, paths, 'train', common)
        validation = CameraData(c, paths, 'validation', common)
        cases = fixed_cases(c, validation)
        if seal_file.exists():
            seal = json.loads(seal_file.read_text())
            verify_contract(seal, c)
        else:
            seal = contract(c, config_file, evidence)
            atomic_json(seal_file, seal)
            atomic_json(out/'CONFIG.json', c)
            preserve_source(out, seal)
            shutil.copyfile(ROOT/c['validation_manifest'], out/'VALIDATION_MANIFEST.json')
            atomic_json(out/'IMPLEMENTATION_AUDIT.json', dict(
                architecture_changed=False, representation_changed=False, sampler_changed=False,
                schedule_changed=False, scene_enabled=False, batch_size=128, quota_per_source=32,
                base_loss='unchanged pipeline.native_motion_loss, before B anchoring',
                existing_supervision=['root/head/neck chordal orientation', 'root-to-head relative rotation',
                    'full-body FK position/GT velocity/GT acceleration', 'toe GT velocity', 'shape rest/native coefficients/template'],
                shape_supplements=['AMASS-only original 16-beta coefficient supervision (mixed EGO wrapper previously zeroed it)',
                    'normalized bone lengths', 'head-to-mean-foot static skeleton height'],
                focused_supplements=['gated SO3 replacement of root chordal contribution', 'full-FK root-to-spine3 and spine3-to-head links',
                    'root-relative lower FK position', 'lower local SO3', 'root-relative lower GT velocity'],
                inference_gt_shape=False, inference_gt_orientation=False,
                evaluation_anchor=c['anchor'], rich_feedback_enabled=False,
                checkpoint_selection='A: shape score; B/C: four-source macro MPJPE, including unmodified A initialization',
                head_stability_guards='predeclared final C-vs-B assessment; do not change checkpoint selection using RICH'))
        atomic_json(out/'STATUS.json', dict(state='running', pid=os.getpid(), at=now(), device=device))
        try:
            plan = json.loads((out/'BUDGET.json').read_text()) if (out/'BUDGET.json').exists() else benchmark(c, paths, data, out, device)
            selected = {}
            for stage in STAGES:
                if STOP:
                    raise InterruptedError('Stop requested; no dependent stage started')
                parent = ROOT/c['initialization']['checkpoint'] if stage == 'A' else selected['A']
                selected[stage] = run_stage(stage, parent, c, paths, data, validation, cases, plan, seal, out, device, resume)
            verify_contract(seal, c)
            result = final_evaluations(c, paths, validation, cases, selected, out, device)
            atomic_json(out/'STATUS.json', dict(state='complete', at=now(), assessment=result['assessment']))
            return result
        except BaseException as error:
            atomic_json(out/'STATUS.json', dict(state='interrupted' if isinstance(error, InterruptedError) else 'failed',
                pid=os.getpid(), at=now(), error=f'{type(error).__name__}: {error}', dependent_stages_not_skipped=True))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight','run'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    c, paths = load_config(args.config), load_paths(args.paths)
    result = preflight(c, paths, args.device, args.config) if args.action == 'preflight' else run(c, paths, args.config, args.device, args.resume)
    print(json.dumps(dict(state=result['state'], output=str(ROOT/c['output']))), flush=True)


if __name__ == '__main__':
    main()
