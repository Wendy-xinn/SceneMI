"""Manually launch an AMASS checkpoint branch with an explicit constant LR.

Reuse the sealed model, trainer, preflights, validation and recovery protocol.
Fork model/AdamW/RNG into a separate run with an explicit LR, batch and budget.
The existing constant-terminal-LR continuation stays immutable.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from scenemi import amass_continuation as previous
from scenemi import amass_training as base
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.evaluation import verify_training_seal

VERSION = 'amass-optimizer-branch-v1'
CONFIG = ROOT/'configs/amass_only_lr_boost.yaml'


def reference_config(path, seen=()):
    path = Path(path).resolve()
    version = yaml.safe_load(path.read_text()).get('version')
    if version == VERSION:
        return load_config(path, seen=seen)
    if version == previous.VERSION:
        return previous.load_config(path)
    if version == base.VERSION:
        return base.load_config(path)
    raise ValueError('Expected an AMASS base, continuation or optimizer branch')


def load_config(path=CONFIG, *, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Cyclic parent configuration')
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') != VERSION:
        raise ValueError('Expected an explicit AMASS optimizer-branch configuration')
    parent_file = (ROOT/spec['reference_config']).resolve()
    parent = reference_config(parent_file, seen=seen + (path,))
    step, additional, lr = spec['parent_step'], spec['additional_updates'], spec['learning_rate']
    if (type(step) is not int or step <= 0 or step > parent['training']['planned_updates']
            or type(additional) is not int or additional <= 0):
        raise ValueError('Use a saved parent step and a positive additional update budget')
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning_rate must be finite and positive')
    original, output = (ROOT/parent['output']).resolve(), (ROOT/spec['output']).resolve()
    if (not output.is_relative_to(ROOT/'training_outputs') or output == ROOT/'training_outputs'
            or output.is_relative_to(original) or original.is_relative_to(output)):
        raise ValueError('Use a separate training_outputs directory for the branch')
    checkpoint = (ROOT/spec['parent_checkpoint']).resolve()
    if checkpoint.parent not in (original/'training', original/'selection'):
        raise ValueError('Parent checkpoint must belong to the reference run')
    if len(spec['parent_sha256']) != 64 or any(x not in '0123456789abcdef' for x in spec['parent_sha256']):
        raise ValueError('Pin the parent checkpoint SHA256')
    c = copy.deepcopy(parent)
    c.update(output=spec['output'], _config_path=str(path),
             continuation=dict(spec, reference_config_sha256=sha256(parent_file),
                               parent_configuration_sha256=hashlib.sha256(
                                   json.dumps(parent, sort_keys=True).encode()).hexdigest()))
    c['training']['planned_updates'] = step + additional
    training = spec.get('training', {})
    if set(training) - {'batch_size'}:
        raise ValueError('Only batch_size may be overridden in branch training settings')
    batch_size = training.get('batch_size', parent['training']['batch_size'])
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    c['training']['batch_size'] = batch_size
    # The shared updater uses global steps. A unit floor prevents a schedule
    # restart or a hidden cosine reduction when the total budget changes.
    c['optimizer'].update(lr=lr, lr_floor_fraction=1., warmup_updates=1)
    return c


def parent_state(c, paths, seal):
    spec = c['continuation']
    checkpoint = ROOT/spec['parent_checkpoint']
    if sha256(checkpoint) != spec['parent_sha256']:
        raise ValueError('Pinned parent checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    old = state['contract']
    parent = reference_config(ROOT/spec['reference_config'])
    verify_training_seal(old, parent, paths, ROOT/c['amass_index'])
    if (state['step'] != spec['parent_step'] or state.get('validation_pending', True)
            or state['device'] != 'cuda'):
        raise ValueError('Use a CUDA checkpoint whose full validation has completed')
    for key in ('index_sha256', 'body_model_sha256', 'environment'):
        if old[key] != seal[key]:
            raise ValueError(f'Parent {key} differs from this branch')
    for name, digest in old['sources'].items():
        if seal['sources'].get(name) != digest:
            raise ValueError(f'Parent source changed: {name}')
    for name, digest in old['configuration_files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Parent configuration changed: {name}')
    run = checkpoint.parent.parent
    metrics_file = run/'validation'/f"step_{state['step']:06d}_full/RESULT.json"
    manifest_file = run/'VALIDATION_MANIFEST.json'
    metrics = json.loads(metrics_file.read_text())
    manifest = json.loads(manifest_file.read_text())
    expected = base.validation_manifest(base.make_data(c, paths, 'val').members)
    if manifest['full'] != expected or metrics['test_used'] is not False:
        raise ValueError('Preserve the parent validation membership; do not select on test')
    score = metrics['selection_score_mm']
    if not math.isfinite(score):
        raise ValueError('Parent full-validation score must be finite')
    groups = state['optimizer']['param_groups']
    if not groups or not state['optimizer']['state']:
        raise ValueError('A resumable AdamW state is required')
    rates = [g['lr'] for g in groups]
    if any(not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError('Invalid saved AdamW learning rate')
    return state, dict(step=state['step'], selection_score_mm=score), dict(
        checkpoint=str(checkpoint), checkpoint_sha256=spec['parent_sha256'],
        contract_sha256=old['sha256'], full_metrics_sha256=sha256(metrics_file),
        validation_manifest_sha256=sha256(manifest_file), previous_learning_rates=rates)


def initialize(c, paths):
    folder = ROOT/c['output']/'training'
    checkpoint = folder/'last.pt'
    if checkpoint.exists():
        raise ValueError('Branch already exists; use --resume')
    seal = base.contract(c, paths)
    base.require_preflights(c, seal)
    state, best, parent = parent_state(c, paths, seal)
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder/'updates.jsonl'
    if journal.exists() and journal.stat().st_size:
        raise ValueError('Refuse to overwrite a previous branch journal')
    journal.touch(exist_ok=True)
    for group in state['optimizer']['param_groups']:
        group['lr'] = c['optimizer']['lr']
    state.update(contract=seal, log_bytes=0, best=best, validation_pending=False,
                 continuation_parent=parent)
    atomic_json(folder/'CONTRACT.json', seal)
    atomic_json(folder/'INITIALIZATION.json', dict(
        initialization='continue_parent_with_explicit_learning_rate', parent=parent,
        parent_step=state['step'], additional_updates=c['continuation']['additional_updates'],
        planned_updates=c['training']['planned_updates'], learning_rate=c['optimizer']['lr'],
        batch_size=c['training']['batch_size'],
        model_preserved=True, adamw_moments_and_steps_preserved=True, rng_preserved=True,
        changed_optimizer_fields=['param_groups[*].lr'], training_log_starts_after_parent=True))
    base.save_checkpoint(state, checkpoint)


def preflight(c, paths, device):
    # Validate the real selected parent as well as the shared updater/resume
    # acceptance. Diagnostic updates are discarded and never enter training.
    state, best, parent = parent_state(c, paths, base.contract(c, paths))
    del state
    result = base.preflight(c, paths, device)
    result.update(parent=parent, parent_selection=best, learning_rate=c['optimizer']['lr'])
    atomic_json(ROOT/c['output']/'preflight'/device/'RESULT.json', result)
    return result


def benchmark(c, paths, batch_sizes, repeats=5):
    """Measure real training-window throughput; all diagnostic updates discarded."""
    if repeats < 1 or not batch_sizes or any(x <= 0 for x in batch_sizes):
        raise ValueError('Positive batch sizes and repeat count required')
    state, best, parent = parent_state(c, paths, base.contract(c, paths))
    data = base.make_data(c, paths, 'train')
    records = []
    for size in batch_sizes:
        config = copy.deepcopy(c)
        config['training']['batch_size'] = size
        base.engine.deterministic(c['seed'], 'cuda')
        model = base.AnchoredAmassPrior(config, paths).to('cuda')
        model.load_state_dict(state['model'], strict=True)
        optimizer = base.engine.optimizer_for(model, config)
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = config['optimizer']['lr']
        rng = np.random.default_rng(c['seed'])
        base.engine.restore_rng(state['rng'], rng, 'cuda')
        torch.cuda.reset_peak_memory_stats()
        seconds = []
        for i in range(repeats + 2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            record = base.train_update(model, optimizer, data, rng, config, 'cuda', state['step'] + i + 1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if len(record['identities']) != size or record['lr'] != c['optimizer']['lr']:
                raise AssertionError('Measured update did not use the declared batch/LR')
            if i >= 2:
                seconds.append(elapsed)
            print(f'Benchmark batch {size}: {i + 1}/{repeats + 2}, {elapsed:.3f}s', flush=True)
        row = dict(batch_size=size, measured_updates=repeats, warmup_updates=2,
            seconds_per_update=float(np.mean(seconds)),
            windows_per_second=size / float(np.mean(seconds)),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            measured_seconds=seconds)
        records.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    result = dict(state='passed', parent=parent, parent_selection=best, records=records,
        formal_training_started=False, diagnostic_weights_discarded=True,
        learning_rate=c['optimizer']['lr'], source_sha256=sha256(Path(__file__)),
        gpu=torch.cuda.get_device_name(), config=c)
    performance = ROOT/c['output']/'performance'
    performance.mkdir(parents=True, exist_ok=True)
    atomic_json(performance/'BATCH_BENCHMARK.json', result)
    return result


def run(c, paths_file):
    runtime = c['runtime']
    print(f"AMASS continuation: {c['continuation']['parent_step']} -> "
          f"{c['training']['planned_updates']}; effective batch={c['training']['batch_size']} "
          f"(one full batch per update); frames={c['data']['frames']}; "
          f"lr={c['optimizer']['lr']:.2e}; output={c['output']}", flush=True)
    for action, device in (('preflight', 'cpu'), ('preflight', 'cuda'), ('train', 'cuda')):
        command = [sys.executable, '-m', 'scenemi', 'resources', 'run',
            '--device', device, '--memory-gib', str(runtime['memory_gib']),
            '--cpu-count', str(runtime['cpu_count']), '--',
            sys.executable, '-u', '-m', 'scenemi.amass_finetuning', action,
            '--config', c['_config_path'], '--paths', str(Path(paths_file).resolve()), '--device', device]
        if action == 'train' and (ROOT/c['output']/'training/last.pt').exists():
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('config', 'run', 'preflight', 'train', 'benchmark'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 128])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args(argv)
    if args.resume and args.action != 'train':
        parser.error('--resume is valid only for train')
    c = load_config(args.config)
    if args.action == 'config':
        print(json.dumps(dict(valid=True, parent_step=c['continuation']['parent_step'],
            additional_updates=c['continuation']['additional_updates'],
            planned_updates=c['training']['planned_updates'], learning_rate=c['optimizer']['lr'],
            batch_size=c['training']['batch_size'],
            validation_batch_size=c['validation']['batch_size'],
            output=str(ROOT/c['output'])), indent=2))
        return
    if args.action == 'run':
        return run(c, args.paths)
    paths = load_paths(args.paths)
    out = ROOT/c['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.configure_runtime(c, args.device)
        if args.action == 'benchmark':
            if args.device != 'cuda':
                parser.error('Throughput benchmark requires CUDA')
            result = benchmark(c, paths, args.batch_sizes, args.repeats)
        elif args.action == 'preflight':
            result = preflight(c, paths, args.device)
        else:
            if args.device != 'cuda':
                parser.error('Formal branch training requires CUDA')
            if not args.resume:
                parent = ROOT/c['continuation']['parent_checkpoint']
                with (parent.parent.parent/'.execution.lock').open('a') as parent_lock:
                    fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    initialize(c, paths)
            result = base.train(c, paths, args.device, resume=True)
    print(json.dumps({k: v for k, v in result.items() if k not in ('contract', 'sampling', 'config')}, indent=2))


if __name__ == '__main__':
    main()
