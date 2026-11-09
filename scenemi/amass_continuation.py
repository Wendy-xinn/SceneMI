"""Continue a completed AMASS prior while preserving its original run.

The existing trainer owns updates, validation, checkpoint selection and exact
recovery. Only output/budget provenance changes; model, AdamW and all RNG states
come from the pinned parent. The saved terminal learning rate stays constant
throughout the extension, independently of the new total update budget.
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

import torch
import yaml

from scenemi import amass_training as base
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256


CONFIG = ROOT/'configs/amass_only_continue.yaml'
VERSION = 'amass-ego-shape-anchor-continuation-v1'


def load_config(path=CONFIG):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') != VERSION:
        raise ValueError('Expected an AMASS continuation configuration')
    parent = base.load_config(ROOT/spec['reference_config'])
    step, total = spec['parent_step'], spec['planned_updates']
    if (type(step) is not int or type(total) is not int or step <= 0 or total <= step
            or step != parent['training']['planned_updates']):
        raise ValueError('Extend the completed parent budget with additional updates')
    output, original = (ROOT/spec['output']).resolve(), (ROOT/parent['output']).resolve()
    if (not output.is_relative_to(ROOT/'training_outputs') or output == ROOT/'training_outputs'
            or output.is_relative_to(original) or original.is_relative_to(output)):
        raise ValueError('Continuation requires a separate training_outputs directory')
    checkpoint = (ROOT/spec['parent_checkpoint']).resolve()
    if checkpoint.parent != original/'training':
        raise ValueError('Parent checkpoint must belong to the reference run')
    opt = parent['optimizer']
    terminal_lr = opt['lr']*opt['lr_floor_fraction']
    learning_rate = spec.get('learning_rate', terminal_lr)
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0 or learning_rate != terminal_lr):
        raise ValueError('Continuation learning_rate must match the parent terminal learning rate')
    c = copy.deepcopy(parent)
    c.update(output=spec['output'], _config_path=str(path),
             continuation=dict(spec, reference_config_sha256=sha256(ROOT/spec['reference_config'])))
    c['training']['planned_updates'] = total
    # The shared updater multiplies lr by a warmup/cosine factor. A unit floor
    # and one-step warmup make that factor exactly one, including in preflight.
    # Parent AdamW moments and step counters are still restored unchanged.
    c['optimizer'].update(lr=learning_rate, lr_floor_fraction=1., warmup_updates=1)
    return c


def parent_state(c, paths, seal):
    """Check the parent against both its seal and the new execution closure."""
    spec = c['continuation']
    checkpoint = ROOT/spec['parent_checkpoint']
    if sha256(checkpoint) != spec['parent_sha256']:
        raise ValueError('Parent checkpoint SHA256 differs from the pinned identity')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    old = state['contract']
    content = {k: v for k, v in old.items() if k != 'sha256'}
    if hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != old['sha256']:
        raise ValueError('Invalid parent contract digest')
    if old['config'] != base.load_config(ROOT/spec['reference_config']):
        raise ValueError('Reference configuration differs from the completed parent')
    if (state['step'] != spec['parent_step'] or state.get('validation_pending', True)
            or state['device'] != 'cuda'):
        raise ValueError('Parent must be a CUDA checkpoint with completed final validation')
    for key in ('index_sha256', 'body_model_sha256', 'environment'):
        if old[key] != seal[key]:
            raise ValueError(f'Parent {key} differs from the current continuation')
    for name, digest in old['sources'].items():
        if seal['sources'].get(name) != digest:
            raise ValueError(f'Parent source changed: {name}')
    for name, digest in old['configuration_files'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Parent configuration changed: {name}')
    metrics_path = checkpoint.parent.parent/'validation'/f"step_{state['step']:06d}_full/RESULT.json"
    metrics = json.loads(metrics_path.read_text())
    manifest_path = checkpoint.parent.parent/'VALIDATION_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    expected = base.validation_manifest(base.make_data(c, paths, 'val').members)
    if manifest['full'] != expected or metrics['test_used'] is not False:
        raise ValueError('Continuation validation membership differs from the parent')
    score = metrics['selection_score_mm']
    if not math.isfinite(score):
        raise ValueError('Parent full-validation score is not finite')
    for group in state['optimizer']['param_groups']:
        if group['lr'] != c['optimizer']['lr']*c['optimizer']['lr_floor_fraction']:
            raise ValueError('Parent optimizer is not at the declared final learning rate')
    return state, dict(step=state['step'], selection_score_mm=score), dict(
        checkpoint=str(checkpoint), checkpoint_sha256=spec['parent_sha256'],
        contract_sha256=old['sha256'], full_metrics_sha256=sha256(metrics_path),
        validation_manifest_sha256=sha256(manifest_path))


def initialize(c, paths):
    """Commit an explicit fork; the parent checkpoint and journal stay intact."""
    out = ROOT/c['output']
    folder = out/'training'
    checkpoint = folder/'last.pt'
    if checkpoint.exists():
        raise ValueError('Continuation already exists; use --resume')
    seal = base.contract(c, paths)
    base.require_preflights(c, seal)
    state, best, parent = parent_state(c, paths, seal)
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder/'updates.jsonl'
    if journal.exists() and journal.stat().st_size:
        raise ValueError('Refuse to overwrite an existing continuation journal')
    journal.touch(exist_ok=True)
    state.update(contract=seal, log_bytes=0, best=best, validation_pending=False,
                 continuation_parent=parent)
    atomic_json(folder/'CONTRACT.json', seal)
    atomic_json(folder/'INITIALIZATION.json', dict(
        initialization='continue_parent_model_optimizer_rng', parent=parent,
        parent_step=state['step'], planned_updates=c['training']['planned_updates'],
        learning_rate=state['optimizer']['param_groups'][0]['lr'],
        model_optimizer_rng_preserved=True, training_log_starts_after_parent=True))
    base.save_checkpoint(state, checkpoint)


def run(c, paths_file):
    """Manually invoked pipeline: bounded checks, followed by bounded training."""
    runtime = c['runtime']
    for action, device in (('preflight', 'cpu'), ('preflight', 'cuda'), ('train', 'cuda')):
        command = [sys.executable, '-m', 'scenemi', 'resources', 'run',
                   '--device', device, '--memory-gib', str(runtime['memory_gib']),
                   '--cpu-count', str(runtime['cpu_count']), '--',
                   sys.executable, '-u', '-m', 'scenemi.amass_continuation', action,
                   '--config', c['_config_path'], '--paths', str(Path(paths_file).resolve()),
                   '--device', device]
        if action == 'train' and (ROOT/c['output']/'training/last.pt').exists():
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('config', 'run', 'preflight', 'train'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and args.action != 'train':
        parser.error('--resume is valid only for train')
    c = load_config(args.config)
    if args.action == 'config':
        print(json.dumps(dict(valid=True, parent_step=c['continuation']['parent_step'],
            planned_updates=c['training']['planned_updates'], output=str(ROOT/c['output']),
            learning_rate=c['optimizer']['lr']*c['optimizer']['lr_floor_fraction']), indent=2))
        return
    if args.action == 'run':
        return run(c, args.paths)
    paths = load_paths(args.paths)
    out = ROOT/c['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            result = base.preflight(c, paths, args.device)
        else:
            if args.device != 'cuda':
                parser.error('Formal continuation training requires CUDA')
            base.configure_runtime(c, args.device)
            if not args.resume:
                # A live parent writer must never race the fork identity check.
                parent = ROOT/c['continuation']['parent_checkpoint']
                with (parent.parent.parent/'.execution.lock').open('a') as parent_lock:
                    fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    initialize(c, paths)
            result = base.train(c, paths, args.device, resume=True)
    print(json.dumps({k: v for k, v in result.items() if k not in ('contract', 'sampling')}, indent=2))


if __name__ == '__main__':
    main()
