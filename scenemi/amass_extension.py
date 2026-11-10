"""Extend a completed AMASS run while retaining its historical best model.

Reuse the sealed branch trainer, updater, preflights and resume protocol. Only
the update budget and output directory change; start from the final checkpoint
and retain a verified reference to the previous validation-selected model.
"""
import copy
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import subprocess
import sys

import torch

from scenemi import amass_finetuning as branch
# Preserve this predecessor's explicit source dependency in child contracts.
from scenemi import amass_cli_compatibility as _legacy_cli_dependency
from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.launch import parent_cli_contract

_parent_state = branch.parent_state
_initialize = branch.initialize
_train = branch.base.train
_load_config = branch.load_config
_preflight = branch.preflight


def load_config(path, *, seen=()):
    c = _load_config(path, seen=seen)
    end = c['continuation'].get('end_step')
    if end is not None and (type(end) is not int or end != c['training']['planned_updates']):
        raise ValueError('end_step must equal parent_step plus additional_updates')
    return c


def inherited_selection(c, state):
    spec = c['continuation']
    reference = state['contract']['config']
    if (Path(spec['parent_checkpoint']).name != 'last.pt'
            or state['step'] != reference['training']['planned_updates']):
        raise ValueError('Budget extensions must start from the completed final checkpoint')
    if (c['training']['batch_size'] != reference['training']['batch_size']
            or c['optimizer'] != reference['optimizer']
            or any(g['lr'] != c['optimizer']['lr'] for g in state['optimizer']['param_groups'])):
        raise ValueError('Budget extensions must preserve the batch and optimizer settings')
    best = state.get('best')
    if not best or not 0 < best['step'] <= state['step']:
        raise ValueError('A saved historical best is required')
    selection = spec['parent_selection']
    checkpoint = (ROOT / selection['checkpoint']).resolve()
    parent_run = (ROOT / spec['parent_checkpoint']).resolve().parent.parent
    if checkpoint.parent != parent_run / 'selection':
        raise ValueError('The inherited best must belong to the parent run')
    if sha256(checkpoint) != selection['sha256']:
        raise ValueError('Pinned historical best checkpoint changed')
    selected = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if (selected['step'] != best['step'] or selected['contract'] != state['contract']
            or selected.get('validation_pending', True) or selected['device'] != 'cuda'
            or selected.get('best') != best):
        raise ValueError('Historical best checkpoint does not match the final checkpoint record')
    metrics_file = parent_run / 'validation' / f"step_{best['step']:06d}_full/RESULT.json"
    metrics = json.loads(metrics_file.read_text())
    if (metrics['test_used'] is not False
            or metrics['selection_score_mm'] != best['selection_score_mm']):
        raise ValueError('Historical best must match its completed full validation')
    return copy.deepcopy(best), dict(best, checkpoint=str(checkpoint),
        checkpoint_sha256=selection['sha256'], full_metrics_sha256=sha256(metrics_file),
        inherited=True)


def parent_state(c, paths, seal):
    with parent_cli_contract(c, seal, branch) as (comparison, migration):
        state, final_selection, provenance = _parent_state(c, paths, comparison)
    best, selection = inherited_selection(c, state)
    provenance.update(final_checkpoint_selection=final_selection,
                      inherited_best=selection, budget_only_extension=True)
    if migration is not None:
        provenance['source_migration'] = migration
    return state, best, provenance


def retain_selection(c):
    output = ROOT / c['output']
    record_file = output / 'training/INITIALIZATION.json'
    record = json.loads(record_file.read_text())
    record.update(initialization='extend_completed_run_budget', changed_optimizer_fields=[],
                  first_update=record['parent_step'] + 1,
                  inherited_best=record['parent']['inherited_best'])
    atomic_json(record_file, record)
    selected = output / 'selection'
    selected.mkdir(exist_ok=True)
    # Keep the predecessor checkpoint under its original contract and path.
    # The trainer replaces this reference after a genuinely better validation.
    if not (selected / 'BEST.json').exists():
        atomic_json(selected / 'BEST.json', record['inherited_best'])


def initialize(c, paths):
    _initialize(c, paths)
    retain_selection(c)


def train(c, paths, device, resume=False):
    # Repair a stop after checkpoint creation but before inherited selection
    # publication; later improved selections are left untouched.
    retain_selection(c)
    return _train(c, paths, device, resume=resume)


@contextmanager
def cpu_preflight_runtime():
    """Keep the discarded CPU resume probe independent of parallel kernels."""
    engine = branch.base.engine
    original = engine.deterministic
    threads = torch.get_num_threads()
    enabled = torch.backends.mkldnn.enabled
    backend_deterministic = torch.backends.mkldnn.deterministic

    def deterministic(seed, device):
        original(seed, device)
        if device == 'cpu':
            torch.set_num_threads(1)

    try:
        torch.backends.mkldnn.enabled = False
        torch.backends.mkldnn.deterministic = True
        engine.deterministic = deterministic
        yield
    finally:
        engine.deterministic = original
        torch.set_num_threads(threads)
        torch.backends.mkldnn.enabled = enabled
        torch.backends.mkldnn.deterministic = backend_deterministic


def preflight(c, paths, device):
    output = ROOT / c['output'] / 'preflight' / device
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / 'RESULT.json'
    result = dict(state='running', device=device, formal_training_started=False)
    if device == 'cpu':
        result['cpu_execution'] = dict(intraop_threads=1, mkldnn_enabled=False,
            mkldnn_deterministic=True, scope='discarded_preflight_only')
    atomic_json(result_file, result)
    try:
        with cpu_preflight_runtime() if device == 'cpu' else nullcontext():
            # The original preflight still requires exact update records and
            # bitwise equality of the complete model, AdamW and RNG state.
            checked = _preflight(c, paths, device)
        result.update(checked)
        atomic_json(result_file, result)
        return result
    except Exception as error:
        result.update(state='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(result_file, result)
        raise
    finally:
        # Diagnostic checkpoints are never used to initialize formal training.
        (output / 'resume.pt').unlink(missing_ok=True)


def run(c, paths_file, *, preflight_only=False):
    runtime = c['runtime']
    print(f"AMASS budget extension: {c['continuation']['parent_step']} -> "
          f"{c['training']['planned_updates']}; effective batch={c['training']['batch_size']} "
          f"(one full batch per update); frames={c['data']['frames']}; "
          f"lr={c['optimizer']['lr']:.2e}; output={c['output']}", flush=True)
    phases = [('preflight', 'cpu'), ('preflight', 'cuda')]
    if not preflight_only:
        paths = branch.load_paths(paths_file)
        seal = branch.base.contract(c, paths)
        try:
            branch.base.require_preflights(c, seal)
        except (OSError, ValueError, KeyError):
            pass
        else:
            phases.clear()
            print('Reusing CPU/CUDA preflights verified against the current code, config and data.', flush=True)
        phases.append(('train', 'cuda'))
    for action, device in phases:
        command = [sys.executable, '-m', 'scenemi', 'resources', 'run',
            '--device', device, '--memory-gib', str(runtime['memory_gib']),
            '--cpu-count', str(runtime['cpu_count']), '--',
            sys.executable, '-u', '-m', 'scenemi.amass_extension', action,
            '--config', c['_config_path'], '--paths', str(Path(paths_file).resolve()), '--device', device]
        if action == 'train' and (ROOT / c['output'] / 'training/last.pt').exists():
            command.append('--resume')
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            result_file = ROOT / c['output'] / (
                f'preflight/{device}/RESULT.json' if action == 'preflight' else 'training/STATUS.json')
            reason = ''
            if result_file.is_file():
                try:
                    result = json.loads(result_file.read_text())
                    if result.get('state') == 'failed':
                        reason = result.get('error', '')
                except (ValueError, OSError):
                    pass
            raise SystemExit(f'{device} {action} failed (exit {completed.returncode}). '
                             f'{reason}\nDetails: {result_file}')
    if preflight_only:
        print('CPU and CUDA preflights passed. Formal training has not started.', flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    preflight_only = '--preflight-only' in argv
    if preflight_only:
        argv.remove('--preflight-only')
        if not argv or argv[0] != 'run':
            raise SystemExit('--preflight-only requires the run action')
    if not any(x in ('-h', '--help', '--config') or x.startswith('--config=') for x in argv):
        raise SystemExit('Use python -m scenemi run --config configs/<experiment>.yaml')
    branch.load_config = load_config
    branch.parent_state = parent_state
    branch.initialize = initialize
    branch.base.train = train
    branch.preflight = preflight
    branch.run = lambda c, paths: run(c, paths, preflight_only=preflight_only)
    return branch.main(argv)


if __name__ == '__main__':
    main()
