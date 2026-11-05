"""Continue a sealed EgoGVAE trial with only a larger update budget.

The retained trainer and its original protocol stay immutable. Preparation
copies full model/optimizer/RNG state into a child run with an explicit parent
identity. Launch training manually through ``python -m scenemi resources``.
"""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import tempfile

import torch
import yaml

from scenemi.config import ROOT
from scenemi.egogvae_sharing import (
    digest, load_contract, object_digest, require_acceptance,
    verify_parent_sources, write_json,
)


def configuration(path):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text())
    link = c['continuation']
    parent_config = (ROOT / link['parent_config']).resolve()
    original, _, overlay, parent = load_contract(parent_config)
    verify_parent_sources(parent, overlay)
    require_acceptance(overlay)
    parent_root = (ROOT / original['output']).resolve()
    output = (ROOT / c['output']).resolve()
    if output == parent_root or not output.is_relative_to(parent_root):
        raise ValueError('Continuation output must be a separate child of the parent run')
    expected = copy.deepcopy(original)
    expected['output'] = c['output']
    expected['training']['planned_updates'] = c['training']['planned_updates']
    expected['training']['budget_kind'] = 'budget_only_continuation_of_initial_adaptation'
    expected['validation']['full_steps'] = c['validation']['full_steps']
    expected['continuation'] = link
    if c != expected:
        raise ValueError('Budget continuation cannot change model/data/loss/optimizer/batch/seed/runtime')
    start = link['step']
    end = c['training']['planned_updates']
    steps = c['validation']['full_steps']
    if (start != original['training']['planned_updates'] or end <= start
            or steps != [0, *range(start + 1000, end, 1000), end]
            or link['parent_protocol_sha256'] != parent['sha256']
            or link['checkpoint'] != str(Path(original['output']) / 'training/last.pt')
            or link['selected_checkpoint'] != str(Path(original['output']) / 'training/selected.pt')):
        raise ValueError('Invalid continuation budget, validation milestones or parent identity')
    for key in ('checkpoint', 'selected_checkpoint'):
        if digest(ROOT / link[key]) != link[key + '_sha256']:
            raise ValueError('Pinned parent checkpoint changed: ' + key)
    status = json.loads((parent_root / 'STATUS.json').read_text())
    if status['state'] != 'complete' or status['step'] != start:
        raise ValueError('Parent must be complete at the pinned continuation step')
    return c, parent, overlay


@contextmanager
def parent_lock(c):
    parent = ROOT / c['continuation']['checkpoint']
    with (parent.parent / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def same_state(a, b):
    """Compare serialized training state, including every optimizer/RNG tensor."""
    import numpy as np
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def copy_checkpoint(source, target, *, parent_seal, seal, expected_step):
    state = torch.load(source, map_location='cpu', weights_only=False)
    if state['protocol_sha256'] != parent_seal or state['step'] != expected_step:
        raise ValueError('Parent checkpoint protocol/step mismatch')
    original = dict(state)
    # Child logs contain only new updates. Optimizer and random streams retain
    # their original state; only provenance and the committed log offset change.
    state['protocol_sha256'] = seal
    state['log_bytes'] = 0
    torch.save(state, target)
    restored = torch.load(target, map_location='cpu', weights_only=False)
    for key in original.keys() - {'protocol_sha256', 'log_bytes'}:
        if not same_state(original[key], restored[key]):
            raise AssertionError('Continuation changed checkpoint state: ' + key)


def protocol_record(c, path, parent, overlay):
    sources = {str(Path(path).resolve()): digest(path), str(Path(__file__).resolve()): digest(__file__)}
    parent_root = ROOT / parent['config']['output']
    for name in ('PROTOCOL.json', 'SHARED_RUNTIME.json', 'SHARING_AUTHORIZATION.json',
                 'SHARED_RUNTIME_ACCEPTANCE.json', 'selection.json', 'STATUS.json',
                 'preflight/cpu.json', 'preflight/cuda.json'):
        file = parent_root / name
        sources[str(file)] = digest(file)
    selected = json.loads((parent_root / 'selection.json').read_text())
    result = parent_root / 'validation' / f"step_{selected['step']:06d}" / 'RESULT.json'
    sources[str(result)] = digest(result)
    payload = dict(config=c, config_path=str(Path(path).resolve()), sources=sources,
                   parent_protocol_sha256=parent['sha256'],
                   parent_runtime_sha256=overlay['sha256'],
                   changes=['output', 'update_budget', 'validation_milestones', 'checkpoint_provenance'],
                   model_data_loss_optimizer_sampler_unchanged=True,
                   parent_preflight_inherited=True, training_launch='manual_only', test_used=False)
    return dict(payload, sha256=object_digest(payload))


def verify(c, seal):
    root = ROOT / c['output']
    record = json.loads((root / 'PROTOCOL.json').read_text())
    payload = {k: v for k, v in record.items() if k != 'sha256'}
    if record['sha256'] != seal or object_digest(payload) != seal or record['config'] != c:
        raise ValueError('Continuation protocol mismatch')
    for file, expected in record['sources'].items():
        if digest(file) != expected:
            raise ValueError('Continuation dependency changed: ' + file)
    configuration(record['config_path'])


def prepare(path):
    c, parent, overlay = configuration(path)
    output = ROOT / c['output']
    record = protocol_record(c, path, parent, overlay)
    seal = record['sha256']
    if output.exists():
        verify(c, seal)
        return c, seal
    parent_root = ROOT / parent['config']['output']
    link = c['continuation']
    selection = json.loads((parent_root / 'selection.json').read_text())
    if selection['step'] > link['step'] or selection['protocol_sha256'] != parent['sha256']:
        raise ValueError('Parent selection is inconsistent with the continuation checkpoint')
    result_path = parent_root / 'validation' / f"step_{selection['step']:06d}" / 'RESULT.json'
    result = json.loads(result_path.read_text())
    if (result['protocol_sha256'] != parent['sha256'] or result['step'] != selection['step']
            or result['dataset_macro_mean']['mpjpe_mm'] != selection['mpjpe_mm']):
        raise ValueError('Parent selection lacks its matching fixed validation result')
    # Publish preparation only after both full checkpoints have been verified.
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '_prepare_', dir=output.parent) as folder:
        staging = Path(folder) / 'run'
        (staging / 'training').mkdir(parents=True)
        (staging / 'preflight').mkdir()
        write_json(staging / 'PROTOCOL.json', record)
        for key, name, step in (('checkpoint', 'last.pt', link['step']),
                                ('selected_checkpoint', 'selected.pt', selection['step'])):
            copy_checkpoint(ROOT / link[key], staging / 'training' / name,
                            parent_seal=parent['sha256'], seal=seal, expected_step=step)
        (staging / 'training/updates.jsonl').touch()
        write_json(staging / 'selection.json', dict(selection, protocol_sha256=seal,
                   inherited_from=str(result_path.relative_to(ROOT)),
                   parent_protocol_sha256=parent['sha256']))
        for device in ('cpu', 'cuda'):
            source = parent_root / 'preflight' / f'{device}.json'
            proof = json.loads(source.read_text())
            if (proof['state'] != 'passed' or proof['protocol_sha256'] != parent['sha256']
                    or not proof['resume_bitwise_exact']):
                raise ValueError('Matching parent CPU/CUDA acceptance is required')
            write_json(staging / 'preflight' / source.name, dict(proof,
                       protocol_sha256=seal, inherited_from=str(source.relative_to(ROOT)),
                       acceptance_kind='inherited_unchanged_training_implementation',
                       fresh_preflight_executed=False, parent_protocol_sha256=parent['sha256']))
        write_json(staging / 'PREPARATION.json', dict(state='prepared', step=link['step'],
                   planned_updates=c['training']['planned_updates'], protocol_sha256=seal,
                   model_optimizer_rng_preserved=True, new_log_starts_at_step=link['step'] + 1,
                   parent_selected_result=str(result_path.relative_to(ROOT)),
                   validation_at_start_inherited=True, training_started=False))
        write_json(staging / 'STATUS.json', dict(state='prepared', step=link['step'],
                   budget=c['training']['planned_updates'], protocol_sha256=seal,
                   best=selection, training_started=False))
        staging.rename(output)
    verify(c, seal)
    return c, seal


def train(path):
    c, seal = prepare(path)
    from scenemi import egogvae_adaptation as stage
    from scenemi import gpu_policy
    original_bounded = stage.bounded

    def bounded(config, device):
        original_bounded(config, 'cpu')
        verify(config, seal)
        if device == 'cuda':
            gpu_policy.configure_cuda()

    def protocol(config, baseline, config_path):
        verify(config, seal)
        return seal

    # Overlay only resource/provenance checks. All training, validation,
    # checkpoint transactions and same-output locking use the retained code.
    originals = stage.bounded, stage.protocol, stage.verify_protocol
    stage.bounded, stage.protocol, stage.verify_protocol = bounded, protocol, verify
    try:
        stage.main(['train', '--config', str(Path(path).resolve()), '--device', 'cuda', '--resume'])
    finally:
        stage.bounded, stage.protocol, stage.verify_protocol = originals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    c, _, _ = configuration(args.config)
    with parent_lock(c):
        if args.action == 'prepare':
            c, seal = prepare(args.config)
            print(json.dumps(dict(state='prepared', output=c['output'],
                  step=c['continuation']['step'], budget=c['training']['planned_updates'],
                  protocol_sha256=seal, training_started=False), indent=2))
        else:
            train(args.config)


if __name__ == '__main__':
    main()
