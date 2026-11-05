"""A budget extension must preserve the next update and the retained 3k run."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scenemi import egogvae_adaptation as stage
from scenemi import egogvae_continuation as continuation


def stochastic_update(model, optimizer, rng):
    values = rng.normal(size=(2, 3))
    inputs = torch.as_tensor(values, dtype=torch.float32) + torch.rand(2, 3) + random.random()
    optimizer.zero_grad()
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    return dict(loss=float(loss.detach()), terms={}), values


def test_migrated_checkpoint_has_the_identical_next_stochastic_update(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model = torch.nn.Linear(3, 2)
    public = SimpleNamespace(egogvae=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    rng = np.random.default_rng(123)
    for _ in range(2):
        stochastic_update(model, optimizer, rng)
    source, target = tmp_path / 'parent.pt', tmp_path / 'child.pt'
    stage.save_checkpoint(source, public, optimizer, rng, 3000, 'parent', log_bytes=1234)
    original_digest = continuation.digest(source)
    expected_row, expected_samples = stochastic_update(model, optimizer, rng)
    expected = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    continuation.copy_checkpoint(source, target, parent_seal='parent', seal='child', expected_step=3000)
    restored = torch.load(target, map_location='cpu', weights_only=False)
    model.load_state_dict(restored['model_state_dict'])
    optimizer.load_state_dict(restored['optimizer_state_dict'])
    stage.restore_rng(restored['rng'], rng)
    actual_row, actual_samples = stochastic_update(model, optimizer, rng)
    assert np.array_equal(actual_samples, expected_samples)
    assert actual_row == expected_row
    assert continuation.same_state(model.state_dict(), expected)
    assert continuation.same_state(optimizer.state_dict(), expected_optimizer)
    assert restored['log_bytes'] == 0 and restored['protocol_sha256'] == 'child'
    assert continuation.digest(source) == original_digest
    with pytest.raises(ValueError, match='protocol/step'):
        continuation.copy_checkpoint(source, target, parent_seal='wrong', seal='child', expected_step=3000)


@pytest.fixture
def prepared_parent(tmp_path, monkeypatch):
    original = yaml.safe_load((continuation.ROOT / 'configs/egogvae_finetune.yaml').read_text())
    child = yaml.safe_load((continuation.ROOT / 'configs/egogvae_finetune_continue_6k.yaml').read_text())
    # Three updates exercise the actual retained trainer without any GPU work.
    child['training']['planned_updates'] = 3003
    child['validation']['full_steps'] = [0, 3003]
    root = tmp_path / original['output']
    (root / 'training').mkdir(parents=True)
    (root / 'preflight').mkdir()
    (root / 'validation/step_003000').mkdir(parents=True)
    (tmp_path / 'configs').mkdir()
    parent = dict(config=original, sha256='parent', sources={})
    overlay = dict(sha256='runtime')
    monkeypatch.setattr(continuation, 'ROOT', tmp_path)
    monkeypatch.setattr(stage, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(continuation, 'load_contract', lambda path: (original, {}, overlay, parent))
    monkeypatch.setattr(continuation, 'verify_parent_sources', lambda *args: None)
    monkeypatch.setattr(continuation, 'require_acceptance', lambda *args: None)
    model = torch.nn.Linear(3, 2)
    optimizer = stage.optimizer_for(SimpleNamespace(egogvae=model), original)
    rng = np.random.default_rng(123)
    stochastic_update(model, optimizer, rng)
    for name in ('last.pt', 'selected.pt'):
        stage.save_checkpoint(root / 'training' / name, SimpleNamespace(egogvae=model),
                              optimizer, rng, 3000, 'parent', log_bytes=1234)
    child['continuation'].update(parent_protocol_sha256='parent',
        checkpoint_sha256=continuation.digest(root / 'training/last.pt'),
        selected_checkpoint_sha256=continuation.digest(root / 'training/selected.pt'))
    selection = dict(step=3000, mpjpe_mm=99.0, protocol_sha256='parent')
    records = {'PROTOCOL.json': parent, 'SHARED_RUNTIME.json': overlay,
               'SHARING_AUTHORIZATION.json': {}, 'SHARED_RUNTIME_ACCEPTANCE.json': {},
               'STATUS.json': dict(state='complete', step=3000), 'selection.json': selection,
               'validation/step_003000/RESULT.json': dict(selection, dataset_macro_mean=dict(mpjpe_mm=99.0)),
               'preflight/cpu.json': dict(state='passed', protocol_sha256='parent', resume_bitwise_exact=True),
               'preflight/cuda.json': dict(state='passed', protocol_sha256='parent', resume_bitwise_exact=True)}
    for name, value in records.items():
        (root / name).write_text(json.dumps(value))
    path = tmp_path / 'configs/continuation.yaml'
    path.write_text(yaml.safe_dump(child))
    return path, root, child


def test_preparation_preserves_parent_and_does_not_overwrite_child_progress(prepared_parent):
    path, parent, child = prepared_parent
    before = {p: continuation.digest(p) for p in parent.rglob('*') if p.is_file()}
    c, seal = continuation.prepare(path)
    output = continuation.ROOT / c['output']
    state = torch.load(output / 'training/last.pt', weights_only=False)
    assert state['step'] == 3000 and state['log_bytes'] == 0 and state['protocol_sha256'] == seal
    assert (output / 'training/updates.jsonl').read_text() == ''
    for device in ('cpu', 'cuda'):
        proof = json.loads((output / 'preflight' / (device + '.json')).read_text())
        assert proof['fresh_preflight_executed'] is False
        assert proof['acceptance_kind'].startswith('inherited_')
    assert all(continuation.digest(p) == value for p, value in before.items())
    (output / 'training/updates.jsonl').write_text('committed progress\n')
    continuation.prepare(path)
    assert (output / 'training/updates.jsonl').read_text() == 'committed progress\n'


@pytest.mark.parametrize('mutation', ['optimizer', 'batch', 'same_output', 'parent_hash'])
def test_continuation_rejects_semantic_changes_or_parent_replacement(prepared_parent, mutation):
    path, parent, child = prepared_parent
    if mutation == 'optimizer':
        child['optimizer']['lr'] *= 2
    elif mutation == 'batch':
        child['training']['effective_batch_size'] *= 2
    elif mutation == 'same_output':
        child['output'] = str(parent.relative_to(continuation.ROOT))
    else:
        child['continuation']['checkpoint_sha256'] = 'wrong'
    path.write_text(yaml.safe_dump(child))
    with pytest.raises(ValueError):
        continuation.prepare(path)


def test_retained_trainer_starts_at_3001_and_keeps_parent_best_if_validation_regresses(
        prepared_parent, monkeypatch):
    path, parent, child = prepared_parent
    original_hash = continuation.digest(parent / 'training/last.pt')
    public = SimpleNamespace(egogvae=torch.nn.Linear(3, 2))
    monkeypatch.setattr(stage, 'configuration', lambda path: (child, {}))
    monkeypatch.setattr(stage, 'setup', lambda *args: (public, None, {}))
    monkeypatch.setattr(stage, 'update', lambda public, data, optimizer, rng, c, device:
                        stochastic_update(public.egogvae, optimizer, rng)[0])
    validated = []

    def evaluate(public, c, step, seal):
        validated.append(step)
        return dict(dataset_macro_mean=dict(mpjpe_mm=100.0), by_dataset={})

    monkeypatch.setattr(stage, 'evaluate', evaluate)
    load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs['map_location'] = 'cpu'
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', cpu_load)
    continuation.train(path)
    output = continuation.ROOT / child['output']
    rows = [json.loads(line) for line in (output / 'training/updates.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [3001, 3002, 3003]
    assert validated == [3003]
    assert torch.load(output / 'training/last.pt', weights_only=False)['step'] == 3003
    assert torch.load(output / 'training/selected.pt', weights_only=False)['step'] == 3000
    assert json.loads((output / 'selection.json').read_text())['step'] == 3000
    assert json.loads((output / 'STATUS.json').read_text())['state'] == 'complete'
    assert continuation.digest(parent / 'training/last.pt') == original_hash
