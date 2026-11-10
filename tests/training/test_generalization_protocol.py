import json
import pytest
from scenemi.generalization_training import fairness, load_config, CONFIG, NAMES, trim_log, restore_selection, contract, ROOT
import torch
import sys
from types import SimpleNamespace


def test_protocol_keeps_equal_quota_and_does_not_unfreeze_shape_with_motion():
    c = load_config(CONFIG)
    assert c['training']['source_quota']*4 == c['training']['effective_batch_size'] == 128
    assert all(x.startswith('core.unet.') for x in c['motion_prefixes'])
    assert c['focused_loss']['max_timestep'] < 999
    assert c['evaluation']['rich_after_selection']


def test_fairness_rejects_mismatched_random_stream_and_parent(tmp_path):
    contract = dict(parent_sha256='same-A', planned_steps=1, trainable={'names':['weight']})
    row = dict(step=1, data_noise_sha256='same-noise', source_counts=dict.fromkeys(('amass','egobody','trumans','grail'),32), lr=1e-6)
    for phase in ('B','C'):
        folder = tmp_path/NAMES[phase]
        folder.mkdir()
        (folder/'CONTRACT.json').write_text(json.dumps(contract))
        (folder/'updates.jsonl').write_text(json.dumps(row)+'\n')
    assert fairness(tmp_path)['data_and_noise_steps_verified'] == 1
    (tmp_path/NAMES['C']/'updates.jsonl').write_text(json.dumps(dict(row, data_noise_sha256='other-noise'))+'\n')
    with pytest.raises(AssertionError, match='random streams'):
        fairness(tmp_path)
    (tmp_path/NAMES['C']/'CONTRACT.json').write_text(json.dumps(dict(contract, parent_sha256='B-checkpoint')))
    with pytest.raises(AssertionError, match='initialization'):
        fairness(tmp_path)


def test_resume_trims_uncommitted_steps_and_truncated_final_line(tmp_path):
    file = tmp_path/'updates.jsonl'
    file.write_text('{"step":1}\n{"step":2}\n{"ste')
    trim_log(file, 1)
    assert file.read_text() == '{"step": 1}\n'


def test_selection_recovers_crash_between_checkpoint_and_metadata(tmp_path):
    torch.save({'step':10}, tmp_path/'control_ft.pt')
    metric = tmp_path/'validation_step_000010'
    metric.mkdir()
    (metric/'RESULT.json').write_text('{"selection_score_mm":90.0}')
    (tmp_path/'SELECTION.json').write_text('{"step":0,"selection_score_mm":100.0}')
    result = restore_selection(tmp_path, 'control_ft')
    assert result['step'] == 10 and result['selection_score_mm'] == 90


def test_dependency_contract_ignores_torch_virtual_relative_files(monkeypatch):
    monkeypatch.setitem(sys.modules, '_test_virtual_torch_classes', SimpleNamespace(__file__='_classes.py'))
    result = contract(load_config(CONFIG), CONFIG, {})
    assert str(ROOT/'_classes.py') not in result['source_sha256']
    assert str(ROOT/'scenemi/generalization_training.py') in result['source_sha256']
