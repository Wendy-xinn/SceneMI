"""AMASS experiment isolation and recovery across interrupted validation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from scenemi import amass_training as job


def test_new_configuration_keeps_motion_shape_and_prior_supervision():
    c = job.load_config()
    assert Path(c['output']).parts[0] == 'training_outputs'
    assert c['model']['initialization'] == 'from_scratch' and c['model']['checkpoint'] is None
    assert c['training']['planned_updates'] == 55000
    assert c['observation']['mode_probabilities'] == {'ego': 1., 'exo': 0.}
    assert all(c['loss']['coefficients'][k] > 0 for k in ('raw_x0', 'position', 'shape_betas', 'shape_rest'))


@pytest.mark.parametrize('output', ['runs/amass_ego128', 'training_outputs', 'training_outputs/../../runs/amass_ego128'])
def test_historical_or_parent_output_is_rejected(tmp_path, output):
    spec = yaml.safe_load(job.CONFIG.read_text())
    spec['output'] = output
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='dedicated training_outputs'):
        job.load_config(path)


def test_quick_validation_samples_families_without_reweighting_long_recordings():
    cases = [dict(corpus=corpus, family=family, start=start) for corpus in ('a', 'b')
             for family in ('one', 'two', 'three') for start in (0, 128, 256)]
    selected = job.quick_cases(cases, 2)
    assert [(x['corpus'], x['family'], x['start']) for x in selected] == [
        ('a', 'one', 0), ('a', 'two', 0), ('b', 'one', 0), ('b', 'two', 0)]


def test_missing_array_cache_does_not_fall_back_to_raw_preprocessing(tmp_path):
    data = object.__new__(job.CachedAmass)
    data.store = SimpleNamespace(root=tmp_path)
    with patch.object(job.PreparedAmass, 'prepare_variant') as fallback:
        with pytest.raises(FileNotFoundError, match='prepared AMASS cache'):
            data.prepare_variant('unavailable.npz', 'missing')
        fallback.assert_not_called()


class TinyPrior(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def to(self, *args, **kwargs):
        return self


def test_interrupted_final_validation_is_replayed_before_marking_complete(tmp_path):
    c = copy.deepcopy(job.load_config())
    c['output'] = 'training_outputs/tiny'
    c['logging']['tensorboard'] = False
    c['training'].update(planned_updates=1, checkpoint_every=1, log_every=1)
    c['validation'].update(every_updates=1, full_every_updates=1)
    paths = {}
    state_for_cpu, restore_for_cpu = job.engine.rng_state, job.engine.restore_rng
    load_for_cpu = torch.load
    seal = {'sha256': 'tiny-same-code-data-config'}

    def update(model, optimizer, data, rng, config, device, step):
        optimizer.zero_grad()
        (model.weight.square()*float(rng.random())).backward()
        optimizer.step()
        return dict(step=step, total=1., terms={}, gradient_norm=1., shape_gradient_norm=1., lr=.001,
                    seconds=.01, identities=[])

    def validation(*args, **kwargs):
        metrics = dict(mpjpe_mm=100., shape_rest_error_mm=20.)
        return dict(raw=dict(family_macro_mean=metrics), anchored=dict(family_macro_mean=metrics),
                    selection_score_mm=100.)

    with patch.object(job, 'ROOT', tmp_path), patch.object(job, 'configure_runtime'), \
         patch.object(job, 'make_data', return_value=SimpleNamespace(members=[])), \
         patch.object(job, 'AnchoredAmassPrior', TinyPrior), patch.object(job, 'contract', return_value=seal), \
         patch.object(job, 'require_preflights'), patch.object(job, 'validation_manifest', return_value=[]), \
         patch.object(job, 'train_update', side_effect=update), \
         patch.object(job.engine, 'rng_state', side_effect=lambda rng, device: state_for_cpu(rng, 'cpu')), \
         patch.object(job.engine, 'restore_rng', side_effect=lambda state, rng, device: restore_for_cpu(state, rng, 'cpu')), \
         patch.object(torch, 'load', side_effect=lambda path, **kw: load_for_cpu(path, **dict(kw, map_location='cpu'))):
        with patch.object(job, 'evaluate', side_effect=RuntimeError('validation worker interrupted')):
            with pytest.raises(RuntimeError, match='worker interrupted'):
                job.train(c, paths, 'cuda')
        out = tmp_path/c['output']
        checkpoint = out/'training/last.pt'
        before = torch.load(checkpoint, weights_only=False)
        assert before['step'] == 1 and before['validation_pending'] is True
        log = out/'training/updates.jsonl'
        with log.open('a') as stream:
            stream.write('{"step": 999, "uncommitted": true}\n')
        with patch.object(job, 'evaluate', side_effect=validation) as evaluate:
            status = job.train(c, paths, 'cuda', resume=True)
            evaluate.assert_called_once()
        after = torch.load(checkpoint, weights_only=False)
        assert torch.equal(before['model']['weight'], after['model']['weight'])
        assert after['validation_pending'] is False
        assert status['state'] == 'complete'
        assert len(log.read_text().splitlines()) == 1
        assert json.loads((out/'selection/BEST.json').read_text())['step'] == 1
        with pytest.raises(ValueError, match='requires --resume'):
            job.train(c, paths, 'cuda')
