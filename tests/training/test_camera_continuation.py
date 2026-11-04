import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scenemi.config import load_ego_config
from scenemi.checkpoint import checkpoint, initialize_continuation, continuation_identity
from scenemi.train import selection_record, require_adaptation_acceptance, contract, SOURCE_ORDER
from scenemi import train as training


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CameraContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_ego_config('configs/ego_only_camera_adaptation.yaml')
        self.config['output'] = str(self.root / 'new_run')
        old = load_ego_config()
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=old['optimizer']['lr'])
        self.model(torch.ones(1, 2)).square().mean().backward(); self.optimizer.step()
        self.rng = np.random.default_rng(5)
        self.path = self.root / 'parent.pt'
        checkpoint(self.model, self.optimizer, self.rng, 'cpu', 14000, {'config': old}, self.path)
        baseline = dict(selection_score_mm=100., by_dataset={s: {'mpjpe_mm': 100.} for s in SOURCE_ORDER})
        for key, content in [('metrics', baseline), ('validation_manifest', {'full': [], 'quick': []})]:
            path = self.root / (key + '.json'); path.write_text(json.dumps(content))
            self.config['continuation'][key] = str(path)
            self.config['continuation'][key + '_sha256'] = sha(path)
        self.config['continuation'].update(checkpoint=str(self.path), checkpoint_sha256=sha(self.path))

    def test_restore_retains_weights_optimizer_and_sampling_rng(self):
        new = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(new.parameters())
        rng = np.random.default_rng(99)
        step = initialize_continuation(new, optimizer, rng, 'cpu', self.config)
        self.assertEqual(step, 14000)
        self.assertEqual(rng.integers(2**31), self.rng.integers(2**31))
        for old, restored in zip(self.model.parameters(), new.parameters()):
            self.assertTrue(torch.equal(old, restored))
        for a, b in zip(self.optimizer.state.values(), optimizer.state.values()):
            self.assertTrue(torch.equal(a['exp_avg'], b['exp_avg']))
            self.assertTrue(torch.equal(a['exp_avg_sq'], b['exp_avg_sq']))

    def test_parent_identity_and_protocol_changes_are_rejected(self):
        model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters())
        c = copy.deepcopy(self.config); c['data']['fps'] = 60
        with self.assertRaisesRegex(ValueError, 'parent data'):
            initialize_continuation(model, optimizer, np.random.default_rng(1), 'cpu', c)
        c = copy.deepcopy(self.config); c['continuation']['checkpoint_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
            continuation_identity(c)
        c = copy.deepcopy(self.config); c['output'] = str(self.root)
        with self.assertRaisesRegex(ValueError, 'new output'):
            continuation_identity(c)

    def test_clean_regression_cannot_be_hidden_by_noisy_gain(self):
        clean = dict(selection_score_mm=101., by_dataset={s: {'mpjpe_mm': 101.} for s in SOURCE_ORDER})
        noisy = {'selection_score_mm': 60.}
        record, eligible = selection_record(clean, noisy, self.config, 15000, 'new.pt', 'metrics.json')
        self.assertTrue(eligible); self.assertEqual(record['selection_score_mm'], 80.5)
        clean['by_dataset']['egobody']['mpjpe_mm'] = 104.
        record, eligible = selection_record(clean, noisy, self.config, 15000, 'new.pt', 'metrics.json')
        self.assertFalse(eligible); self.assertFalse(record['clean_guards']['egobody'])

    def test_step_and_device_checks_are_not_bypassed(self):
        model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters())
        c = copy.deepcopy(self.config); c['continuation']['step'] = 13999
        with self.assertRaisesRegex(ValueError, 'step differs'):
            initialize_continuation(model, optimizer, np.random.default_rng(1), 'cpu', c)
        with self.assertRaisesRegex(ValueError, 'parent device'):
            initialize_continuation(model, optimizer, np.random.default_rng(1), 'cuda', self.config)

    def test_acceptance_cannot_be_reused_after_source_or_config_changes(self):
        source = self.root / 'source.py'; source.write_text('accepted = True\n')
        seal = dict(config=self.config, source_sha256={str(source): sha(source)}, asset_sha256={'body': 'identity'})
        folder = Path(self.config['output']) / 'preflight'; folder.mkdir(parents=True)
        (folder / 'cpu.json').write_text(json.dumps(dict(state='passed', device='cpu', contract=seal)))
        require_adaptation_acceptance(self.config, seal, 'cpu')
        with self.assertRaisesRegex(ValueError, 'preflight --device cuda'):
            require_adaptation_acceptance(self.config, seal, 'cuda')
        changed = copy.deepcopy(self.config); changed['camera_noise']['rho'] = .9
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            require_adaptation_acceptance(changed, seal, 'cpu')
        source.write_text('accepted = False\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            require_adaptation_acceptance(self.config, seal, 'cpu')

    def test_source_contract_ignores_dynamic_relative_module_filenames(self):
        c = load_ego_config()
        index = self.root / 'index.json'; index.write_text('{}')
        c['amass_index'] = str(index)
        mount = self.root / 'mount.json'; mount.write_text('{}')
        model = self.root / 'neutral/model.npz'; model.parent.mkdir(); model.write_bytes(b'body identity')
        with patch.dict('sys.modules', {'dynamic_relative_module': SimpleNamespace(__file__='_classes.py')}), \
             patch('scenemi.train._mount', return_value=({}, mount)):
            seal = contract(c, {'smplh_models': self.root})
        self.assertNotIn('_classes.py', seal['source_sha256'])

    def test_initial_continuation_checkpoint_preserves_parent_step_before_validation(self):
        model = torch.nn.Linear(2, 1)
        with patch.object(training, 'configure_runtime'), \
             patch.object(training, 'load_paths', return_value={}), \
             patch.object(training, 'build_model', return_value=model), \
             patch.object(training, 'FourEgoData'), \
             patch.object(training, 'contract', return_value={'config': self.config}), \
             patch.object(training, 'require_adaptation_acceptance'), \
             patch.object(training, 'source_cases', return_value=[]), \
             patch.object(training, 'TrainingReporter'), \
             patch.object(training, 'validation_pair', side_effect=RuntimeError('stop before validation')):
            with self.assertRaisesRegex(RuntimeError, 'stop before validation'):
                training.train(self.config, 'cpu')
        saved = torch.load(Path(self.config['output']) / 'training/last.pt', weights_only=False)
        self.assertEqual(saved['step'], 14000)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(saved['model'][name], value))


if __name__ == '__main__':
    unittest.main()
