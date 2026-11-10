import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from unittest.mock import patch
from scenemi.replay.validation_runner import load_state, admission
from scenemi.replay.validation_sampling import reference_noise


class NativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.body = nn.Linear(2, 2)
        self.base.native_body = nn.Linear(2, 2)
        self.base.register_buffer('pv_mount', torch.zeros(3))


class ValidationRunnerTests(unittest.TestCase):
    def test_baseline_requires_complete_exact_initializer(self):
        model = NativeModel()
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        state = {k: v for k, v in initial.items() if not k.startswith(('base.pv_', 'base.native_body.'))}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'parent.pt'
            torch.save({'model': state, 'step': 0}, path)
            with self.assertRaises(ValueError):
                load_state(model, path, baseline=True)
            with torch.no_grad():
                model.base.native_body.weight.fill_(99)
            loaded = load_state(model, path, baseline=True, initializer=initial)
            self.assertEqual(loaded['step'], 0)
            for key, value in initial.items():
                self.assertTrue(torch.equal(model.state_dict()[key], value))
            state.pop('base.body.bias')
            torch.save({'model': state}, path)
            with self.assertRaises(ValueError):
                load_state(model, path, baseline=True, initializer=initial)

    def test_candidate_requires_exact_state(self):
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidate.pt'
            torch.save({'model': model.state_dict(), 'step': 1}, path)
            load_state(model, path, baseline=False)
            state = model.state_dict(); state.pop('weight')
            torch.save({'model': state, 'step': 1}, path)
            with self.assertRaises(RuntimeError):
                load_state(model, path, baseline=False)

    def test_noise_keeps_original_groups_and_tail(self):
        for logical, physical, total in ((4,16,39), (8,32,77)):
            expected = torch.cat([torch.randn((min(logical,total-offset),128,201),
                generator=torch.Generator().manual_seed(777)) for offset in range(0,total,logical)])
            actual = torch.cat([reference_noise(offset,min(physical,total-offset),total,
                logical,device='cpu') for offset in range(0,total,physical)])
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(expected[3:29], reference_noise(3,26,total,logical,device='cpu')))

    def test_sharing_recalculates_admission_and_retains_reserve(self):
        report = dict(admitted=False,memory_admitted=True,gpu=dict(policy='original',
            unknown_compute_pids=[9],memory_free_mib=9500,admitted=False))
        config = dict(runtime=dict(memory_gib=12,cuda_allowance_gib=1,gpu_physical_reserve_gib=4))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); (path/'SHARING_AUTHORIZATION.json').write_text('{}')
            with patch('scenemi.replay.validation_runner.training.configuration', return_value=config), \
                 patch('scenemi.replay.validation_runner.authorization', return_value=dict(scope='test')), \
                 patch('scenemi.replay.validation_runner.parallel_resources.admission', return_value=report), \
                 patch('scenemi.replay.validation_runner.parallel_resources.configure_cuda'), \
                 patch.dict('os.environ', SCENEMI_ALLOCATOR_GIB='4'):
                admission(path/'config.yaml',path)
                self.assertTrue(report['admitted'])
                self.assertEqual(report['gpu']['required_free_mib'],9216)
                self.assertEqual(report['gpu']['user_authorized_external_pids'],[9])


if __name__ == '__main__':
    unittest.main()
