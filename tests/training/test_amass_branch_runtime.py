from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scenemi import amass_branch_runtime as job
from scenemi.data.amass_prepare import sha256


class BranchRuntimeTests(unittest.TestCase):
    config = 'configs/amass_only_130k_to_140k_lr4e5.yaml'

    def test_short_lr_branch_preserves_batch_validation_and_model(self):
        c = job.acceptance.load_config(self.config)
        parent = job.branch.reference_config(job.ROOT / c['continuation']['reference_config'])
        self.assertEqual(c['continuation']['parent_step'], 130000)
        self.assertEqual(c['training']['planned_updates'], 140000)
        self.assertEqual(c['training']['batch_size'], 128)
        self.assertEqual(c['optimizer']['lr'], 4e-5)
        self.assertEqual(c['optimizer']['warmup_updates'], 1)
        self.assertEqual(c['optimizer']['lr_floor_fraction'], 1.)
        for key in ('model', 'loss', 'anchor', 'validation', 'observation', 'runtime', 'seed'):
            self.assertEqual(c[key], parent[key])

    def test_preflight_only_cannot_launch_formal_training(self):
        c = job.acceptance.load_config(self.config)
        with patch.object(job.subprocess, 'run') as execute:
            execute.return_value.returncode = 0
            job.run(c, 'configs/amass_paths.local.yaml', preflight_only=True)
        self.assertEqual(execute.call_count, 2)
        for call in execute.call_args_list:
            self.assertIn('preflight', call.args[0])
            self.assertNotIn('train', call.args[0])
            self.assertIn('resources', call.args[0])

    def test_failed_preflight_blocks_formal_training(self):
        c = job.acceptance.load_config(self.config)
        with tempfile.TemporaryDirectory() as tmp, patch.object(job, 'ROOT', Path(tmp)), \
                patch.object(job.branch, 'load_paths', return_value={}), \
                patch.object(job.branch.base, 'contract', return_value={}), \
                patch.object(job.branch.base, 'require_preflights', side_effect=ValueError('stale')), \
                patch.object(job.subprocess, 'run') as execute:
            execute.return_value.returncode = 1
            with self.assertRaises(SystemExit):
                job.run(c, 'paths.yaml')
            self.assertEqual(execute.call_count, 1)
            self.assertNotIn('train', execute.call_args.args[0])

    def test_verified_parent_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'parent.pt'
            torch.save(dict(model=dict(weight=torch.ones(3))), path)
            expected = sha256(path)
            state = job.verified_parent(path, expected)
            self.assertTrue(torch.equal(state['model']['weight'], torch.ones(3)))
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'Pinned parent checkpoint changed'):
                job.verified_parent(path, expected)

    def test_parent_reader_restores_module_on_failure(self):
        c = job.acceptance.load_config(self.config)
        original = job.branch.torch
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with job.parent_reader(c):
                self.assertIsNot(job.branch.torch, original)
                self.assertIs(job.branch.torch.nn, original.nn)
                raise RuntimeError('probe')
        self.assertIs(job.branch.torch, original)
