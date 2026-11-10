"""LR branches preserve AdamW moments, RNG and exact checkpoint recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from scenemi import amass_finetuning as job


class TinyPrior(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def to(self, *args, **kwargs):
        return self


class TinyMotionPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_encoder = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer('alphas', torch.full((1000,), .5))

    def forward(self, noisy, times, public, **kwargs):
        return noisy*self.shape_encoder.weight.reshape(1, 1, 1), {}

    def final_motion(self, raw, details):
        return raw


class BranchTests(unittest.TestCase):
    def test_config_changes_only_declared_settings(self):
        c = job.load_config()
        parent = job.reference_config(job.ROOT/c['continuation']['reference_config'])
        self.assertEqual(c['training']['planned_updates'], 100000)
        self.assertEqual(c['continuation']['parent_step'], 80000)
        self.assertEqual(c['continuation']['additional_updates'], 20000)
        self.assertEqual(c['optimizer']['lr'], 2e-5)
        self.assertEqual(c['training']['batch_size'], 128)
        for key in ('model', 'loss', 'anchor', 'validation', 'observation', 'runtime', 'seed'):
            self.assertEqual(c[key], parent[key])
        self.assertNotEqual(c['output'], parent['output'])

    def test_rejects_invalid_branch_settings(self):
        spec = yaml.safe_load(job.CONFIG.read_text())
        for changes in ({'additional_updates': 0}, {'additional_updates': -1},
                        {'parent_step': 86000}, {'parent_step': True},
                        {'learning_rate': 0}, {'learning_rate': float('nan')},
                        {'output': 'training_outputs/amass_ego_shape_anchor_continue'},
                        {'output': 'runs/another'}, {'parent_checkpoint': 'unrelated.pt'},
                        {'parent_sha256': 'missing'}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'config.yaml'
                path.write_text(yaml.safe_dump(dict(spec, **changes)))
                with self.assertRaises(ValueError):
                    job.load_config(path)

    def test_batch_override_is_explicit_and_validated(self):
        spec = yaml.safe_load(job.CONFIG.read_text())
        for batch in (0, -1, True, 1.5):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp)/'config.yaml'
                file.write_text(yaml.safe_dump(dict(spec, training=dict(batch_size=batch))))
                with self.assertRaises(ValueError):
                    job.load_config(file)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'config.yaml'
            spec.pop('training')
            file.write_text(yaml.safe_dump(spec))
            self.assertEqual(job.load_config(file)['training']['batch_size'], 8)

    def test_shared_updater_uses_boosted_lr_at_global_steps(self):
        c = job.load_config()
        model = TinyMotionPrior()
        optimizer = job.base.engine.optimizer_for(model, c)
        with patch.object(job.base.engine, 'batch_for', return_value=(
                 dict(motion=torch.ones(1, 4, 1)), [dict(mode='ego', split='train')])), \
             patch.object(job.base, 'inference_payload', return_value={}), \
             patch.object(job.base, 'motion_loss', side_effect=lambda pred, target, *a, **kw:
                          dict(total=(pred-target).square().mean())):
            for step in (80001, 85000, 95000, 100000):
                result = job.base.train_update(model, optimizer, None, np.random.default_rng(7), c, 'cpu', step)
                self.assertEqual(result['lr'], 2e-5)

    def exercise_branch(self, interrupted):
        base = job.base
        c = job.load_config()
        c.update(output='training_outputs/child')
        c['training'].update(planned_updates=3, checkpoint_every=1, log_every=1)
        c['validation'].update(every_updates=1, full_every_updates=1)
        c['logging']['tensorboard'] = False
        cpu_rng, cpu_restore = base.engine.rng_state, base.engine.restore_rng
        real_load = torch.load

        def update(model, optimizer, data, rng, config, device, step):
            optimizer.zero_grad()
            loss = model.weight.square()*(float(rng.random())+torch.rand(()))
            loss.backward()
            optimizer.step()
            return dict(step=step, total=float(loss.detach()), terms={}, gradient_norm=1.,
                        shape_gradient_norm=1., lr=optimizer.param_groups[0]['lr'], seconds=.01, identities=[])

        def evaluate(*args, **kwargs):
            metric = dict(mpjpe_mm=110., shape_rest_error_mm=20.)
            return dict(selection_score_mm=110., raw=dict(family_macro_mean=metric),
                        anchored=dict(family_macro_mean=metric))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TinyPrior()
            optimizer = base.engine.optimizer_for(model, c)
            optimizer.param_groups[0]['lr'] = 1e-5
            rng = np.random.default_rng(7)
            for step in (1, 2):
                update(model, optimizer, None, rng, c, 'cpu', step)
            parent = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=cpu_rng(rng, 'cpu'), device='cuda', step=2, contract={},
                best=dict(step=2, selection_score_mm=100.), log_bytes=123, validation_pending=False)
            parent_file = root/'parent.pt'
            base.save_checkpoint(parent, parent_file)
            parent_hash = job.sha256(parent_file)
            seal = dict(config=c, sha256='new-explicit-optimizer-branch')
            with patch.object(job, 'ROOT', root), patch.object(base, 'ROOT', root), \
                 patch.object(job, 'parent_state', return_value=(copy.deepcopy(parent), parent['best'], {})), \
                 patch.object(base, 'configure_runtime'), patch.object(base, 'require_preflights'), \
                 patch.object(base, 'contract', return_value=seal), \
                 patch.object(base, 'make_data', return_value=type('Data', (), dict(members=[]))()), \
                 patch.object(base, 'validation_manifest', return_value=[]), \
                 patch.object(base, 'AnchoredAmassPrior', TinyPrior), \
                 patch.object(base, 'train_update', side_effect=update), \
                 patch.object(base.engine, 'rng_state', side_effect=lambda r, d: cpu_rng(r, 'cpu')), \
                 patch.object(base.engine, 'restore_rng', side_effect=lambda s, r, d: cpu_restore(s, r, 'cpu')), \
                 patch.object(torch, 'load', side_effect=lambda p, **kw: real_load(p, **dict(kw, map_location='cpu'))):
                job.initialize(c, {})
                checkpoint = root/c['output']/'training/last.pt'
                fork = torch.load(checkpoint, weights_only=False)
                for key in ('model', 'rng'):
                    self.assertEqual(base.engine.tensor_digest(fork[key]), base.engine.tensor_digest(parent[key]))
                self.assertEqual(base.engine.tensor_digest(fork['optimizer']['state']),
                                 base.engine.tensor_digest(parent['optimizer']['state']))
                self.assertEqual(fork['optimizer']['param_groups'][0]['lr'], 2e-5)
                self.assertEqual(parent['optimizer']['param_groups'][0]['lr'], 1e-5)
                self.assertEqual(fork['log_bytes'], 0)
                model.load_state_dict(fork['model'])
                optimizer.load_state_dict(fork['optimizer'])
                cpu_restore(fork['rng'], rng, 'cpu')
                update(model, optimizer, None, rng, c, 'cpu', 3)
                expected = base.engine.tensor_digest((model.state_dict(), optimizer.state_dict(), cpu_rng(rng, 'cpu')))
                if interrupted:
                    with patch.object(base, 'evaluate', side_effect=RuntimeError('interrupted validation')):
                        with self.assertRaisesRegex(RuntimeError, 'interrupted validation'):
                            base.train(c, {}, 'cuda', resume=True)
                    self.assertTrue(torch.load(checkpoint, weights_only=False)['validation_pending'])
                with patch.object(base, 'evaluate', side_effect=evaluate):
                    status = base.train(c, {}, 'cuda', resume=True)
                saved = torch.load(checkpoint, weights_only=False)
                actual = base.engine.tensor_digest((saved['model'], saved['optimizer'], saved['rng']))
                self.assertEqual(expected, actual)
                self.assertEqual(saved['step'], 3)
                self.assertEqual(status['state'], 'complete')
                self.assertFalse(saved['validation_pending'])
                self.assertEqual(job.sha256(parent_file), parent_hash)
                selected = torch.load(root/c['output']/'selection/best.pt', weights_only=False)
                self.assertEqual(selected['step'], 2)
                rows = [json.loads(s) for s in (checkpoint.parent/'updates.jsonl').read_text().splitlines()]
                self.assertEqual([r['step'] for r in rows], [3])
                self.assertEqual(rows[0]['lr'], 2e-5)

    def test_branch_preserves_moments_rng_and_next_update(self):
        self.exercise_branch(False)

    def test_interrupted_validation_resumes_without_extra_update(self):
        self.exercise_branch(True)


if __name__ == '__main__':
    unittest.main()
