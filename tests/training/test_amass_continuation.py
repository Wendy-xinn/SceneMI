"""Continuation keeps parent AdamW/RNG and never overwrites its run."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from scenemi import amass_continuation as job


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


class ContinuationTests(unittest.TestCase):
    def test_extension_keeps_terminal_learning_rate(self):
        c = job.load_config()
        self.assertEqual(c['training']['planned_updates'], yaml.safe_load(job.CONFIG.read_text())['planned_updates'])
        self.assertEqual(c['continuation']['parent_step'], 55000)
        self.assertEqual(c['optimizer']['lr']*c['optimizer']['lr_floor_fraction'], 1e-5)
        self.assertNotEqual(c['output'], job.base.load_config()['output'])

    def test_unsafe_budget_and_output_changes_are_rejected(self):
        original = yaml.safe_load(job.CONFIG.read_text())
        for changes in ({'planned_updates': 55000}, {'planned_updates': 0},
                        {'output': 'training_outputs/amass_ego_shape_anchor'},
                        {'output': 'runs/another'}, {'parent_step': 54000},
                        {'learning_rate': 1e-4}, {'learning_rate': 0}, {'learning_rate': float('nan')}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'config.yaml'
                path.write_text(yaml.safe_dump(dict(original, **changes)))
                with self.assertRaises(ValueError):
                    job.load_config(path)

    def test_real_updater_keeps_lr_constant_when_budget_is_extended(self):
        original = yaml.safe_load(job.CONFIG.read_text())
        batch = dict(motion=torch.ones(1, 4, 1))
        identities = [dict(mode='ego', split='train')]
        for total in (65000, 85000, 100000):
            with self.subTest(total=total), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'config.yaml'
                path.write_text(yaml.safe_dump(dict(original, planned_updates=total)))
                c = job.load_config(path)
                model = TinyMotionPrior()
                optimizer = job.base.engine.optimizer_for(model, c)
                with patch.object(job.base.engine, 'batch_for', return_value=(batch, identities)), \
                     patch.object(job.base, 'inference_payload', return_value={}), \
                     patch.object(job.base, 'motion_loss', side_effect=lambda pred, target, *a, **kw:
                                  dict(total=(pred-target).square().mean())):
                    for step in (1, 55001, (55000+total)//2, total):
                        record = job.base.train_update(model, optimizer, None, np.random.default_rng(7), c, 'cpu', step)
                        self.assertEqual(record['lr'], 1e-5)

    def fork_and_continue(self, final_score, interrupt=False):
        base = job.base
        original = base.load_config()
        c = copy.deepcopy(original)
        c.update(output='training_outputs/child', continuation=dict(
            parent_checkpoint='training_outputs/parent/training/last.pt',
            parent_step=2, reference_config='configs/parent.yaml'))
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
                        shape_gradient_norm=1., lr=1e-5, seconds=.01, identities=[])

        def evaluate(*args, **kwargs):
            metric = dict(mpjpe_mm=final_score, shape_rest_error_mm=20.)
            return dict(selection_score_mm=final_score,
                        raw=dict(family_macro_mean=metric), anchored=dict(family_macro_mean=metric))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root/c['continuation']['parent_checkpoint']
            file.parent.mkdir(parents=True)
            folder = file.parent.parent/'validation/step_000002_full'
            folder.mkdir(parents=True)
            (folder/'RESULT.json').write_text(json.dumps(dict(selection_score_mm=100., test_used=False)))
            (file.parent.parent/'VALIDATION_MANIFEST.json').write_text(json.dumps(dict(full=[])))
            old = dict(config=original, sources={}, configuration_files={},
                       index_sha256='same-index', body_model_sha256='same-body', environment={})
            old['sha256'] = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
            seal = dict(old, config=c, sha256='new-explicit-continuation-contract')
            model = TinyPrior()
            optimizer = base.engine.optimizer_for(model, c)
            for group in optimizer.param_groups:
                group['lr'] = 1e-5
            rng = np.random.default_rng(7)
            for step in (1, 2):
                update(model, optimizer, None, rng, c, 'cpu', step)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         rng=cpu_rng(rng, 'cpu'), device='cuda', step=2, contract=old,
                         best=dict(step=2, selection_score_mm=100.), log_bytes=123, validation_pending=False)
            base.save_checkpoint(state, file)
            identity = job.sha256(file)
            c['continuation']['parent_sha256'] = identity
            with patch.object(job, 'ROOT', root), patch.object(base, 'ROOT', root), \
                 patch.object(base, 'load_config', return_value=original), \
                 patch.object(base, 'configure_runtime'), patch.object(base, 'require_preflights'), \
                 patch.object(base, 'contract', return_value=seal), \
                 patch.object(base, 'make_data', return_value=SimpleNamespace(members=[])), \
                 patch.object(base, 'validation_manifest', return_value=[]), \
                 patch.object(base, 'AnchoredAmassPrior', TinyPrior), \
                 patch.object(base, 'train_update', side_effect=update), \
                 patch.object(base.engine, 'rng_state', side_effect=lambda r, d: cpu_rng(r, 'cpu')), \
                 patch.object(base.engine, 'restore_rng', side_effect=lambda s, r, d: cpu_restore(s, r, 'cpu')), \
                 patch.object(torch, 'load', side_effect=lambda p, **kw: real_load(p, **dict(kw, map_location='cpu'))):
                job.initialize(c, {})
                child_file = root/c['output']/'training/last.pt'
                fork = torch.load(child_file, weights_only=False)
                parent = torch.load(file, weights_only=False)
                for key in ('model', 'optimizer', 'rng'):
                    self.assertEqual(base.engine.tensor_digest(fork[key]), base.engine.tensor_digest(parent[key]))
                self.assertEqual(fork['log_bytes'], 0)
                # Establish the exact next update from the original parent.
                model.load_state_dict(parent['model'])
                optimizer.load_state_dict(parent['optimizer'])
                cpu_restore(parent['rng'], rng, 'cpu')
                update(model, optimizer, None, rng, c, 'cpu', 3)
                expected = base.engine.tensor_digest((model.state_dict(), optimizer.state_dict(), cpu_rng(rng, 'cpu')))
                if interrupt:
                    with patch.object(base, 'evaluate', side_effect=RuntimeError('validation interrupted')):
                        with self.assertRaisesRegex(RuntimeError, 'validation interrupted'):
                            base.train(c, {}, 'cuda', resume=True)
                    saved = torch.load(child_file, weights_only=False)
                    self.assertTrue(saved['validation_pending'])
                with patch.object(base, 'evaluate', side_effect=evaluate):
                    status = base.train(c, {}, 'cuda', resume=True)
                saved = torch.load(child_file, weights_only=False)
                actual = base.engine.tensor_digest((saved['model'], saved['optimizer'], saved['rng']))
                self.assertEqual(expected, actual)
                self.assertEqual(job.sha256(file), identity)
                self.assertEqual(status['state'], 'complete')
                self.assertEqual(saved['step'], 3)
                self.assertFalse(saved['validation_pending'])
                log = (child_file.parent/'updates.jsonl').read_text().splitlines()
                self.assertEqual([json.loads(row)['step'] for row in log], [3])
                best = torch.load(root/c['output']/'selection/best.pt', weights_only=False)
                self.assertEqual(best['step'], 3 if final_score < 100. else 2)

    def test_fork_preserves_next_update_and_selects_improvement(self):
        self.fork_and_continue(90.)

    def test_regression_keeps_parent_as_best(self):
        self.fork_and_continue(110.)

    def test_interrupted_final_validation_resumes_without_an_extra_update(self):
        self.fork_and_continue(90., interrupt=True)


if __name__ == '__main__':
    unittest.main()
