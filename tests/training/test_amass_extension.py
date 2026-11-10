"""Budget extensions preserve the trajectory and its earlier selected best."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scenemi import amass_extension as extension
from scenemi import amass_finetuning as branch
from tests.training.test_amass_finetuning import TinyPrior


class ExtensionTests(unittest.TestCase):
    def test_cpu_probe_settings_restore_after_failure(self):
        engine = branch.base.engine
        deterministic = engine.deterministic
        threads = torch.get_num_threads()
        enabled = torch.backends.mkldnn.enabled
        backend_deterministic = torch.backends.mkldnn.deterministic
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            with extension.cpu_preflight_runtime():
                engine.deterministic(7, 'cpu')
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertFalse(torch.backends.mkldnn.enabled)
                self.assertTrue(torch.backends.mkldnn.deterministic)
                raise RuntimeError('probe failed')
        self.assertIs(engine.deterministic, deterministic)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(torch.backends.mkldnn.enabled, enabled)
        self.assertEqual(torch.backends.mkldnn.deterministic, backend_deterministic)

    def test_cpu_probe_still_rejects_changed_resume_state(self):
        base = branch.base
        c = extension.load_config('configs/amass_only_115k_to_135k.yaml')
        c['output'] = 'training_outputs/probe'
        seal = dict(sha256='tiny-exact-resume')
        original_load = base.engine.load_state

        def corrupt_load(path, model, optimizer, rng, device, contract):
            state = original_load(path, model, optimizer, rng, device, contract)
            with torch.no_grad():
                model.weight.add_(.1)
            return state

        def update(model, optimizer, data, rng, config, device, step):
            optimizer.zero_grad()
            loss = model.weight.square() * (float(rng.random()) + torch.rand(()))
            loss.backward()
            optimizer.step()
            return dict(step=step, total=float(loss.detach()), shape_gradient_norm=1., seconds=.01)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(extension, 'ROOT', root), patch.object(branch, 'ROOT', root), \
                 patch.object(base, 'ROOT', root), patch.object(base, 'configure_runtime'), \
                 patch.object(branch, 'parent_state', return_value=({}, {}, {})), \
                 patch.object(base, 'contract', return_value=seal), \
                 patch.object(base, 'make_data'), patch.object(base, 'AnchoredAmassPrior', TinyPrior), \
                 patch.object(base, 'train_update', side_effect=update), \
                 patch.object(base.engine, 'load_state', side_effect=corrupt_load):
                with self.assertRaisesRegex(AssertionError, 'Resume differs'):
                    extension.preflight(c, {}, 'cpu')
            result_file = root/c['output']/'preflight/cpu/RESULT.json'
            result = json.loads(result_file.read_text())
            self.assertEqual(result['state'], 'failed')
            self.assertFalse(result['formal_training_started'])
            self.assertNotIn('resume_bitwise_exact', result)
            self.assertEqual(result['cpu_execution']['intraop_threads'], 1)
            self.assertFalse((result_file.parent/'resume.pt').exists())

    def test_cuda_probe_keeps_original_execution_settings(self):
        deterministic = branch.base.engine.deterministic
        threads = torch.get_num_threads()
        enabled = torch.backends.mkldnn.enabled

        def probe(*args):
            self.assertIs(branch.base.engine.deterministic, deterministic)
            self.assertEqual(torch.get_num_threads(), threads)
            self.assertEqual(torch.backends.mkldnn.enabled, enabled)
            return dict(state='passed', resume_bitwise_exact=True)

        with tempfile.TemporaryDirectory() as tmp, patch.object(extension, 'ROOT', Path(tmp)), \
             patch.object(extension, '_preflight', side_effect=probe):
            result = extension.preflight(dict(output='probe'), {}, 'cuda')
        self.assertNotIn('cpu_execution', result)

    def exercise_extension(self, score):
        base = branch.base
        c = branch.load_config('configs/amass_only_115k_to_135k.yaml')
        c['continuation'].pop('source_migration', None)
        reference = copy.deepcopy(c)
        reference['training']['planned_updates'] = 2
        reference['output'] = 'training_outputs/parent'
        c['output'] = 'training_outputs/child'
        c['training'].update(planned_updates=3, checkpoint_every=1, log_every=1)
        c['validation'].update(every_updates=1, full_every_updates=1)
        c['logging']['tensorboard'] = False
        c['continuation'].update(parent_checkpoint='training_outputs/parent/training/last.pt',
            parent_step=2, additional_updates=1)
        cpu_rng, cpu_restore, real_load = base.engine.rng_state, base.engine.restore_rng, torch.load

        def update(model, optimizer, data, rng, config, device, step):
            optimizer.zero_grad()
            loss = model.weight.square() * (float(rng.random()) + torch.rand(()))
            loss.backward()
            optimizer.step()
            return dict(step=step, total=float(loss.detach()), terms={}, gradient_norm=1.,
                shape_gradient_norm=1., lr=optimizer.param_groups[0]['lr'], seconds=.01, identities=[])

        def evaluate(*args, **kwargs):
            metrics = dict(mpjpe_mm=score, shape_rest_error_mm=20.)
            return dict(selection_score_mm=score, raw=dict(family_macro_mean=metrics),
                        anchored=dict(family_macro_mean=metrics))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_file = root/'training_outputs/parent/selection/best.pt'
            parent_file = root/c['continuation']['parent_checkpoint']
            selected_file.parent.mkdir(parents=True)
            parent_file.parent.mkdir(parents=True)
            model = TinyPrior()
            optimizer = base.engine.optimizer_for(model, c)
            rng = np.random.default_rng(7)
            old_seal = dict(config=reference)
            best = dict(step=1, selection_score_mm=100.)
            for step in (1, 2):
                update(model, optimizer, None, rng, c, 'cpu', step)
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=cpu_rng(rng, 'cpu'), device='cuda', step=step, contract=old_seal,
                    best=best, log_bytes=123, validation_pending=False)
                base.save_checkpoint(state, selected_file if step == 1 else parent_file)
            parent = torch.load(parent_file, weights_only=False)
            parent_hash, selected_hash = branch.sha256(parent_file), branch.sha256(selected_file)
            c['continuation']['parent_selection'] = dict(
                checkpoint=str(selected_file.relative_to(root)), sha256=selected_hash)
            metrics_file = selected_file.parent.parent/'validation/step_000001_full/RESULT.json'
            metrics_file.parent.mkdir(parents=True)
            metrics_file.write_text(json.dumps(dict(test_used=False, selection_score_mm=100.)))
            seal = dict(config=c, sha256='budget-extension')
            with patch.object(extension, 'ROOT', root), patch.object(branch, 'ROOT', root), \
                 patch.object(base, 'ROOT', root), \
                 patch.object(extension, '_parent_state', return_value=(parent,
                     dict(step=2, selection_score_mm=110.), {})):
                loaded, inherited, provenance = extension.parent_state(c, {}, seal)
                self.assertEqual(inherited, best)
                self.assertEqual(loaded['step'], 2)
                self.assertEqual(provenance['inherited_best']['checkpoint'], str(selected_file))
                for changes in ({'batch_size': 8},):
                    changed = copy.deepcopy(c)
                    changed['training'].update(changes)
                    with self.assertRaisesRegex(ValueError, 'preserve the batch'):
                        extension.inherited_selection(changed, parent)
                changed = copy.deepcopy(c)
                changed['optimizer']['lr'] = 5e-5
                with self.assertRaisesRegex(ValueError, 'preserve the batch'):
                    extension.inherited_selection(changed, parent)
                changed = copy.deepcopy(c)
                changed['continuation']['parent_selection']['sha256'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    extension.inherited_selection(changed, parent)

                with patch.object(branch, 'parent_state', return_value=(parent, inherited, provenance)), \
                     patch.object(base, 'configure_runtime'), patch.object(base, 'require_preflights'), \
                     patch.object(base, 'contract', return_value=seal), \
                     patch.object(base, 'make_data', return_value=type('Data', (), dict(members=[]))()), \
                     patch.object(base, 'validation_manifest', return_value=[]), \
                     patch.object(base, 'AnchoredAmassPrior', TinyPrior), \
                     patch.object(base, 'train_update', side_effect=update), \
                     patch.object(base, 'evaluate', side_effect=evaluate), \
                     patch.object(base.engine, 'rng_state', side_effect=lambda r, d: cpu_rng(r, 'cpu')), \
                     patch.object(base.engine, 'restore_rng', side_effect=lambda s, r, d: cpu_restore(s, r, 'cpu')), \
                     patch.object(torch, 'load', side_effect=lambda p, **kw: real_load(p, **dict(kw, map_location='cpu'))):
                    extension.initialize(c, {})
                    output = root/c['output']
                    checkpoint = output/'training/last.pt'
                    fork = torch.load(checkpoint, weights_only=False)
                    for key in ('model', 'optimizer', 'rng'):
                        self.assertEqual(base.engine.tensor_digest(fork[key]),
                                         base.engine.tensor_digest(parent[key]))
                    self.assertEqual(fork['step'], 2)
                    self.assertEqual(fork['best'], best)
                    self.assertEqual(fork['log_bytes'], 0)
                    initialization = json.loads((output/'training/INITIALIZATION.json').read_text())
                    self.assertEqual(initialization['first_update'], 3)
                    self.assertEqual(initialization['changed_optimizer_fields'], [])
                    (output/'selection/BEST.json').unlink()  # interrupted metadata publication
                    model.load_state_dict(fork['model'])
                    optimizer.load_state_dict(fork['optimizer'])
                    cpu_restore(fork['rng'], rng, 'cpu')
                    update(model, optimizer, None, rng, c, 'cpu', 3)
                    expected = base.engine.tensor_digest((model.state_dict(), optimizer.state_dict(), cpu_rng(rng, 'cpu')))
                    status = extension.train(c, {}, 'cuda', resume=True)
                    saved = torch.load(checkpoint, weights_only=False)
                    self.assertEqual(status['state'], 'complete')
                    self.assertEqual(expected, base.engine.tensor_digest((saved['model'], saved['optimizer'], saved['rng'])))
                    rows = [json.loads(line) for line in (output/'training/updates.jsonl').read_text().splitlines()]
                    self.assertEqual([r['step'] for r in rows], [3])
                    selected = json.loads((output/'selection/BEST.json').read_text())
                    if score >= 100.:
                        self.assertEqual(saved['best'], best)
                        self.assertEqual(selected['checkpoint'], str(selected_file))
                        self.assertTrue(selected['inherited'])
                    else:
                        self.assertEqual(saved['best']['step'], 3)
                        self.assertEqual(selected['checkpoint'], str(output/'selection/best.pt'))
                        self.assertEqual(selected['selection_score_mm'], score)
                    extension.retain_selection(c)
                    self.assertEqual(json.loads((output/'selection/BEST.json').read_text()), selected)
                    self.assertEqual(branch.sha256(parent_file), parent_hash)
                    self.assertEqual(branch.sha256(selected_file), selected_hash)

    def test_worse_validation_preserves_earlier_best_and_exact_next_update(self):
        self.exercise_extension(110.)

    def test_better_validation_replaces_inherited_selection(self):
        self.exercise_extension(90.)


if __name__ == '__main__':
    unittest.main()
