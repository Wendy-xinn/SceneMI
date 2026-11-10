"""Public training interfaces must preserve run identity and resume selection."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scenemi import train as training, pipeline
from scenemi.__main__ import main, resource_main
from scenemi.monitoring import TrainingReporter


class TrainingInterfaceTests(unittest.TestCase):
    def test_resource_launch_repairs_stale_runtime_only_for_the_call(self):
        runtime = f'/run/user/{os.getuid()}'
        seen = []
        with patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/missing/ide/session'}), \
             patch('scenemi.execution.Path.is_socket', return_value=True), \
             patch('scenemi.execution.Path.stat') as info, \
             patch('scenemi.resource_runner.main',
                   side_effect=lambda argv: seen.append((argv, os.environ['XDG_RUNTIME_DIR']))), \
             patch('sys.stderr', io.StringIO()):
            info.return_value.st_uid = os.getuid()
            resource_main(['check', '--device', 'cpu'])
            self.assertEqual(os.environ['XDG_RUNTIME_DIR'], '/missing/ide/session')
        self.assertEqual(seen, [(['check', '--device', 'cpu'], runtime)])

    def test_new_default_is_independent_of_completed_run(self):
        config = training.load_config()
        self.assertEqual(config['output'], 'runs/ego_only_4datasets')
        self.assertFalse(training.CONFIG.is_symlink())

    def test_public_commands_are_discoverable_and_pass_arguments(self):
        output = io.StringIO()
        with patch('sys.stdout', output), self.assertRaises(SystemExit) as exited:
            main(['--help'])
        self.assertEqual(exited.exception.code, 0)
        for name in ('train', 'evaluate', 'view', 'preflight'):
            self.assertIn(name, output.getvalue())
        with patch.object(training, 'main') as entry:
            main(['train', '--config', 'my.yaml'])
        entry.assert_called_once_with(['train', '--config', 'my.yaml'])

    def test_model_dimensions_reach_constructor(self):
        config = training.load_config()
        config['model'].update(latent_dim=128, dim_mults=[1, 2])
        model = nn.Module()
        model.core = nn.Module()
        model.core.scene_embedding = nn.Linear(1, 1)
        with patch.object(pipeline, 'load_camera_mount', return_value=({}, None)), \
             patch.object(pipeline, 'MotionPipeline', return_value=model) as constructor:
            training.build_model(config, {'smplh_models': Path('/models')})
        self.assertEqual(constructor.call_args.kwargs, dict(latent_dim=128, dim_mults=(1, 2)))
        self.assertFalse(model.core.scene_embedding.weight.requires_grad)

    def test_refusing_existing_run_does_not_rewrite_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / 'CONTRACT.json'
            identity.write_bytes(b'original run identity')
            config = training.load_config()
            config['output'] = directory
            with patch.object(training, 'configure_runtime') as runtime:
                with self.assertRaisesRegex(ValueError, 'Existing run'):
                    training.train(config, 'cpu')
            runtime.assert_not_called()
            self.assertEqual(identity.read_bytes(), b'original run identity')
            with self.assertRaisesRegex(ValueError, 'existing last.pt'):
                training.train(config, 'cpu', resume=True)

    def test_checkpoint_is_atomic_and_contract_mismatch_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            model = nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            rng = np.random.default_rng(5)
            training.checkpoint(model, optimizer, rng, 'cpu', 12, {'identity': 1}, path)
            before = path.read_bytes()
            with patch.object(torch, 'save', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    training.checkpoint(model, optimizer, rng, 'cpu', 13, {'identity': 1}, path)
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(path.with_name('last.pt.tmp').exists())
            with self.assertRaisesRegex(ValueError, 'contract/device'):
                training.load_checkpoint(model, optimizer, rng, 'cpu', {'identity': 2}, path)
            self.assertEqual(training.load_checkpoint(model, optimizer, rng, 'cpu', {'identity': 1}, path), 12)

    def test_resume_retains_best_and_ignores_future_validations(self):
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory)
            (selection / 'candidates').mkdir()
            for step, score in [(100, 90.), (200, 91.), (300, 80.)]:
                (selection / f'full_step_{step:06d}.json').write_text(json.dumps({'selection_score_mm': score}))
                torch.save({'step': step}, selection / 'candidates' / f'step_{step:06d}.pt')
            best = training.restore_selection(selection, 200, training.load_config())
            self.assertEqual(best['step'], 100)
            self.assertEqual(torch.load(selection / 'best.pt', weights_only=False)['step'], 100)

    def test_resume_discards_uncommitted_log_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'updates.jsonl'
            path.write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n{"step":')
            training.trim_update_log(path, 2)
            self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()], [1, 2])

    def test_tensorboard_contains_update_terms_and_per_dataset_validation(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            config = training.load_config()
            with patch('sys.stderr', io.StringIO()), patch('sys.stdout', io.StringIO()):
                reporter = TrainingReporter(directory, config, initial_step=10)
                reporter.update(dict(step=11, total=2., gradient_norm=.3, lr=.0001,
                                     terms={'position': .4}), .5)
                reporter.validation(dict(selection_score_mm=90., by_dataset={'grail': {'mpjpe_mm': 40.}},
                                         dataset_macro_mean={'mpjpe_mm': 90.}), 11, 'validation_full')
                reporter.close()
            events = EventAccumulator(str(Path(directory) / 'tensorboard')).Reload()
            for tag in ('train/loss_total', 'train/loss/position', 'train/learning_rate',
                        'validation_full/grail/mpjpe_mm', 'validation_full/selection_score_mm'):
                self.assertEqual(events.Scalars(tag)[0].step, 11)
            self.assertEqual(events.Scalars('train/seconds_per_step')[0].value, .5)


if __name__ == '__main__':
    unittest.main()
