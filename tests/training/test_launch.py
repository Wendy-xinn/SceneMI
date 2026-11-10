"""The public launch command requires a named experiment and authenticates CLI changes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scenemi import __main__ as cli
from scenemi import amass_extension as extension
from scenemi import amass_finetuning as branch
from scenemi import launch


class LaunchTests(unittest.TestCase):
    config = 'configs/amass_only_115k_to_135k.yaml'

    def test_public_command_dispatches_without_training(self):
        args = ['--config', self.config, '--check']
        with patch.object(launch, 'main') as entry:
            cli.main(['run'] + args)
        entry.assert_called_once_with(args)

    def test_config_is_required_for_both_entry_points(self):
        with self.assertRaises(SystemExit):
            launch.main([])
        with self.assertRaises(SystemExit):
            extension.main(['run'])

    def test_check_and_run_select_the_named_experiment(self):
        for check, action in [(True, 'config'), (False, 'run')]:
            with self.subTest(action=action), patch.object(extension, 'main') as entry:
                launch.main(['--config', self.config] + (['--check'] if check else []))
                args = entry.call_args.args[0]
                self.assertEqual(args[:3], [action, '--config', self.config])
                self.assertEqual(args[3:], ['--paths', str(launch.ROOT/'configs/amass_paths.local.yaml')])

    def test_public_preflight_only_stops_before_formal_training(self):
        with patch.object(extension, 'main') as entry:
            launch.main(['--config', self.config, '--preflight-only'])
        self.assertEqual(entry.call_args.args[0][-1], '--preflight-only')
        c = extension.load_config(self.config)
        with patch.object(extension.subprocess, 'run') as child:
            child.return_value.returncode = 0
            extension.run(c, 'configs/amass_paths.local.yaml', preflight_only=True)
        self.assertEqual(child.call_count, 2)
        for call, device in zip(child.call_args_list, ('cpu', 'cuda')):
            command = call.args[0]
            self.assertIn('preflight', command)
            self.assertEqual(command[-1], device)
            self.assertNotIn('train', command)
            self.assertIn('resources', command)
            self.assertIn('--memory-gib', command)

    def test_failed_preflight_reports_cause_and_blocks_training(self):
        c = extension.load_config(self.config)
        with tempfile.TemporaryDirectory() as tmp, patch.object(extension, 'ROOT', Path(tmp)):
            result = Path(tmp)/c['output']/'preflight/cpu/RESULT.json'
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps(dict(state='failed', error='AssertionError: exact resume failed')))
            with patch.object(extension.subprocess, 'run') as child, \
                 patch.object(branch, 'load_paths', return_value={}), \
                 patch.object(branch.base, 'contract', return_value={}), \
                 patch.object(branch.base, 'require_preflights', side_effect=ValueError('stale')):
                child.return_value.returncode = 1
                with self.assertRaisesRegex(SystemExit, 'AssertionError: exact resume failed'):
                    extension.run(c, 'configs/amass_paths.local.yaml')
                self.assertEqual(child.call_count, 1)

    def test_only_current_acceptance_can_skip_preflights(self):
        c = extension.load_config(self.config)
        seal = dict(sha256='current-source-config-data')
        for failure in (None, ValueError('stale acceptance'), FileNotFoundError('missing acceptance')):
            with self.subTest(failure=failure), \
                 patch.object(extension.subprocess, 'run') as child, \
                 patch.object(branch, 'load_paths', return_value={}), \
                 patch.object(branch.base, 'contract', return_value=seal), \
                 patch.object(branch.base, 'require_preflights', side_effect=failure) as check:
                child.return_value.returncode = 0
                extension.run(c, 'configs/amass_paths.local.yaml')
                check.assert_called_once_with(c, seal)
                self.assertEqual(child.call_count, 1 if failure is None else 3)
                self.assertIn('train', child.call_args_list[-1].args[0])

    def test_declared_end_step_is_checked(self):
        spec = yaml.safe_load(Path(self.config).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'invalid.yaml'
            spec['end_step'] += 1
            path.write_text(yaml.safe_dump(spec))
            with self.assertRaisesRegex(ValueError, 'end_step'):
                extension.load_config(path)

    def test_only_the_audited_cli_extension_is_accepted(self):
        original = b'# original CLI\n'
        current = original + launch.RUN_DISPATCH.encode() + launch.RUN_PARSER.encode()
        spec = dict(kind='unified_run_cli_dispatch_only', source=launch.CLI,
            parent_sha256=hashlib.sha256(original).hexdigest(),
            current_sha256=hashlib.sha256(current).hexdigest())
        self.assertTrue(launch.verify_cli_extension(spec, source=current)['reconstructed_parent_verified'])
        other = current + b'# unrelated edit\n'
        with self.assertRaisesRegex(ValueError, 'beyond the run'):
            launch.verify_cli_extension(dict(spec, current_sha256=hashlib.sha256(other).hexdigest()), source=other)

    def test_verification_preserves_parent_and_restores_verifier(self):
        spec = yaml.safe_load(Path(self.config).read_text())['source_migration']
        old = dict(sources={launch.CLI: spec['parent_sha256'], 'model.py': 'model-digest'})
        old['sha256'] = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
        before = copy.deepcopy(old)
        seal = dict(sources={launch.CLI: spec['current_sha256'], 'model.py': 'model-digest'})
        c = dict(continuation=dict(source_migration=spec))
        with patch.object(branch, 'verify_training_seal') as verifier:
            with launch.parent_cli_contract(c, seal, branch) as (comparison, migration):
                self.assertEqual(comparison['sources'][launch.CLI], spec['parent_sha256'])
                branch.verify_training_seal(old, {}, {}, 'index')
                view = verifier.call_args.args[0]
                self.assertEqual(view['sources'][launch.CLI], spec['current_sha256'])
                self.assertEqual(view['sources']['model.py'], 'model-digest')
                self.assertEqual(old, before)
                self.assertTrue(migration['reconstructed_parent_verified'])
            self.assertIs(branch.verify_training_seal, verifier)
            old['sha256'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'Parent training contract changed'):
                with launch.parent_cli_contract(c, seal, branch):
                    branch.verify_training_seal(old, {}, {}, 'index')
            self.assertIs(branch.verify_training_seal, verifier)


if __name__ == '__main__':
    unittest.main()
