"""User-managed GPU execution retains CPU/RAM isolation and source compatibility."""
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import yaml
import torch

from scenemi.config import ROOT, load_ego_config
from scenemi import execution, parallel_resources, resource_runner


class PublicJobIdentityTests(unittest.TestCase):
    def recognized(self, arguments, *, uid=None, cwd=ROOT):
        proc = MagicMock()
        proc.stat.return_value.st_uid = os.getuid() if uid is None else uid
        proc.joinpath.return_value.read_bytes.return_value = b'\0'.join(s.encode() for s in arguments)
        proc.joinpath.return_value.resolve.return_value = cwd
        proc_root = MagicMock()
        proc_root.__truediv__.return_value = proc
        with patch.object(resource_runner, 'Path', return_value=proc_root):
            return resource_runner.own_public_job(123)

    def test_public_training_preflight_and_evaluation_are_recognized(self):
        for action in ('train', 'preflight', 'evaluate'):
            with self.subTest(action=action):
                self.assertTrue(self.recognized(['python', '-m', 'scenemi', action]))
        self.assertFalse(self.recognized(['python', '-m', 'scenemi.native_ego4_training', 'train']))

    def test_other_users_directories_and_arbitrary_public_commands_are_unknown(self):
        words = ['python', '-m', 'scenemi', 'train']
        self.assertFalse(self.recognized(words, uid=os.getuid() + 1))
        self.assertFalse(self.recognized(words, cwd=Path('/another/repository')))
        for action in ('resources', 'view', None):
            with self.subTest(action=action):
                self.assertFalse(self.recognized(words[:3] + ([action] if action else [])))


class ResourceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.args=['run','--gpu-sharing','--device','cuda','--memory-gib','10',
                   '--cpu-count','4','--','python','-m','scenemi','train',
                   '--config','configs/ego_only_clean_continuation.yaml','--device','cuda']

    def test_old_flags_reuse_cpu_ram_scope_without_gpu_gates(self):
        from scenemi import gpu_policy
        with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
             patch.object(gpu_policy.legacy,'inspect_gpu',side_effect=AssertionError('GPU gate called')), \
             patch.object(gpu_policy.legacy,'scope_command',return_value=['bounded-scope']) as scope, \
             patch.object(gpu_policy.subprocess,'call',return_value=0) as launch, \
             patch('sys.stdout',io.StringIO()),self.assertRaises(SystemExit) as exited:
            resource_runner.main(self.args)
        self.assertEqual(exited.exception.code,0)
        scope.assert_called_once_with(self.args[self.args.index('--')+1:],10,4)
        self.assertEqual(launch.call_args.kwargs['env']['OMP_NUM_THREADS'],'4')

    def test_insufficient_host_ram_still_prevents_launch(self):
        from scenemi import gpu_policy
        with patch.object(gpu_policy.legacy,'available_memory',return_value=3*2**30), \
             patch.object(gpu_policy.subprocess,'call') as launch, \
             patch('sys.stdout',io.StringIO()),self.assertRaisesRegex(SystemExit,'Host RAM'):
            resource_runner.main(self.args)
        launch.assert_not_called()

    def test_scope_failure_does_not_fall_back_to_unbounded_cpu_ram_execution(self):
        from scenemi import gpu_policy
        with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
             patch.object(gpu_policy.legacy,'scope_command',return_value=['bounded-scope']), \
             patch.object(gpu_policy.subprocess,'call',return_value=7) as launch, \
             patch('sys.stdout',io.StringIO()),self.assertRaises(SystemExit) as exited:
            resource_runner.main(self.args)
        self.assertEqual(exited.exception.code,7)
        self.assertEqual(launch.call_count,1)

    def test_checks_never_start_a_program_or_require_known_gpu_pids(self):
        from scenemi import gpu_policy
        with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
             patch.object(gpu_policy.legacy,'inspect_gpu',side_effect=AssertionError('PID gate called')), \
             patch.object(gpu_policy.subprocess,'call') as launch,patch('sys.stdout',io.StringIO()):
            resource_runner.main(['check','--device','cuda'])
        launch.assert_not_called()

    def test_historical_allocator_values_no_longer_install_gpu_caps(self):
        for cap in (None,5.5):
            config=dict(runtime=dict(memory_gib=10,cpu_count=4,allocator_gib=cap))
            with patch.object(execution,'ensure_bounded'), \
                 patch.object(execution.Path,'read_text',side_effect=['0::/test-scope',str(10*2**30)]), \
                 patch.object(execution.os,'sched_getaffinity',return_value={0,1}), \
                 patch('scenemi.gpu_policy.load_policy',return_value={}), \
                 patch('torch.cuda.set_per_process_memory_fraction') as fraction, \
                 patch('torch.cuda.get_device_properties') as properties:
                execution.configure_runtime(config,'cuda')
            fraction.assert_not_called();properties.assert_not_called()


if __name__=='__main__':unittest.main()
