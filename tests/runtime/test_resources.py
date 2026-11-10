"""Current GPU policy ignores historical budgets and retains host RAM bounds."""
import io
import unittest
import torch
from unittest.mock import patch

from scenemi import gpu_policy
from scenemi.resources import admission,configure_cuda,main


class SharedResourceTests(unittest.TestCase):
    def test_gpu_occupancy_and_unknown_pids_are_not_launch_conditions(self):
        with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
             patch.object(gpu_policy.legacy,'inspect_gpu',side_effect=AssertionError('GPU gate called')):
            result=admission('cuda',12,config={})
        self.assertTrue(result['admitted'])
        self.assertTrue(result['gpu_limits_disabled'])

    def test_no_gpu_policy_can_override_insufficient_host_ram(self):
        with patch.object(gpu_policy.legacy,'available_memory',return_value=3*2**30):
            self.assertFalse(admission('cuda',12,config={})['admitted'])

    def test_no_pytorch_memory_fraction_is_installed(self):
        with patch('torch.cuda.set_per_process_memory_fraction') as cap:
            result=configure_cuda({})
        cap.assert_not_called()
        self.assertIsNone(result['pytorch_allocator_limit_bytes'])

    def test_legacy_amass_wrapper_forwards_its_training_config(self):
        with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
             patch.object(gpu_policy.legacy,'scope_command',return_value=['scope']) as scope, \
             patch.object(gpu_policy.subprocess,'call',return_value=0), \
             patch('sys.stdout',io.StringIO()),self.assertRaises(SystemExit) as exited:
            main(['run','--config','custom.yaml','--','python','-m','scenemi','amass','train'])
        self.assertEqual(exited.exception.code,0)
        self.assertEqual(scope.call_args.args[0][-2:],['--config','custom.yaml'])


if __name__=='__main__':unittest.main()
