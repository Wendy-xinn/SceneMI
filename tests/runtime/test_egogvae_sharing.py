"""The user-managed policy cannot gate GPU usage or change model sources."""
from pathlib import Path
import hashlib
import tempfile
import io
from types import SimpleNamespace
from unittest.mock import patch

from scenemi import gpu_policy
from scenemi.egogvae_sharing import verify_parent_sources


def test_gpu_launch_never_queries_memory_processes_or_utilization():
    with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
         patch.object(gpu_policy.legacy,'inspect_gpu',side_effect=AssertionError('GPU gate called')):
        report=gpu_policy.admission('cuda',10)
    assert report['admitted'] and report['gpu_limits_disabled']
    assert report['gpu']['allocator_gib'] is None
    assert not report['gpu']['process_count_checked']
    assert not report['gpu']['minimum_free_enforced']


def test_legacy_cuda_launcher_also_has_no_gpu_gates():
    with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
         patch.object(gpu_policy.legacy,'inspect_gpu',side_effect=AssertionError('Legacy GPU gate called')):
        report=gpu_policy.legacy.admission('cuda',10)
    assert report['admitted'] and report['gpu_limits_disabled']


def test_cuda_configuration_does_not_install_an_allocator_limit():
    import torch
    with patch.object(torch.cuda,'set_per_process_memory_fraction',side_effect=AssertionError('Allocator capped')):
        assert gpu_policy.configure_cuda()['allocator_gib'] is None
        from scenemi.parallel_resources import configure_cuda as parallel
        from scenemi.resources import configure_cuda as amass
        assert parallel()['allocator_gib'] is None
        assert amass({})['allocator_gib'] is None


def test_host_memory_is_still_isolated():
    with patch.object(gpu_policy.legacy,'available_memory',return_value=3*2**30):
        assert not gpu_policy.admission('cuda',10)['admitted']


def test_a_runtime_overlay_cannot_whitelist_model_edits():
    with tempfile.TemporaryDirectory() as directory:
        file=Path(directory)/'model.py';file.write_text('new model')
        parent=dict(sources={str(file):'old'})
        overlay=dict(parent_runtime_source_updates={str(file):dict(before='old',after='new')})
        try:verify_parent_sources(parent,overlay)
        except ValueError:return
        raise AssertionError('Model change was incorrectly accepted as execution-only')


def test_unchanged_parent_code_is_accepted_without_resealing_weights():
    with tempfile.TemporaryDirectory() as directory:
        file=Path(directory)/'model.py';file.write_text('original model')
        expected=hashlib.sha256(file.read_bytes()).hexdigest()
        verify_parent_sources(dict(sources={str(file):expected}),{})


def test_all_current_admission_entrypoints_use_the_same_policy():
    from scenemi import resources,parallel_resources,resource_runner
    with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
         patch.object(gpu_policy.legacy,'inspect_gpu',side_effect=AssertionError('Legacy GPU gate called')):
        reports=[resources.admission('cuda',10,config={}),parallel_resources.admission('cuda',10,999),
                 resource_runner.shared_admission('cuda',10,999)]
    assert all(r['admitted'] and r['gpu_limits_disabled'] for r in reports)


def test_replay_runtime_check_does_not_query_gpu_headroom():
    from scenemi.replay import runtime
    with patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
         patch.object(runtime.os,'sched_getaffinity',return_value={0,1}), \
         patch.object(runtime.subprocess,'check_output',side_effect=AssertionError('GPU headroom queried')), \
         patch('sys.argv',['runtime','check','--cpu-count','2','--cpu-list','0,1']), \
         patch('sys.stdout',io.StringIO()):
        runtime.main()


def test_replay_training_launch_preserves_cpu_settings_without_gpu_gates():
    from scenemi.replay import training
    with tempfile.TemporaryDirectory() as directory:
        c=dict(output='run',runtime=dict(memory_gib=10,cpu_affinity=[0,1],gpu_physical_reserve_gib=4))
        args=SimpleNamespace(config=Path(directory)/'config.yaml',device='cuda',task='train')
        with patch.object(training,'ROOT',Path(directory)), \
             patch.object(training,'configuration',return_value=c), \
             patch.object(gpu_policy.legacy,'available_memory',return_value=30*2**30), \
             patch.object(training.os,'sched_getaffinity',return_value={0,1}), \
             patch.object(training.os,'sched_setaffinity'), \
             patch.object(training.subprocess,'check_output',side_effect=AssertionError('GPU gate queried')), \
             patch.object(training.legacy,'scope_command',return_value=['bounded-scope']), \
             patch.object(training.subprocess,'call',return_value=0) as launch, \
             patch('sys.stdout',io.StringIO()):
            assert training.launch(args)==0
        assert launch.call_args.kwargs['env']['OMP_NUM_THREADS']=='1'
