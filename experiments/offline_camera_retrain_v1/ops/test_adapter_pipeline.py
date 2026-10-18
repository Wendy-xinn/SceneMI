"""CPU-only completion guards; no training or motion sampling in these tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import adapter_pipeline as pipeline


class CompletionTests(unittest.TestCase):
    def fixture(self, root):
        stage=Path(root);(stage/'pipeline').mkdir();(stage/'prepared').mkdir();(stage/'training').mkdir()
        job=dict(supervisor_pid=111,child_pid=112,launched_at='now',command=['python','stage'],source_commit='commit',
                 returncode=0,status='failed',independent_cpu_verified=False)
        pipeline.write(stage/'JOB.json',job)
        pipeline.write(stage/'LAUNCH.json',dict(supervisor_source="verified=check.get('status')=='independent_cpu_passed'"))
        config=dict(warm_start=dict(sha256='parent'))
        pipeline.write(stage/'config.json',config)
        pipeline.write(stage/'prepared/effective_config.json',dict(config=config,code_sha256={}))
        (stage/'training/last.pt').write_bytes(b'full-final-checkpoint')
        checksum=pipeline.digest(stage/'training/last.pt')
        cpu=dict(status=pipeline.CPU_STATUS,matched_input_arrays=True,EGO_serialized_outputs_exact=True,CUDA_initialized=False,
                 GPU_used_for_recheck=False,frozen_base_sha256=pipeline.BASE_HASH,final_checkpoint_sha256=checksum,
                 parent_checkpoint_sha256='parent',files_sha256={})
        pipeline.write(stage/'CPU_RECHECK.json',cpu)
        pipeline.write(stage/'STATUS.json',dict(status='completed',completed_stage_updates=1000,frozen_base_sha256=pipeline.BASE_HASH,EGO_serialized_outputs_exact=True))
        pipeline.write(stage/'training/protocol.json',dict(completed_steps=1000,planned_steps=1000,frozen_base_sha256=pipeline.BASE_HASH))
        pipeline.write(stage/'prepared/input_checks.json',dict(status='passed',TEST_used=False,nonzero_adapter_EGO_bypass_exact=True,
            hidden_truth_poison_unchanged=True,frozen_EXO_without_body_labels=True,
            zero_init_equivalence={mode:dict(exact_equal=True) for mode in ('ego','exo','mixed')}))
        gates=dict(status='failed')
        pipeline.write(stage/'ACCEPTANCE.json',gates)
        pipeline.write(stage/'COMPARISON.json',dict(status='matched_primary_metrics_CPU_verified',TEST_used=False,
            independent_CPU_metric_recheck_pending=False,CPU_recheck_sha256=pipeline.digest(stage/'CPU_RECHECK.json'),acceptance=gates))
        (stage/'training/trace.jsonl').write_text(''.join(json.dumps(dict(step=step))+'\n' for step in range(1,1001)))
        return stage,job,cpu

    def test_real_status_and_exact1000_identity_required_but_acceptance_may_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,job,cpu=self.fixture(directory)
            evidence=pipeline.verify_completed_stage(stage)
            self.assertEqual(evidence['completed_stage_updates'],1000)
            self.assertEqual(evidence['engineering_acceptance'],'failed')
            for key,bad in [('status','independent_cpu_passed'),('EGO_serialized_outputs_exact',False),
                            ('CUDA_initialized',True),('GPU_used_for_recheck',True),('final_checkpoint_sha256','wrong')]:
                altered=dict(cpu,**{key:bad});pipeline.write(stage/'CPU_RECHECK.json',altered)
                with self.subTest(key=key),self.assertRaises(ValueError):pipeline.verify_completed_stage(stage)
            pipeline.write(stage/'CPU_RECHECK.json',cpu)
            (stage/'training/trace.jsonl').write_text('{"step":1000}\n')
            with self.assertRaises(ValueError):pipeline.verify_completed_stage(stage)

    def test_failed_command_never_relabelled_and_label_poison_proof_required(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,job,cpu=self.fixture(directory)
            job['returncode']=1;pipeline.write(stage/'JOB.json',job)
            with self.assertRaises(ValueError):pipeline.verify_completed_stage(stage)
            job['returncode']=0;pipeline.write(stage/'JOB.json',job)
            checks=pipeline.read(stage/'prepared/input_checks.json');checks['hidden_truth_poison_unchanged']=False
            pipeline.write(stage/'prepared/input_checks.json',checks)
            with self.assertRaises(ValueError):pipeline.verify_completed_stage(stage)

    def test_job_classification_fix_archives_exact_bytes_after_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,job,cpu=self.fixture(directory)
            original=(stage/'JOB.json').read_bytes()
            plan=dict(stage_job_identity=pipeline.job_identity(job),processes=dict(supervisor=None,child=None))
            evidence=pipeline.verify_completed_stage(stage)
            result=pipeline.correct_classification(stage,plan,evidence)
            self.assertEqual(result['status'],'classification_corrected_only')
            self.assertEqual((stage/'pipeline/JOB.original.json').read_bytes(),original)
            fixed=pipeline.read(stage/'JOB.json')
            self.assertEqual(fixed['returncode'],0);self.assertEqual(fixed['status'],'completed')
            self.assertTrue(fixed['independent_cpu_verified'])

    def test_never_rewrite_a_live_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,job,cpu=self.fixture(directory)
            plan=dict(stage_job_identity=pipeline.job_identity(job),processes=dict(supervisor={'pid':1},child=None))
            with mock.patch.object(pipeline,'alive',return_value=True),self.assertRaises(ValueError):
                pipeline.correct_classification(stage,plan,pipeline.verify_completed_stage(stage))
            self.assertFalse((stage/'pipeline/JOB.original.json').exists())


class PipelineLifecycleTests(unittest.TestCase):
    fixture = CompletionTests.fixture
    def test_full_dry_lifecycle_negative_report_continues_and_GPU_is_last(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,job,cpu=self.fixture(directory)
            out=stage/'expanded';out.mkdir()
            plan=dict(stage_job_identity=pipeline.job_identity(job),processes=dict(supervisor=None,child=None),
                      source_sha256={},fixed_files_sha256={})
            calls=[]
            checksum=cpu['final_checkpoint_sha256']
            def popen(arguments,**kwargs):
                calls.append(arguments)
                if 'deployment-check' in arguments:
                    pipeline.write(stage/'DEPLOYMENT_LOAD_CHECK.json',dict(status='passed',step=1000,checkpoint_sha256=checksum,CUDA_initialized=False))
                    code=0
                elif 'report' in arguments:
                    self.assertEqual(pipeline.read(stage/'JOB.json')['status'],'completed')
                    (stage/'RESULTS_zh.md').write_text('Valid negative engineering result')
                    pipeline.write(stage/'pipeline/REPORT_CHECK.json',dict(status='valid_report',report_returncode=1,engineering_acceptance_passed=False,
                        report_sha256=pipeline.digest(stage/'RESULTS_zh.md'),acceptance_sha256=pipeline.digest(stage/'ACCEPTANCE.json'),CPU_recheck_sha256=pipeline.digest(stage/'CPU_RECHECK.json')))
                    code=1
                else:
                    self.assertEqual(arguments[3],pipeline.EVALUATOR)
                    self.assertEqual(len(calls),3)
                    pipeline.write(out/'CPU_RECHECK.json',dict(status='independent_cpu_passed',predictions=384,CUDA_initialized=False))
                    pipeline.write(out/'SUMMARY.json',dict(status='complete_CPU_verified'))
                    code=0
                return mock.Mock(pid=100+len(calls),returncode=code,poll=mock.Mock(return_value=code))
            with mock.patch.object(pipeline,'make_plan',return_value=plan),mock.patch.object(pipeline.subprocess,'Popen',side_effect=popen):
                pipeline.run_pipeline(stage,out,poll=1)
            result=pipeline.read(stage/'PIPELINE.json')
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['expanded_result']['predictions'],384)
            self.assertFalse(result['stage_report']['engineering_acceptance_passed'])
            self.assertEqual(result['visuals']['status'],'skipped_no_manifest')
            with self.assertRaises(ValueError):pipeline.run_pipeline(stage,out,poll=1)

    def test_optional_visual_failure_is_separate_and_no_GPU_started(self):
        with tempfile.TemporaryDirectory() as directory:
            stage,job,cpu=self.fixture(directory)
            (stage/'VIEWER_CASES.json').write_text('{"cases":[]}')
            command=mock.Mock(return_value=2)
            result=pipeline.optional_visuals(stage,command)
            self.assertEqual(result['status'],'visuals_failed')
            self.assertFalse(result['GPU_used'])
            self.assertEqual(command.call_count,1)
            self.assertIn('--prepare-only',command.call_args.args[1])


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.job=dict(supervisor_pid=1,child_pid=2,launched_at='now',command=['stage'],source_commit='commit')
        self.expected=pipeline.job_identity(self.job)

    def test_no_forever_wait_for_dead_supervisor_or_nonzero_exit(self):
        with self.assertRaises(ValueError):pipeline.wait_state(self.job,self.expected,False,False)
        with self.assertRaises(ValueError):pipeline.wait_state(dict(self.job,returncode=1),self.expected,False,False)
        self.assertEqual(pipeline.wait_state(self.job,self.expected,True,True),'running')
        self.assertEqual(pipeline.wait_state(self.job,self.expected,True,False),'wait_supervisor_commit')
        self.assertEqual(pipeline.wait_state(dict(self.job,returncode=0),self.expected,True,False),'wait_supervisor_exit')
        self.assertEqual(pipeline.wait_state(dict(self.job,returncode=0),self.expected,False,False),'finished')

    def test_pid_reuse_and_zombie_are_not_live(self):
        original=dict(pid=123,start_ticks=10,state='S')
        for current in [None,dict(pid=123,start_ticks=11,state='S'),dict(pid=123,start_ticks=10,state='Z')]:
            with mock.patch.object(pipeline,'process_identity',return_value=current):self.assertFalse(pipeline.alive(original))
        with mock.patch.object(pipeline,'process_identity',return_value=original):self.assertTrue(pipeline.alive(original))

    def test_changed_job_identity_rejected(self):
        with self.assertRaises(ValueError):pipeline.wait_state(dict(self.job,child_pid=999),self.expected,True,True)


class ReportTests(unittest.TestCase):
    def test_valid_negative_report_can_continue_but_ambiguous_exit1_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);(stage/'pipeline').mkdir()
            for name in ('RESULTS_zh.md','ACCEPTANCE.json','CPU_RECHECK.json'):(stage/name).write_text('evidence')
            result=dict(status='valid_report',report_returncode=1,engineering_acceptance_passed=False,
                report_sha256=pipeline.digest(stage/'RESULTS_zh.md'),acceptance_sha256=pipeline.digest(stage/'ACCEPTANCE.json'),CPU_recheck_sha256=pipeline.digest(stage/'CPU_RECHECK.json'))
            pipeline.write(stage/'pipeline/REPORT_CHECK.json',result)
            self.assertEqual(pipeline.accept_report_exit(stage,1)['status'],'valid_report')
            result['status']='failed';pipeline.write(stage/'pipeline/REPORT_CHECK.json',result)
            with self.assertRaises(ValueError):pipeline.accept_report_exit(stage,1)
            result['status']='valid_report';pipeline.write(stage/'pipeline/REPORT_CHECK.json',result)
            with self.assertRaises(ValueError):pipeline.accept_report_exit(stage,2)

    def test_dependency_mutation_stops_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.py';source.write_text('original')
            plan=dict(source_sha256={str(source):pipeline.digest(source)},fixed_files_sha256={})
            pipeline.verify_plan(plan);source.write_text('edited')
            with self.assertRaises(ValueError):pipeline.verify_plan(plan)


if __name__=='__main__':unittest.main()
