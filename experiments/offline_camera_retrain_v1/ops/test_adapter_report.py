"""Tiny completed/failed evidence fixtures; no model, dataset, NumPy or CUDA."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.offline_camera_retrain_v1.ops import adapter_report as report


class AdapterReportTest(unittest.TestCase):
    def fixture(self, run, accepted=True):
        def write(name, value):
            path = run/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        (run/'training').mkdir()
        (run/'training/last.pt').write_bytes(b'tiny-checkpoint-identity-fixture')
        write('STATUS.json', dict(status='completed', completed_stage_updates=2,
            updated_at='2026-10-18T02:10:00+08:00', frozen_base_sha256='base', EGO_serialized_outputs_exact=True))
        write('LAUNCH.json', dict(launched_at='2026-10-18T02:00:00+08:00', supervisor_pid=41))
        write('training/protocol.json', dict(completed_steps=2, planned_steps=2, frozen_base_sha256='base'))
        (run/'training/trace.jsonl').write_text('\n'.join(json.dumps(dict(step=step, update_s=30.)) for step in (1, 2)))
        write('STORAGE.json', dict(stage='before', free_bytes={'/': 12345}, retention='one checkpoint'))
        write('CPU_RECHECK.json', dict(status='independent_CPU_primary_metrics_passed',
            CUDA_initialized=False, GPU_used_for_recheck=False, matched_input_arrays=True,
            EGO_serialized_outputs_exact=True, frozen_base_sha256='base', files_sha256={},
            final_checkpoint_sha256=report.digest(run/'training/last.pt')))
        groups=[]
        for dataset in ('trumans', 'egobody', 'rich'):
            for regime in ('clean', 'noisy'):
                groups.append(dict(dataset=dataset, regime=regime, windows=1,
                    baseline_hidden_mm=100., final_hidden_mm=90., hidden_no_material_regression_pass=True,
                    baseline_visible_mm=10., final_visible_mm=9., visible_guard_pass=True,
                    baseline_low_speed_foot_m_s=.1, final_low_speed_foot_m_s=.09, foot_speed_guard_pass=True,
                    motion_guards={key:dict(baseline=20., final=19., passed=True) for key in
                        ('bilateral_motion_rms_error_mm', 'bilateral_stride_change_abs_error_mm')}))
        gates = dict(status='pass' if accepted else 'failed', groups=groups, EGO_serialized_outputs_exact=True,
            hidden_macro=[dict(regime=r, baseline_hidden_mm=100., final_hidden_mm=90., passed=accepted) for r in ('clean', 'noisy')])
        write('ACCEPTANCE.json', gates)
        write('COMPARISON.json', dict(status='matched_primary_metrics_CPU_verified',
            TEST_used=False, independent_CPU_metric_recheck_pending=False, acceptance=gates,
            CPU_recheck_sha256=report.digest(run/'CPU_RECHECK.json'), per_window_rows=12))
        keys=('raw_gt_world_rms_mm', 'diagnostic_gt_window_rigid_fit_weighted_rms_residual_mm',
            'diagnostic_gt_window_similarity_fit_weighted_rms_residual_mm', 'observed_window_robust_similarity_fit_scale')
        write('ALIGNMENT_DIAGNOSTICS.json', dict(rows=[dict(model=stage, mode='exo', source='metrics',
            regime=regime, dataset=dataset, metrics=dict(zip(keys, (100., 95., 94., 1.))))
            for stage in ('baseline', 'final') for regime in ('clean', 'noisy') for dataset in ('trumans', 'egobody', 'rich')]))

    def test_verified_completed_report_and_distinct_training_time(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            self.fixture(run)
            (run/'DEPLOYMENT_LOAD_CHECK.json').write_text(json.dumps(dict(status='passed', step=1,
                checkpoint_sha256=report.digest(run/'training/last.pt'))))
            self.assertEqual(report.main(['--run', str(run)]), 0)
            body=(run/'RESULTS_zh.md').read_text()
            self.assertIn('工程验收通过', body)
            self.assertIn('实际更新累计 1.0 分钟', body)
            self.assertIn('复核完成 10.0 分钟', body)
            self.assertIn('最终独立部署加载：待核验', body)
            self.assertNotIn('http://localhost', body)
            storage=json.loads((run/'STORAGE.json').read_text())
            self.assertEqual(storage['free_bytes']['/'], 12345)
            self.assertIn('after', storage)

    def test_completed_but_acceptance_failed_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            self.fixture(run, accepted=False)
            self.assertEqual(report.main(['--run', str(run)]), 1)
            body=(run/'RESULTS_zh.md').read_text()
            self.assertIn('工程验收未通过', body)
            self.assertIn('hidden_macro[noisy].passed', body)
            self.assertNotIn('工程验收通过', body)

    def test_changed_checkpoint_refuses_report(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            self.fixture(run)
            (run/'training/last.pt').write_bytes(b'changed-after-recheck')
            self.assertEqual(report.main(['--run', str(run)]), 1)
            self.assertIn('checkpoint SHA 不匹配', (run/'RESULTS_zh.md').read_text())

    def test_new_status_ignores_old_failed_supervisor(self):
        status=dict(status='training', pid=200, updated_at='2026-10-18T02:01:00+08:00')
        job=dict(status='failed', child_pid=100, supervisor_pid=40, returncode=1)
        launch=dict(launched_at='2026-10-18T02:00:00+08:00', supervisor_pid=41)
        self.assertIsNone(report.terminal_error(status, job, launch))
        status.update(pid=100)
        self.assertIsNone(report.terminal_error(status, job, launch))
        job.update(supervisor_pid=41)
        self.assertIn('returncode=1', report.terminal_error(status, job, launch))

    def test_actual_failed_attempt_records_error_without_result(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/'STATUS.json').write_text(json.dumps(dict(status='failed', error='actual loader CRC mismatch')))
            self.assertEqual(report.main(['--run', str(run)]), 1)
            self.assertIn('actual loader CRC mismatch', (run/'RESULTS_zh.md').read_text())


if __name__ == '__main__':
    unittest.main()
