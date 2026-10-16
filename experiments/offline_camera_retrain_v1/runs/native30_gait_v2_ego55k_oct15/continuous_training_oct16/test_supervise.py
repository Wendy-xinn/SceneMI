"""CPU tests for preserving the full-run supervision contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import supervise


class SupervisionTests(unittest.TestCase):
    def test_valid_checkpoint_retains_every_resume_field(self):
        state = {name: {} for name in supervise.REQUIRED_STATE}
        state.update(step=5200, planned_steps=55000, effective_config={'same': True},
                     optimizer={'state': {1: {'step': 5200}}})
        self.assertEqual(supervise.validate_checkpoint(state, {'same': True}), 5200)
        del state['cuda_rng']
        with self.assertRaisesRegex(ValueError, 'optimizer/RNG'):
            supervise.validate_checkpoint(state, {'same': True})

    def test_changed_schedule_or_configuration_is_rejected(self):
        state = {name: {} for name in supervise.REQUIRED_STATE}
        state.update(step=5200, planned_steps=10000, optimizer={'state': {1: 1}})
        with self.assertRaisesRegex(ValueError, '55000'):
            supervise.validate_checkpoint(state, {})
        state.update(planned_steps=55000, effective_config={'changed': True})
        with self.assertRaisesRegex(ValueError, 'configuration'):
            supervise.validate_checkpoint(state, {})

    def test_tail_ignores_partial_line_and_does_not_need_full_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'trace.jsonl'
            path.write_bytes((b'x' * 500000) + b'\n' + b'{"step": 5201}\n{"step": 5202')
            self.assertEqual(supervise.tail_json(path, 256), {'step': 5201})

    def test_crc_and_temporary_io_allow_retry(self):
        self.assertTrue(supervise.restartable_io_error('BadZipFile: Bad CRC-32 for data'))
        self.assertTrue(supervise.restartable_io_error('OSError: [Errno 5] Input/output error'))

    def test_integrity_or_numeric_failures_never_retry(self):
        for text in ('ValueError: Cache SHA256 mismatch', 'Nonfinite native30 objective',
                     'Code/calibration changed since preparation',
                     'Resume requires identical configuration, dependencies, and schedule',
                     'Bad CRC followed by AssertionError', 'unclassified worker failure'):
            with self.subTest(text=text):
                self.assertFalse(supervise.restartable_io_error(text))

    def test_failure_budget_resets_only_after_checkpoint_progress(self):
        self.assertEqual(supervise.retry_count(2, 5200, 5200), 3)
        self.assertEqual(supervise.retry_count(2, 5200, 5300), 1)
        self.assertEqual(supervise.retry_count(3, 5200, 5200), 4)

    def test_exact_process_script_detection_ignores_similar_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'runs/baseline/continuous'
            root.mkdir(parents=True)
            proc = Path(temporary) / 'proc'
            for pid, name in ((222, 'worker.py'), (223, 'worker.py.bak')):
                folder = proc / str(pid)
                folder.mkdir(parents=True)
                (folder / 'cmdline').write_bytes(b'python\0' + str(root / name).encode() + b'\0')
                (folder / 'cwd').symlink_to(root, target_is_directory=True)
            self.assertEqual(supervise.conflicting_processes(root, proc),
                             [{'pid': 222, 'script': str(root / 'worker.py')}])

    def test_manifest_must_include_all_execution_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'runs/baseline/continuous'
            root.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'omits'):
                supervise.verify_sources({'expected_source_sha256': {'other': 'x'}}, root)

    def run_controller(self, temporary, steps, returncode, output=''):
        root = Path(temporary) / 'runs/baseline/continuous'
        root.mkdir(parents=True)
        (root / 'PROTOCOL.json').write_text('{}')
        child = mock.Mock(pid=654321, returncode=returncode)
        child.poll.return_value = returncode
        def launch(*args, **kwargs):
            kwargs['stdout'].write(output.encode())
            return child
        with mock.patch.object(supervise, 'verify_sources'), \
             mock.patch.object(supervise, 'conflicting_processes', return_value=[]), \
             mock.patch.object(supervise, 'checkpoint_step', side_effect=steps), \
             mock.patch.object(supervise.subprocess, 'Popen', side_effect=launch) as popen:
            supervise.supervise(root)
        return root, popen

    def test_success_requires_actual_55000_checkpoint_and_zero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, popen = self.run_controller(temporary, [5200, 55000], 0)
            status = json.loads((root / 'STATUS.json').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(status['latest_committed_step'], 55000)
            self.assertEqual(popen.call_count, 1)

    def test_early_zero_exit_is_failure_instead_of_false_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'before verified completion'):
                self.run_controller(temporary, [5200, 5200], 0)

    def test_three_io_restarts_then_stop_without_resetting_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, 'consecutive_failures=4'):
                self.run_controller(temporary, [5200] * 5, 1,
                                    'BadZipFile: Bad CRC-32 for array\n')
            events = [json.loads(line) for line in
                      (Path(temporary) / 'runs/baseline/continuous/EVENTS.jsonl')
                      .read_text().splitlines()]
            self.assertEqual(sum(row['event'] == 'worker_started' for row in events), 4)
            self.assertEqual(sum(row['event'] == 'transient_io_restart' for row in events), 3)


if __name__ == '__main__':
    unittest.main()
