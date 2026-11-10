"""Monitoring must not turn a restored step count into apparent throughput."""
import json
from pathlib import Path
import tempfile
import unittest

from scenemi.monitoring import measure_progress, progress_snapshot
from experiments.offline_camera_retrain_v1.ops.__main__ import read_status


class TrainingMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run/'training').mkdir()
        self.log = self.run/'training/updates.jsonl'

    def write(self, steps):
        self.log.write_text(''.join(json.dumps(dict(step=s))+'\n' for s in steps))

    def snapshot(self, monotonic):
        value = progress_snapshot(self.run)
        value['_monotonic'] = monotonic
        return value

    def test_restored_500_updates_are_not_counted_as_new_work(self):
        self.write(range(1, 501))
        before = self.snapshot(100)
        self.write(range(1, 503))
        result = measure_progress(before, self.snapshot(160), 3000)
        self.assertEqual(result['new_updates'], 2)
        self.assertEqual(result['seconds_per_update'], 30)
        self.assertEqual(result['remaining_seconds'], 2498*30)

    def test_resume_rewrites_are_not_counted_even_if_step_recovers(self):
        self.write(range(1, 503))
        before = self.snapshot(100)
        rows = self.log.read_text().splitlines()
        rows[-1] = json.dumps(dict(step=502, resumed=True))
        self.log.write_text('\n'.join(rows)+'\n'+json.dumps(dict(step=503))+'\n')
        result = measure_progress(before, self.snapshot(160), 3000)
        self.assertEqual(result['state'], 'log_rewritten_or_noncontiguous')
        self.assertIsNone(result['remaining_seconds'])

    def test_partial_update_is_ignored_and_no_activity_is_not_called_dead(self):
        self.write([500])
        before = self.snapshot(100)
        with self.log.open('a') as stream:
            stream.write('{"step": 501')
        result = measure_progress(before, self.snapshot(160))
        self.assertEqual(result['state'], 'no_updates_observed')
        self.assertIsNone(result['seconds_per_update'])

    def test_duplicate_updates_are_reported_without_throughput(self):
        self.write([500, 501])
        before = self.snapshot(100)
        self.write([500, 501, 500, 501, 502])
        result = measure_progress(before, self.snapshot(160))
        self.assertEqual(result['state'], 'log_rewritten_or_noncontiguous')

    def test_current_status_and_log_are_both_visible(self):
        self.write([500, 501])
        (self.run/'training/STATUS.json').write_text(json.dumps(dict(completed_updates=500)))
        result = read_status(self.run)
        self.assertEqual(result['training']['completed_updates'], 500)
        self.assertEqual(result['observed_progress']['last_update'], 501)


if __name__ == '__main__':
    unittest.main()
