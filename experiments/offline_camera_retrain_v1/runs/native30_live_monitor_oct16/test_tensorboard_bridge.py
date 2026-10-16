import datetime as dt
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SPEC = importlib.util.spec_from_file_location('tensorboard_bridge', Path(__file__).with_name('tensorboard_bridge.py'))
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class BridgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'trace.jsonl'
        self.source.write_text('')
        self.config = {'sources': [dict(key='train', run='new_native30', role='native_trace', path=str(self.source))]}
        self.instance = bridge.Bridge(self.root / 'monitor', self.config)

    def tearDown(self):
        self.instance.close()
        self.temporary.cleanup()

    def append(self, row):
        with self.source.open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def events(self, tag, run='new_native30'):
        accumulator = EventAccumulator(str(self.root / 'monitor' / 'tensorboard' / run), size_guidance={'scalars': 0})
        accumulator.Reload()
        return accumulator.Scalars(tag)

    def restart(self):
        self.instance.close()
        self.instance = bridge.Bridge(self.root / 'monitor', self.config)

    def test_real_event_reader_receives_original_supervision_values(self):
        self.append(dict(step=1, loss=3., terms={'base': {'total': 2., 'raw_x0': .25}, 'geometry': {'contact': .5}}, elapsed_s=2., update_s=1.5))
        self.instance.poll()
        self.assertEqual(self.events('train/terms/base/total')[0].value, 2.)
        self.assertEqual(self.events('train/terms/base/raw_x0')[0].value, .25)
        self.assertEqual(self.events('train/terms/geometry/contact')[0].value, .5)

    def test_restart_and_repoll_do_not_duplicate_events(self):
        self.append(dict(step=1, loss=1., elapsed_s=3., update_s=2.))
        self.instance.poll()
        self.restart()
        self.instance.poll()
        self.append(dict(step=2, loss=.5, elapsed_s=5., update_s=2.))
        self.instance.poll()
        self.assertEqual([e.step for e in self.events('train/loss')], [1, 2])

    def test_crash_after_database_commit_before_publish_is_recovered(self):
        self.append(dict(step=1, loss=1., elapsed_s=3.))
        self.instance.ingest_source(self.config['sources'][0])
        self.restart()
        self.instance.poll()
        self.assertEqual(len(self.events('train/loss')), 1)
        self.assertEqual(self.instance.status()['sources']['train']['rows'], 1)

    def test_crash_after_atomic_publish_before_mark_is_idempotent(self):
        self.append(dict(step=1, loss=1., elapsed_s=3.))
        self.instance.poll()
        self.instance.db.execute('UPDATE batches SET published=0')
        self.instance.db.commit()
        self.restart()
        self.instance.poll()
        self.assertEqual(len(self.events('train/loss')), 1)

    def test_partial_line_waits_and_clocks_commit_only_once(self):
        payload = json.dumps(dict(step=1, loss=1., elapsed_s=3., update_s=2.))
        self.source.write_text(payload[:15])
        self.instance.poll()
        self.assertEqual(self.instance.status()['sources']['train']['offset'], 0)
        with self.source.open('a') as handle:
            handle.write(payload[15:] + '\n')
        self.instance.poll()
        self.assertEqual(len(self.events('train/loss')), 1)
        self.assertEqual(self.instance.status()['sources']['train']['update_seconds'], 2.)

    def test_resume_elapsed_reset_accumulates_measured_sessions(self):
        self.append(dict(step=1000, loss=1., elapsed_s=2000., update_s=2.))
        self.append(dict(step=1001, loss=1., elapsed_s=5., update_s=3.))
        self.append(dict(step=1002, loss=1., elapsed_s=7., update_s=2.))
        self.instance.poll()
        clock = self.instance.status()['sources']['train']
        self.assertEqual(clock['active_seconds'], 2007.)
        self.assertEqual(clock['update_seconds'], 7.)
        self.assertEqual([e.step for e in self.events('train/loss', 'new_native30__active_seconds')], [2000, 2005, 2007])

    def test_truncated_source_rebuilds_without_stale_points(self):
        self.append(dict(step=1, loss=1., elapsed_s=3.))
        self.instance.poll()
        self.source.write_text('')
        self.instance.poll()
        self.append(dict(step=1, loss=2., elapsed_s=2., update_s=1.))
        self.instance.poll()
        self.assertEqual([e.value for e in self.events('train/loss')], [2.])
        self.assertTrue((self.root / 'monitor' / 'REBUILD_EVENTS.jsonl').exists())

    def test_changed_imported_bytes_rebuilds_and_running_reader_purges_old_points(self):
        self.append(dict(step=1, loss=1., elapsed_s=3.))
        self.append(dict(step=2, loss=1., elapsed_s=5.))
        self.instance.poll()
        reader = EventAccumulator(str(self.root / 'monitor' / 'tensorboard' / 'new_native30'), size_guidance={'scalars': 0})
        reader.Reload()
        self.assertEqual(len(reader.Scalars('train/loss')), 2)
        self.source.write_text(json.dumps(dict(step=1, loss=2., elapsed_s=3.)) + '\n')
        self.instance.poll()
        reader.Reload()
        self.assertEqual([e.value for e in reader.Scalars('train/loss')], [2.])

    def test_missing_timer_does_not_invent_update_time(self):
        self.config['sources'][0]['role'] = 'legacy_train'
        self.append(dict(step=20, losses={'total': 1.2}, elapsed_s=20.))
        self.instance.poll()
        accumulator = EventAccumulator(str(self.root / 'monitor' / 'tensorboard' / 'new_native30'))
        accumulator.Reload()
        self.assertNotIn('time/train_update_hours', accumulator.Tags()['scalars'])

    def test_source_absolute_walltime_is_preserved(self):
        supplied = '2026-10-16T13:00:00+08:00'
        self.append(dict(step=1, loss=1., checked_at=supplied))
        self.instance.poll()
        self.assertEqual(self.events('train/loss')[0].wall_time, dt.datetime.fromisoformat(supplied).timestamp())

    def test_validation_and_feet_do_not_double_count_call_duration(self):
        validation = self.root / 'validation.jsonl'
        feet = self.root / 'feet.jsonl'
        validation.write_text(json.dumps(dict(step=1000, summary={'fixed/clean': {'MPJPE_mm': 200.}})) + '\n')
        feet.write_text(json.dumps(dict(step=1000, validation_seconds=120., summary={'fixed/clean': {'body_local_foot_error_mm': 300.}})) + '\n')
        self.config['sources'] += [dict(key='val', run='new_native30', role='native_validation', path=str(validation)), dict(key='feet', run='new_native30', role='native_feet', path=str(feet))]
        self.instance.poll()
        self.assertAlmostEqual(self.events('time/validation_hours')[0].value, 120./3600., places=6)
        self.assertEqual(self.events('validation/fixed/clean/MPJPE_mm')[0].value, 200.)
        self.assertEqual(self.events('feet/fixed/clean/body_local_foot_error_mm')[0].value, 300.)

    def test_custom_dashboard_published_once_and_uses_existing_metric_tags(self):
        self.append(dict(step=1, loss=1., elapsed_s=3.))
        self.instance.poll()
        self.restart()
        self.instance.poll()
        accumulator = EventAccumulator(str(self.root / 'monitor' / 'tensorboard' / 'new_native30'), size_guidance={'tensors': 0}).Reload()
        events = accumulator.Tensors('custom_scalars__config__')
        self.assertEqual(len(events), 1)
        layout = bridge.layout_pb2.Layout.FromString(events[0].tensor_proto.string_val[0])
        self.assertEqual(len(layout.category), 10)
        self.assertFalse(layout.category[0].closed)
        self.assertTrue(layout.category[4].closed)
        chart_patterns = [pattern for category in layout.category for chart in category.chart for pattern in chart.multiline.tag]
        self.assertIn(r'^loss/total$', chart_patterns)
        self.assertIn(r'^time/train_(update|compute|preparation)_hours$', chart_patterns)


if __name__ == '__main__':
    unittest.main()
