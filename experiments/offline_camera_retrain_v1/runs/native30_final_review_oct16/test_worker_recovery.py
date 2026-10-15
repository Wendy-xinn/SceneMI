"""CPU-only checks for checkpoint-boundary log recovery and paired inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_module('native30_recovery_worker', 'worker.py')
manager = load_module('native30_recovery_manager', 'run_review.py')


def json_lines(rows):
    return ''.join(json.dumps(row) + '\n' for row in rows)


class TrimUncommittedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'trace.jsonl'
        self.archive = self.path.with_suffix('.jsonl.uncommitted')

    def test_keeps_committed_rows_and_archives_future_step_and_partial_json(self):
        committed = [dict(step=10, loss=1.2), dict(step=11, loss=1.1)]
        future = json.dumps(dict(step=12, loss=1.0))
        partial = '{"step": 13, "loss":'
        self.path.write_text(json_lines(committed) + future + '\n' + partial)

        worker.trim_uncommitted(self.path, committed_step=11)

        self.assertEqual(self.path.read_text(), json_lines(committed))
        self.assertEqual(self.archive.read_text(), future + '\n' + partial + '\n')
        self.assertFalse(self.path.with_suffix('.jsonl.recovering').exists())

    def test_second_recovery_does_not_duplicate_discarded_rows(self):
        self.path.write_text(json_lines([dict(step=11), dict(step=12)]) + '{"step":')
        worker.trim_uncommitted(self.path, committed_step=11)
        recovered = self.path.read_bytes()
        discarded = self.archive.read_bytes()

        worker.trim_uncommitted(self.path, committed_step=11)

        self.assertEqual(self.path.read_bytes(), recovered)
        self.assertEqual(self.archive.read_bytes(), discarded)

    def test_already_committed_log_is_preserved_byte_for_byte(self):
        original = b'{ "step" : 10, "loss": 1.0 }\n{"step":11}'
        self.path.write_bytes(original)

        worker.trim_uncommitted(self.path, committed_step=11)

        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(self.archive.exists())

    def test_missing_log_remains_missing(self):
        worker.trim_uncommitted(self.path, committed_step=11)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.archive.exists())

    def test_log_with_only_uncommitted_content_becomes_empty(self):
        original = json_lines([dict(step=12)]) + '{"step":13'
        self.path.write_text(original)

        worker.trim_uncommitted(self.path, committed_step=11)

        self.assertEqual(self.path.read_text(), '')
        self.assertEqual(self.archive.read_text(), original + '\n')


class PairedInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patch = mock.patch.object(manager, 'ROOT', self.root)
        patch.start()
        self.addCleanup(patch.stop)
        self.inputs = [dict(step=step, timesteps=[25, 300], scene=True,
                            noisy_sha256='noise-' + str(step),
                            trajectory_sha256='trajectory-' + str(step),
                            observation_meta_sha256='meta-' + str(step))
                       for step in (10, 11, 12, 13)]
        self.training = [dict(step=step, timesteps=[25, 300], scene=True,
                              identities=[dict(sequence_id='source-A', start=step * 64,
                                               group='trumans', mode='ego')])
                         for step in (10, 11, 12, 13)]

    def write_arm(self, name, *, inputs=None, training=None):
        arm = self.root / 'arms' / name
        (arm / 'training').mkdir(parents=True)
        (arm / 'input_trace.jsonl').write_text(json_lines(self.inputs if inputs is None else inputs))
        (arm / 'training' / 'trace.jsonl').write_text(json_lines(self.training if training is None else training))
        return arm

    def test_matching_interval_passes_and_ignores_rows_outside_interval(self):
        self.write_arm('control')
        inputs = copy.deepcopy(self.inputs)
        training = copy.deepcopy(self.training)
        inputs[0]['noisy_sha256'] = 'different-outside-interval'
        training[-1]['identities'][0]['sequence_id'] = 'different-outside-interval'
        self.write_arm('candidate', inputs=inputs, training=training)

        manager.paired_inputs(('control', 'candidate'), 'phase1', start=10, end=12)

        report = json.loads((self.root / 'phase1_PAIRING.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['counts'], dict(control=2, candidate=2))
        self.assertEqual((report['start_step'], report['end_step']), (10, 12))

    def test_changed_noise_or_observation_hash_is_rejected(self):
        self.write_arm('control')
        self.write_arm('candidate')
        for field in ('noisy_sha256', 'trajectory_sha256', 'observation_meta_sha256'):
            with self.subTest(field=field):
                inputs = copy.deepcopy(self.inputs)
                inputs[1][field] = 'unexpected-input'
                (self.root / 'arms/candidate/input_trace.jsonl').write_text(json_lines(inputs))
                with self.assertRaisesRegex(AssertionError, 'Noisy/observed inputs differ'):
                    manager.paired_inputs(('control', 'candidate'), field, start=10, end=12)
                self.assertFalse((self.root / (field + '_PAIRING.json')).exists())

    def test_changed_source_identity_is_rejected(self):
        self.write_arm('control')
        training = copy.deepcopy(self.training)
        training[1]['identities'][0]['start'] += 1
        self.write_arm('candidate', training=training)

        with self.assertRaisesRegex(AssertionError, 'Sample/timestep/scene differs'):
            manager.paired_inputs(('control', 'candidate'), 'identity', start=10, end=12)
        self.assertFalse((self.root / 'identity_PAIRING.json').exists())

    def test_missing_committed_step_is_rejected(self):
        self.write_arm('control')
        self.write_arm('candidate', inputs=[row for row in self.inputs if row['step'] != 11])

        with self.assertRaises(AssertionError):
            manager.paired_inputs(('control', 'candidate'), 'missing', start=10, end=12)
        self.assertFalse((self.root / 'missing_PAIRING.json').exists())


if __name__ == '__main__':
    unittest.main()
