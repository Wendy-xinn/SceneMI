"""CPU-only preparation recovery tests; no source exports, body models, or GPU."""
import fcntl
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from . import grail_prepare as prepare


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='grail_prepare_test_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.py'
        self.source.write_text('immutable source\n')
        self.seal = {str(self.source): prepare.grail_source.file_sha256(self.source)}
        self.rows = [dict(sequence_id=name, object_family_id='fixture/' + name)
                     for name in ('ok', 'bad', 'review', 'last')]
        self.inventory = self.root / 'inventory.jsonl'
        self.inventory.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        self.addCleanup(patch.stopall)
        patch.object(prepare, 'source_hashes', return_value=self.seal).start()
        patch.object(prepare, 'MINIMUM_FREE_BYTES', 0).start()

    def fake_export(self, row, output, **options):
        if row['sequence_id'] == 'bad':
            raise ValueError('Invalid source pose shape')
        folder = (Path(output) / prepare.grail_export.grail_data.family_split(
                  prepare.grail_export.normalized_family(row)) / row['sequence_id'])
        folder.mkdir(parents=True, exist_ok=True)
        metadata = folder / 'metadata.json'
        if not metadata.exists():
            metadata.write_text(json.dumps(dict(sequence_id=row['sequence_id'])))
        return dict(path=str(folder), trainable=row['sequence_id'] != 'review', source_hashes={})

    def test_limit_to_full_resume_preserves_job_and_compact_references(self):
        output = self.root / 'out'
        with patch.object(prepare.grail_export, 'export_record', side_effect=self.fake_export):
            first = prepare.run(self.inventory, output, limit=2)
            job = (output / 'JOB.json').read_bytes()
            final = prepare.run(self.inventory, output)
        self.assertEqual(first['state'], 'invocation_limit_reached')
        self.assertEqual((first['completed'], first['failures']), (1, 1))
        self.assertEqual(final['state'], 'complete')
        self.assertEqual((final['completed'], final['trainable'], final['quarantine'], final['failures']), (3, 2, 1, 1))
        self.assertEqual((output / 'JOB.json').read_bytes(), job)
        self.assertEqual(len((output / 'manifest.jsonl').read_text().splitlines()), 2)
        self.assertEqual(len((output / 'review_manifest.jsonl').read_text().splitlines()), 1)
        self.assertTrue(all(set(json.loads(line)) == {'path', 'metadata_sha256'}
                            for line in (output / 'manifest.jsonl').read_text().splitlines()))

    def test_partial_tail_recovery_preserves_evidence_but_complete_corruption_fails(self):
        ledger = self.root / 'ledger.jsonl'
        ledger.write_bytes(b'{"valid":1}\n{"partial":')
        self.assertEqual(prepare.recover_events(ledger), [{'valid': 1}])
        self.assertEqual(ledger.read_bytes(), b'{"valid":1}\n')
        evidence = list(self.root.glob('ledger.jsonl.interrupted.*'))
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].read_bytes(), b'{"partial":')
        ledger.write_bytes(b'{broken}\n')
        with self.assertRaises(prepare.JobIdentityError):
            prepare.recover_events(ledger)
        self.assertEqual(ledger.read_bytes(), b'{broken}\n')

    def test_changed_metadata_manifest_and_source_fail_without_replacing_job(self):
        output = self.root / 'out'
        with patch.object(prepare.grail_export, 'export_record', side_effect=self.fake_export):
            prepare.run(self.inventory, output, limit=1)
            job = (output / 'JOB.json').read_bytes()
            entry = json.loads((output / 'manifest.jsonl').read_text())
            metadata = Path(entry['path']) / 'metadata.json'
            original = metadata.read_bytes()
            metadata.write_bytes(original + b' ')
            with self.assertRaisesRegex(prepare.JobIdentityError, 'metadata checksum'):
                prepare.run(self.inventory, output, limit=1)
            metadata.write_bytes(original)
            self.source.write_text('changed source\n')
            with self.assertRaisesRegex(prepare.JobIdentityError, 'source/template changed'):
                prepare.run(self.inventory, output, limit=1)
        self.assertEqual((output / 'JOB.json').read_bytes(), job)
        status = json.loads((output / 'STATUS.json').read_text())
        self.assertEqual(status['state'], 'stopped_fatal')

    def test_inventory_and_options_changes_reject_resume(self):
        output = self.root / 'out'
        with patch.object(prepare.grail_export, 'export_record', side_effect=self.fake_export):
            prepare.run(self.inventory, output, limit=1)
            job = (output / 'JOB.json').read_bytes()
            with self.assertRaises(prepare.JobIdentityError):
                prepare.run(self.inventory, output, limit=1, scope='pilot')
            self.inventory.write_text(self.inventory.read_text() + '\n')
            with self.assertRaises(prepare.JobIdentityError):
                prepare.run(self.inventory, output, limit=1)
        self.assertEqual((output / 'JOB.json').read_bytes(), job)

    def test_storage_template_and_existing_bundle_errors_are_fatal(self):
        cases = (('storage', OSError('disk I/O error'), OSError),
                 ('template', prepare.grail_export.UnverifiedBodyTemplate('unverified template'),
                  prepare.grail_export.UnverifiedBodyTemplate),
                 ('existing', ValueError('existing bundle checksum mismatch'), prepare.JobIdentityError))
        for name, error, expected in cases:
            with self.subTest(name=name):
                output = self.root / name
                if name == 'existing':
                    row = self.rows[0]
                    (output / prepare.grail_export.grail_data.family_split(
                     prepare.grail_export.normalized_family(row)) / row['sequence_id']).mkdir(parents=True)
                with patch.object(prepare.grail_export, 'export_record', side_effect=error) as exporter:
                    with self.assertRaises(expected):
                        prepare.run(self.inventory, output)
                self.assertEqual(exporter.call_count, 1)
                self.assertFalse((output / 'FAILURES.jsonl').exists())
                self.assertEqual(json.loads((output / 'STATUS.json').read_text())['state'], 'stopped_fatal')

    def test_sigterm_finishes_current_record_and_preserves_stop_state(self):
        output, calls = self.root / 'out', []
        original_handler = signal.getsignal(signal.SIGTERM)

        def request_stop(row, output, **options):
            calls.append(row['sequence_id'])
            # Exercise our installed callback without sending any OS signal.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return self.fake_export(row, output, **options)

        with patch.object(prepare.grail_export, 'export_record', side_effect=request_stop):
            status = prepare.run(self.inventory, output)
        self.assertEqual(calls, ['ok'])
        self.assertEqual((status['state'], status['completed'], status['stop_signal']),
                         ('stopped', 1, signal.SIGTERM))
        self.assertEqual(signal.getsignal(signal.SIGTERM), original_handler)
        self.assertIsNone(status['current_sequence'])

    def test_concurrent_writer_cannot_clobber_running_status(self):
        output = self.root / 'out'
        output.mkdir()
        original = '{"state":"running","pid":123}\n'
        (output / 'STATUS.json').write_text(original)
        with (output / 'PREPARE.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(prepare.JobIdentityError, 'Another preparation process'):
                prepare.run(self.inventory, output)
        self.assertEqual((output / 'STATUS.json').read_text(), original)

    def test_pinned_input_identity_mismatch_is_fatal(self):
        row = dict(sequence_id='fixture', source_identities=dict(recon=dict(sha256='a' * 64)))
        with self.assertRaises(prepare.JobIdentityError):
            prepare.check_pinned_input_hashes(row, dict(source_hashes=dict(recon='b' * 64)))


if __name__ == '__main__':
    unittest.main()
