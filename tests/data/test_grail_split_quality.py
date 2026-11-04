"""Held-out QC must preserve time gaps, source splits and ledger identities."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scenemi.data.grail_contact import intersect_windows
from scenemi.data.grail_split_quality import (PROTOCOL, discover, resume_records,
                                            usable_frames, write_rows)


class QualitySelectionTest(unittest.TestCase):
    def test_short_clean_segments_cannot_be_concatenated(self):
        bad = np.zeros(400, bool)
        bad[150:180] = True
        ranges, _ = intersect_windows([[0, 272]], bad)
        self.assertEqual(ranges, [[0, 22], [180, 272]])
        self.assertEqual(usable_frames(ranges, 128, 400), 370)
        bad[240:] = True  # Remaining second fragment is only 60 frames.
        ranges, _ = intersect_windows([[0, 272]], bad)
        self.assertEqual(usable_frames(ranges, 128, 400), 150)

    def test_frame_union_does_not_double_count_overlapping_spans(self):
        self.assertEqual(usable_frames([[0, 10], [30, 40]], 128, 200), 168)
        for ranges in ([[50, 50], [0, 0]], [[0, 90]], [[-1, 2]]):
            with self.assertRaises(ValueError):
                usable_frames(ranges, 128, 200)

    def test_metadata_family_overlap_or_changed_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for split in ('train', 'validation', 'test'):
                p = root/split
                p.mkdir()
                meta = dict(sequence_id=split, object_family_id=split, split=split,
                            dataset='grail', fps=30, trainable=True, category='curb', frames=301)
                raw = json.dumps(meta).encode()
                (p/'metadata.json').write_bytes(raw)
                rows.append(dict(path=str(p), metadata_sha256=hashlib.sha256(raw).hexdigest()))
            manifest = root/'manifest.jsonl'
            write_rows(manifest, rows)
            selected, audit = discover(manifest, ['validation', 'test'])
            self.assertEqual({r['split'] for r in selected}, {'validation', 'test'})
            self.assertTrue(audit['source_family_split_intersection_empty'])
            path = root/'test/metadata.json'
            meta = json.loads(path.read_text())
            meta['object_family_id'] = 'train'
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                discover(manifest, ['validation', 'test'])
            rows[-1]['metadata_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_rows(manifest, rows)
            with self.assertRaisesRegex(ValueError, 'crosses source splits'):
                discover(manifest, ['validation', 'test'])

    def test_resume_cannot_mix_source_or_frame_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'records.jsonl'
            row = dict(sequence_id='x', path='/x', frames=200, metadata_sha256='a')
            result = dict(row, protocol=PROTOCOL, eligible_start_ranges=[[0, 72]],
                          eligible_windows=73, usable_frames=200)
            write_rows(path, [result])
            self.assertEqual(len(resume_records(path, [row])), 1)
            with self.assertRaises(ValueError):
                resume_records(path, [dict(row, metadata_sha256='b')])
            write_rows(path, [dict(result, usable_frames=199)])
            with self.assertRaises(ValueError):
                resume_records(path, [row])


if __name__ == '__main__':
    unittest.main()
