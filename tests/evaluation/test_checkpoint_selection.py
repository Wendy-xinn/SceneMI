"""Selection must cover recordings and preserve a coherent checkpoint inode."""
import os
import tempfile
import unittest
from pathlib import Path

from scenemi.selection import pin_checkpoint, validation_manifest, window_starts
from scenemi.evaluation import summarize


class SelectionTests(unittest.TestCase):
    def test_complete_coverage_and_no_duplicate_tail(self):
        for length in (128, 129, 255, 256, 257, 2000):
            starts = window_starts(length)
            self.assertEqual(starts, sorted(set(starts)))
            covered = {frame for start in starts for frame in range(start, start+128)}
            self.assertEqual(covered, set(range(length)))
        with self.assertRaises(ValueError):
            window_starts(127)

    def test_no_test_or_reflection_in_selection(self):
        row = dict(split='val', family='family', corpus='corpus', variants=[
            dict(reflected=True, sha256='reflected', frames_after_relative_pose_drop=1000),
            dict(reflected=False, sha256='original', frames_after_relative_pose_drop=257)])
        manifest = validation_manifest([row])
        self.assertEqual([r['start'] for r in manifest], [0, 128, 129])
        self.assertTrue(all(r['variant_sha256']=='original' for r in manifest))
        with self.assertRaises(ValueError):
            validation_manifest([dict(row, split='test')])

    def test_pin_survives_atomic_last_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source, pinned, replacement = [Path(directory)/name for name in ('last', 'pinned', 'new')]
            source.write_bytes(b'old coherent checkpoint')
            pin_checkpoint(source, pinned)
            replacement.write_bytes(b'new coherent checkpoint')
            os.replace(replacement, source)
            self.assertEqual(pinned.read_bytes(), b'old coherent checkpoint')
            self.assertEqual(source.read_bytes(), b'new coherent checkpoint')

    def test_equal_family_weight(self):
        rows = [dict(identity=dict(family=family, corpus='corpus'), metrics=dict(mpjpe_mm=value))
                for family, value in [('long', 100), ('long', 100), ('short', 400)]]
        result = summarize(rows)
        self.assertEqual(result['window_mean']['mpjpe_mm'], 200)
        self.assertEqual(result['family_macro_mean']['mpjpe_mm'], 250)


if __name__ == '__main__':
    unittest.main()
