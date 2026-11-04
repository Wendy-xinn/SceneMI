"""Causality and split-preservation checks for the EGO visibility selection."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scenemi.data.grail_visibility import (
    DEFAULT_CONFIG, apply, eligibility, load_policy, scan, sha256,
)


class GrailVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(DEFAULT_CONFIG)
        self.profile = self.policy['profiles']['standard']

    def test_past_observation_survives_turning_away(self):
        counts = np.zeros(301, int)
        counts[:16] = 123
        result = eligibility(counts, self.profile)
        self.assertEqual(result['history_ready_frame'], 15)
        self.assertEqual(result['eligible_start_ranges'], [[0, 173]])
        self.assertGreater(result['windows_with_no_current_but_known_history'], 0)

    def test_later_visibility_does_not_qualify_earlier_windows(self):
        counts = np.zeros(301, int)
        counts[260:276] = 123
        result = eligibility(counts, self.profile)
        self.assertEqual(result['history_ready_frame'], 275)
        self.assertEqual(result['eligible_start_ranges'], [[163, 173]])

    def test_tiny_hits_and_isolated_glimpses_are_insufficient(self):
        self.assertEqual(eligibility(np.full(151, 30), self.profile)['eligible_windows'], 0)
        counts = np.zeros(301, int)
        counts[::2] = 123
        self.assertEqual(eligibility(counts, self.profile)['eligible_windows'], 0)
        self.assertEqual(eligibility(np.full(127, 123), self.profile)['possible_windows'], 0)

    def source(self, root, *, crossing_family=False):
        rows = []
        specs = [('train', 'strong', 200), ('train', 'weak', 30),
                 ('validation', 'val', 200), ('test', 'test', 200)]
        for split, name, pixels in specs:
            folder = root/name
            folder.mkdir()
            frame_stats = [dict(frame=t, rays=12288, static=500, ground_y0=500-pixels,
                                object_above_ground=pixels) for t in range(151)]
            meta = dict(sequence_id=name, fps=30, trainable=True, dataset='grail', split=split,
                object_family_id='same_family' if crossing_family else name, frames=151, category='sitting',
                virtual_camera=dict(width=128, height=96, hfov_deg=66.56, vfov_deg=40.49,
                                    pitch_down_deg=0., near_m=.05, far_m=4.),
                primary_visibility_by_frame=frame_stats)
            path = folder/'metadata.json'
            path.write_text(json.dumps(meta))
            rows.append(dict(path=str(folder), metadata_sha256=sha256(path)))
        manifest = root/'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return manifest

    def test_training_manifest_and_original_evaluation_are_separate(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            manifest = self.source(root)
            original = sha256(manifest)
            output = root/'selection'
            scan(manifest, output)
            state = apply(output)
            kept = [json.loads(row) for row in (output/'train/manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(kept), 1)
            self.assertEqual(Path(kept[0]['path']).name, 'strong')
            self.assertEqual(state['counts']['train_sequences'], 2)
            audit = [json.loads(row) for row in (output/'evaluation_coverage_audit.jsonl').read_text().splitlines()]
            self.assertEqual({row['split'] for row in audit}, {'validation', 'test'})
            self.assertTrue(all(row['mode'] == 'ego' for row in audit))
            self.assertFalse((output/'test/manifest.jsonl').exists())
            self.assertEqual(sha256(manifest), original)
            with (output/'visibility_statistics.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'verified statistics'):
                apply(output)

    def test_cross_split_object_families_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            manifest = self.source(root, crossing_family=True)
            with self.assertRaisesRegex(ValueError, 'family crosses'):
                scan(manifest, root/'selection')

    def test_changed_source_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            manifest = self.source(root)
            with (root/'strong/metadata.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                scan(manifest, root/'selection')


if __name__ == '__main__':
    unittest.main()
