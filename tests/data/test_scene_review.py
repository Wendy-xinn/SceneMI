"""Conservative source-scene review flags severe persistent conflicts only."""
import unittest
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scenemi.data.scene_review import load_policy, severe_support_events, severe_seat_events, remaining_starts, start_ranges
from scenemi.data.scene_ego import apply_scene_review
from scenemi.data.amass_prepare import sha256


class SceneReviewTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_small_gaps_swing_and_short_airborne_phases_are_not_vetoed(self):
        n = 128
        gap = np.full((n, 2), .04); coverage = np.zeros((n, 2))
        speed = np.zeros((n, 2)); probability = np.ones((n, 2)); valid = np.ones(n, bool)
        self.assertFalse(any(severe_support_events(gap, coverage, speed, probability, valid, self.policy).values()))
        gap[:] = .2; probability[:] = .1
        self.assertFalse(any(severe_support_events(gap, coverage, speed, probability, valid, self.policy).values()))
        probability[:] = 1.; speed[:] = 1.
        self.assertFalse(any(severe_support_events(gap, coverage, speed, probability, valid, self.policy).values()))
        speed[:] = 0.; gap[:] = 0.; gap[20:30] = .2
        self.assertFalse(any(severe_support_events(gap, coverage, speed, probability, valid, self.policy).values()))

    def test_user_case_25_style_persistent_stance_gap_is_flagged_per_foot(self):
        n = 128; gap = np.zeros((n, 2)); gap[71:89, 0] = .18
        values = severe_support_events(gap, np.zeros((n, 2)), np.zeros((n, 2)),
            np.ones((n, 2)), np.ones(n, bool), self.policy)
        self.assertEqual(values['sustained_stance_clearance_0'], [[71, 88]])
        self.assertFalse(values['sustained_stance_clearance_1'])

    def test_unknown_support_and_invalid_frames_do_not_prove_a_conflict(self):
        n = 128
        values = severe_support_events(np.full((n, 2), np.inf), np.zeros((n, 2)),
            np.zeros((n, 2)), np.ones((n, 2)), np.ones(n, bool), self.policy)
        self.assertFalse(any(values.values()))
        values = severe_support_events(np.full((n, 2), .2), np.zeros((n, 2)),
            np.zeros((n, 2)), np.ones((n, 2)), np.zeros(n, bool), self.policy)
        self.assertFalse(any(values.values()))

    def test_chair_cushion_small_overlap_and_absent_surface_do_not_veto(self):
        n = 128
        for gap, ready in ((-.042, True), (-.2, False)):
            events = severe_seat_events(np.full(n, gap), np.full(n, ready), np.ones(n, bool),
                np.full(n, np.nan), np.ones(n, bool), self.policy)
            self.assertFalse(any(events.values()))

    def test_stable_deep_seat_overlap_requires_duration_and_correspondence(self):
        n = 128; gap = np.zeros(n); gap[30:50] = -.12
        events = severe_seat_events(gap, np.ones(n, bool), np.ones(n, bool),
            np.zeros(n), np.ones(n, bool), self.policy)
        self.assertEqual(events['seated_deep_support_overlap'], [[30, 49]])
        events = severe_seat_events(gap, np.ones(n, bool), np.zeros(n, bool),
            np.zeros(n), np.ones(n, bool), self.policy)
        self.assertFalse(any(events.values()))

    def test_exclusion_keeps_only_existing_complete_windows_without_padding(self):
        starts = np.array([0, 1, 2, 128, 129, 130, 260])
        kept = remaining_starts(starts, [[129, 129]])
        np.testing.assert_array_equal(kept, [0, 1, 130, 260])
        self.assertEqual(start_ranges(kept), [[0, 1], [130, 130], [260, 260]])
        np.testing.assert_array_equal(remaining_starts(starts, [[500, 510]]), starts)

    def test_overlay_filters_only_scene_training_and_preserves_source_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); selection = root / 'selection.json'; selection.write_text('{}')
            source = [dict(path='human', dataset='egobody', family='human', starts=np.array([0])),
                dict(path='other', dataset='trumans', family='other', starts=np.array([0])),
                dict(path='bad', dataset='grail', family='bad', metadata_sha256='bad-sha', starts=np.arange(5)),
                dict(path='good', dataset='grail', family='good', metadata_sha256='good-sha', starts=np.arange(5))]
            original = copy.deepcopy(source)
            records = [dict(split=split, path=path, metadata_sha256=path+'-sha', original_start_ranges=[[0,4]],
                            proposed_scene_start_ranges=[] if path=='bad' else [[2,4]], evidence={})
                       for split in ('train','validation') for path in ('bad','good')]
            overlay = root / 'review.json'
            overlay.write_text(json.dumps(dict(protocol='grail-scene-review-overlay-v1', state='complete',
                selection_sha256=sha256(selection), validation_membership_changed=False, source_arrays_modified=False,
                records=records)))
            c = dict(selection=str(selection), scene_stage=dict(quality_review=dict(file=str(overlay),sha256=sha256(overlay))))
            training = SimpleNamespace(split='train', records=source)
            apply_scene_review(training,c)
            self.assertEqual([r['path'] for r in training.records],['human','other','good'])
            np.testing.assert_array_equal(training.records[-1]['starts'],[2,3,4])
            for r, before in zip(source,original):
                np.testing.assert_array_equal(r['starts'],before['starts'])
            validation = SimpleNamespace(split='validation', records=copy.deepcopy(source))
            before = validation.records
            self.assertEqual(apply_scene_review(validation,c),{})
            self.assertIs(validation.records,before)
            np.testing.assert_array_equal(validation.records[-2]['starts'],np.arange(5))
            c['scene_stage']['quality_review']['sha256']='changed'
            with self.assertRaisesRegex(ValueError,'identity differs'):
                apply_scene_review(validation,c)


if __name__ == '__main__':
    unittest.main()
