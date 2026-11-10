"""Checks for quality diagnostics that could otherwise mislabel valid motion."""
import unittest

import numpy as np

from scenemi.data.grail_quality import contact_probabilities, ranges, support_summary, vertical_gaps


class GrailQualityTest(unittest.TestCase):
    def test_contacts_keep_left_and_right_channels_separate(self):
        # Both channels of one side must indicate contact; opposite legs differ.
        raw = np.array([[.9, .95, .1, .99], [.2, .9, .95, .96]])
        result = contact_probabilities(raw, 3, 1, 2)
        np.testing.assert_allclose(result, [[.9, .1], [.55, .525], [.2, .95]])
        with self.assertRaises(ValueError):
            contact_probabilities(raw, 4, 1, 2)

    def test_high_swing_foot_is_not_marked_bad_contact(self):
        gaps = np.array([[[.005, .01], [.30, .35]], [[.25, .3], [.002, .02]]])
        probabilities = np.array([[.99, .01], [.01, .99]])
        result = support_summary(gaps, probabilities)
        self.assertEqual(result['source_contact_above_5cm_foot_frames'], 0)
        self.assertEqual(result['both_feet_above_5cm_frames'], 0)
        self.assertEqual(result['source_contact_foot_frames'], 2)
        mapped = support_summary(gaps + .1, probabilities, frame_ids=[200, 205])
        self.assertEqual(mapped['both_feet_above_5cm_ranges'], [[200, 205]])
        self.assertEqual(mapped['range_frame_stride'], 5)

    def test_vertical_gap_uses_highest_surface_and_preserves_missing(self):
        from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
        # Two overlapping horizontal surfaces: the support height is the top one.
        bottom = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], np.float32)
        top = bottom + [0, .3, 0]
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        scene = static_ray_scene(np.concatenate([bottom, top]), faces)
        gaps = vertical_gaps(np.array([[0, .35, 0], [0, .25, 0], [2, 0, 2]]), scene, 2.)
        np.testing.assert_allclose(gaps[:2], [.05, -.05], atol=1e-6)
        self.assertTrue(np.isnan(gaps[2]))
        self.assertEqual(ranges([False, True, True, False, True]), [[1, 2], [4, 4]])


if __name__ == '__main__':
    unittest.main()
