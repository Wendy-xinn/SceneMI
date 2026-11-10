"""Source review must preserve timing and never reveal unavailable observations."""
import unittest
from unittest.mock import patch

import numpy as np

from scenemi.native_visualization import camera_motion, observations


class NativeReviewTests(unittest.TestCase):
    def test_windowed_observations_preserve_source_clock_and_mask_hidden_positions(self):
        count = 8
        reference = np.full((count, 22), 12, np.int32)
        visible = reference.copy()
        visible[:, 6] = 0
        valid = np.ones(count, bool)
        valid[3] = False
        available = ((visible > 0) & valid[:, None]).astype(np.float32)
        trajectory = np.broadcast_to(np.arange(count)[:, None, None], (count, 22, 3)).copy().astype(np.float32)
        arrays = dict(exo_reference_counts=reference, exo_visible_counts=visible,
            exo_static_counts=reference-visible, exo_person_counts=np.zeros_like(reference),
            exo_rigid_counts=np.zeros_like(reference), observation_valid=valid,
            exo_position_available=available, exo_position_confidence=available,
            exo_trajectory_world=trajectory)
        with patch('scenemi.native_visualization.readonly_array', side_effect=lambda _, k: arrays[k]):
            result = observations(dict(start=2), 3)
        observation, meta = result['observations'], result['observation_meta']
        self.assertEqual(observation.shape, (3, 22, 9))
        np.testing.assert_array_equal(observation[[0, 2], 0, :3]*2, trajectory[[2, 4], 0])
        self.assertTrue(np.all(observation[:, 6] == 0))
        self.assertTrue(np.all(observation[1] == 0))
        self.assertTrue(np.all(meta[..., 1] == 0))  # No invented joint rotations.
        np.testing.assert_array_equal(result['source_occluded'][:, 6], [True, False, True])

    def test_camera_translation_and_rotation_are_reported_separately(self):
        matrices = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        angle = np.deg2rad(30)
        matrices[-1, :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                               [-np.sin(angle), 0, np.cos(angle)]]
        result = camera_motion(matrices)
        self.assertEqual(result['position_path_m'], 0.)
        self.assertAlmostEqual(result['rotation_max_from_first_deg'], 30.)
        matrices[:, 0, 3] = [0, .5, 1.]
        result = camera_motion(matrices)
        self.assertAlmostEqual(result['position_path_m'], 1.)
        self.assertAlmostEqual(result['position_max_from_first_m'], 1.)


if __name__ == '__main__':
    unittest.main()
