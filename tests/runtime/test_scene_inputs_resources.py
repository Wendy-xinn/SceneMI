"""User-managed GPU launches and unchanged target-person scene exclusions."""
import unittest
from unittest.mock import patch

import numpy as np

from scenemi import parallel_resources as resources
from scenemi.data.scene_dataset import bounded_frame_points, SCENE_INPUT_KEYS


class ParallelAdmissionTests(unittest.TestCase):
    def report(self, free_mib, known):
        with patch.object(resources.legacy, 'admission', return_value=dict(memory_admitted=True, admitted=True)), \
             patch.object(resources.legacy, 'inspect_gpu', return_value=dict(compute_pids=[123], admitted=False)), \
             patch.object(resources, 'own_job', return_value=known), \
             patch.object(resources.subprocess, 'check_output', return_value=f'24455, {free_mib}'):
            return resources.admission('cuda', 12, 8)

    def test_foreign_or_uninspectable_pid_no_longer_blocks_launch(self):
        result = self.report(22000, False)
        self.assertTrue(result['admitted'])
        self.assertFalse(result['gpu']['process_identity_checked'])

    def test_no_gpu_headroom_threshold_is_enforced(self):
        for free in (0,11263,11264):
            self.assertTrue(self.report(free,True)['admitted'])


class SceneInputTests(unittest.TestCase):
    def test_target_person_and_invalid_points_excluded_without_frame_mixing(self):
        points = np.ones((2, 5, 3), np.float32)
        points[0, 2] = np.nan
        owners = np.array([[0, 100, 0, 3, 255], [100, 255, 0, 1, 2]])
        sampled, ids = bounded_frame_points(points, owners, count=3)
        np.testing.assert_array_equal(ids[0], [0, 3, -1])
        original = sampled[0].copy()
        points[1] *= 999
        changed, _ = bounded_frame_points(points, owners, count=3)
        np.testing.assert_array_equal(original, changed[0])
        self.assertTrue(np.isfinite(sampled).all())

    def test_shape_labels_not_in_inference_payload(self):
        for forbidden in ('target_native_betas', 'target_template', 'target_rest', 'motion', 'native_body'):
            self.assertNotIn(forbidden, SCENE_INPUT_KEYS)


if __name__ == '__main__':
    unittest.main()
