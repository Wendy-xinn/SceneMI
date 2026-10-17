"""Synthetic invariance, missing-data and leak checks for scale diagnostics."""
import json
import unittest

import numpy as np

from experiments.offline_camera_retrain_v1.ops.alignment_metrics import (
    alignment_metrics, visible_alignment_metrics)


class AlignmentMetricsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(197)
        self.rest = rng.normal(size=(22, 3)) * .2
        self.joints = np.broadcast_to(self.rest, (8, 22, 3)).copy()
        self.joints += np.arange(8)[:, None, None] * np.array([.03, 0., -.01])
        self.mask = np.ones((8, 22), bool)
        self.conf = np.ones((8, 22))

    def test_known_twofold_body_and_cloud_scale(self):
        observed = self.joints * 2.
        result = alignment_metrics(self.joints, observed, observed, self.mask, self.conf, self.rest)
        self.assertAlmostEqual(result['observed_window_robust_similarity_fit_scale'], 2., places=10)
        self.assertAlmostEqual(result['observed_bone_to_rest_ratio_median'], 2., places=10)
        self.assertAlmostEqual(result['prediction_bone_to_rest_ratio_mean'], 1., places=10)
        self.assertLess(result['diagnostic_gt_window_similarity_fit_weighted_residual_mm'], 1e-8)
        self.assertGreater(result['diagnostic_gt_window_rigid_fit_weighted_residual_mm'], 100.)

    def test_translation_is_separate_from_size(self):
        shift = np.array([1., -.3, .2])
        observed = self.joints + shift
        result = alignment_metrics(self.joints, observed, observed, self.mask, self.conf, self.rest)
        self.assertAlmostEqual(result['observed_window_robust_similarity_fit_scale'], 1.)
        self.assertAlmostEqual(result['observed_bone_to_rest_ratio_median'], 1.)
        self.assertAlmostEqual(result['raw_gt_root_translation_error_mm'], np.linalg.norm(shift) * 1000.)
        self.assertLess(result['diagnostic_gt_root_relative_mpjpe_mm'], 1e-8)
        self.assertLess(result['diagnostic_gt_window_translation_aligned_mpjpe_mm'], 1e-8)

    def test_proper_rigid_rotation_and_root_orientation(self):
        angle = .4
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        observed = self.joints @ rotation
        result = alignment_metrics(self.joints, observed, observed, self.mask, self.conf, self.rest)
        self.assertLess(result['diagnostic_gt_window_rigid_fit_weighted_residual_mm'], 1e-8)
        self.assertAlmostEqual(result['diagnostic_gt_root_orientation_error_deg'], np.rad2deg(angle))
        self.assertAlmostEqual(result['observed_window_robust_similarity_fit_scale'], 1.)

    def test_no_per_frame_fit_hides_motion_error(self):
        observed = self.joints.copy()
        observed[:, :, 0] += np.linspace(-.4, .4, len(observed))[:, None]
        result = alignment_metrics(self.joints, observed, observed, self.mask, self.conf, self.rest)
        self.assertGreater(result['diagnostic_gt_window_translation_aligned_mpjpe_mm'], 200.)
        self.assertGreater(result['diagnostic_gt_window_rigid_fit_weighted_residual_mm'], 100.)
        self.assertAlmostEqual(result['observed_bone_to_rest_ratio_median'], 1.)

    def test_masked_nan_and_confidence_zero_never_enter_fit_or_modify_input(self):
        observed = self.joints.copy()
        mask = self.mask.copy()
        confidence = self.conf.copy()
        mask[:, 3:12] = False
        observed[:, 3:12] = np.nan
        confidence[:, 18:] = 0.
        observed[:, 18:] = 1e10
        before = mask.copy()
        result = visible_alignment_metrics(self.joints, observed, mask, confidence, self.rest)
        self.assertEqual(result['observed_valid_samples'], 8 * (22 - 9 - 4))
        self.assertAlmostEqual(result['observed_window_robust_similarity_fit_scale'], 1.)
        np.testing.assert_array_equal(mask, before)
        json.dumps(result, allow_nan=False)

    def test_missing_or_collinear_points_reject_rigid_fit(self):
        for empty in (False, True):
            points = np.zeros_like(self.joints)
            points[..., 0] = np.arange(22)
            mask = np.zeros_like(self.mask) if empty else self.mask
            result = visible_alignment_metrics(points, points, mask, self.conf, self.rest)
            self.assertFalse(result['observed_window_robust_rigid_fit_valid'])
            self.assertIsNone(result['observed_window_robust_similarity_fit_scale'])
            json.dumps(result, allow_nan=False)

    def test_robust_visible_fit_resists_one_gross_tracker_outlier(self):
        observed = self.joints * 1.2 + np.array([.1, .2, -.1])
        observed[0, 20] += np.array([10., -5., 12.])
        result = visible_alignment_metrics(self.joints, observed, self.mask, self.conf, self.rest)
        self.assertAlmostEqual(result['observed_window_robust_similarity_fit_scale'], 1.2, places=3)
        self.assertLess(result['observed_window_robust_similarity_fit_residual_median_mm'], 1.)

    def test_visible_fit_cannot_depend_on_hidden_truth(self):
        mask = self.mask.copy()
        mask[:, 4:12] = False
        truth = self.joints.copy()
        first = alignment_metrics(self.joints, truth, self.joints, mask, self.conf, self.rest)
        truth[:, 4:12] += 400.
        second = alignment_metrics(self.joints, truth, self.joints, mask, self.conf, self.rest)
        for key in first:
            if key.startswith(('observed_', 'prediction_')):
                self.assertEqual(first[key], second[key], key)
        self.assertGreater(second['raw_gt_world_mpjpe_mm'], first['raw_gt_world_mpjpe_mm'])

    def test_invalid_visible_coordinates_are_not_silently_discarded(self):
        observed = self.joints.copy()
        observed[0, 0] = np.nan
        with self.assertRaises(ValueError):
            visible_alignment_metrics(self.joints, observed, self.mask, self.conf, self.rest)

    def test_no_reflection_and_fit_objective_is_window_rms(self):
        reflected = self.joints * np.array([-1., 1., 1.])
        result = alignment_metrics(self.joints, reflected, reflected, self.mask, self.conf, self.rest)
        self.assertGreater(result['diagnostic_gt_window_similarity_fit_weighted_residual_mm'], 100.)
        self.assertLessEqual(result['diagnostic_gt_window_rigid_fit_weighted_rms_residual_mm'], result['raw_gt_world_rms_mm'] + 1e-8)
        self.assertLessEqual(result['diagnostic_gt_window_similarity_fit_weighted_rms_residual_mm'], result['diagnostic_gt_window_rigid_fit_weighted_rms_residual_mm'] + 1e-8)


if __name__ == '__main__':
    unittest.main()
