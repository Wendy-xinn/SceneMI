"""Check display labels against evaluator masks and first-hit evidence."""
import unittest

import numpy as np

from experiments.offline_camera_retrain_v1.ops import viewer_observation as observation


class ObservationTests(unittest.TestCase):
    def metadata(self):
        return np.zeros((2, 22, 5), dtype=np.float32)

    def test_ego_missing_body_is_not_claimed_occluded(self):
        meta = self.metadata()
        meta[:, 15, (0, 2)] = 1
        classes, colors, explanation = observation.observation_classes(meta, "ego")
        self.assertEqual(np.count_nonzero(classes == observation.TRUSTED), 2)
        self.assertEqual(np.count_nonzero(classes == observation.MISSING), 42)
        self.assertEqual(colors.dtype, np.uint8)
        self.assertEqual(colors.shape, (2, 22, 3))
        self.assertFalse(explanation["missing_is_occlusion"])
        self.assertIn("未提供", explanation["labels"]["0"])

    def test_low_confidence_remains_in_exact_original_conditioned_group(self):
        meta = self.metadata()
        meta[..., 0] = 1
        meta[..., 2] = 1
        meta[:, 7, 2] = 1e-8
        meta[:, 8, 2] = .99
        pred = np.zeros((2, 22, 3))
        truth = pred.copy()
        pred[:, 7, 0] = .3
        classes, _, _ = observation.observation_classes(meta, "exo")
        self.assertTrue(np.all(classes[:, [7, 8]] == observation.LOW_CONFIDENCE))
        result = observation.masked_joint_metrics(pred, truth, meta)
        self.assertEqual(result["conditioned_mpjpe_mm"]["count"], 44)
        self.assertAlmostEqual(result["conditioned_mpjpe_mm"]["value"], 600 / 44)
        self.assertEqual(result["unconditioned_mpjpe_mm"], dict(value=None, count=0))

    def test_masked_metrics_use_metres_without_alignment_or_confidence_reweighting(self):
        meta = self.metadata()
        meta[0, 0, (0, 2)] = 1
        meta[1, 1, 0] = .00001
        meta[1, 1, 2] = .25
        truth = np.zeros((2, 22, 3))
        pred = truth.copy()
        pred[0, 0] = [.003, .004, 0]
        pred[1, 1] = [.009, .012, 0]
        result = observation.masked_joint_metrics(pred, truth, meta)
        self.assertAlmostEqual(result["conditioned_mpjpe_mm"]["value"], 10.)
        self.assertAlmostEqual(result["mpjpe_mm"]["value"], 20 / 44)
        self.assertEqual(result["conditioned_mpjpe_mm"], result["visible_mpjpe_mm"])
        self.assertEqual(result["hidden_mpjpe_mm"], dict(value=0., count=42))

    def test_absent_metadata_errors_not_fabricated_zero(self):
        meta = self.metadata()
        result = observation.masked_joint_metrics(np.ones((2, 22, 3)), np.zeros((2, 22, 3)), meta)
        self.assertIsNone(result["conditioned_mpjpe_mm"]["value"])
        self.assertAlmostEqual(result["hidden_mpjpe_mm"]["value"], np.sqrt(3) * 1000)

    def test_invalid_saved_data_is_not_silently_filtered(self):
        meta = self.metadata()
        with self.assertRaisesRegex(ValueError, "shape"):
            observation.masked_joint_metrics(np.zeros((2, 21, 3)), np.zeros((2, 21, 3)), meta)
        pred = np.zeros((2, 22, 3))
        pred[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            observation.masked_joint_metrics(pred, np.zeros_like(pred), meta)
        meta[0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "zero confidence"):
            observation.observation_classes(meta, "exo")

    def first_hit_fixture(self):
        reference, visible = np.zeros((2, 22), int), np.zeros((2, 22), int)
        blockers = {name: np.zeros_like(reference) for name in ("person", "static", "rigid")}
        reference[:, 0] = 12
        visible[:, 0] = 12
        reference[:, 1] = 12
        visible[:, 1] = 1
        blockers["static"][:, 1] = 11
        reference[:, 2] = 20
        visible[:, 2] = 3
        blockers["person"][:, 2] = 17
        return reference, visible, blockers

    def test_first_hit_separates_occlusion_partial_visibility_and_no_support(self):
        result = observation.exo_first_hit_evidence(*self.first_hit_fixture())
        self.assertTrue(np.all(result["exported_position_available"][:, 0]))
        self.assertFalse(np.any(result["exported_position_available"][:, 1]))
        self.assertTrue(np.all(result["has_occluding_surface"][:, 1]))
        # Exactly 0.15 passes; confidence remains 0.15 * (3/12).
        self.assertTrue(np.all(result["exported_position_available"][:, 2]))
        np.testing.assert_allclose(result["exported_position_confidence"][:, 2], .0375)
        self.assertTrue(np.all(result["has_occluding_surface"][:, 2]))
        self.assertTrue(np.all(result["no_reference_support"][:, 3]))
        self.assertFalse(np.any(result["has_occluding_surface"][:, 3]))

    def test_first_hit_source_invalidity_and_partition_corruption(self):
        reference, visible, blockers = self.first_hit_fixture()
        result = observation.exo_first_hit_evidence(reference, visible, blockers, valid=[True, False])
        self.assertFalse(np.any(result["exported_position_available"][1]))
        self.assertFalse(np.any(result["has_occluding_surface"][1]))
        blockers["static"][0, 1] = 10
        with self.assertRaisesRegex(ValueError, "partition"):
            observation.exo_first_hit_evidence(reference, visible, blockers)


if __name__ == "__main__":
    unittest.main()
