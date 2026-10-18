"""Coordinate and observation-identity checks for shared display layers."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.offline_camera_retrain_v1.ops import viewer_geometry as geometry


class GeometryTests(unittest.TestCase):
    def identity(self, mode="ego"):
        return dict(mode=mode, start=1, origin=[10., 2., 3.],
            anchor=[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], group="rich")

    def test_anchor_transforms_positions_and_frustum_directions(self):
        identity = self.identity()
        arrays = dict(camera_position_scenemi_yup=np.array([[0., 0., 0.], [10., 2., 4.], [11., 2., 4.]]),
            camera_rotation_scenemi_yup=np.tile(np.eye(3), (3, 1, 1)),
            camera_intrinsics_128x96=np.tile([128., 96., 64., 48.], (3, 1)))
        result, evidence = geometry.camera_layers(identity, 2, lambda _, name: arrays[name], frustum_depth_m=1.)
        np.testing.assert_allclose(result["camera_positions"], [[1., 0., 0.], [1., 0., -1.]])
        # Top-left EGO is local (-.5,+.5,+1), then rotated by anchor.
        np.testing.assert_allclose(result["camera_frustum_segments"][0, 0, 1], [2., .5, .5])
        np.testing.assert_allclose(result["camera_axis_segments"][0, 2, 1], [1.65, 0., 0.])
        self.assertEqual(result["camera_path"].shape, (1, 2, 3))
        self.assertIn("+Y up", evidence["optical_convention"])

    def test_ego_and_exo_pixel_y_are_opposite(self):
        ego = geometry.optical_corner_rays(np.array([1., 1., .5, .5]), mode="ego")
        exo = geometry.optical_corner_rays(np.array([1., 1., .5, .5]), mode="exo")
        np.testing.assert_array_equal(ego[:, (0, 2)], exo[:, (0, 2)])
        np.testing.assert_array_equal(ego[:, 1], -exo[:, 1])
        self.assertGreater(ego[0, 1], 0.)
        self.assertLess(exo[0, 1], 0.)

    def test_exo_fixed_camera_broadcasts_and_preserves_source_kind(self):
        identity = self.identity("exo")
        identity["exo_camera"] = dict(kind="synthetic_training_observer", distortion=[0.] * 8)
        matrix = np.eye(4)
        matrix[:3, 3] = identity["origin"]
        arrays = dict(exo_sensor_to_world=matrix, exo_intrinsics=np.array([1., 1., .5, .5]))
        result, evidence = geometry.camera_layers(identity, 3, lambda _, name: arrays[name])
        np.testing.assert_array_equal(result["camera_positions"], np.zeros((3, 3)))
        self.assertEqual(result["camera_frustum_segments"].shape, (3, 8, 2, 3))
        self.assertEqual(evidence["source_kind"], "synthetic_training_observer")

    def test_exo_distortion_matches_export_camera_rays_formula(self):
        import cv2
        intr = np.array([.9, 1.1, .45, .55])
        distortion = np.array([.08, -.01, .002, -.003, 0.])
        rays = geometry.optical_corner_rays(intr, mode="exo", distortion=distortion)
        pixels = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        k = np.array([[.9, 0., .45], [0., 1.1, .55], [0., 0., 1.]])
        expected = cv2.undistortPointsIter(pixels[:, None], k, distortion, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10))[:, 0]
        np.testing.assert_allclose(rays[:, :2], expected, atol=1e-7)

    def test_current_scene_preserves_static_and_partner_excludes_target(self):
        identity = self.identity()
        points = np.zeros((3, 512, 3), np.float32)
        points[:] = [10., 2., 4.]
        owners = np.full((3, 512), -1, np.int32)
        owners[1, :5] = [0, 7, 101, 100, -1]
        arrays = dict(visible_frame_points_scenemi_yup=points, visible_frame_owner=owners)
        result, evidence = geometry.instant_scene_layers(identity, 2, lambda _, name: arrays[name])
        np.testing.assert_array_equal(result["current_static_0"], [[1., 0., 0.]])
        np.testing.assert_array_equal(result["current_dynamic_0"], [[1., 0., 0.], [1., 0., 0.]])
        self.assertEqual(len(result["current_static_1"]), 0)
        self.assertEqual(evidence["excluded_target_samples"], 1)
        self.assertTrue(evidence["target_skin_excluded"])

    def test_readonly_original_fallback_hashes_without_cache_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "sample.npy", np.array([1., 2.]))
            digest = hashlib.sha256((root / "sample.npy").read_bytes()).hexdigest()
            identity = dict(path=str(root), output_sha256={"sample.npy": digest})
            missing_mirror = root / "missing_mirror"
            result = geometry.readonly_array(identity, "sample", mirror_root=missing_mirror)
            np.testing.assert_array_equal(result, [1., 2.])
            self.assertFalse(result.flags.writeable)
            self.assertFalse(missing_mirror.exists())
            # An existing but corrupted mirror must not silently fall through.
            missing_mirror.mkdir()
            np.save(missing_mirror / (digest + ".npy"), np.array([99.]))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                geometry.readonly_array(identity, "sample", mirror_root=missing_mirror)

    def test_missing_optional_layers_are_explicit(self):
        def missing(*_):
            raise FileNotFoundError("fixture intentionally absent")
        result, metadata = geometry.build_scene_layers(self.identity(), {}, 2, missing)
        self.assertFalse(result)
        self.assertEqual(metadata["camera"]["status"], "not_available")
        self.assertEqual(metadata["current_scene"]["status"], "not_available")
        self.assertIn("FileNotFoundError", metadata["camera"]["error"])


if __name__ == "__main__":
    unittest.main()
