"""Verify frontend coordinate handling without consulting personal GT."""
import io
import pickle
import unittest

import numpy as np

from scenemi.data.scene_exports import WorldTransform, arrays, camera


class SceneExportTests(unittest.TestCase):
    def test_common_similarity_preserves_body_and_scene_pixel_projections(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        change = WorldTransform(rotation, 2.7, [.3, -.2, 1.])
        source_camera = np.eye(4)
        source_camera[:3, 3] = [.5, 1., -3.]
        points = np.array([[0., 0., 1.], [1., 2., 3.], [-1., 1., 2.]])
        local = points-source_camera[:3, 3]
        moved_camera = change.camera_to_world(source_camera)
        moved_local = (change.points(points)-moved_camera[:3, 3])@moved_camera[:3, :3]
        np.testing.assert_allclose(moved_local, local*2.7, atol=1e-12)
        np.testing.assert_allclose(moved_local[:, :2]/moved_local[:, 2:],
                                   local[:, :2]/local[:, 2:], atol=1e-12)
        # A missing estimated body must stay missing after normalization.
        self.assertTrue(np.isnan(change.points(np.full((22, 3), np.nan))).all())

    def test_invalid_scale_and_reflected_rotation_are_rejected(self):
        for scale in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                WorldTransform(np.eye(3), scale, np.zeros(3))
        with self.assertRaises(AssertionError):
            WorldTransform(np.diag([-1., 1., 1.]), 1, np.zeros(3))

    def test_camera_inverse_and_intrinsics_are_verified(self):
        value = dict(cameras={'hsi': dict(camera_to_world_4x4=np.eye(4),
            world_to_camera_4x4=np.eye(4))}, intrinsic_3x3=np.eye(3), image_hw=[384, 688])
        actual, _, _ = camera(value)
        np.testing.assert_equal(actual, np.eye(4))
        value['cameras']['hsi']['world_to_camera_4x4'][0, 3] = 1.
        with self.assertRaises(AssertionError):
            camera(value)

    def test_restricted_loader_accepts_arrays_but_rejects_python_globals(self):
        value = {'joints': np.arange(12, dtype=np.float32).reshape(4, 3)}
        restored = arrays(pickle.dumps(value, protocol=5))
        np.testing.assert_equal(restored['joints'], value['joints'])
        with self.assertRaises(ValueError):
            arrays(pickle.dumps(io.StringIO('not an array payload')))


if __name__ == '__main__':
    unittest.main()
