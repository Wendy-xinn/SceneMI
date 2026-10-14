"""Exact query samples/masks, including radius boundaries and frame owners."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.native30_overnight import YawSurface
from experiments.offline_camera_retrain_v1.native30_query_runtime import (
    select_radius_samples, select_radius_samples_cuda)


def fixture():
    rng = np.random.default_rng(618)
    frames, count = 4, 120
    points = rng.normal(0, .19, (frames, count, 3)).astype(np.float32)
    points[:, :50, 1] = 0
    owners = np.broadcast_to(np.r_[np.zeros(50), np.ones(35), np.full(35, 101)], (frames, count)).astype(np.int64).copy()
    object_positions = np.zeros((frames, 1, 3), np.float32)
    object_positions[:, 0, 0] = np.arange(frames) * .01
    object_rotations = np.broadcast_to(np.eye(3), (frames, 1, 3, 3)).copy()
    scene = Native30SurfaceWindow(points, owners, np.tile([0, 1, 0], (frames, 1)),
        object_positions=object_positions, object_rotations=object_rotations,
        object_valid=np.ones((frames, 1), bool))
    probes = rng.normal(0, .22, (frames, 19, 3)).astype(np.float32)
    probes[:, -1] = 10
    rotations = np.broadcast_to(np.eye(3), (frames, 19, 3, 3))
    return scene, probes, rotations


class RuntimeQueryTests(unittest.TestCase):
    def assert_query_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(np.array_equal(expected[key], actual[key]), key)

    def test_cpu_all_fields_and_neighbor_counts(self):
        scene, probes, rotations = fixture()
        for dtype in (np.float32, np.float64):
            for neighbors in (1, 4, 8, 12):
                with self.subTest(dtype=dtype, neighbors=neighbors):
                    self.assert_query_equal(scene.query_reference(probes.astype(dtype), rotations, neighbors),
                                            scene.query(probes.astype(dtype), rotations, neighbors))

    def test_wide_and_nearest_keep_consumed_fields_exact(self):
        scene, probes, rotations = fixture()
        expected = scene.query_reference(probes, rotations)
        wide = scene.query_wide(probes, rotations)
        nearest = scene.query_nearest(probes, rotations)
        for key in expected:
            self.assertTrue(np.array_equal(expected[key][:, :, 2], wide[key][:, :, 2]), key)
            self.assertTrue(np.array_equal(expected[key][:, :, 2, 0], nearest[key][:, :, 2, 0]), key)
        self.assertFalse(wide['valid'][:, :, :2].any())

    def test_empty_and_zero_anchor_queries(self):
        scene = Native30SurfaceWindow(np.zeros((4, 0, 3)), np.zeros((4, 0), np.int64), np.zeros((4, 3)))
        for anchors in (0, 19):
            probes = np.zeros((4, anchors, 3))
            rotations = np.broadcast_to(np.eye(3), (4, anchors, 3, 3))
            self.assert_query_equal(scene.query_reference(probes, rotations), scene.query(probes, rotations))

    def test_radius_boundary_and_equal_distance_order(self):
        p = np.array([[.08, 0, 0], [-.08, 0, 0], [.25, 0, 0], [-.25, 0, 0],
                      [.6, 0, 0], [-.6, 0, 0], [0, .25, 0], [0, -.25, 0]], np.float32)
        scene = Native30SurfaceWindow(np.broadcast_to(p, (2, *p.shape)), np.ones((2, len(p)), np.int64), np.zeros((2, 3)))
        probes = np.zeros((2, 3, 3), np.float32)
        rotations = np.broadcast_to(np.eye(3), (2, 3, 3, 3))
        self.assert_query_equal(scene.query_reference(probes, rotations), scene.query(probes, rotations))

    def test_yaw_frame_owners_and_velocities(self):
        scene, probes, rotations = fixture()
        rotation = Rotation.from_rotvec([0, .37, 0]).as_matrix().astype(np.float32)
        yaw = YawSurface(scene, rotation)
        expected = scene.query_reference(probes @ rotation, rotation.T @ rotations)
        for key in ('points', 'normals', 'velocities'):
            expected[key] = expected[key] @ rotation.T
        self.assert_query_equal(expected, yaw.query(probes, rotations))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is needed for implementation equivalence')
    def test_cuda_all_fields_and_precisions(self):
        scene, probes, rotations = fixture()
        for dtype in (np.float32, np.float64):
            for neighbors in (1, 8):
                with self.subTest(dtype=dtype, neighbors=neighbors):
                    expected = scene.query_reference(probes.astype(dtype), rotations, neighbors)
                    actual = scene.query_motion(probes.astype(dtype), rotations, 'cuda', neighbors)
                    self.assert_query_equal(expected, actual)
                    wide = scene.query_wide_motion(probes.astype(dtype), rotations, 'cuda', neighbors)
                    for key in expected:
                        self.assertTrue(np.array_equal(expected[key][:, :, 2], wide[key][:, :, 2]), key)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is needed for implementation equivalence')
    def test_cuda_first_index_ties_and_missing_rows(self):
        points = np.array([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        probes = np.zeros((3, 3), np.float32)
        hits = [[0, 1, 2, 3], [], [2]]
        expected, valid = select_radius_samples(points, probes, hits, [[], [], []], len(points), 8)
        padded = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [2, 0, 0, 0]])
        actual, actual_valid = select_radius_samples_cuda(torch.from_numpy(points).cuda(), probes,
            padded, np.array([4, 0, 1]), 8)
        self.assertTrue(np.array_equal(valid, actual_valid))
        self.assertTrue(np.array_equal(expected[valid], actual[valid]))

    def test_mismatched_clock_rejected(self):
        scene, probes, rotations = fixture()
        with self.assertRaises(ValueError):
            scene.query(probes[:-1], rotations[:-1])


if __name__ == '__main__':
    unittest.main()
