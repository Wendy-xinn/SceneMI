"""CPU geometry tests for the experimental reference-centred rotation guard."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .refinement_guard import (IDENTITY_6D, PARENTS, forward_kinematics_numpy,
                               motion_rotations, project_local_rotations)


def rest_skeleton():
    rng = np.random.default_rng(538)
    rest = np.empty((22, 3))
    rest[0] = [.02, .93, -.04]
    for joint in range(1, 22):
        offset = rng.normal(size=3)
        offset = offset / np.linalg.norm(offset) * (.08 + .009 * joint)
        rest[joint] = rest[PARENTS[joint]] + offset
    return rest


def set_rotation(motion, joint, matrix):
    six = np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1) - IDENTITY_6D
    motion[..., 3 + joint * 6:9 + joint * 6] = six


def consistent_motion(rest, dtype=np.float64, frames=3):
    rng = np.random.default_rng(694)
    motion = np.zeros((frames, 201), dtype=dtype)
    motion[:, :3] = rng.normal(size=(frames, 3)) * .2
    matrices = Rotation.from_rotvec(rng.normal(size=(frames * 22, 3)) * .2).as_matrix()
    matrices = matrices.reshape(frames, 22, 3, 3)
    motion[:, 3:135] = (np.concatenate((matrices[..., 0], matrices[..., 1]), -1)
                        - IDENTITY_6D).reshape(frames, 132)
    motion[:, 135:] = (forward_kinematics_numpy(motion, rest) / 2).reshape(frames, 66)
    return motion


def angles_between(reference, prediction):
    a, b = motion_rotations(reference), motion_rotations(prediction)
    relative = np.swapaxes(a, -1, -2) @ b
    return np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(a.shape[:-2])


class RefinementGuardTests(unittest.TestCase):
    def test_noncommuting_composition_and_shortest_geodesic(self):
        rest = rest_skeleton()
        reference = consistent_motion(rest, frames=1)
        prediction = reference.copy()
        a = Rotation.from_euler('xyz', [51, -37, 24], degrees=True).as_matrix()
        # 210 degrees follows the opposite, shortest 150-degree path.
        delta = Rotation.from_rotvec(np.deg2rad(210) * np.array([.2, .7, -.3])
                                    / np.linalg.norm([.2, .7, -.3])).as_matrix()
        b = a @ delta
        set_rotation(reference, 5, a)
        set_rotation(prediction, 5, b)
        result = project_local_rotations(reference, prediction, rest, 60)
        shortest = Rotation.from_matrix(delta).as_rotvec()
        expected = a @ Rotation.from_rotvec(shortest * np.deg2rad(60)
                                            / np.linalg.norm(shortest)).as_matrix()
        np.testing.assert_allclose(motion_rotations(result)[0, 5], expected, atol=1e-12)
        self.assertAlmostEqual(angles_between(reference, result)[0, 5], 60, places=10)
        # Applying the same bounded delta on the wrong side must differ.
        wrong = Rotation.from_rotvec(shortest * .4).as_matrix() @ a
        self.assertGreater(np.max(np.abs(expected - wrong)), .1)

    def test_zero_cap_keeps_reference_local_rotations_and_prediction_root(self):
        rest = rest_skeleton()
        reference = consistent_motion(rest)
        prediction = reference.copy()
        prediction[:, :3] += [.3, -.2, .1]
        set_rotation(prediction, 0, Rotation.from_euler('z', 41, degrees=True).as_matrix())
        set_rotation(prediction, 5, Rotation.from_euler('x', 170, degrees=True).as_matrix())
        result = project_local_rotations(reference, prediction, rest, 0)
        np.testing.assert_array_equal(result[:, :9], prediction[:, :9])
        np.testing.assert_allclose(motion_rotations(result)[:, 1:],
                                   motion_rotations(reference)[:, 1:], atol=1e-12)
        np.testing.assert_allclose(result[:, 135:].reshape(-1, 22, 3) * 2,
                                   forward_kinematics_numpy(result, rest), atol=1e-12)

    def test_disabled_within_cap_and_large_cap_are_exact_copies(self):
        rest = rest_skeleton()
        reference = consistent_motion(rest, np.float32)
        prediction = reference.copy()
        set_rotation(prediction, 5, Rotation.from_euler('x', 12, degrees=True).as_matrix())
        prediction[:, 135:] = (forward_kinematics_numpy(prediction, rest) / 2).reshape(-1, 66)
        for cap in (None, 90, 180):
            result = project_local_rotations(reference, prediction, rest, cap)
            self.assertEqual(result.dtype, prediction.dtype)
            np.testing.assert_array_equal(result, prediction)
            self.assertFalse(np.shares_memory(result, prediction))

    def test_float32_bound_root_invariance_fk_bones_and_no_input_mutation(self):
        rest = rest_skeleton().astype(np.float32)
        reference = consistent_motion(rest, np.float32)
        prediction = reference.copy()
        prediction[:, :3] += [.3, -.4, .2]
        for joint, axis in ((0, 'y'), (5, 'x'), (14, 'z')):
            set_rotation(prediction, joint, Rotation.from_euler(axis, 165, degrees=True).as_matrix())
        original_reference, original_prediction, original_rest = reference.copy(), prediction.copy(), rest.copy()
        result = project_local_rotations(reference, prediction, rest, 30)
        self.assertLessEqual(angles_between(reference, result)[:, 1:].max(), 30 + 3e-6)
        np.testing.assert_array_equal(result[:, :9], prediction[:, :9])
        np.testing.assert_array_equal(result[:, 9:3 + 5 * 6], prediction[:, 9:3 + 5 * 6])
        joints = result[:, 135:].reshape(-1, 22, 3) * 2
        self.assertLess(np.max(np.abs(joints - forward_kinematics_numpy(result, rest))), 1.2e-7)
        for joint in range(1, 22):
            actual = np.linalg.norm(joints[:, joint] - joints[:, PARENTS[joint]], axis=-1)
            expected = np.linalg.norm(rest[joint] - rest[PARENTS[joint]])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
        np.testing.assert_array_equal(reference, original_reference)
        np.testing.assert_array_equal(prediction, original_prediction)
        np.testing.assert_array_equal(rest, original_rest)

    def test_batched_separate_rest_skeletons(self):
        first = rest_skeleton()
        rest = np.stack((first, first * 1.3))
        reference = np.stack([consistent_motion(item) for item in rest])
        prediction = reference.copy()
        set_rotation(prediction, 5, Rotation.from_euler('z', 130, degrees=True).as_matrix())
        result = project_local_rotations(reference, prediction, rest, 60)
        for batch in range(2):
            expected = project_local_rotations(reference[batch], prediction[batch], rest[batch], 60)
            np.testing.assert_array_equal(result[batch], expected)
        shared = project_local_rotations(reference, prediction, first, 60)
        self.assertEqual(shared.shape, prediction.shape)

    def test_invalid_inputs(self):
        rest = rest_skeleton()
        motion = consistent_motion(rest)
        for cap in (-1, 181, np.nan, np.inf, True, [30], '30', 1j):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                project_local_rotations(motion, motion, rest, cap)
        for bad in (motion[:, :-1], motion[0], motion[:0], motion.astype(int), motion.astype(complex)):
            with self.assertRaises(ValueError):
                project_local_rotations(motion, bad, rest, 30)
        for bad in (np.zeros((21, 3)), np.zeros((3, 22, 3)), rest * np.nan):
            with self.assertRaises(ValueError):
                project_local_rotations(motion, motion, bad, 30)
        nonfinite = motion.copy()
        nonfinite[0, 9] = np.inf
        with self.assertRaises(ValueError):
            project_local_rotations(nonfinite, motion, rest, 30)
        degenerate = motion.copy()
        degenerate[0, 9:15] = -IDENTITY_6D
        with self.assertRaises(ValueError):
            project_local_rotations(motion, degenerate, rest, 30)
        collinear = motion.copy()
        collinear[0, 9:15] = np.array([1, 0, 0, 2, 0, 0]) - IDENTITY_6D
        with self.assertRaises(ValueError):
            project_local_rotations(motion, collinear, rest, 30)


if __name__ == '__main__':
    unittest.main()
