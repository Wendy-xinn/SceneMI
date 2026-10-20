"""CPU geometry tests for the experimental reference-centred rotation guard."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .refinement_guard import (IDENTITY_6D, PARENTS, forward_kinematics_numpy,
                               motion_rotations, project_local_rotations,
                               unobserved_rotation_mask)


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


def global_rotations(motion):
    local = motion_rotations(motion)
    global_ = local.copy()
    for joint in range(1, 22):
        global_[..., joint, :, :] = global_[..., PARENTS[joint], :, :] @ local[..., joint, :, :]
    return global_


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

    def test_unobserved_tree_dependencies(self):
        meta = np.zeros((5, 22, 4))
        # Frame 0: every non-root local rotation is eligible.
        # Frame 1: knee self-position does not depend on its local rotation.
        meta[1, 5, [0, 2]] = 1
        # Frame 2: ankle position depends on its knee and hip, not itself.
        meta[2, 8, [0, 2]] = 1
        # Frame 3: ankle global orientation depends on ankle, knee and hip.
        meta[3, 8, [1, 3]] = 1
        # Frame 4: knee global orientation protects knee and ancestors.
        meta[4, 5, [1, 3]] = 1
        eligible = unobserved_rotation_mask(meta)
        self.assertFalse(eligible[:, 0].any())
        self.assertTrue(eligible[0, 1:].all())
        self.assertTrue(eligible[1, 5])
        self.assertFalse(eligible[1, 2])
        self.assertFalse(eligible[2, 5])
        self.assertFalse(eligible[2, 2])
        self.assertTrue(eligible[2, 8])
        self.assertTrue(eligible[2, 11])
        for joint in (2, 5, 8):
            self.assertFalse(eligible[3, joint])
        self.assertTrue(eligible[3, 11])
        self.assertFalse(eligible[4, 5])
        self.assertTrue(eligible[4, 8])
        np.testing.assert_array_equal(unobserved_rotation_mask(np.stack((meta, meta))),
                                      np.stack((eligible, eligible)))

    def test_unobserved_availability_confidence_and_nonfinite_metadata(self):
        meta = np.zeros((1, 22, 6))
        for availability, confidence in ((1, 0), (0, 1), (-1, 1), (1, -1),
                                         (np.nan, 1), (1, np.inf)):
            for flag_channel, confidence_channel in ((0, 2), (1, 3)):
                meta[:] = 0
                meta[0, 8, flag_channel] = availability
                meta[0, 8, confidence_channel] = confidence
                self.assertTrue(unobserved_rotation_mask(meta)[0, 1:].all())
        meta[:] = 0
        meta[0, 8, [0, 2]] = [1, .01]
        meta[..., 4:] = np.nan  # Extra metadata is not an observation channel.
        self.assertFalse(unobserved_rotation_mask(meta)[0, 5])
        self.assertTrue(unobserved_rotation_mask(meta)[0, 8])

    def test_unobserved_projection_preserves_all_measured_fk_positions_and_rotations(self):
        rest = rest_skeleton()
        reference = consistent_motion(rest)
        prediction = reference.copy()
        rng = np.random.default_rng(819)
        arbitrary = Rotation.random(3 * 22, random_state=rng).as_matrix().reshape(3, 22, 3, 3)
        prediction[:, 3:135] = (np.concatenate((arbitrary[..., 0], arbitrary[..., 1]), -1)
                                - IDENTITY_6D).reshape(3, 132)
        prediction[:, :3] += [.3, -.4, .2]
        prediction[:, 135:] = (forward_kinematics_numpy(prediction, rest) / 2).reshape(3, 66)
        meta = np.zeros((3, 22, 4))
        for frame, position_joints, rotation_joint in ((0, (5, 15, 20), 6),
                                                       (1, (8,), 20), (2, (7, 18), 21)):
            meta[frame, position_joints, 0] = 1
            meta[frame, position_joints, 2] = .7
            meta[frame, rotation_joint, [1, 3]] = 1
        original_meta = meta.copy()
        result = project_local_rotations(reference, prediction, rest, 0, observation_meta=meta)
        visible_position, visible_rotation = meta[..., 0] > 0, meta[..., 1] > 0
        np.testing.assert_allclose(forward_kinematics_numpy(result, rest)[visible_position],
                                   forward_kinematics_numpy(prediction, rest)[visible_position], atol=1e-12)
        np.testing.assert_allclose(global_rotations(result)[visible_rotation],
                                   global_rotations(prediction)[visible_rotation], atol=1e-12)
        np.testing.assert_array_equal(result[:, :9], prediction[:, :9])
        eligible = unobserved_rotation_mask(meta)
        np.testing.assert_allclose(motion_rotations(result)[eligible],
                                   motion_rotations(reference)[eligible], atol=1e-12)
        unchanged = ~eligible
        np.testing.assert_array_equal(result[:, 3:135].reshape(3, 22, 6)[unchanged],
                                      prediction[:, 3:135].reshape(3, 22, 6)[unchanged])
        self.assertGreater(np.max(np.abs(result - prediction)), .1)
        np.testing.assert_array_equal(meta, original_meta)
        hidden = np.zeros_like(meta)
        np.testing.assert_array_equal(project_local_rotations(reference, prediction, rest, 30,
                                                               observation_meta=hidden),
                                      project_local_rotations(reference, prediction, rest, 30))

    def test_unobserved_invalid_shapes_and_all_observed_passthrough(self):
        rest = rest_skeleton()
        motion = consistent_motion(rest)
        for bad in (np.zeros((22, 4)), np.zeros((3, 21, 4)), np.zeros((3, 22, 3)),
                    np.zeros((0, 22, 4)), np.zeros((3, 22, 4), dtype=complex)):
            with self.assertRaises(ValueError):
                unobserved_rotation_mask(bad)
        with self.assertRaises(ValueError):
            project_local_rotations(motion, motion, rest, 30,
                                    observation_meta=np.ones((2, 22, 4)))
        prediction = motion.copy()
        set_rotation(prediction, 5, Rotation.from_euler('x', 165, degrees=True).as_matrix())
        meta = np.ones((3, 22, 4))
        self.assertFalse(unobserved_rotation_mask(meta).any())
        np.testing.assert_array_equal(project_local_rotations(motion, prediction, rest, 0,
                                                               observation_meta=meta), prediction)


if __name__ == '__main__':
    unittest.main()
