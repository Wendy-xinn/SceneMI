"""Temporal gap, evidence, label-isolation and ego-bypass reference contracts."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from experiments.offline_camera_retrain_v1 import native30_refinement as refinement
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.ops import reference_temporal as temporal
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, PARENTS, forward_kinematics
from experiments.offline_camera_retrain_v1.test_native30_refinement import example


def gap_batch():
    batch = example()
    t = batch["trajectory"].shape[1]
    rotations = torch.zeros(1, t, 22, 3)
    rotations[:, :, 0, 1] = torch.linspace(.2, .8, t)
    rotations[:, :, 18, 2] = torch.linspace(.9, 1.2, t)
    truth = reconcile_motion(torch.zeros(1, t, 201), batch["rest"], rotation_axis_angle=rotations)
    batch["trajectory"].zero_()
    batch["trajectory"][..., :3] = forward_kinematics(truth, batch["rest"]) / 2
    batch["observation_meta"].zero_()
    batch["observation_meta"][..., 0] = 1
    batch["observation_meta"][..., 2] = 1
    batch["observation_meta"][:, (0, 2, 3, 7), 18, :4] = 0
    batch["trajectory"][:, (0, 2, 3, 7), 18] = float("nan")
    batch["motion"] = truth
    return batch


def step_degrees(motion, joint=20):
    r = global_rotations(motion)[:, :, joint]
    return matrix_to_axis_angle(r[:, 1:] @ r[:, :-1].transpose(-1, -2)).norm(dim=-1).rad2deg()


class TemporalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gaps_are_interpolated_and_endpoints_held_in_local_coordinates(self):
        batch = gap_batch()
        old, confidence = temporal._ORIGINAL_REFERENCE(batch)
        fixed, fixed_confidence = temporal.reference_from_observations_temporal(batch)
        self.assertGreater(float(step_degrees(old).max()), 20)
        self.assertLess(float(step_degrees(fixed).max()), 6)
        self.assertTrue(torch.equal(confidence, fixed_confidence))
        self.assertTrue(torch.equal(old[..., :9], fixed[..., :9]))
        valid = temporal.rotation_evidence(batch)
        old_r, fixed_r = global_rotations(old), global_rotations(fixed)
        for joint in range(1, 22):
            self.assertTrue(torch.allclose(old_r[:, :, joint][valid[:, :, joint]],
                                           fixed_r[:, :, joint][valid[:, :, joint]], atol=3e-6))
        # The repaired hand follows its moving parent, rather than freezing its
        # world orientation or resetting its local rotation to identity.
        local = refinement.rotation_from_6d(fixed[..., 3:135].reshape(1, 8, 22, 6))
        self.assertTrue(torch.allclose(local[:, 0, 18], local[:, 1, 18], atol=2e-6))
        self.assertTrue(torch.allclose(local[:, 7, 18], local[:, 6, 18], atol=2e-6))
        self.assertGreater(float((fixed_r[:, 0, 20] - fixed_r[:, 1, 20]).norm()), 1e-3)

    def test_rotation_measurements_take_priority_even_if_positions_are_missing(self):
        batch = gap_batch()
        rotation = axis_angle_to_matrix(torch.tensor([.4, -.7, .2]))
        batch["trajectory"][0, 2, 18, 3:] = torch.cat((rotation[:, 0], rotation[:, 1]))
        batch["observation_meta"][0, 2, 18, 1] = 1
        batch["observation_meta"][0, 2, 18, 3] = .7
        fixed, confidence = temporal.reference_from_observations_temporal(batch)
        self.assertTrue(torch.allclose(global_rotations(fixed)[0, 2, 18], rotation, atol=2e-6))
        self.assertEqual(float(confidence[0, 2, 18, 1]), float(torch.tensor(.7)))

    def test_so3_gap_crosses_pi_along_short_arc_instead_of_averaging_to_identity(self):
        batch = gap_batch()
        batch["trajectory"].zero_()
        batch["observation_meta"].zero_()
        for frame, degrees in ((1, 150.), (5, -150.)):
            rotation = axis_angle_to_matrix(torch.tensor([0., np.deg2rad(degrees), 0.], dtype=torch.float32))
            batch["trajectory"][0, frame, 18, 3:] = torch.cat((rotation[:, 0], rotation[:, 1]))
            batch["observation_meta"][0, frame, 18, 1] = 1
            batch["observation_meta"][0, frame, 18, 3] = .8
        fixed, confidence = temporal.reference_from_observations_temporal(batch)
        expected_midpoint = axis_angle_to_matrix(torch.tensor([0., torch.pi, 0.]))
        self.assertTrue(torch.allclose(global_rotations(fixed)[0, 3, 18], expected_midpoint, atol=2e-6))
        self.assertEqual(float(confidence[0, 3, 18, 1]), float(torch.tensor(.05)))
        local = refinement.rotation_from_6d(fixed[..., 3:135].reshape(1, 8, 22, 6))
        self.assertTrue(torch.allclose(local[0, 0, 18], local[0, 1, 18], atol=2e-6))
        self.assertTrue(torch.allclose(local[0, 5:, 18], local[0, 5, 18].expand(3, 3, 3), atol=2e-6))

    def test_labels_do_not_affect_reference_or_confidence(self):
        batch = gap_batch()
        first = temporal.reference_from_observations_temporal(batch)
        batch["motion"].fill_(float("nan"))
        batch["contact_truth"] = torch.full((1, 8, 22), float("nan"))
        batch["gt_rotations"] = torch.full((1, 8, 22, 3, 3), float("nan"))
        second = temporal.reference_from_observations_temporal(batch)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))

    def test_ego_is_bitwise_unchanged_in_individual_and_mixed_batches(self):
        ego = gap_batch()
        ego["mode"] = ["ego"]
        ego["observation_meta"][:, :, :15] = 0
        ego["observation_meta"][:, :, 16:] = 0
        for old, fixed in zip(temporal._ORIGINAL_REFERENCE(ego), temporal.reference_from_observations_temporal(ego)):
            self.assertTrue(torch.equal(old, fixed))
        exo = gap_batch()
        mixed = {key: torch.cat((value, ego[key]), 0) if isinstance(value, torch.Tensor) else value
                 for key, value in exo.items()}
        mixed["mode"] = ["exo", "ego"]
        for old, fixed in zip(temporal._ORIGINAL_REFERENCE(mixed), temporal.reference_from_observations_temporal(mixed)):
            self.assertTrue(torch.equal(old[1], fixed[1]))

    def test_all_missing_is_finite_has_fixed_bones_and_no_new_confidence(self):
        batch = gap_batch()
        batch["observation_meta"].zero_()
        batch["trajectory"].fill_(float("nan"))
        old = temporal._ORIGINAL_REFERENCE(batch)
        fixed, confidence = temporal.reference_from_observations_temporal(batch)
        self.assertTrue(torch.isfinite(fixed).all())
        self.assertTrue(torch.equal(fixed, old[0]))
        self.assertTrue(torch.equal(confidence, old[1]))
        self.assertTrue(bool((confidence == .05).all()))
        joints = forward_kinematics(fixed, batch["rest"])
        for joint in range(1, 22):
            actual = (joints[:, :, joint] - joints[:, :, PARENTS[joint]]).norm(dim=-1)
            wanted = (batch["rest"][:, joint] - batch["rest"][:, PARENTS[joint]]).norm(dim=-1)
            self.assertTrue(torch.allclose(actual, wanted[:, None], atol=2e-6))

    def test_input_tensors_unchanged_and_override_restored(self):
        batch = gap_batch()
        before = copy.deepcopy(batch)
        original = refinement.reference_from_observations
        with temporal.installed():
            with temporal.installed():
                self.assertIs(refinement.reference_from_observations, temporal.reference_from_observations_temporal)
                refinement.reference_from_observations(batch)
            self.assertIs(refinement.reference_from_observations, temporal.reference_from_observations_temporal)
        self.assertIs(refinement.reference_from_observations, original)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)

    def test_real_egobody_left_arm_regression_if_fixture_is_retained(self):
        path = Path(__file__).resolve().parents[1] / "runs/native30_exo_adapter_balanced_v2/baseline_validation/validation_latest.npz"
        if not path.exists():
            self.skipTest("Retained local EXO regression archive unavailable")
        with np.load(path, allow_pickle=False) as source:
            batch = {"rest": torch.from_numpy(source["8_rest"]).unsqueeze(0), "mode": ["exo"]}
            for regime in ("clean", "noisy"):
                batch["trajectory"] = torch.from_numpy(source[f"8_{regime}_observations"]).unsqueeze(0)
                batch["observation_meta"] = torch.from_numpy(source[f"8_{regime}_observation_meta"]).unsqueeze(0)
                old, old_confidence = temporal._ORIGINAL_REFERENCE(batch)
                fixed, confidence = temporal.reference_from_observations_temporal(batch)
                # Reject an archive mismatch rather than silently testing a
                # different case as the 20220215/1984 left-arm regression.
                self.assertEqual(int((step_degrees(old) > 30).sum()), 30)
                self.assertLess(int((step_degrees(fixed) > 30).sum()), 3)
                self.assertLess(float(torch.quantile(step_degrees(fixed).flatten(), .95)), 15)
                self.assertTrue(torch.equal(old_confidence, confidence))
                self.assertTrue(torch.equal(old[..., :9], fixed[..., :9]))


if __name__ == "__main__":
    unittest.main()
