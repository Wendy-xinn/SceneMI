"""CPU invariants for the process-local reference-prior ablation."""
from __future__ import annotations

import copy
import unittest

import torch
from pytorch3d.transforms import axis_angle_to_matrix

from experiments.offline_camera_retrain_v1 import native30_refinement as refinement
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, forward_kinematics
from experiments.offline_camera_retrain_v1.ops import reference_prior as prior


def rotation(yaw=0., pitch=0., roll=0.):
    vector = lambda x: torch.tensor(x, dtype=torch.float32)
    ry = axis_angle_to_matrix(vector([0, yaw, 0]))
    rx = axis_angle_to_matrix(vector([pitch, 0, 0]))
    rz = axis_angle_to_matrix(vector([0, 0, roll]))
    return ry @ rx @ rz


def batch(modes=('ego',), frames=5):
    """Synthetic fixed bones, camera/head observations; no model/assets loaded."""
    rest = torch.tensor([
        [0, .9, 0], [-.1, .88, 0], [.1, .88, 0], [0, 1.05, 0],
        [-.1, .48, 0], [.1, .48, 0], [0, 1.2, 0], [-.1, .08, 0],
        [.1, .08, 0], [0, 1.4, 0], [-.1, .03, .12], [.1, .03, .12],
        [0, 1.55, 0], [-.15, 1.45, 0], [.15, 1.45, 0], [0, 1.72, .02],
        [-.3, 1.45, 0], [.3, 1.45, 0], [-.55, 1.45, 0], [.55, 1.45, 0],
        [-.75, 1.45, 0], [.75, 1.45, 0]], dtype=torch.float32)
    b = len(modes)
    observation = torch.zeros(b, frames, 22, 9)
    meta = torch.zeros(b, frames, 22, 5)
    meta[..., 4] = 1
    head = torch.tensor([.2, 1.7, -.3])
    head_r = rotation(yaw=.4, pitch=.7, roll=.2)
    observation[:, :, 15, :3] = head / 2
    observation[:, :, 15, 3:] = torch.cat((head_r[:, 0], head_r[:, 1]))
    meta[:, :, 15, :4] = 1
    value = dict(trajectory=observation, observation_meta=meta, rest=rest[None].repeat(b, 1, 1),
        mode=list(modes), camera=torch.zeros(b, frames, 9), camera_available=torch.ones(b, frames, 2),
        camera_confidence=torch.ones(b, frames, 2), camera_intrinsics=torch.ones(b, frames, 4),
        camera_intrinsics_valid=torch.ones(b, frames), camera_context=torch.zeros(b, 6),
        observation_uncertainty=torch.zeros(b, frames, 22, 2), occupancy=torch.zeros(b, 1),
        bps=torch.zeros(b, frames, 67, 3), bps_valid=torch.zeros(b, frames, 67, dtype=torch.bool),
        body_type=torch.zeros(b, 2), body_scale=torch.ones(b, 1), surface_windows=[None] * b,
        skin_assets=[None] * b, mount_rotation=torch.eye(3)[None].repeat(b, 1, 1),
        mount_translation_m=torch.zeros(b, 3), mount_valid=torch.zeros(b), mount_uncertainty=torch.zeros(b, 2))
    return value


class ReferencePriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tearDown(self):
        prior.uninstall()

    def test_disabled_math_is_original_tensor_for_tensor(self):
        item = batch(('ego', 'exo'))
        original = prior._ORIGINAL_REFERENCE(item)
        reproduced = prior._build_reference(item, yaw_only=False)
        for expected, actual in zip(original, reproduced):
            self.assertTrue(torch.equal(expected, actual))

    def test_yaw_root_preserves_head_position_full_rotation_and_confidence(self):
        item = batch()
        saved = copy.deepcopy(item)
        old_ref, old_conf = prior._ORIGINAL_REFERENCE(item)
        ref, confidence = prior.yaw_only_reference_from_observations(item)
        r = global_rotations(ref)
        jp = forward_kinematics(ref, item['rest'])
        expected_head_r = rotation(yaw=.4, pitch=.7, roll=.2)
        self.assertTrue(torch.allclose(r[:, :, 0], rotation(yaw=.4), atol=2e-6, rtol=0))
        self.assertTrue(torch.allclose(jp[:, :, 15], item['trajectory'][:, :, 15, :3] * 2, atol=2e-6, rtol=0))
        self.assertTrue(torch.allclose(r[:, :, 15], expected_head_r, atol=2e-6, rtol=0))
        self.assertTrue(torch.equal(confidence, old_conf))
        self.assertFalse(torch.equal(ref, old_ref))
        for key in ('trajectory', 'observation_meta', 'camera', 'rest'):
            self.assertTrue(torch.equal(item[key], saved[key]))

    def test_mixed_exo_is_unchanged(self):
        item = batch(('ego', 'exo'))
        expected = prior._ORIGINAL_REFERENCE(item)
        actual = prior.yaw_only_reference_from_observations(item)
        for x, y in zip(expected, actual):
            self.assertTrue(torch.equal(x[1], y[1]))
        only_exo = batch(('exo',))
        for x, y in zip(prior._ORIGINAL_REFERENCE(only_exo), prior.yaw_only_reference_from_observations(only_exo)):
            self.assertTrue(torch.equal(x, y))

    def test_reliable_torso_fit_is_unchanged(self):
        item = batch()
        true_r = rotation(yaw=.3, pitch=.2, roll=-.15)
        points = torch.einsum('ij,btqj->btqi', true_r, item['rest'][:, None, :4].expand(-1, 5, -1, -1))
        item['trajectory'][:, :, :4, :3] = points / 2
        item['observation_meta'][:, :, :4, 0] = 1
        item['observation_meta'][:, :, :4, 2] = 1
        for x, y in zip(prior._ORIGINAL_REFERENCE(item), prior.yaw_only_reference_from_observations(item)):
            self.assertTrue(torch.equal(x, y))

    def test_missing_and_gapped_inputs_are_finite(self):
        for all_missing in (False, True):
            item = batch()
            item['observation_meta'][:, :, 15, :4] = 0 if all_missing else 1
            if not all_missing:
                item['observation_meta'][:, 1:4, 15, :4] = 0
            ref, confidence = prior.yaw_only_reference_from_observations(item)
            self.assertTrue(torch.isfinite(ref).all())
            self.assertTrue(torch.isfinite(confidence).all())
            self.assertTrue((confidence >= 0).all() and (confidence <= 1).all())

    def test_near_vertical_heading_gap_is_continuous_and_finite(self):
        item = batch()
        for t, pitch in enumerate((1.2, 1.5707, 1.5708, 1.5709, 1.2)):
            r = rotation(yaw=.6, pitch=pitch)
            item['trajectory'][0, t, 15, 3:] = torch.cat((r[:, 0], r[:, 1]))
        ref, _ = prior.yaw_only_reference_from_observations(item)
        root = global_rotations(ref)[0, :, 0]
        self.assertTrue(torch.isfinite(root).all())
        self.assertTrue(torch.allclose(root, rotation(yaw=.6), atol=2e-6, rtol=0))
        self.assertLess(float((root[1:] - root[:-1]).abs().max()), 2e-6)
        # With an entirely vertical window no heading is fabricated.
        r = rotation(yaw=.6, pitch=torch.pi / 2)
        item['trajectory'][0, :, 15, 3:] = torch.cat((r[:, 0], r[:, 1]))
        vertical, _ = prior.yaw_only_reference_from_observations(item)
        self.assertTrue(torch.allclose(global_rotations(vertical)[0, :, 0], torch.eye(3), atol=2e-6, rtol=0))

    def test_vertical_translation_is_equivariant_and_pitch_does_not_lift_feet(self):
        item = batch()
        ref, _ = prior.yaw_only_reference_from_observations(item)
        original = forward_kinematics(ref, item['rest'])
        moved = copy.deepcopy(item)
        moved['trajectory'][:, :, 15, 1] += .2  # m/2: a 0.4m change of head height.
        shifted, _ = prior.yaw_only_reference_from_observations(moved)
        displacement = forward_kinematics(shifted, item['rest']) - original
        self.assertTrue(torch.allclose(displacement[..., 1], torch.full_like(displacement[..., 1], .4), atol=2e-6, rtol=0))
        self.assertTrue(torch.allclose(displacement[..., (0, 2)], torch.zeros_like(displacement[..., (0, 2)]), atol=2e-6, rtol=0))
        steps = copy.deepcopy(item)
        levels = torch.tensor([-.15, 0., .1, .3, .2])
        steps['trajectory'][:, :, 15, 1] += levels / 2
        step_ref, _ = prior.yaw_only_reference_from_observations(steps)
        changes = forward_kinematics(step_ref, steps['rest']) - original
        self.assertTrue(torch.allclose(changes[..., 1], levels[None, :, None].expand_as(changes[..., 1]), atol=2e-6, rtol=0))
        upright = batch()
        r = rotation(yaw=.4)
        upright['trajectory'][:, :, 15, 3:] = torch.cat((r[:, 0], r[:, 1]))
        no_pitch, _ = prior.yaw_only_reference_from_observations(upright)
        self.assertTrue(torch.allclose(original[:, :, (10, 11), 1], forward_kinematics(no_pitch, upright['rest'])[:, :, (10, 11), 1], atol=2e-6, rtol=0))

    def test_ego_anchor_still_allows_a_tilted_generated_root(self):
        item = batch()
        tilted_motion, _ = prior._ORIGINAL_REFERENCE(item)
        with prior.installed():
            known = refinement.model_inputs({'diffusion_target': 'reference_residual'}, item)
        result = refinement.confidence_anchor(tilted_motion, known, torch.ones(1), .9)
        before = global_rotations(tilted_motion)[:, :, 0]
        after = global_rotations(result)[:, :, 0]
        self.assertTrue(torch.allclose(before, after, atol=2e-6, rtol=0))
        self.assertGreater(float((after - global_rotations(known['reference_motion'])[:, :, 0]).abs().max()), .2)

    def test_install_reaches_existing_model_inputs_alias_and_external_reference_stays_original(self):
        from experiments.offline_camera_retrain_v1.train_native30 import model_inputs as trainer_alias
        item = batch()
        original_known = trainer_alias({'diffusion_target': 'reference_residual'}, item)
        restore = prior.install()
        installed_known = trainer_alias({'diffusion_target': 'reference_residual'}, item)
        self.assertFalse(torch.equal(original_known['reference_motion'], installed_known['reference_motion']))
        self.assertTrue(torch.equal(original_known['trajectory'], installed_known['trajectory']))
        self.assertTrue(torch.equal(original_known['observation_meta'], installed_known['observation_meta']))
        external = copy.deepcopy(item)
        external['reference_motion'] = original_known['reference_motion']
        external['reference_confidence'] = original_known['reference_confidence']
        actual = trainer_alias({'diffusion_target': 'reference_residual'}, external)
        restore()
        expected = trainer_alias({'diffusion_target': 'reference_residual'}, external)
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(actual[key], value), key)
        self.assertIs(refinement.reference_from_observations, prior._ORIGINAL_REFERENCE)

    def test_cpu_only(self):
        ref, confidence = prior.yaw_only_reference_from_observations(batch())
        self.assertEqual(ref.device.type, 'cpu')
        self.assertEqual(confidence.device.type, 'cpu')
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
