import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scenemi.config import load_config
from scenemi.data.camera_noise import augment_camera_batch, case_seed, validate_policy
from scenemi.data.ego import payload
from scenemi.observations import public_observations, camera_supervision
from scenemi.pipeline import native_motion_loss
from tests.model.test_amass_model import public_batch
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D


def policy(**changes):
    result = dict(version='correlated-camera-error-v1', probability=1., position_std_m=.005,
        rotation_std_deg=.5, drift_position_std_m=.015, drift_rotation_std_deg=.5,
        rho=.95, ramp_updates=500, seed=11)
    result.update(changes)
    return result


def batch():
    result = {k: v[:1].expand(2, *v.shape[1:]).clone() for k, v in public_batch().items()}
    rest = torch.zeros(2, 22, 3)
    rest[:, :, 1] = torch.arange(22) * .03
    rest[:, :, 0] = torch.arange(22) * .01
    motion = torch.zeros(2, 128, 201)
    motion[:, :, 0] = torch.arange(128) * .001
    result.update(motion=reconcile_motion(motion, rest), target_rest=rest,
                  body_format=torch.zeros(2, dtype=torch.long), sensor_profile=torch.zeros(2, dtype=torch.long))
    return result


class CameraNoiseTests(unittest.TestCase):
    def test_disabled_and_zero_amplitudes_are_exact_noops(self):
        original = batch()
        self.assertIs(augment_camera_batch(original, policy(probability=0)), original)
        self.assertIs(augment_camera_batch(original, policy(), strength=0), original)
        self.assertIs(augment_camera_batch(original, policy(position_std_m=0, rotation_std_deg=0,
            drift_position_std_m=0, drift_rotation_std_deg=0)), original)
        with self.assertRaises(ValueError):
            validate_policy(policy(rho=1.))

    def test_clean_training_control_still_accepts_matched_stress_validation(self):
        original = batch(); control = policy(probability=0.)
        self.assertIs(augment_camera_batch(original, control), original)
        stressed = augment_camera_batch(original, control, probability=1., seed=7)
        self.assertFalse(torch.equal(stressed['camera'], original['camera']))
        self.assertTrue(stressed['camera_noise_selected'].all())

    def test_reproducible_valid_rotations_no_mutation_and_no_oracle_confidence(self):
        original = batch(); saved = copy.deepcopy(original)
        a = augment_camera_batch(original, policy(), seed=13)
        b = augment_camera_batch(original, policy(), seed=13)
        for k in original:
            self.assertTrue(torch.equal(original[k], saved[k]), k)
        self.assertTrue(torch.equal(a['camera'], b['camera']))
        self.assertFalse(torch.equal(a['camera'], a['supervision_camera']))
        self.assertTrue(torch.equal(a['camera_confidence'], original['camera_confidence']))
        self.assertTrue(torch.equal(a['camera_available'], original['camera_available']))
        rotation = rotation_from_6d(a['camera'][..., 3:] - a['camera'].new_tensor(IDENTITY_6D))
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2, 128), atol=1e-6, rtol=0)
        torch.testing.assert_close(a['camera'][:, 0, [0, 2]], torch.zeros(2, 2), atol=1e-7, rtol=0)

    def test_reanchoring_preserves_physical_motion_and_fk(self):
        original = batch(); a = augment_camera_batch(original, policy(), seed=23)
        truth = forward_kinematics(original['motion'], original['target_rest'])
        transformed = forward_kinematics(a['motion'], a['target_rest'])
        torch.testing.assert_close(transformed, a['motion'][..., 135:].reshape(2, 128, 22, 3) * 2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(torch.cdist(truth, truth), torch.cdist(transformed, transformed), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(torch.diff(truth, dim=1).norm(dim=-1),
                                   torch.diff(transformed, dim=1).norm(dim=-1), atol=1e-6, rtol=1e-5)

    def test_corruption_does_not_depend_on_targets(self):
        original = batch(); poisoned = dict(original, motion=torch.full_like(original['motion'], float('nan')),
                                            target_rest=torch.full_like(original['target_rest'], float('nan')))
        a = augment_camera_batch(original, policy(), seed=31)
        b = augment_camera_batch(poisoned, policy(), seed=31)
        self.assertTrue(torch.equal(a['camera'], b['camera']))
        self.assertNotIn('supervision_camera', payload(a))
        self.assertNotIn('camera_noise_selected', payload(a))

    def test_case_noise_is_independent_of_evaluation_batch_size(self):
        original = batch()
        seeds = [case_seed(dict(dataset='amass', family='x', member=i, start=0), 3) for i in range(2)]
        a = augment_camera_batch(original, policy(), seeds=seeds)
        for i in range(2):
            single = {k: v[i:i+1] for k, v in original.items()}
            b = augment_camera_batch(single, policy(), seeds=seeds[i:i+1])
            torch.testing.assert_close(a['camera'][i:i+1], b['camera'], atol=0, rtol=0)

    def test_jitter_is_temporally_correlated(self):
        a = augment_camera_batch(batch(), policy(drift_position_std_m=0), seed=73)
        error = (a['camera'][..., :3] - a['supervision_camera'][..., :3]) * 2
        self.assertLess(float(torch.diff(error, dim=1).square().mean().sqrt()), .005)

    def test_camera_loss_uses_clean_reference_and_correct_sensor_mount(self):
        a = augment_camera_batch(batch(), policy(), seed=5)
        a['sensor_profile'][1] = 1
        mount = torch.eye(3).expand(2, 3, 3).clone()
        mount[1] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        details = dict(observations=public_observations(a), predicted_betas=torch.zeros(2, 16),
                       conditioned_observations={'mount_rotation': mount})
        clean = camera_supervision(a, details)
        noisy_reference = details['observations']['camera'].clone()
        self.assertFalse(torch.equal(clean['camera'], details['observations']['camera']))
        captured = {}
        def objective(*args, **kwargs):
            captured.update(kwargs['details'])
            return {'total': torch.tensor(0.)}
        with patch('scenemi.pipeline.motion_loss', side_effect=objective):
            native_motion_loss(a['motion'], a['motion'], a, torch.ones(2), details=details, config=load_config())
        self.assertTrue(torch.equal(captured['observations']['camera'], clean['camera']))
        self.assertTrue(torch.equal(details['observations']['camera'], noisy_reference))


if __name__ == '__main__':
    unittest.main()
