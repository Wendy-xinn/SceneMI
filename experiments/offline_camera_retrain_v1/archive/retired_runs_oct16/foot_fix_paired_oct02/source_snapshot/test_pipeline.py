"""Fast correctness gates for the offline SceneMI retraining adapter."""
import unittest

import torch

from experiments.offline_camera_retrain_v1.control import HEAD, sample_control_masks
from experiments.offline_camera_retrain_v1.data import OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


class OfflineSceneMIPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation = OfflineSceneMIData('validation', seed=777)

    def test_zero_residual_is_identity_rotation(self):
        residual = torch.zeros(4, 22, 6)
        matrix = rotation_from_6d(residual)
        identity = torch.eye(3).expand_as(matrix)
        torch.testing.assert_close(matrix, identity, atol=1e-6, rtol=0.)

    def test_clean_targets_close_under_forward_kinematics(self):
        # TRUMANS joints and fitted SMPL-X have a known ~2.4 cm source-domain
        # discrepancy; EgoBody should be substantially tighter.
        thresholds_m = {'trumans': .08, 'camera_wearer': .02, 'interactee': .02}
        for group, threshold in thresholds_m.items():
            with self.subTest(group=group):
                sample, _ = self.validation.sample(128, group)
                batch = collate([sample])
                predicted = forward_kinematics(batch['motion'], batch['rest'])
                error = (predicted - batch['joints'] * 2).norm(dim=-1).mean()
                self.assertLess(float(error), threshold)

    def test_trajectory_is_full_body_and_head_is_camera(self):
        sample, _ = self.validation.sample(64, 'camera_wearer')
        self.assertEqual(sample['trajectory'].shape, (64, 22, 9))
        torch.testing.assert_close(torch.from_numpy(sample['trajectory'][:, HEAD]),
                                   torch.from_numpy(sample['camera']))

    def test_control_mixture_is_head_dominant_and_non_head_capable(self):
        import numpy as np
        rng = np.random.default_rng(2026)
        mask, modes = sample_control_masks(2000, 64, rng, 'cpu')
        head_fraction = float(mask[:, :, HEAD].any(dim=1).float().mean())
        self.assertGreater(head_fraction, .8)
        self.assertLess(head_fraction, .92)
        self.assertTrue(any(mode == 'pelvis' for mode in modes))
        self.assertTrue(any(mode == 'sparse_wrists' for mode in modes))


if __name__ == '__main__':
    unittest.main()
