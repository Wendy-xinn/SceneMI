"""Synthetic regression tests; neither dataset nor CUDA is required."""
import unittest
import torch
from experiments.offline_camera_retrain_v1.supervision import gait_losses


class GaitLossTest(unittest.TestCase):
    def target(self):
        joints = torch.zeros(2, 32, 22, 3)
        joints[:, :, 0, 0] = torch.arange(32) * .015
        # One planted foot and one swinging foot, then switch roles.
        joints[:, 16:, 10, 0] = torch.arange(16) * .04
        joints[:, :16, 11, 0] = torch.arange(16) * .04
        joints[:, 16:, 11, 0] = .60
        return joints

    def test_exact_target_zero(self):
        target = self.target()
        for value in gait_losses(target, target).values():
            self.assertEqual(float(value), 0.)

    def test_detects_sliding_and_absent_swing(self):
        target = self.target()
        predicted = target.clone()
        predicted[:, :, (10, 11), 0] = predicted[:, :, 0:1, 0]
        result = gait_losses(predicted, target)
        for name in ('gait_stance_velocity_m2_s2', 'gait_swing_velocity_m2_s2',
                     'gait_stride_displacement_m2'):
            self.assertGreater(float(result[name]), 0.)

    def test_support_height_and_finite_gradient(self):
        target = self.target()
        predicted = (target + torch.tensor([0., .1, 0.])).requires_grad_()
        result = gait_losses(predicted, target)
        self.assertAlmostEqual(float(result['gait_support_height_m2']), .01, places=6)
        sum(result.values()).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_zero_signal_disables_phase_supervision(self):
        target = self.target()
        for value in gait_losses(target + .1, target, torch.zeros(2)).values():
            self.assertEqual(float(value), 0.)

    def test_empty_stance_or_swing_is_finite(self):
        for target in (torch.zeros(2, 32, 22, 3),
                       torch.arange(32)[None, :, None, None].expand(2, 32, 22, 3).float()):
            result = gait_losses(target + .1, target)
            self.assertTrue(all(torch.isfinite(v) for v in result.values()))


if __name__ == '__main__':
    unittest.main()
