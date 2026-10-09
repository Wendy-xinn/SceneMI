"""Synthetic regression tests; neither dataset nor CUDA is required."""
import unittest
import torch
from experiments.offline_camera_retrain_v1.supervision import gait_losses, phase_free_support_losses, supervised_losses


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
        self.assertAlmostEqual(float(result['gait_support_height_m2'].detach()), .01, places=6)
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

    def test_support_is_invariant_to_left_right_phase_swap(self):
        target = self.target()
        predicted = target.clone()
        predicted[:, :, (10, 11)] = target[:, :, (11, 10)]
        for value in phase_free_support_losses(predicted, target).values():
            self.assertEqual(float(value), 0.)

    def test_support_detects_floating_penetration_and_sliding(self):
        target = self.target()
        for displacement in (-.1, .1):
            p = target + torch.tensor([0., displacement, 0.])
            result = phase_free_support_losses(p, target)
            self.assertAlmostEqual(float(result['support_envelope_m2']), .01, places=6)
        sliding = target.clone()
        sliding[:, :, (10, 11), 0] = sliding[:, :, 0:1, 0]
        self.assertGreater(float(phase_free_support_losses(sliding, target)['phase_free_planting_m2_s2']), 0.)

    def test_support_all_flight_has_finite_backward(self):
        target = torch.arange(32)[None, :, None, None].expand(2, 32, 22, 3).float()
        prediction = target.clone().requires_grad_()
        values = phase_free_support_losses(prediction, target)
        self.assertEqual(float(values['phase_free_planting_m2_s2'].detach()), 0.)
        sum(values.values()).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_complete_profiles_zero_on_consistent_static_truth(self):
        truth = torch.zeros(2, 64, 201)
        batch = {'rest': torch.zeros(2, 22, 3), 'joints': torch.zeros(2, 64, 22, 3)}
        for profile in ('baseline', 'gait_v1', 'gait_v2', 'coordination_v1', 'orientation_v1', 'orientation_v2'):
            predicted = truth.clone().requires_grad_()
            loss = supervised_losses(predicted, truth, batch, profile=profile,
                                     signal_weight=torch.tensor([0., 1.]))['total']
            self.assertEqual(float(loss.detach()), 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(predicted.grad).all())


if __name__ == '__main__':
    unittest.main()
