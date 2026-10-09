import unittest
import torch
from experiments.offline_camera_retrain_v1.coordination import coordination_losses


class CoordinationTest(unittest.TestCase):
    def target(self, same_side=False):
        p = torch.zeros(2, 64, 22, 3)
        t = torch.arange(64) * .31
        wave = .2 * (t.sin() + .2 * (2.3 * t).cos())
        p[:, :, 20, 0] = wave
        p[:, :, 21, 0] = -wave
        p[:, :, 10, 0] = wave if same_side else -wave
        p[:, :, 11, 0] = -p[:, :, 10, 0]
        p[:, :, 15, 2] = .08 * (t * .7).sin()
        return p

    def test_exact_any_phase_zero(self):
        for same in (False, True):
            target = self.target(same)
            for value in coordination_losses(target, target).values():
                self.assertEqual(float(value), 0.)

    def test_wrong_coordination_detected_without_amplitude_change(self):
        values = coordination_losses(self.target(True), self.target(False))
        self.assertGreater(float(values['coordination_relation_mse']), .05)
        self.assertEqual(float(values['coordination_amplitude_m2']), 0.)

    def test_freezing_and_amplitude_collapse_penalized(self):
        target = self.target()
        for scale in (0., .1):
            values = coordination_losses(target * scale, target)
            self.assertGreater(float(values['coordination_amplitude_m2']), 0.)
            self.assertGreater(float(values['coordination_relation_mse']), 0.)

    def test_stationary_lower_body_cooperative_hands_and_torso(self):
        # Non-locomotion: two hands cooperate with a torso bend. Correct
        # in-phase arms must not be penalized for violating a walking rule.
        target = torch.zeros(1, 64, 22, 3)
        wave = .15 * torch.sin(torch.arange(64) * .2)
        target[:, :, (20, 21), 2] = wave[:, None]
        target[:, :, 9, 2] = wave * .4
        self.assertEqual(float(sum(coordination_losses(target, target).values())), 0.)
        wrong = target.clone()
        wrong[:, :, 21, 2] *= -1
        self.assertGreater(float(coordination_losses(wrong, target)['coordination_relation_mse']), 0.)
        wrong = target.clone()
        wrong[:, :, 9] = 0
        self.assertGreater(float(coordination_losses(wrong, target)['coordination_amplitude_m2']), 0.)

    def test_translation_and_rotation_invariant(self):
        truth, pred = self.target(), self.target(True)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        shift = torch.randn(2, 64, 1, 3)
        a = coordination_losses(pred, truth)
        b = coordination_losses(pred @ rotation + shift, truth @ rotation + shift)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=1e-7, rtol=1e-5)

    def test_static_and_short_finite_backward(self):
        for length in (2, 16, 64, 128, 192):
            truth = torch.zeros(2, length, 22, 3)
            pred = truth.clone().requires_grad_()
            sum(coordination_losses(pred, truth).values()).backward()
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_zero_signal_and_finite_nonzero_gradient(self):
        truth = self.target()
        pred = self.target(True).requires_grad_()
        self.assertEqual(float(sum(coordination_losses(pred, truth, torch.zeros(2)).values()).detach()), 0.)
        sum(coordination_losses(pred, truth).values()).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
