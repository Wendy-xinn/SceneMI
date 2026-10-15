"""Check collapsed gait detection, masks, physical lags and baseline parity."""
import copy
import json
from pathlib import Path
import unittest
import zipfile

import torch
from experiments.offline_camera_retrain_v1.native30_gait_supervision import DEFAULT, gait_terms
from experiments.offline_camera_retrain_v1.native30_losses import base_losses


class GaitSupervisionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.truth = torch.zeros(1, 96, 22, 3)
        phase = torch.arange(96) * (2 * torch.pi / 30)
        self.truth[0, :, 10, 2] = .2 * phase.sin()
        self.truth[0, :, 11, 2] = -.2 * phase.sin()
        self.quality = torch.ones(1, 96, 22)

    def terms(self, prediction, truth=None, quality=None, signal=None):
        return gait_terms(prediction, self.truth if truth is None else truth,
            self.quality if quality is None else quality,
            torch.ones(1) if signal is None else signal, copy.deepcopy(DEFAULT))

    def test_identical_motion_zero_and_finite_gradient(self):
        p = self.truth.clone().requires_grad_()
        terms = self.terms(p)
        self.assertEqual(float(terms['total']), 0)
        terms['total'].backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_frozen_feet_detected_and_can_receive_gradient(self):
        p = torch.zeros_like(self.truth).requires_grad_()
        terms = self.terms(p)
        self.assertGreater(float(terms['stride_displacement'].detach()), .001)
        self.assertGreater(float(terms['coordination_amplitude'].detach()), 0)
        terms['total'].backward()
        self.assertGreater(float(p.grad[:, :, (10, 11)].norm()), 0)

    def test_root_relative_terms_ignore_common_body_translation(self):
        p = self.truth * .3
        offset = torch.arange(96)[None, :, None, None] * torch.tensor([.004, .001, -.002])
        a, b = self.terms(p), self.terms(p + offset, self.truth + offset)
        for key in ('relative_position', 'stride_displacement', 'coordination_amplitude', 'coordination_relation'):
            torch.testing.assert_close(a[key], b[key], atol=1e-7, rtol=1e-5)

    def test_stride_uses_six_and_twelve_frames(self):
        p = torch.zeros_like(self.truth)
        relative = self.truth[:, :, (10, 11)] - self.truth[:, :, :1]
        expected = sum((relative[:, k:] - relative[:, :-k]).square().sum(-1).mean() / 2 for k in (6, 12))
        torch.testing.assert_close(self.terms(p)['stride_displacement'], expected)

    def test_unreliable_pulse_does_not_become_velocity_or_stride_target(self):
        truth = torch.zeros_like(self.truth)
        truth[:, 20, 10, 2] = 2
        q = self.quality.clone()
        q[:, 20, 10] = .25
        terms = self.terms(torch.zeros_like(truth), truth, q)
        for key in ('stance_velocity', 'swing_velocity', 'stride_displacement', 'coordination_amplitude', 'coordination_relation'):
            self.assertEqual(float(terms[key]), 0, key)
        self.assertGreater(float(terms['relative_position']), 0)

    def test_zero_signal_and_no_reliable_labels_are_safe(self):
        for q, signal in ((self.quality, torch.zeros(1)), (torch.zeros_like(self.quality), torch.ones(1))):
            p = torch.zeros_like(self.truth).requires_grad_()
            terms = self.terms(p, quality=q, signal=signal)
            self.assertEqual(float(terms['total']), 0)
            terms['total'].backward()
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_disabled_profile_matches_completed_sealed_objective(self):
        folder = Path(__file__).parent / 'runs/native30_gait_restore_probe_oct15'
        if not (folder / 'baseline_sealed_sources.zip').exists():
            self.skipTest('Historical source snapshot is a local integration fixture')
        record = json.loads((folder / 'BASELINE_SOURCE_SNAPSHOT.json').read_text())
        path = next(p for p in record['original_code_sha256'] if p.endswith('/native30_losses.py'))
        with zipfile.ZipFile(folder / 'baseline_sealed_sources.zip') as archive:
            source = archive.read('sources/' + path.lstrip('/')).decode()
        scope = {}
        exec(compile(source, path, 'exec'), scope)
        torch.manual_seed(34)
        truth = torch.randn(1, 20, 201) * .01
        prediction = truth + torch.randn_like(truth) * .01
        rest = torch.randn(1, 22, 3) * .1
        batch, signal = dict(rest=rest), torch.tensor([.5])
        old = scope['base_losses'](prediction, truth, batch, signal)
        new = base_losses(prediction, truth, batch, signal)
        for key in old:
            torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
        for function in (scope['base_losses'], base_losses):
            p = prediction.clone().requires_grad_()
            function(p, truth, batch, signal)['total'].backward()
            if function is base_losses:
                torch.testing.assert_close(old_gradient, p.grad, atol=0, rtol=0)
            else:
                old_gradient = p.grad.clone()


if __name__ == '__main__':
    unittest.main()
