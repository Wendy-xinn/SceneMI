import unittest
import torch
from experiments.offline_camera_retrain_v1.soft_track_robustness import perturb_head_track


class TrackProbeTests(unittest.TestCase):
    def test_bias_gap_and_no_target_mutation(self):
        trajectory = torch.zeros(2, 64, 22, 9)
        trajectory[..., 3] = trajectory[..., 7] = 1
        batch = {'trajectory': trajectory, 'motion': torch.randn(2, 64, 201),
                 'camera': trajectory[:, :, 15].clone(), 'bps': torch.randn(2, 64, 22, 9)}
        altered, mask = perturb_head_track(batch, 'drift_gap')
        self.assertTrue(torch.equal(batch['trajectory'], trajectory))
        self.assertIs(altered['motion'], batch['motion'])
        self.assertIs(altered['camera'], batch['camera'])
        self.assertIs(altered['bps'], batch['bps'])
        self.assertTrue(torch.equal(altered['trajectory'][:, :, :15], trajectory[:, :, :15]))
        self.assertEqual(int(mask[:, :, 15].sum()), 2 * (64 - 10))
        self.assertAlmostEqual(float(altered['trajectory'][0, 0, 15, 0] * 2), .03, places=6)
        columns = altered['trajectory'][:, :, 15, 3:].reshape(2, 64, 2, 3)
        identity = columns @ columns.transpose(-1, -2)
        self.assertTrue(torch.allclose(identity, torch.eye(2).expand_as(identity), atol=1e-6))
        clean, clean_mask = perturb_head_track(batch, 'clean')
        self.assertIs(clean['trajectory'], trajectory)
        _, empty = perturb_head_track(batch, 'none')
        self.assertEqual(int(empty.sum()), 0)
        self.assertEqual(int(clean_mask.sum()), 128)

if __name__ == '__main__':
    unittest.main()
