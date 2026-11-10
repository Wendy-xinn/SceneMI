"""Observable translation constraints must preserve pose and missingness."""
import unittest

import torch

from scenemi.exo_projection import translation_correction


class TranslationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(71)
        self.predicted = torch.randn(2, 128, 22, 3, generator=generator)

    def test_constant_rigid_displacement_is_recovered_without_changing_pose(self):
        delta = torch.tensor([.4, -.2, .7])[None, None]
        observed = self.predicted+delta[:, :, None]
        valid = torch.ones(2, 128, 22, dtype=torch.bool)
        correction = translation_correction(self.predicted, observed, valid, valid.float())
        torch.testing.assert_close(correction, delta.expand(2,128,3), atol=2e-5, rtol=0)
        moved = self.predicted+correction[:, :, None]
        torch.testing.assert_close(moved-moved[:, :, :1],
                                   self.predicted-self.predicted[:, :, :1], atol=5e-7, rtol=0)

    def test_missing_payload_does_not_matter_and_absent_track_is_unchanged(self):
        observed = self.predicted+.2
        valid = torch.zeros(2,128,22,dtype=torch.bool)
        valid[0, :, 0] = True
        actual = translation_correction(self.predicted, observed, valid, valid.float())
        observed[~valid] = float('nan')
        poisoned = translation_correction(self.predicted, observed, valid, valid.float())
        self.assertTrue(torch.equal(actual, poisoned))
        self.assertTrue(torch.equal(actual[1], torch.zeros_like(actual[1])))

    def test_short_unobserved_interval_is_supported_by_neighbor_frames(self):
        delta = torch.tensor([.2,.3,-.4])[None,None]
        observed = self.predicted+delta[:, :, None]
        valid = torch.ones(2,128,22,dtype=torch.bool)
        valid[:,48:64] = False
        correction = translation_correction(self.predicted, observed, valid, valid.float())
        torch.testing.assert_close(correction[:,48:64],delta.expand(2,16,3),atol=.002,rtol=0)


if __name__ == '__main__':
    unittest.main()
