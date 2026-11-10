import unittest
import torch
from scenemi.geometry.temporal import continuation_proposal, missing_intervals
from scenemi.exo_basis import continuation_proposal as original


class TemporalCompletionTests(unittest.TestCase):
    def test_multiple_outages_preserve_observed_frames_and_recover_linear_motion(self):
        truth=torch.arange(128,dtype=torch.float32)[None,:,None].expand(1,128,201).clone()
        available=torch.ones(1,128);available[:,:10]=0;available[:,40:50]=0;available[:,110:]=0
        initial=truth.clone();initial[available==0]=-999.
        result=continuation_proposal(initial,available)
        self.assertEqual(missing_intervals(available[0]),[(0,10),(40,50),(110,128)])
        self.assertTrue(torch.equal(result[available.bool()],truth[available.bool()]))
        self.assertTrue(torch.allclose(result[:,:,:135],truth[:,:,:135],atol=2e-5))
        self.assertTrue(torch.equal(result[:,:,135:],initial[:,:,135:]))

    def test_single_outage_is_bitwise_compatible_with_original(self):
        initial=torch.randn(1,128,201)
        for start,end in [(0,8),(30,50),(120,128)]:
            available=torch.ones(1,128);available[:,start:end]=0
            self.assertTrue(torch.equal(continuation_proposal(initial,available),original(initial,available)))

    def test_rejects_missing_initialization_instead_of_using_hidden_values(self):
        with self.assertRaisesRegex(ValueError,'actual estimated anchors'):
            continuation_proposal(torch.zeros(1,128,201),torch.zeros(1,128))


if __name__=='__main__':unittest.main()
