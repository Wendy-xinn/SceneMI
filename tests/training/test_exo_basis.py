"""Continuation directions must come only from available estimated neighbors."""
import unittest

import torch

from scenemi.exo_basis import continuation_proposal


class BasisTests(unittest.TestCase):
    def test_linear_motion_is_reconstructed_at_all_three_gap_placements(self):
        expected=torch.arange(128).float()[None,:,None].expand(3,-1,201).clone()*.003
        initial=expected.clone();available=torch.ones(3,128)
        for b,(start,end) in enumerate([(0,16),(48,72),(112,128)]):
            available[b,start:end]=0
            initial[b,start:end,:135]=12345.
        proposal=continuation_proposal(initial,available)
        torch.testing.assert_close(proposal[:,:,:135],expected[:,:,:135],atol=5e-7,rtol=0)
        torch.testing.assert_close(proposal[available>0],initial[available>0],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
