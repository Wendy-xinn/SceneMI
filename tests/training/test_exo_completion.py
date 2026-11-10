import unittest
import torch
from torch import nn

from scenemi.exo_completion import BoundaryEncoder, boundary_features


class CompletionTests(unittest.TestCase):
    def features(self):
        values=torch.zeros(3,128,383)
        values[...,0]=torch.arange(128)*.01
        values[...,358:360]=1
        for b,(start,end) in enumerate([(0,16),(48,72),(112,128)]):
            values[b,start:end,359]=0
        return values

    def test_boundary_context_distinguishes_one_and_two_sided_gaps(self):
        out=boundary_features(self.features())
        self.assertEqual(tuple(out.shape),(3,128,150))
        self.assertEqual(out[0,5,2:4].tolist(),[0.,1.])
        self.assertEqual(out[1,60,2:4].tolist(),[1.,1.])
        self.assertEqual(out[2,120,2:4].tolist(),[1.,0.])
        self.assertTrue(torch.isfinite(out).all())

    def test_missing_estimates_are_never_selected_as_boundary_anchors(self):
        values=self.features();first=boundary_features(values)
        # At the first tracked frame after the gap, all nearest anchors and
        # their finite-difference endpoints must remain actual observations.
        values[1,48:72,:3]=12345
        values[0,:16,:3]=12345
        values[2,112:,:3]=12345
        second=boundary_features(values)
        torch.testing.assert_close(first[1,72],second[1,72],atol=0,rtol=0)
        torch.testing.assert_close(first[0,16],second[0,16],atol=0,rtol=0)
        torch.testing.assert_close(first[2,111],second[2,111],atol=0,rtol=0)
        self.assertEqual(second[1,60,0].item(),13/128)
        self.assertEqual(second[1,60,1].item(),12/128)

    def test_zero_new_projection_preserves_initializer_and_gets_gradient(self):
        torch.manual_seed(7)
        original=nn.Conv1d(383,16,1);encoder=BoundaryEncoder(original,16)
        values=self.features().transpose(1,2)
        torch.testing.assert_close(encoder(values),original(values),atol=0,rtol=0)
        encoder(values).square().mean().backward()
        self.assertGreater(float(encoder.boundary[-1].weight.grad.norm()),0)

    def test_observation_features_detach_without_blocking_encoder_learning(self):
        torch.manual_seed(11)
        original=nn.Conv1d(383,16,1).requires_grad_(False)
        encoder=BoundaryEncoder(original,16)
        values=self.features().transpose(1,2).requires_grad_()
        untouched=values.detach().clone()
        frozen={name:p.detach().clone() for name,p in original.named_parameters()}
        optimizer=torch.optim.AdamW(encoder.boundary.parameters(),lr=.01)
        self.assertFalse(boundary_features(values.transpose(1,2)).requires_grad)
        with torch.autograd.detect_anomaly():
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                encoder(values).square().mean().backward()
                # Zero initialization blocks the first layer only on the first
                # update. Once the output projection learns, both layers learn.
                self.assertGreater(float(encoder.boundary[-1].weight.grad.norm()),0)
                if step:
                    self.assertGreater(float(encoder.boundary[0].weight.grad.norm()),0)
                for p in encoder.boundary.parameters():
                    self.assertTrue(torch.isfinite(p.grad).all())
                optimizer.step()
        torch.testing.assert_close(values.detach(),untouched,atol=0,rtol=0)
        for name,p in original.named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p,frozen[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
