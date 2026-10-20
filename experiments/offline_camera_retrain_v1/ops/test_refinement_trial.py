"""Loss contracts: reject harmful edits without penalizing useful large edits."""
import unittest

import torch

from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.ops.refinement_trial import hidden_reference_nondegradation


class NondegradationTests(unittest.TestCase):
    def setUp(self):
        self.rest = torch.zeros(1,22,3)
        self.rest[0,1] = torch.tensor([-.1,0.,0.])
        self.rest[0,4] = torch.tensor([-.1,-.4,0.])
        self.rest[0,7] = torch.tensor([-.1,-.8,0.])
        self.rest[0,10] = torch.tensor([-.1,-.8,.15])
        self.base = reconcile_motion(torch.zeros(1,3,201),self.rest)
        self.meta = torch.zeros(1,3,22,5)
        self.meta[...,0] = 1
        self.meta[:,:,7,0] = 0
        self.meta[:,:,10,0] = 0

    def rotated(self, angle):
        delta = torch.zeros(1,3,22,3)
        delta[:,:,4,0] = angle
        return reconcile_motion(self.base,self.rest,rotation_axis_angle=delta)

    def loss(self,pred,truth,ref,quality=None):
        return hidden_reference_nondegradation(pred,truth,ref,self.rest,self.meta,quality)[0]

    def test_good_reference_and_identity_edit_have_zero_penalty(self):
        self.assertEqual(float(self.loss(self.base,self.base,self.base)),0.)

    def test_wrong_hidden_leg_has_finite_positive_gradient(self):
        pred = self.rotated(1.).detach().requires_grad_()
        value = self.loss(pred,self.base,self.base)
        value.backward()
        self.assertGreater(float(value),0.)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad[...,27:33].abs().sum()),0.)

    def test_large_correct_repair_is_not_penalized(self):
        truth = self.rotated(2.)
        self.assertEqual(float(self.loss(truth,truth,self.base)),0.)

    def test_visible_only_and_empty_mask_are_zero(self):
        self.meta[...,0] = 1
        self.assertEqual(float(self.loss(self.rotated(1.),self.base,self.base)),0.)

    def test_quality_attenuates_instead_of_cancelling_normalization(self):
        pred = self.rotated(1.)
        full = self.loss(pred,self.base,self.base)
        soft = self.loss(pred,self.base,self.base,torch.full((1,3,22),.25))
        self.assertAlmostEqual(float(soft/full),.25,places=6)

    def test_targets_and_reference_are_stop_gradient(self):
        pred = self.rotated(1.).detach().requires_grad_()
        truth = self.base.clone().requires_grad_()
        ref = self.base.clone().requires_grad_()
        self.loss(pred,truth,ref).backward()
        self.assertIsNone(truth.grad)
        self.assertIsNone(ref.grad)


if __name__=='__main__': unittest.main()
