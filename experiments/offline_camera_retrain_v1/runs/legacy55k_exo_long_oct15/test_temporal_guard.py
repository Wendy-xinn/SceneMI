import unittest
import torch
from temporal_guard import smooth_reference,SmoothResidual,tapered_permissions,alternation_loss
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


class TemporalTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        self.rest=torch.randn(1,22,3)*.1
        self.motion=reconcile_motion(torch.zeros(1,64,201),self.rest)

    def test_smoothing_preserves_constant_pose_and_bones(self):
        p=smooth_reference(self.motion,self.rest)
        torch.testing.assert_close(p,self.motion,atol=1e-7,rtol=1e-6)
        impulse=self.motion.clone();impulse[:,30,:3]=.2
        smoothed=smooth_reference(impulse,self.rest)
        self.assertLess(float(smoothed[:,:,:3].diff(n=2,dim=1).square().sum()),float(impulse[:,:,:3].diff(n=2,dim=1).square().sum()))

    def test_taper_zero_at_boundary_and_invisible_flicker(self):
        allowed=torch.ones(1,40,1,dtype=torch.bool);allowed[:,10:15]=False
        weight=tapered_permissions(allowed)
        self.assertEqual(float(weight[:,9]),0);self.assertEqual(float(weight[:,15]),0)
        self.assertEqual(float(weight[:,10:15].sum()),0)
        self.assertEqual(float(weight[:,22]),1)
        allowed[:]=False;allowed[:,20]=True
        self.assertEqual(float(tapered_permissions(allowed).sum()),0)

    def test_training_and_eval_guard_preserves_available_positions(self):
        model=SmoothResidual(width=32)
        torch.nn.init.normal_(model.output.weight,std=.01)
        features=torch.randn(1,64,578);features[...,267:289]=torch.rand(1,64,22)>.5
        known=dict(initial=self.motion,rest=self.rest,features=features,scene=torch.randn(1,64,528))
        p,d=model(known)
        mask=features[...,267:289]>0
        a=forward_kinematics(self.motion,self.rest);b=forward_kinematics(p,self.rest)
        torch.testing.assert_close(a[mask],b[mask],atol=1e-6,rtol=1e-6)
        p.square().mean().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None))
        model.eval();q,_=model(known)
        torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_alternation_loss_identical_is_zero(self):
        truth=self.motion.clone();truth[:,:,3:135]=torch.randn_like(truth[:,:,3:135])*.1
        truth=reconcile_motion(truth,self.rest)
        loss=alternation_loss(truth,truth,self.rest,torch.ones(1,64,22))
        self.assertEqual(float(loss),0)


if __name__=='__main__':unittest.main()
