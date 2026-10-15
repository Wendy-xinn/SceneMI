import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import unittest
import torch
from torch import nn
from run_reference_probe import ReferenceResidual
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


class ReferencePriorTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        rest=torch.randn(1,22,3)*.1
        self.initial=reconcile_motion(torch.zeros(1,64,201),rest)
        self.known=dict(initial=self.initial,rest=rest,features=torch.randn(1,64,578),scene=torch.randn(1,64,528))

    def test_zero_residual_ignores_bad_prior_and_preserves_reference(self):
        model=ReferenceResidual(width=32);self.known['features'][...,377:]*=100
        p,_=model(self.known)
        torch.testing.assert_close(p,self.initial,rtol=0,atol=1e-7)

    def test_prior_off_cannot_see_poisoned_prior_features(self):
        model=ReferenceResidual(width=32);nn.init.normal_(model.output.weight,std=.01)
        a,_=model(self.known,use_prior=False)
        self.known['features'][...,377:]=float('nan')
        b,_=model(self.known,use_prior=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        b.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_scene_and_prior_factorial_controls_are_separate(self):
        model=ReferenceResidual(width=32);nn.init.normal_(model.output.weight,std=.01)
        a,_=model(self.known,use_prior=False,use_scene=False)
        self.known['features'][...,377:]*=200;self.known['scene']*=200
        b,_=model(self.known,use_prior=False,use_scene=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        c,_=model(self.known,use_prior=True,use_scene=False)
        self.assertGreater(float((c-b).abs().sum()),0)
        d,_=model(self.known,use_prior=False,use_scene=True)
        self.assertGreater(float((d-b).abs().sum()),0)


if __name__=='__main__':unittest.main()
