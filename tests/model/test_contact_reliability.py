from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from scenemi.contact_loss import (
    load_policy, contact_reliability, contact_terms, reliable_native_motion_loss,
)


class ContactReliabilityTest(unittest.TestCase):
    def setUp(self):
        self.policy=load_policy()

    def test_per_foot_stance_support_and_missing_evidence(self):
        p=np.array([[1,0],[1,1],[1,1],[1,1]],np.float32)
        gaps=np.array([[0,.3],[.035,0],[np.inf,0],[0,0]],np.float32)
        coverage=np.ones_like(p);speed=np.zeros_like(p)
        value=contact_reliability(p,gaps,coverage,speed,[True,True,True,False],self.policy)
        np.testing.assert_allclose(value['weight'],[[1,0],[.5,1],[0,1],[0,0]],atol=1e-6)
        # Swing is not a bad motion frame. Its other foot can still have contact.
        self.assertFalse(value['stance'][0,1]);self.assertTrue(value['stance'][0,0])

    def test_support_speed_and_large_source_penetration_reduce_contact(self):
        value=contact_reliability(np.ones((2,2)),np.array([[0,0],[-.06,0]]),
            np.array([[.1,1],[1,1]]),np.array([[0,.3],[0,0]]),[True,True],self.policy)
        np.testing.assert_array_equal(value['weight'],[[0,0],[0,1]])

    def geometry(self,weight,valid=True):
        gap=torch.full((1,4,2),.06,requires_grad=True)
        velocity=torch.full((1,3,2,3),.2,requires_grad=True)
        w=torch.full((1,4,2),weight,requires_grad=True)
        losses=contact_terms(gap,velocity,weight=w,stance=torch.ones_like(w,dtype=torch.bool),
            distance_valid=torch.full_like(w,valid,dtype=torch.bool),
            velocity_valid=torch.full((1,3,2),valid,dtype=torch.bool),
            signal=torch.ones(1),policy=self.policy)
        sum(losses.values()).backward()
        return losses,gap,velocity,w

    def test_uniform_low_confidence_reduces_loss_and_gradient(self):
        full,gap1,_,w1=self.geometry(1.)
        low,gap2,_,w2=self.geometry(.1)
        for name in full:
            self.assertAlmostEqual(float(low[name].detach()),float(full[name].detach())*.1,places=6)
        torch.testing.assert_close(gap2.grad,gap1.grad*.1)
        self.assertIsNone(w1.grad);self.assertIsNone(w2.grad)

    def test_all_masked_and_unknown_observed_surfaces_have_zero_gradients(self):
        for weight,valid in ((0.,True),(1.,False)):
            losses,gap,velocity,_=self.geometry(weight,valid)
            self.assertTrue(all(float(v.detach())==0 for v in losses.values()))
            self.assertEqual(float(gap.grad.abs().sum()+velocity.grad.abs().sum()),0.)

    def test_stance_transition_has_no_sliding_constraint(self):
        gap=torch.full((1,3,2),.1,requires_grad=True)
        velocity=torch.ones((1,2,2,3),requires_grad=True)
        stance=torch.tensor([[[True,False],[False,True],[True,False]]])
        losses=contact_terms(gap,velocity,weight=torch.ones_like(gap),stance=stance,
            distance_valid=torch.ones_like(stance),velocity_valid=torch.ones((1,2,2),dtype=torch.bool),
            signal=torch.ones(1),policy=self.policy)
        self.assertEqual(float(losses['scene_contact_sliding'].detach()),0.)

    def test_mask_changes_only_planting_in_base_motion_loss(self):
        from scenemi.scene_model import native_motion_loss
        b,t=1,16
        target=torch.zeros((b,t,201));prediction=target.clone()
        prediction[:,:,0]=torch.arange(t)*.01;prediction.requires_grad_()
        rest=torch.zeros((b,22,3));rest[:,:,1]=torch.arange(22)*.02
        batch=dict(target_rest=rest,target_betas=torch.zeros(b,16),body_format=torch.tensor([3]))
        obs=dict(camera=torch.zeros(b,t,9),camera_available=torch.zeros(b,t,2),
            camera_confidence=torch.zeros(b,t,2),camera_context=torch.zeros(b,6),
            trajectory=torch.zeros(b,t,22,9),observation_meta=torch.zeros(b,t,22,5))
        details=dict(rest=rest,raw_motion=prediction,predicted_betas=torch.zeros(b,16),
            head_camera_translation=torch.zeros(b,3),observations=obs)
        config=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/amass_ego128.yaml').read_text())
        baseline=native_motion_loss(prediction,target,batch,torch.ones(b),details=details,config=config)
        full=reliable_native_motion_loss(prediction,target,batch,torch.ones(b),details=details,
            config=config,contact_weight=torch.ones(b,t,2))
        masked=reliable_native_motion_loss(prediction,target,batch,torch.ones(b),details=details,
            config=config,contact_weight=torch.zeros(b,t,2))
        self.assertGreater(float(baseline['planting'].detach()),0)
        for name,value in baseline.items():
            torch.testing.assert_close(value,full[name])
            if name not in ('planting','total'):torch.testing.assert_close(value,masked[name])
        self.assertEqual(float(masked['planting'].detach()),0)
        masked['total'].backward()
        self.assertGreater(float(prediction.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
