"""Counterfactual checks: actual leg articulation, phase freedom and GT masks."""
import copy
import unittest
import torch
from experiments.offline_camera_retrain_v1.native30_gait_supervision_v2 import DEFAULT,gait_terms,validate_config
from experiments.offline_camera_retrain_v1.native30_losses import base_losses
from experiments.offline_camera_retrain_v1.supervision import PARENTS,forward_kinematics
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


class BilateralGaitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.target=torch.zeros(1,96,22,3)
        phase=torch.arange(96)*(2*torch.pi/30)
        self.target[0,:,10,2]=.2*phase.sin()
        self.target[0,:,11,2]=-.2*phase.sin()
        self.q=torch.ones(1,96,22)
    def terms(self,p,t=None,q=None,signal=None,**kwargs):
        return gait_terms(p,self.target if t is None else t,self.q if q is None else q,
            torch.ones(1) if signal is None else signal,copy.deepcopy(DEFAULT),**kwargs)
    def test_correct_motion_zero_finite_gradient_and_no_target_gradient(self):
        target=self.target.clone().requires_grad_();p=target.detach().clone().requires_grad_()
        terms=self.terms(p,target);self.assertEqual(float(terms['total']),0)
        terms['total'].backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertIsNone(target.grad)
    def test_whole_body_translation_does_not_fake_leg_articulation(self):
        offset=torch.arange(96)[None,:,None,None]*torch.tensor([.01,0.,.004])
        p=torch.zeros_like(self.target).add(offset).requires_grad_()
        terms=self.terms(p,self.target+offset)
        for k in ('bilateral_stride','bilateral_amplitude','bilateral_relation'):
            self.assertGreater(float(terms[k]),0,k)
        terms['total'].backward()
        self.assertGreater(float(p.grad[:,:,(10,11)].norm()),0)
    def test_same_amplitude_in_phase_feet_are_detected(self):
        p=self.target.clone();p[:,:,11]=p[:,:,10]
        terms=self.terms(p)
        self.assertGreater(float(terms['bilateral_relation']),.1)
        self.assertGreater(float(terms['bilateral_stride']),.001)
    def test_hopping_is_not_forced_to_alternate(self):
        t=self.target.clone();t[:,:,11]=t[:,:,10]
        self.assertEqual(float(self.terms(t,t)['total']),0)
    def test_body_local_terms_invariant_to_time_varying_root_motion(self):
        p=self.target*.3
        angles=torch.arange(96)*.02;c=angles.cos();s=angles.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
        r=torch.stack((c,z,s,z,o,z,-s,z,c),-1).reshape(1,96,3,3)
        offset=torch.arange(96)[None,:,None,None]*torch.tensor([.01,.002,.004])
        rotate=lambda x:(r[:,:,None]@x[...,None]).squeeze(-1)+offset
        eye=torch.eye(3).expand(1,96,3,3)
        a=self.terms(p,root_rotations=eye,target_root_rotations=eye)
        b=self.terms(rotate(p),rotate(self.target),root_rotations=r,target_root_rotations=r)
        for key in ('bilateral_stride','bilateral_amplitude','bilateral_relation'):
            torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=1e-5)
    def test_six_and_twelve_frame_bilateral_displacements(self):
        diff=self.target[:,:,10]-self.target[:,:,11]
        expected=sum((diff[:,k:]-diff[:,:-k]).square().sum(-1).mean()/2 for k in (6,12))
        torch.testing.assert_close(self.terms(torch.zeros_like(self.target))['bilateral_stride'],expected)
    def test_isolated_fit_pulse_skips_temporal_stencils(self):
        t=torch.zeros_like(self.target);t[:,20,10,2]=2
        q=self.q.clone();q[:,20,10]=.25
        terms=self.terms(torch.zeros_like(t),t,q)
        self.assertEqual(float(terms['bilateral_stride']),0)
        self.assertEqual(float(terms['stance_velocity']),0)
        self.assertEqual(float(terms['swing_velocity']),0)
        self.assertGreater(float(terms['relative_position']),0)
    def test_zero_signal_and_no_reliable_points_are_safe(self):
        for q,signal in ((self.q,torch.zeros(1)),(torch.zeros_like(self.q),torch.ones(1))):
            p=torch.zeros_like(self.target).requires_grad_();terms=self.terms(p,q=q,signal=signal)
            self.assertEqual(float(terms['total']),0);terms['total'].backward()
            self.assertTrue(torch.isfinite(p.grad).all())
    def test_stationary_height_errors_and_planting_are_not_hidden(self):
        t=torch.zeros_like(self.target);p=t.clone();p[:,:,(10,11),1]=.1
        terms=self.terms(p,t)
        self.assertGreater(float(terms['support_height']),0)
        self.assertGreater(float(terms['support_envelope']),0)
        p[:,:,(10,11),1]=-.1
        self.assertGreater(float(self.terms(p,t)['support_penetration']),0)
    def test_raw_output_is_supervised_before_solver(self):
        rest=torch.zeros(1,22,3)
        for j in range(1,22):rest[:,j]=rest[:,PARENTS[j]]+torch.tensor([.02,-.1,.01])
        t=reconcile_motion(torch.zeros(1,48,201),rest)
        raw=t.clone();raw[:,:,3+4*6+1]=.1*torch.arange(48).sin();raw.requires_grad_()
        p=t.clone().requires_grad_();options=copy.deepcopy(DEFAULT)
        a=base_losses(p,t,{'rest':rest},torch.ones(1),raw,options)
        self.assertEqual(float(a['gait_total']),0)
        self.assertGreater(float(a['gait_raw_total']),0)
        a['gait_raw_total'].backward()
        self.assertGreater(float(raw.grad.norm()),0)
    def test_invalid_parameters_fail(self):
        for key,value in (('raw_gait_weight',2.),('stance_speed_m_s',0.),('signal_gate','none')):
            options=copy.deepcopy(DEFAULT);options[key]=value
            with self.assertRaises(ValueError):validate_config(options)


if __name__=='__main__':unittest.main()
