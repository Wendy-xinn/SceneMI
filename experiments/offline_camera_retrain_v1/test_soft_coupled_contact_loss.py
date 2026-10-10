import unittest
import torch
from experiments.offline_camera_retrain_v1.typed_head_condition import prepare_observation
from experiments.offline_camera_retrain_v1.test_typed_head_condition import TypedObservationTests
from experiments.offline_camera_retrain_v1.soft_coupled_contact_loss import soft_coupled_losses,contact_geometry_proxy_losses
from experiments.offline_camera_retrain_v1.control import fixed_control_mask

class LossTests(unittest.TestCase):
 def batch(self):
  batch=TypedObservationTests().batch();batch=prepare_observation(batch,'joint')
  batch['contact_target']=torch.zeros(2,64,22);batch['contact_valid']=torch.zeros(2,64,22,dtype=torch.bool)
  return batch
 def test_soft_availability_confidence_and_high_noise(self):
  b=self.batch();p=b['motion'].clone();p[:,:,95]=.2;p.requires_grad_()
  mask=fixed_control_mask(2,64,'head','cpu');zero=torch.zeros(2)
  full=soft_coupled_losses(p,b['motion'],b,mask,zero)
  self.assertGreater(float(full['soft_observed_head_orientation']),0)
  self.assertEqual(float(full['head_body_relative_orientation']),0)
  b['observation_meta'][:,:,15,3]=.1
  low=soft_coupled_losses(p,b['motion'],b,mask,zero)
  self.assertTrue(torch.allclose(low['soft_observed_head_orientation'],full['soft_observed_head_orientation']*.1))
  b['observation_meta'][:,:,15,1]=0
  absent=soft_coupled_losses(p,b['motion'],b,mask,zero);absent['soft_coupled_total'].backward()
  self.assertEqual(float(absent['soft_coupled_total']),0);self.assertEqual(float(p.grad.abs().max()),0)
 def test_valid_head_torso_difference_not_forced_to_identity(self):
  b=self.batch();b['motion'][:,:,95]=.3;b=prepare_observation(b,'joint')
  values=soft_coupled_losses(b['motion'],b['motion'],b,fixed_control_mask(2,64,'head','cpu'),torch.ones(2))
  self.assertAlmostEqual(float(values['soft_coupled_total']),0,places=6)
 def test_contact_missing_labels_and_identical_reference(self):
  b=self.batch();p=b['motion'].clone().requires_grad_();p2=p+1
  v=contact_geometry_proxy_losses(p2,b['motion'],b,torch.ones(2));v['contact_geometry_proxy_total'].backward()
  self.assertEqual(float(v['contact_geometry_proxy_total']),0);self.assertEqual(float(p.grad.abs().max()),0)
  b['contact_target'][:,:,10]=1;b['contact_valid'][:,:,10]=True
  b['motion'][:,:,0]=torch.linspace(0,1,64)
  v=contact_geometry_proxy_losses(b['motion'],b['motion'],b,torch.ones(2))
  self.assertEqual(float(v['contact_geometry_proxy_total']),0)
 def test_contact_height_bound_and_signal_gate(self):
  b=self.batch();b['contact_target'][:,:,10]=1;b['contact_valid'][:,:,10]=True
  p=b['motion'].clone();p[:,:,1]=-.1
  v=contact_geometry_proxy_losses(p,b['motion'],b,torch.ones(2))
  self.assertGreater(float(v['labeled_foot_below_reference_m2']),0)
  self.assertEqual(float(contact_geometry_proxy_losses(p,b['motion'],b,torch.zeros(2))['contact_geometry_proxy_total']),0)
  b['contact_valid'].zero_()
  self.assertEqual(float(contact_geometry_proxy_losses(p,b['motion'],b,torch.ones(2))['contact_geometry_proxy_total']),0)

if __name__=='__main__':unittest.main()
