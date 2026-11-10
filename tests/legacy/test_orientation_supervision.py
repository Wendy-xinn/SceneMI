import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.offline_camera_retrain_v1.orientation_supervision import orientation_losses

def motion(angles):
 m=torch.zeros(1,len(angles),201)
 r=Rotation.from_euler('y',angles).as_matrix();six=np.concatenate((r[:,:,0],r[:,:,1]),axis=-1)-np.array([1,0,0,0,1,0])
 m[0,:,3:9]=torch.tensor(six,dtype=torch.float32)
 return m
class OrientationTests(unittest.TestCase):
 def test_identical_and_weight_mask(self):
  t=motion(np.linspace(0,1,16));self.assertLess(float(orientation_losses(t,t)['orientation_total']),1e-9)
  self.assertEqual(float(orientation_losses(motion(-np.linspace(0,1,16)),t,torch.zeros(1))['orientation_total']),0)
 def test_signed_direction_and_gradient(self):
  t=motion(np.linspace(-1,1,16));p=motion(np.linspace(1,-1,16)).requires_grad_();loss=orientation_losses(p,t);self.assertGreater(float(loss['signed_turn_rate_loss'].detach()),1);loss['orientation_total'].backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
 def test_accumulated_turn_and_gradient(self):
  t=motion(np.zeros(64));p=motion(np.linspace(0,2*np.pi,64)).requires_grad_();values=orientation_losses(p,t,accumulated=True);self.assertGreater(float(values['accumulated_heading_loss'].detach()),10);values['orientation_total'].backward();self.assertTrue(torch.isfinite(p.grad).all())
 def test_duration_normalization(self):
  values=[]
  for n in (64,128,192):
   t=motion(np.zeros(n));p=motion(np.arange(n)/20*np.deg2rad(10)).requires_grad_()
   raw=orientation_losses(p,t,accumulated=True);normalized=orientation_losses(p,t,accumulated=True,normalize_duration=True)
   if n==128:self.assertTrue(torch.equal(raw["orientation_total"],normalized["orientation_total"]))
   values.append(float(normalized["accumulated_heading_loss"].detach()))
   normalized["orientation_total"].backward();self.assertTrue(torch.isfinite(p.grad).all())
  self.assertLess(max(values)-min(values),1e-5)
 def test_noise_gate_isolates_temporal_terms(self):
  target=motion(np.zeros(64));pred=motion(np.linspace(0,1,64))
  raw=orientation_losses(pred,target,torch.tensor([.5]),accumulated=True,normalize_duration=True)
  gated=orientation_losses(pred,target,torch.tensor([.5]),accumulated=True,normalize_duration=True,gate_turn_noise=True)
  self.assertTrue(torch.equal(raw['global_head_orientation_loss'],gated['global_head_orientation_loss']))
  for k in ('signed_turn_rate_loss','accumulated_heading_loss'):
   self.assertTrue(torch.allclose(gated[k],raw[k]*.5))
  full=orientation_losses(pred,target,torch.ones(1),accumulated=True,normalize_duration=True,gate_turn_noise=True)
  baseline=orientation_losses(pred,target,torch.ones(1),accumulated=True,normalize_duration=True)
  self.assertTrue(torch.equal(full['orientation_total'],baseline['orientation_total']))
 def test_neck_parent_gradient_and_zero_signal(self):
  target=motion(np.zeros(16));pred=target.clone()
  # Alter joint12 (neck), leaving root and head-local orientation unchanged.
  pred[:,:,75:81]=motion(np.full(16,.5))[:,:,3:9]
  pred.requires_grad_()
  values=orientation_losses(pred,target,torch.tensor([.5]),supervise_neck=True,gate_turn_noise=True)
  values['global_neck_parent_orientation_loss'].backward()
  self.assertTrue(torch.isfinite(pred.grad).all())
  self.assertGreater(float(pred.grad[:,:,75:81].abs().sum()),0)
  self.assertEqual(float(pred.grad[:,:,93:99].abs().sum()),0)
  zero=orientation_losses(pred,target,torch.zeros(1),accumulated=True,normalize_duration=True,gate_turn_noise=True,supervise_neck=True)
  self.assertEqual(float(zero['orientation_total'].detach()),0)
 def test_angle_wrap(self):
  a=np.array([3.0,3.1,-3.08,-2.98]);t=motion(a);same=motion(a+2*np.pi);self.assertLess(float(orientation_losses(same,t)['orientation_total']),1e-9)
if __name__=='__main__':unittest.main()
