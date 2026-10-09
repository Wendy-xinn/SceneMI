import unittest
import torch
from experiments.offline_camera_retrain_v1.head_constraint import project_head_orientation
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, IDENTITY_6D

class HeadConstraintTests(unittest.TestCase):
    def test_exact_rotation_preserves_fk_and_unobserved(self):
        torch.manual_seed(41)
        motion=torch.randn(2,64,201)*.1;camera=torch.zeros(2,64,9)
        camera[...,3:]=camera.new_tensor(IDENTITY_6D)
        mask=torch.zeros(2,64,dtype=torch.bool);mask[0]=True
        projected=project_head_orientation(motion,camera,mask)
        self.assertTrue(torch.equal(projected[1],motion[1]))
        self.assertTrue(torch.allclose(global_rotations(projected)[0,:,15],torch.eye(3).expand(64,3,3),atol=1e-5))
        rest=torch.randn(2,22,3)*.1
        self.assertTrue(torch.equal(forward_kinematics(projected,rest),forward_kinematics(motion,rest)))
    def test_parent_gradient_and_control_mask(self):
        motion=(torch.randn(1,64,201)*.1).requires_grad_();camera=torch.zeros(1,64,9);camera[...,3:]=camera.new_tensor(IDENTITY_6D)
        enabled=torch.ones(1,64,dtype=torch.bool)
        projected=project_head_orientation(motion,camera,enabled)
        projected[...,93:99].square().mean().backward()
        self.assertTrue(torch.isfinite(motion.grad).all())
        self.assertGreater(float(motion.grad[...,3:9].abs().sum()),0)
        self.assertEqual(float(motion.grad[...,93:99].abs().sum()),0)
    def test_nonidentity_mount(self):
        motion=torch.zeros(1,64,201);camera=torch.zeros(1,64,9);camera[...,3:]=camera.new_tensor(IDENTITY_6D)
        mount=torch.diag(torch.tensor([-1.,1.,-1.]))
        result=project_head_orientation(motion,camera,torch.ones(1,64,dtype=torch.bool),mount)
        self.assertTrue(torch.allclose(global_rotations(result)[:,:,15],mount.expand(1,64,3,3),atol=1e-5))

if __name__=='__main__':unittest.main()
