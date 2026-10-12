"""Physical-time, masking, mount and skeleton contracts; no training steps."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from experiments.offline_camera_retrain_v1.native30_observations import (
    observation_derivatives,sanitize_observations,inverse_head)
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow,rigid_surface_velocity
from experiments.offline_camera_retrain_v1.native30_losses import base_losses
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,PARENTS,IDENTITY_6D


class Native30ContractTests(unittest.TestCase):
    def test_physical_derivatives_and_gap(self):
        t=torch.arange(16,dtype=torch.float64)/30
        x=torch.zeros(1,16,1,9,dtype=torch.float64);x[0,:,0,0]=(.5*t.square())/2
        r=Rotation.from_rotvec(np.c_[np.zeros(16),t.numpy()*.3,np.zeros(16)]).as_matrix()
        x[0,:,0,3:]=torch.from_numpy(np.c_[r[:,:,0],r[:,:,1]])
        meta=torch.ones(1,16,1,5,dtype=torch.float64)
        v,a,w,m=observation_derivatives(x,meta)
        self.assertTrue(torch.allclose(a[:,2:,...,0],torch.ones_like(a[:,2:,...,0]),atol=1e-10))
        self.assertTrue(torch.allclose(w[:,1:,...,1],torch.full_like(w[:,1:,...,1],.3),atol=1e-9))
        meta[:,7,:,2:4]=0;x[:,7]=float('nan')
        v,a,w,m=observation_derivatives(x,meta)
        self.assertFalse(m[:,7:9,...,0].any());self.assertFalse(m[:,7:10,...,1].any())
        self.assertTrue(torch.isfinite(v).all() and torch.isfinite(a).all() and torch.isfinite(w).all())

    def test_zero_confidence_removes_poison(self):
        meta=torch.ones(1,8,22,5);meta[...,2:4]=0
        x=torch.randn(1,8,22,9)*1e6;x[:,2]=float('nan')
        safe,effective=sanitize_observations(x,meta)
        self.assertEqual(float(safe.abs().sum()),0.)
        self.assertEqual(float(effective[...,:4].abs().sum()),0.)

    def test_degenerate_rotation_is_unavailable(self):
        x=torch.zeros(1,8,1,9);meta=torch.ones(1,8,1,5)
        x[:,2,:,3:6]=1;x[:,2,:,6:]=2  # collinear columns cannot define SO(3)
        safe,effective=sanitize_observations(x,meta)
        self.assertFalse(effective[...,1].any())
        _,_,w,mask=observation_derivatives(x,meta)
        self.assertFalse(mask[...,2].any());self.assertEqual(float(w.abs().sum()),0.)

    def test_mount_inverse_and_missing_rotation(self):
        head=Rotation.from_rotvec([[.1,.2,.3],[.3,-.4,.2]]).as_matrix()
        mount_r=Rotation.from_rotvec([.03,-.01,.08]).as_matrix();offset=np.array([.01,.08,.1])
        p=np.array([[1,2,3],[2,3,4]],float);camera_r=head@mount_r
        camera_p=p+np.einsum('tij,j->ti',head,offset)
        camera=np.c_[camera_p/2,camera_r[:,:,0],camera_r[:,:,1]]
        mount=dict(rotation=mount_r.tolist(),translation_m=offset.tolist(),position_p90_m=.06,rotation_p90_rad=.2)
        result,mask,c=inverse_head(camera,np.ones((2,2)),np.ones((2,2)),mount)
        self.assertTrue(np.allclose(result[:,:3]*2,p,atol=1e-6))
        self.assertTrue(np.allclose(result[:,3:],np.c_[head[:,:,0],head[:,:,1]],atol=1e-6))
        available=np.ones((2,2));available[1,1]=0
        result,mask,c=inverse_head(camera,available,np.ones((2,2)),mount)
        self.assertFalse(mask[1].any());self.assertEqual(float(np.abs(result[1]).sum()),0.)

    def test_position_and_rotation_preserve_bones(self):
        motion=torch.zeros(1,8,201);rest=torch.randn(1,22,3)*.1
        baseline=reconcile_motion(motion,rest);delta=torch.zeros(1,8,3);delta[...,0]=.07
        moved=reconcile_motion(baseline,rest,delta)
        joints=forward_kinematics(moved,rest);original=forward_kinematics(baseline,rest)
        self.assertTrue(torch.allclose(joints-original,delta[:,:,None].expand_as(joints),atol=1e-6))
        rotation=torch.randn(1,8,22,3)*.1;rotated=reconcile_motion(baseline,rest,delta,rotation)
        joints=forward_kinematics(rotated,rest)
        self.assertTrue(torch.allclose(rotated[...,135:].reshape(1,8,22,3)*2,joints,atol=1e-6))
        for j in range(1,22):
            length=(joints[:,:,j]-joints[:,:,PARENTS[j]]).norm(dim=-1)
            target=(rest[:,j]-rest[:,PARENTS[j]]).norm(dim=-1)[:,None]
            self.assertTrue(torch.allclose(length,target.expand_as(length),atol=1e-6))

    def test_degenerate_predicted_six_keeps_a_proper_skeleton(self):
        motion=torch.zeros(1,8,201);motion[...,3:135]=-motion.new_tensor(IDENTITY_6D).repeat(22)
        rest=torch.randn(1,22,3)*.1;result=reconcile_motion(motion,rest)
        joints=forward_kinematics(result,rest)
        for j in range(1,22):
            self.assertTrue(torch.allclose((joints[:,:,j]-joints[:,:,PARENTS[j]]).norm(dim=-1),
                (rest[:,j]-rest[:,PARENTS[j]]).norm(dim=-1)[:,None].expand(1,8),atol=1e-6))

    def test_dynamic_current_owner_velocity_and_unknown(self):
        p=np.array([[x,0,z] for x in np.linspace(-.1,.1,12) for z in np.linspace(-.1,.1,12)],np.float32)
        points=np.stack((p,p+[.03,0,0],p+[.06,0,0]));owners=np.ones(points.shape[:2],np.int32)
        positions=np.array([[[0,0,0]],[[.03,0,0]],[[.06,0,0]]],np.float32)
        rotations=np.broadcast_to(np.eye(3),(3,1,3,3)).copy();valid=np.ones((3,1),bool)
        window=Native30SurfaceWindow(points,owners,np.tile([0,1,0],(3,1)),
            object_positions=positions,object_rotations=rotations,object_valid=valid)
        self.assertEqual(len(window.points),0)
        query=window.query(np.array([[[0,.01,0]],[[.03,.01,0]],[[3,2,1]]],np.float32),
            np.broadcast_to(np.eye(3),(3,1,3,3)))
        self.assertFalse(query['velocity_valid'][0].any());self.assertTrue(query['velocity_valid'][1].any())
        self.assertTrue(np.allclose(query['velocities'][1][query['valid'][1]], [.9,0,0],atol=1e-5))
        self.assertFalse(query['valid'][2].any())
        self.assertTrue(np.all(query['owners'][query['valid']]==1))
        with self.assertRaises(ValueError):
            Native30SurfaceWindow(points,np.full_like(owners,100),np.tile([0,1,0],(3,1)))
        self.assertTrue(np.allclose(rigid_surface_velocity([1,0,0],[0,0,0],[0,0,0],[0,1,0]),[0,0,-1]))

    def test_absolute_root_detects_constant_turn_offset(self):
        rest=torch.randn(1,22,3)*.1;truth=reconcile_motion(torch.zeros(1,16,201),rest)
        delta=torch.zeros(1,16,22,3);delta[:,:,0,1]=np.pi/2
        predicted=reconcile_motion(truth,rest,rotation_axis_angle=delta)
        losses=base_losses(predicted,truth,dict(rest=rest),torch.ones(1))
        self.assertGreater(float(losses['root_absolute_chordal']),1.)
        self.assertLess(float(losses['relative_turning_s2']),1e-8)

    def test_raw_six_objective_has_gradient_at_180_degree_error(self):
        rest=torch.randn(1,22,3)*.1
        raw=torch.zeros(1,16,201,requires_grad=True);prediction=reconcile_motion(raw,rest)
        delta=torch.zeros(1,16,22,3);delta[:,:,0,1]=np.pi
        truth=reconcile_motion(torch.zeros_like(raw),rest,rotation_axis_angle=delta).detach()
        loss=base_losses(prediction,truth,dict(rest=rest),torch.ones(1),raw)['raw_x0']
        gradient=torch.autograd.grad(loss,raw)[0]
        self.assertGreater(float(gradient[...,3:9].norm()),0.)


if __name__=='__main__':
    unittest.main()
