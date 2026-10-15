"""Input isolation, exact zero-residual behavior and fixed-bone corrections."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from experiments.offline_camera_retrain_v1.legacy55k_exo_residual import (
    ExoResidual,frozen_initial_pose,residual_inputs,scene_features)
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics,PARENTS


class FakePrior(nn.Module):
    def forward(self,state,t,batch,**kwargs):
        assert not kwargs['use_control'] and not kwargs['use_scene']
        assert not {'motion','joints','contact_target'} & set(batch)
        assert torch.isfinite(batch['trajectory']).all()
        return .98*state


class ResidualTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        self.rest=torch.randn(1,22,3)*.1
        self.initial=reconcile_motion(torch.zeros(1,64,201),self.rest)
        p=forward_kinematics(self.initial,self.rest)
        self.batch=dict(rest=self.rest,trajectory=torch.zeros(1,64,22,9),
            observation_meta=torch.zeros(1,64,22,5),body_type=torch.zeros(1,2),
            body_scale=torch.ones(1,1),occupancy=torch.zeros(1,1),bps=torch.zeros(1,64,67,3),
            bps_valid=torch.zeros(1,64,67,dtype=torch.bool))
        self.batch['trajectory'][:,:,:4,:3]=p[:,:,:4]/2
        self.batch['observation_meta'][:,:,:4,0]=1
        self.batch['observation_meta'][:,:,:4,2]=1

    def test_hidden_and_label_poisoning_cannot_change_initial_or_features(self):
        a,_=frozen_initial_pose(FakePrior(),self.batch,27)
        known=residual_inputs(a,self.batch,torch.zeros(1,64,528))
        self.batch.update(motion=torch.full_like(a,float('nan')),joints=torch.randn(1,64,22,3),
                          contact_target=torch.randn(1,64,22))
        self.batch['trajectory'][:,:,4:]=float('nan')
        b,_=frozen_initial_pose(FakePrior(),self.batch,27)
        other=residual_inputs(b,self.batch,torch.zeros(1,64,528))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(known['features'],other['features'],rtol=0,atol=0)

    def test_zero_residual_preserves_initial_pose(self):
        model=ExoResidual(width=32)
        known=residual_inputs(self.initial,self.batch,torch.randn(1,64,528))
        p,_=model(known)
        torch.testing.assert_close(p,self.initial,atol=1e-7,rtol=1e-6)
        known['motion']=self.initial
        with self.assertRaises(ValueError):model(known)

    def test_scene_off_is_invariant_and_corrections_preserve_bones(self):
        model=ExoResidual(width=32)
        nn.init.normal_(model.output.weight,std=.01)
        known=residual_inputs(self.initial,self.batch,torch.randn(1,64,528))
        a,_=model(known,use_scene=False)
        known['scene']=torch.randn_like(known['scene'])*100
        b,_=model(known,use_scene=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        p=forward_kinematics(a,self.rest)
        for j in range(1,22):
            expected=(self.rest[:,j]-self.rest[:,PARENTS[j]]).norm(dim=-1)[:,None].expand(1,64)
            torch.testing.assert_close((p[:,:,j]-p[:,:,PARENTS[j]]).norm(dim=-1),expected,atol=1e-6,rtol=1e-5)
        a.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_unknown_scene_has_zero_features_and_target_owner_rejected(self):
        empty=dict(points=np.empty((0,3)),owners=np.empty(0,dtype=int))
        scene=SimpleNamespace(static=empty,dynamic=[empty]*64)
        f=scene_features(scene,np.zeros((64,22,3)))
        self.assertEqual(f.shape,(64,528));self.assertEqual(float(abs(f).sum()),0)
        scene.dynamic[0]=dict(points=np.zeros((1,3)),owners=np.array([100]))
        with self.assertRaises(ValueError):scene_features(scene,np.zeros((64,22,3)))

    def test_cached_yaw_surface_rotates_geometry_without_changing_masks(self):
        cloud=dict(points=np.array([[.3,0,.2]],np.float32),owners=np.array([0]),
                   normals=np.array([[0,1,0]],np.float32),reliable=np.array([True]))
        empty=dict(points=np.empty((0,3)),owners=np.empty(0,dtype=int))
        base=SimpleNamespace(static=cloud,dynamic=[empty]*64,_static_tree=cKDTree(cloud['points']))
        rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]],np.float32)
        wrapped=SimpleNamespace(base=base,rotation=rotation)
        points=np.ones((64,22,3),np.float32)*.1
        original=scene_features(base,points).reshape(64,22,2,12)
        transformed=scene_features(wrapped,points@rotation.T).reshape(64,22,2,12)
        np.testing.assert_allclose(transformed[...,:3],original[...,:3]@rotation.T,atol=1e-7)
        np.testing.assert_allclose(transformed[...,3:6],original[...,3:6]@rotation.T,atol=1e-7)
        np.testing.assert_allclose(transformed[...,6:],original[...,6:],atol=1e-7)


if __name__=='__main__':unittest.main()
