"""Geometry semantics and internal residual injection, without model assets."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from scenemi.geometry.scene_features import measured_free_space, relation_features, FEATURE_NAMES
from scenemi.scene_adapter import RelationAdapter, inject_residuals, resize_temporal


POLICY = dict(neighbours=16, query_radius_m=.6, plane_radius_m=.15, minimum_points=6,
    maximum_planarity_ratio=.03, minimum_spread_m=.005, reliability_distance_m=.3,
    ray_tube_m=.015, ray_endpoint_margin_m=.05, ray_minimum_depth_m=.1)


class SceneFeatureTests(unittest.TestCase):
    def test_free_ray_stops_before_the_measured_surface_and_not_beside_it(self):
        query = torch.tensor([[[0.,0.,1.],[0.,0.,2.],[0.,0.,3.],[.1,0.,1.]]])
        p = torch.tensor([[[0.,0.,2.]]])
        args = (query, torch.zeros(1,3), p, torch.zeros(1,1,dtype=torch.long), torch.ones(1,dtype=torch.bool))
        expected = torch.tensor([[True,False,False,False]])
        result = measured_free_space(*args, tube_m=.015, endpoint_margin_m=.05, minimum_depth_m=.1)
        self.assertTrue(torch.equal(result, expected))

    def test_invalid_first_hits_and_missing_camera_do_not_prove_free_space(self):
        anchors = torch.tensor([[[0.,0.,1.]]])
        points = torch.tensor([[[0.,0.,2.],[float('nan'),0.,2.]]])
        owners = torch.tensor([[-1,0]])
        result = measured_free_space(anchors,torch.zeros(1,3),points,owners,torch.ones(1,dtype=torch.bool),
            tube_m=.015,endpoint_margin_m=.05,minimum_depth_m=.1)
        self.assertFalse(result.any())

    def scene(self, valid=True):
        x,z = torch.meshgrid(torch.linspace(-.04,.04,4),torch.linspace(-.04,.04,4),indexing='ij')
        points = torch.stack((x.flatten(),torch.zeros(16),z.flatten()),-1)
        camera = torch.zeros(1,1,9)
        camera[...,1] = .5  # Optical position is normalized by 2.
        return dict(local_history_points=points[None], local_history_valid=torch.full((1,16),valid),
            local_frame_points=torch.zeros(1,1,1,3),local_frame_owner=torch.full((1,1,1),-1),
            camera=camera,camera_available=torch.ones(1,1,2),camera_confidence=torch.ones(1,1,2))

    def test_real_plane_normal_offset_and_unknown_mask_enter_forward_features(self):
        anchors = torch.tensor([[[[0.,.08,0.]]]])
        rotations = torch.eye(3).reshape(1,1,1,3,3)
        f,gate,diag = relation_features(anchors,rotations,self.scene(),POLICY)
        self.assertEqual(f.shape[-1],len(FEATURE_NAMES))
        self.assertGreater(float(f[...,5]),.99)
        self.assertAlmostEqual(float(f[...,7]),.08/.6,places=5)
        self.assertEqual(float(f[...,8]),1.)
        self.assertGreater(float(gate),0.)
        f,gate,diag = relation_features(anchors,rotations,self.scene(False),POLICY)
        self.assertEqual(float(f[...,13]),1.)
        self.assertEqual(float(f[...,11]),0.)
        self.assertEqual(float(gate),0.)

    def test_measured_free_space_is_distinct_from_unknown_without_nearby_surface(self):
        scene=self.scene(False)
        scene['camera'].zero_()
        scene['local_frame_points']=torch.tensor([[[[0.,0.,2.]]]])
        scene['local_frame_owner'].zero_()
        q=torch.tensor([[[[0.,0.,1.]]]])
        f,gate,_=relation_features(q,torch.eye(3).reshape(1,1,1,3,3),scene,POLICY)
        self.assertEqual(float(f[...,11]),1.)
        self.assertEqual(float(f[...,13]),0.)
        self.assertEqual(float(gate),1.)


class InjectionTests(unittest.TestCase):
    def test_temporal_reduction_matches_linear_values_and_gradients(self):
        from torch.nn import functional as F
        for length in (128, 64, 32):
            value=torch.randn(2,3,128,requires_grad=True)
            actual=resize_temporal(value,length)
            expected=F.interpolate(value,size=length,mode='linear',align_corners=False)
            self.assertTrue(torch.equal(actual,expected))
            g=torch.randn_like(actual)
            gradient=torch.autograd.grad(actual,value,g,retain_graph=True)[0]
            reference=torch.autograd.grad(expected,value,g)[0]
            self.assertTrue(torch.equal(gradient,reference))

    def test_zero_initialization_and_gradients_at_every_resolution(self):
        adapter=RelationAdapter([16,8,4],hidden=8)
        body=torch.randn(1,8,67,12)
        geo=torch.randn(1,8,67,len(FEATURE_NAMES))
        labels=(torch.arange(67)%22)[None]
        output=adapter(body,geo,torch.ones(1,8,67),labels,torch.ones(1))
        self.assertTrue(all(torch.count_nonzero(v)==0 for v in output))
        sum(v.sum() for v in output).backward()
        self.assertTrue(all(layer.weight.grad.norm()>0 for layer in adapter.projections))

    def test_scoped_hooks_affect_backbone_and_are_removed_on_failure(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.mid_block2=nn.Identity()
                self.ups=nn.ModuleList([nn.ModuleList([nn.Identity(),nn.Identity()])])
                self.final_conv=nn.Sequential(nn.Identity())
        model=Toy()
        x=torch.zeros(1,4,8)
        with self.assertRaises(RuntimeError):
            with inject_residuals(model,[torch.ones_like(x),torch.ones_like(x)],torch.ones(1,8)):
                self.assertTrue(torch.equal(model.mid_block2(x),torch.ones_like(x)))
                raise RuntimeError('interrupted forward')
        self.assertTrue(torch.equal(model.mid_block2(x),x))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_unknown_geometry_cannot_inject_bias(self):
        model=SimpleNamespace(mid_block2=nn.Identity(),ups=[],final_conv=nn.Sequential(nn.Identity()))
        x=torch.randn(1,4,8)
        with inject_residuals(model,[torch.ones_like(x)*100],torch.zeros(1,8)):
            self.assertTrue(torch.equal(model.mid_block2(x),x))


if __name__=='__main__':
    unittest.main()
