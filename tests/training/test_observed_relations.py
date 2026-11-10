import unittest

import torch

from scenemi.geometry.observed_relations import local_planes, relation_loss, masked_bce


class ObservedRelationTests(unittest.TestCase):
    def plane(self):
        x,z=torch.meshgrid(torch.linspace(-.04,.04,3),torch.linspace(-.04,.04,3),indexing='ij')
        p=torch.stack((x.flatten(),torch.zeros(9),z.flatten()),-1)[None,None]
        return p,torch.zeros(1,1,9,dtype=torch.long),torch.ones(1,1,9,dtype=torch.bool)

    def test_plane_rejects_line_and_mixed_owner_neighbours(self):
        p,o,v=self.plane();self.assertTrue(local_planes(p,o,v)['valid'].all())
        p[...,2]=0;self.assertFalse(local_planes(p,o,v)['valid'].any())
        p,o,v=self.plane();o[...,3:]=1
        self.assertFalse(local_planes(p,o,v)['valid'].any())

    def test_offset_supervision_preserves_target_clearance_and_tangential_freedom(self):
        p,o,v=self.plane();planes=local_planes(p,o,v)
        target=torch.tensor([[[0.,.08,0.]]]);pred=target.clone().requires_grad_()
        self.assertEqual(float(relation_loss(pred,target,planes)),0.)
        shifted=pred+torch.tensor([.02,0.,.02])
        self.assertEqual(float(relation_loss(shifted,target,planes)),0.)
        loss=relation_loss(pred+torch.tensor([0.,.02,0.]),target,planes)
        loss.backward();self.assertGreater(float(pred.grad[...,1].sum()),0.)
        self.assertEqual(float(pred.grad[...,[0,2]].abs().sum()),0.)

    def test_unknown_geometry_has_zero_loss_and_gradient(self):
        p,o,v=self.plane();planes=local_planes(p,o,torch.zeros_like(v))
        pred=torch.ones(1,1,3,requires_grad=True)
        loss=relation_loss(pred,torch.zeros_like(pred),planes);loss.backward()
        self.assertEqual(float(loss),0.);self.assertEqual(float(pred.grad.abs().sum()),0.)
        logits=torch.ones(2,requires_grad=True)
        loss=masked_bce(logits,torch.zeros(2),torch.zeros(2,dtype=torch.bool));loss.backward()
        self.assertEqual(float(loss),0.);self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_plane_loss_does_not_depend_on_normal_sign_or_world_rotation(self):
        p,o,v=self.plane();plane=local_planes(p,o,v)
        pred=torch.tensor([[[.01,.02,.03]]]);target=torch.zeros_like(pred)
        original=relation_loss(pred,target,plane)
        flipped=dict(plane,normal=-plane['normal'])
        self.assertTrue(torch.equal(original,relation_loss(pred,target,flipped)))
        r=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rotated=dict(plane,normal=plane['normal']@r.T)
        self.assertTrue(torch.allclose(original,relation_loss(pred@r.T,target@r.T,rotated)))


if __name__=='__main__':unittest.main()
