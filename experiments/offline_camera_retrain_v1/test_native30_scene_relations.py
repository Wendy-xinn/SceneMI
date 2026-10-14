"""Metric relation geometry, label isolation and safeguarded articulated guidance."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from experiments.offline_camera_retrain_v1.native30_geometry import Native30SurfaceWindow
from experiments.offline_camera_retrain_v1.native30_scene_relations import relation_query
from experiments.offline_camera_retrain_v1.native30_physical_guidance import guide_motion
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_observations import inference_payload,INPUT_KEYS
from experiments.offline_camera_retrain_v1.supervision import PARENTS,forward_kinematics


def fixture():
    rest=torch.tensor([[0,1,0],[-.1,.92,0],[.1,.92,0],[0,1.1,0],[-.1,.5,0],[.1,.5,0],
                       [0,1.25,0],[-.1,.04,0],[.1,.04,0],[0,1.4,0],[-.1,.04,.1],[.1,.04,.1],
                       [0,1.5,0],[-.08,1.45,0],[.08,1.45,0],[0,1.65,0],[-.2,1.45,0],
                       [.2,1.45,0],[-.5,1.4,0],[.5,1.4,0],[-.7,1.35,0],[.7,1.35,0]],dtype=torch.float32)
    labels=torch.cat((torch.arange(22).repeat_interleave(3),torch.tensor([15])))
    vertices=rest[labels].clone();vertices[torch.isin(labels,torch.tensor([7,8,10,11])),1]-=.04
    asset=dict(labels=labels,vertices=vertices,joints=rest,parents=torch.tensor(PARENTS),
               weights=torch.nn.functional.one_hot(labels,22).float(),posedirs=torch.zeros(21*9,67*3),
               extra_rotations=torch.empty(0,3,3),scale=1.)
    t=8;raw=torch.zeros(1,t,201);raw[...,:3]=torch.tensor([0.,-.02,0.])
    motion=reconcile_motion(raw,rest[None])
    grid=np.linspace(-.5,.5,31);points=np.asarray([[x,0,z] for x in grid for z in grid],np.float32)
    scene=Native30SurfaceWindow(np.broadcast_to(points,(t,*points.shape)),np.zeros((t,len(points)),np.int64),
                                np.broadcast_to([0.,2.,0.],(t,3)))
    batch=dict(rest=rest[None],skin_assets=[asset],surface_windows=[scene],mode=['ego'],
               trajectory=torch.zeros(1,t,22,9),observation_meta=torch.zeros(1,t,22,5),
               observation_uncertainty=torch.zeros(1,t,22,2))
    p=torch.zeros(1,t,22);p[:,:,(7,8,10,11)]=8.
    details={'contact_logits':p}
    config=json.loads((Path(__file__).parent/'native30_scene_relations_config.json').read_text())['physical_guidance']
    return motion,batch,details,config


class RelationTests(unittest.TestCase):
    def test_wide_upward_context_without_false_local_surface_completion(self):
        _,batch,_,_=fixture();scene=batch['surface_windows'][0]
        centres=np.broadcast_to([0.,1.65,0.],(8,22,3)).copy()
        result=relation_query(scene,centres,radius=.6)
        self.assertFalse(result['valid'][:,:,:4].any())
        self.assertTrue(result['valid'][:,:,4:].any())
        self.assertTrue(result['wide'][:,:,4:][result['valid'][:,:,4:]].all())
        self.assertTrue(np.all(result['points'][:,:,4:,1]==0.))

    def test_empty_scene_is_unknown(self):
        scene=Native30SurfaceWindow(np.zeros((8,0,3)),np.zeros((8,0),np.int64),np.zeros((8,3)))
        result=relation_query(scene,np.zeros((8,22,3)))
        self.assertFalse(result['valid'].any())

    def test_yaw_relation_equivariance(self):
        _,batch,_,_=fixture();scene=batch['surface_windows'][0]
        centres=np.broadcast_to([.08,1.,.06],(8,22,3)).copy()
        from experiments.offline_camera_retrain_v1.native30_overnight import YawSurface
        r=axis_angle_to_matrix(torch.tensor([0.,.63,0.])).numpy()
        a=relation_query(scene,centres);b=relation_query(YawSurface(scene,r),centres@r.T)
        self.assertTrue(np.array_equal(a['valid'],b['valid']))
        self.assertTrue(np.allclose(a['points']@r.T,b['points'],atol=1e-6))

    def test_guidance_reduces_energy_and_keeps_bones(self):
        motion,batch,details,config=fixture()
        with torch.no_grad():fixed,report=guide_motion(motion,torch.tensor([0]),batch,details,config,lambda m,t,b:m)
        self.assertLess(float(report['physical_energy_after'][0]),float(report['physical_energy_before'][0]))
        self.assertGreater(float(report['physical_accepted_steps'][0]),0)
        a=forward_kinematics(motion,batch['rest']);b=forward_kinematics(fixed,batch['rest'])
        for j in range(1,22):
            self.assertTrue(torch.allclose((a[:,:,j]-a[:,:,PARENTS[j]]).norm(dim=-1),
                                          (b[:,:,j]-b[:,:,PARENTS[j]]).norm(dim=-1),atol=1e-5))
        leaf=motion.detach().requires_grad_();fixed,_=guide_motion(leaf,torch.tensor([0]),batch,details,config,lambda m,t,b:m)
        fixed.square().mean().backward();self.assertTrue(torch.isfinite(leaf.grad).all())

    def test_guidance_protects_trusted_observations(self):
        motion,batch,details,config=fixture()
        batch['trajectory'][...,:3]=forward_kinematics(motion,batch['rest'])/2
        batch['observation_meta'][...,0]=1;batch['observation_meta'][...,2]=1
        fixed,_=guide_motion(motion,torch.tensor([0]),batch,details,config,lambda m,t,b:m)
        difference=(forward_kinematics(fixed,batch['rest'])-batch['trajectory'][...,:3]*2).norm(dim=-1)
        self.assertLessEqual(float(difference.max()),config['trusted_tolerance_m']+1e-6)

    def test_no_scene_evidence_does_not_move_body(self):
        motion,batch,details,config=fixture()
        batch['surface_windows']=[Native30SurfaceWindow(np.zeros((8,0,3)),np.zeros((8,0),np.int64),np.zeros((8,3)))]
        fixed,report=guide_motion(motion,torch.tensor([0]),batch,details,config,lambda m,t,b:m)
        self.assertTrue(torch.allclose(fixed,motion,atol=1e-6));self.assertEqual(float(report['physical_accepted_steps'].sum()),0.)

    def test_supervision_is_not_inference_input(self):
        batch={k:torch.zeros(1) for k in INPUT_KEYS}
        batch.update(surface_windows=[],skin_assets=[],mode=['ego'],contact_weight=torch.ones(1),
                     support_points=torch.full((1,),float('nan')),support_vertices=torch.full((1,),float('nan')))
        known=inference_payload(batch)
        self.assertNotIn('support_points',known);self.assertNotIn('support_vertices',known)
        self.assertNotIn('contact_weight',known)


if __name__=='__main__':unittest.main()
