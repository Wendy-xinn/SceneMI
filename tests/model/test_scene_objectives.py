import numpy as np
import torch

from scenemi.geometry.scene_objectives import geometry_terms


class Body:
    def skin_points(self,motion,details,formats):
        labels=torch.cat((torch.arange(22).repeat_interleave(3),torch.tensor([15])))
        return motion.reshape(*motion.shape[:2],67,3),labels[None].expand(len(motion),-1)


def fixture():
    t=8
    target=torch.zeros(1,t,201)
    x,z=torch.meshgrid(torch.linspace(-.1,.1,5),torch.linspace(-.1,.1,5),indexing='ij')
    points=torch.stack((x.flatten(),torch.zeros(x.numel()),z.flatten()),-1)[None]
    normals=torch.zeros_like(points);normals[...,1]=1
    batch=dict(motion=target,target_rest=torch.zeros(1,22,3),target_native_betas=torch.zeros(1,10),
        target_template=torch.zeros(1,dtype=torch.long),body_format=torch.ones(1,dtype=torch.long),
        static_points=points,static_normals=normals,static_valid=torch.ones(points.shape[:2],dtype=torch.bool),
        static_normal_valid=torch.ones(points.shape[:2],dtype=torch.bool),frame_scene_points=torch.zeros(1,t,1,3),
        frame_scene_owner=torch.full((1,t,1),-1),frame_normals=torch.zeros(1,t,1,3),
        frame_normal_valid=torch.zeros(1,t,1,dtype=torch.bool),contact_weight=torch.ones(1,t,2))
    policy=dict(loss_frame_stride=1,query_radius_m=.2,source_penetration_tolerance_m=.02,
        source_contact_tolerance_m=.04,penetration_tolerance_m=.01,contact_tolerance_m=.02,
        distance_scale_m=.02,velocity_scale_mps=.1,stance_speed_mps=.2,support_normal_y=.8,
        sliding_tolerance_mps=.02)
    return batch,policy


def test_floor_crossing_pushes_predicted_skin_up():
    batch,policy=fixture();pred=batch['motion'].clone().reshape(1,8,67,3)
    pred[...,1]=-.04;pred=pred.reshape(1,8,201).requires_grad_()
    terms,stats=geometry_terms(Body(),pred,{},batch,torch.ones(1),policy)
    assert terms['scene_nonpenetration']>0
    terms['scene_nonpenetration'].backward()
    assert torch.all(pred.grad.reshape(1,8,67,3)[...,1]<0)
    assert stats['observed_surface_penetration_mm'].item()>29.9


def test_unknown_or_unoriented_geometry_has_zero_loss_and_gradient():
    for unknown in [True,False]:
        batch,policy=fixture()
        batch['static_valid' if unknown else 'static_normal_valid'].zero_()
        pred=torch.full_like(batch['motion'],-.04,requires_grad=True)
        terms,stats=geometry_terms(Body(),pred,{},batch,torch.ones(1),policy)
        assert all(v.item()==0 for v in terms.values())
        sum(terms.values()).backward();assert torch.count_nonzero(pred.grad)==0
        assert stats['observed_surface_coverage'].item()==0


def test_source_penetration_is_not_supervised_as_a_valid_target():
    batch,policy=fixture();batch['motion'].reshape(1,8,67,3)[...,1]=-.06
    pred=batch['motion'].clone().requires_grad_()
    terms,stats=geometry_terms(Body(),pred,{},batch,torch.ones(1),policy)
    assert all(v.item()==0 for v in terms.values())
    assert stats['observed_surface_coverage'].item()==0


def test_target_far_in_front_does_not_mask_a_predicted_crossing():
    batch,policy=fixture();batch['motion'].reshape(1,8,67,3)[...,1]=.6
    pred=batch['motion'].clone().reshape(1,8,67,3)
    pred[...,1]=-.04;pred=pred.reshape(1,8,201).requires_grad_()
    terms,_=geometry_terms(Body(),pred,{},batch,torch.ones(1),policy)
    assert terms['scene_nonpenetration']>0


def test_static_stance_sliding_is_penalized_but_swing_is_not_planted():
    batch,policy=fixture();pred=batch['motion'].clone().reshape(1,8,67,3)
    pred[...,0]=torch.arange(8)[None,:,None]*.01;pred=pred.reshape(1,8,201).requires_grad_()
    terms,_=geometry_terms(Body(),pred,{},batch,torch.ones(1),policy)
    assert terms['scene_sliding']>0
    batch['motion']=pred.detach().clone()
    terms,_=geometry_terms(Body(),pred,{},batch,torch.ones(1),policy)
    assert terms['scene_support']==0 and terms['scene_sliding']==0
