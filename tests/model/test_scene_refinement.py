"""Structural identities, coordinate conventions and differentiable readout."""
import copy
import math

import pytest
import torch
from torch import nn

from scenemi.body_diagnostic_metrics import motion_from_rotations
from scenemi.geometry.temporal_head_anchor import TemporalHeadAnchor
from scenemi.refinement.model import (ANCHOR_JOINTS, SceneRefinementPipeline,
    apply_rotations, rotation_increment)
from scenemi.refinement.losses import objective, contact_targets, masked_mean
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def policies():
    refiner = dict(width=24, heads=4, blocks=2, root_yaw_limit_deg=5., torso_limit_deg=8., lower_limit_deg=15.)
    anchor = dict(data_loss='l2', data_weight=1., acceleration_weight=.03**4,
        jerk_weight=.03**6, huber_delta_m=.02, irls_iterations=5, missing_ridge=1e-8)
    geometry = dict(loss_frame_stride=2, query_radius_m=.2, source_penetration_tolerance_m=.02,
        source_contact_tolerance_m=.04, penetration_tolerance_m=.01, contact_tolerance_m=.02,
        distance_scale_m=.02, velocity_scale_mps=.1, stance_speed_mps=.2, support_normal_y=.8,
        sliding_tolerance_mps=.02)
    loss = dict(position_scale_m=.5, velocity_scale_mps=1., head_position_scale_m=.05,
        contact=dict(positive_gap_m=.03, negative_gap_m=.06, positive_speed_mps=.2,
                     negative_speed_mps=.3, support_normal_y=.8, positive_class_weight=2.),
        weights={k:1. for k in ('pose_rotation','fk_position','lower_fk','lower_velocity',
            'head_position','head_orientation','contact','residual','scene_nonpenetration','scene_support','scene_sliding')})
    return refiner, anchor, geometry, loss


class SparseSkin(nn.Module):
    def forward(self, motion, shape, formats):
        labels = torch.tensor(ANCHOR_JOINTS, device=motion.device)[None].expand(len(motion), -1)
        points = forward_kinematics(motion, shape['rest'])[:, :, labels[0]]
        offset = torch.zeros_like(points)
        foot = ((labels==10)|(labels==11))[:, None]
        ankle = ((labels==7)|(labels==8))[:, None]
        offset[..., 1] = -.02*foot-.1*ankle
        return points+offset, labels


def fixture(batch_size=2, frames=8):
    torch.manual_seed(5)
    rp, ap, gp, lp = policies()
    pipeline = SceneRefinementPipeline(nn.Linear(1, 1), {}, gp, ap, rp).double()
    pipeline.skin = SparseSkin()
    pipeline.anchor = TemporalHeadAnchor(ap, frames=frames)
    rest = torch.tensor([[0,1,0],[-.1,.95,0],[.1,.95,0],[0,1.1,0],[-.1,.55,0],
        [.1,.55,0],[0,1.2,0],[-.1,.1,0],[.1,.1,0],[0,1.3,0],[-.1,.02,.1],[.1,.02,.1],
        [0,1.4,0],[-.1,1.4,0],[.1,1.4,0],[0,1.6,0],[-.2,1.4,0],[.2,1.4,0],
        [-.4,1.4,0],[.4,1.4,0],[-.6,1.4,0],[.6,1.4,0]], dtype=torch.float64)[None].repeat(batch_size,1,1)
    local = torch.eye(3, dtype=torch.float64).expand(batch_size, frames, 22, 3, 3).clone()
    root = rest[:, None, 0].expand(-1, frames, -1).clone()
    raw = motion_from_rotations(local, root, rest)
    mount = torch.tensor([[.02,.04,.11]], dtype=torch.float64).repeat(batch_size,1)
    camera_position = forward_kinematics(raw, rest)[:, :, 15]+mount[:, None]
    camera_position[:, :, 0] += torch.sin(torch.arange(frames, dtype=torch.float64)*1.7)*.02
    camera = torch.cat((camera_position/2, local[:, :, 15, :, 0], local[:, :, 15, :, 1]), -1)
    available = torch.ones(batch_size, frames, 2, dtype=torch.float64)
    details = dict(rest=rest, head_camera_translation=mount,
        predicted_native_betas=torch.zeros(batch_size,10,dtype=torch.float64),
        predicted_template=torch.zeros(batch_size,dtype=torch.long),
        observations=dict(camera=camera, camera_available=available))
    x,z = torch.meshgrid(torch.linspace(-1,1,21,dtype=torch.float64),torch.linspace(-1,1,21,dtype=torch.float64),indexing='ij')
    points = torch.stack((x.flatten(),torch.zeros(x.numel(),dtype=torch.float64),z.flatten()), -1)[None].repeat(batch_size,1,1)
    normals = torch.zeros_like(points); normals[...,1]=1
    public = dict(camera=camera, camera_available=available, body_format=torch.ones(batch_size,dtype=torch.long),
        static_points=points, static_normals=normals, static_valid=torch.ones(points.shape[:2],dtype=torch.bool),
        static_normal_valid=torch.ones(points.shape[:2],dtype=torch.bool),
        frame_scene_points=torch.zeros(batch_size,frames,1,3,dtype=torch.float64),
        frame_scene_owner=torch.full((batch_size,frames,1),-1),
        frame_normals=torch.zeros(batch_size,frames,1,3,dtype=torch.float64),
        frame_normal_valid=torch.zeros(batch_size,frames,1,dtype=torch.bool))
    batch = dict(public, motion=raw.clone(), target_rest=rest.clone(),
        target_native_betas=details['predicted_native_betas'].clone(),
        target_template=details['predicted_template'].clone(),
        contact_weight=torch.ones(batch_size,frames,2,dtype=torch.float64))
    return pipeline, raw, details, public, batch, lp


def test_rodrigues_zero_and_nonzero_match_matrix_exp_and_have_gradients():
    omega = torch.tensor([[0.,0.,0.],[.1,-.2,.3]], dtype=torch.float64, requires_grad=True)
    x,y,z = omega.unbind(-1); zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    torch.testing.assert_close(rotation_increment(omega)+torch.eye(3),torch.matrix_exp(skew),atol=1e-12,rtol=1e-12)
    assert torch.count_nonzero(rotation_increment(omega[:1]))==0
    assert torch.autograd.gradcheck(rotation_increment,(omega,),atol=1e-7,rtol=1e-5)


def test_zero_initialization_is_exactly_one_B_not_two():
    model, raw, details, public, _, _ = fixture()
    baseline = model.locate(raw,details)
    assert not torch.equal(model.locate(baseline,details),baseline)
    final, extra = model.refine(raw,details,public)
    assert extra['anchor_hit'].any()
    assert torch.count_nonzero(extra['rotation_residual'])==0
    assert torch.equal(final,baseline)


def test_disabled_unknown_and_camera_only_inputs_restore_baseline_after_learning():
    model, raw, details, public, _, _ = fixture()
    nn.init.normal_(model.refiner.rotation.weight,std=.1)
    baseline = model.locate(raw,details)
    camera_only = {k:v for k,v in public.items() if k in ('camera','camera_available','body_format')}
    for inputs, use_scene in ((public,False),(dict(public,static_valid=torch.zeros_like(public['static_valid'])),True),(camera_only,True)):
        final, extra = model.refine(raw,details,inputs,use_scene=use_scene)
        assert torch.equal(final,baseline)
        assert torch.count_nonzero(extra['rotation_residual'])==0
    final, extra = model.refine(raw,details,public,use_scene=torch.tensor([True,False]))
    assert torch.equal(final[1],baseline[1]) and extra['scene_active'].tolist()==[True,False]


def test_residual_is_bounded_and_preserves_unselected_rotations_and_bones():
    model,raw,details,public,_,_=fixture()
    nn.init.normal_(model.refiner.rotation.weight,std=10)
    final,extra=model.refine(raw,details,public)
    assert (extra['rotation_residual'].norm(dim=-1)<=model.refiner.limits[None,None]+1e-12).all()
    assert torch.count_nonzero(extra['rotation_residual'][:,:,0,[0,2]])==0
    unchanged=(12,13,14,15,16,17,18,19,20,21)
    six=lambda m:m[...,3:135].reshape(*m.shape[:2],22,6)
    assert torch.equal(six(final)[:,:,unchanged],six(raw)[:,:,unchanged])
    r=rotation_from_6d(six(final))
    torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3,dtype=r.dtype).expand_as(r),atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(torch.linalg.det(r),torch.ones_like(r[...,0,0]),atol=1e-10,rtol=1e-10)
    joints=forward_kinematics(final,details['rest'])
    from scenemi.geometry.body import PARENTS
    lengths=lambda p:(p[:,:,1:]-p[:,:,list(PARENTS[1:])]).norm(dim=-1)
    torch.testing.assert_close(lengths(joints),lengths(forward_kinematics(raw,details['rest'])),atol=1e-10,rtol=1e-10)


def test_local_right_composition_and_root_world_yaw_are_distinct():
    model,raw,details,_,_,_=fixture()
    base_rot=rotation_from_6d(raw[...,3:135].reshape(*raw.shape[:2],22,6))
    base_rot[:,:,1]=torch.matrix_exp(torch.tensor([[0.,0.,.2],[0.,0.,0.],[-.2,0.,0.]],dtype=raw.dtype))
    base_rot[:,:,0]=torch.matrix_exp(torch.tensor([[0.,0.,0.],[0.,0.,-.2],[0.,.2,0.]],dtype=raw.dtype))
    raw=motion_from_rotations(base_rot,forward_kinematics(raw,details['rest'])[:,:,0],details['rest'])
    omega=torch.zeros(*raw.shape[:2],22,3,dtype=raw.dtype);omega[:,:,1,0]=.1;omega[:,:,0,1]=.05
    out=apply_rotations(raw,omega,details['rest'])
    actual=rotation_from_6d(out[...,3:135].reshape(*out.shape[:2],22,6))
    exp=rotation_increment(omega)+torch.eye(3,dtype=raw.dtype)
    torch.testing.assert_close(actual[:,:,1],base_rot[:,:,1]@exp[:,:,1])
    torch.testing.assert_close(actual[:,:,0],exp[:,:,0]@base_rot[:,:,0])


def test_scene_query_features_and_refinement_are_equivariant_to_world_yaw_translation():
    model,raw,details,public,_,_=fixture()
    nn.init.normal_(model.refiner.rotation.weight,std=.02)
    first, extra=model.refine(raw,details,public)
    q=rotation_increment(raw.new_tensor([0.,.7,0.]))+torch.eye(3,dtype=raw.dtype)
    translation=raw.new_tensor([.31,0.,-.27])
    transform=lambda p:(q@p[...,None])[...,0]+translation
    local=rotation_from_6d(raw[...,3:135].reshape(*raw.shape[:2],22,6)).clone();local[:,:,0]=q@local[:,:,0]
    transformed=motion_from_rotations(local,transform(forward_kinematics(raw,details['rest'])[:,:,0]),details['rest'])
    camera_rotation=q@global_rotations(raw)[:,:,15]
    camera=torch.cat((transform(public['camera'][...,:3]*2)/2,camera_rotation[..., :,0],camera_rotation[..., :,1]),-1)
    other_public=dict(public,camera=camera,static_points=transform(public['static_points']),static_normals=(q@public['static_normals'][...,None])[...,0])
    other_details=dict(details,observations=dict(details['observations'],camera=camera))
    second,other=model.refine(transformed,other_details,other_public)
    torch.testing.assert_close(other['rotation_residual'],extra['rotation_residual'],atol=1e-9,rtol=1e-8)
    torch.testing.assert_close(forward_kinematics(second,details['rest']),transform(forward_kinematics(first,details['rest'])),atol=1e-9,rtol=1e-8)


def test_private_targets_rejected_and_frozen_components_stay_frozen():
    model,raw,details,public,batch,_=fixture()
    with pytest.raises(ValueError,match='Labels'):
        model.refine(raw,details,dict(public,motion=batch['motion']))
    model.train()
    assert not model.base.training and not model.skin.training and model.refiner.training
    assert all(not p.requires_grad for p in model.base.parameters())


def test_final_B_and_FK_propagate_pose_and_contact_gradients():
    model,raw,details,public,batch,policy=fixture()
    batch['motion']=batch['motion'].clone();batch['motion'][...,9]+=.1
    final,extra=model.refine(raw,details,public)
    terms,stats=objective(model,final,extra,batch,policy)
    terms['total'].backward()
    assert torch.isfinite(terms['total'])
    for name in ('rotation.weight','rotation.bias','contact.2.weight'):
        gradient=model.refiner.get_parameter(name).grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm()>0,name
    assert all(p.grad is None for p in model.base.parameters())
    assert stats['contact_valid_count']>0 and stats['contact_positive_count']>0


def test_contact_masks_unknown_normals_source_conflicts_and_uncertain_bands():
    model,raw,details,public,batch,policy=fixture()
    label,weight=contact_targets(model,batch,policy['contact'])
    assert (label*weight).sum()>0
    for replacement in (dict(static_normal_valid=torch.zeros_like(batch['static_normal_valid'])),
                        dict(contact_weight=torch.zeros_like(batch['contact_weight']))):
        _, weight=contact_targets(model,dict(batch,**replacement),policy['contact'])
        assert weight.sum()==0
    for height in (-.2, .045):
        changed=batch['motion'].clone();changed[...,1]+=height/2
        _,weight=contact_targets(model,dict(batch,motion=changed),policy['contact'])
        assert weight.sum()==0
    value=torch.randn_like(label,requires_grad=True)
    zero=masked_mean(value,torch.zeros_like(weight));zero.backward()
    assert zero.item()==0 and torch.count_nonzero(value.grad)==0


def test_missing_normals_are_explicit_and_do_not_remove_distance_features():
    model,raw,details,public,_,_=fixture()
    public=dict(public,static_normal_valid=torch.zeros_like(public['static_normal_valid']))
    _,features,hit=model.features(model.locate(raw,details),details,public,True)
    assert hit.any() and torch.isfinite(features).all()
    assert torch.count_nonzero(features[...,3:7])==0
    assert torch.count_nonzero(features[...,9])==0


def test_pose_loss_alone_reaches_part_geometry_without_contact_supervision():
    model,raw,details,public,batch,policy=fixture()
    nn.init.normal_(model.refiner.rotation.weight,std=.05)
    final,extra=model.refine(raw,details,public)
    terms,_=objective(model,final,extra,batch,policy)
    (terms['fk_position']+terms['lower_fk']).backward()
    gradient=model.refiner.geometry[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm()>0
    shifted=dict(public,static_points=public['static_points']+raw.new_tensor([0.,.05,0.]))
    other,other_details=model.refine(raw,details,shifted)
    assert not torch.allclose(other_details['rotation_residual'],extra['rotation_residual'])
