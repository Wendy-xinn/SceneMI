"""Head orientation is distinct from translation anchoring and local motion."""
import math

import torch

from scenemi.refinement.diagnostics import motion_diagnostics, residual_statistics, rotation_angle
from scenemi.refinement.model import apply_rotations, rotation_increment
from tests.model.test_scene_refinement import fixture


def test_so3_diagnostic_preserves_zero_and_pi():
    identity=torch.eye(3,dtype=torch.float64)
    rotations=rotation_increment(torch.tensor([[0.,0.,0.],[0.,math.pi,0.]],dtype=torch.float64))+identity
    torch.testing.assert_close(rotation_angle(identity,rotations),torch.tensor([0.,math.pi],dtype=torch.float64))


def test_second_B_preserves_position_but_cannot_restore_head_orientation():
    model,raw,details,public,batch,_=fixture()
    baseline=model.locate(raw,details)
    omega=torch.zeros(*raw.shape[:2],22,3,dtype=raw.dtype);omega[:,:,0,1]=math.radians(4)
    final=model.locate(apply_rotations(raw,omega,details['rest']),details)
    extra=dict(details,rotation_residual=omega,correction_gate=torch.ones_like(omega[...,0],dtype=torch.bool))
    stats=motion_diagnostics(model,final,baseline,extra,batch)
    torch.testing.assert_close(stats['head_observation_orientation_deg'],torch.full((2,),4.,dtype=raw.dtype))
    assert (stats['head_observation_position_mm']<10).all()
    torch.testing.assert_close(stats['head_orientation_root_only_deg'],stats['head_observation_orientation_deg'])
    assert not stats['head_orientation_spine_only_deg'].any()
    omega.zero_();omega[:,:,4,0]=.1
    lower=apply_rotations(baseline,omega,details['rest'])
    stats=motion_diagnostics(model,lower,baseline,extra,batch)
    assert not stats['head_observation_orientation_deg'].any()


def test_saturation_is_normalized_by_eligible_joints_only():
    model,raw,*_=fixture()
    limits=model.refiner.limits
    omega=torch.zeros(*raw.shape[:2],22,3,dtype=raw.dtype)
    omega[:,:,4,0]=limits[4]*.95
    gate=torch.zeros_like(omega[...,0],dtype=torch.bool);gate[:,:,4]=True
    stats=residual_statistics(omega,limits,gate)
    assert (stats['residual_lower_saturation_fraction']==1).all()
    assert (stats['residual_lower_saturated_count']==raw.shape[1]).all()
    assert not stats['residual_root_saturation_fraction'].any()
    assert not stats['residual_spine_enabled_fraction'].any()
    # A single-axis output is saturated while its vector norm remains below
    # 90% of the nominal joint angle budget.
    omega[:,:,4,0]=limits[4]*.95/math.sqrt(3)
    stats=residual_statistics(omega,limits,gate)
    assert not stats['residual_lower_saturation_fraction'].any()
    assert (stats['residual_lower_component_saturation_fraction']==1).all()
