import torch
from scenemi.pose_oracles import replace_root_orientation
from scenemi.body_diagnostic_metrics import motion_from_rotations
from scenemi.metrics_batch import metric_tensors
from experiments.offline_camera_retrain_v1.supervision import PARENTS,forward_kinematics


def test_root_only_changes_no_local_rotations_or_translation_and_ignores_other_gt():
    generator=torch.Generator().manual_seed(41)
    rest=torch.zeros(1,22,3,dtype=torch.float64)
    for j in range(1,22):rest[:,j]=rest[:,PARENTS[j]]+torch.randn(3,generator=generator)*.1
    rotation=torch.eye(3,dtype=torch.float64).expand(1,8,22,3,3).clone()
    root=torch.randn(1,8,3,dtype=torch.float64,generator=generator)
    original=motion_from_rotations(rotation,root,rest)
    rotation[:,:,0]=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    target=motion_from_rotations(rotation,root+10,rest)
    changed=replace_root_orientation(original,target,rest)
    assert torch.equal(changed[...,:3],original[...,:3])
    assert torch.equal(changed[...,9:135],original[...,9:135])
    assert torch.equal(changed[...,3:9],target[...,3:9])
    other=target.clone();other[...,:3]+=99;other[...,9:]=torch.randn(other[...,9:].shape,generator=generator)
    assert torch.equal(changed,replace_root_orientation(original,other,rest))
    assert torch.equal(original,motion_from_rotations(torch.eye(3,dtype=torch.float64).expand(1,8,22,3,3),root,rest))
    points=forward_kinematics(changed,rest);expected=forward_kinematics(target,rest)
    before=metric_tensors(forward_kinematics(original,rest),expected)
    after=metric_tensors(points,expected)
    torch.testing.assert_close(before['pa_mpjpe_mm'],after['pa_mpjpe_mm'],atol=1e-8,rtol=0)
    assert after['root_relative_mpjpe_mm'].abs().max()<1e-8
    assert before['root_relative_mpjpe_mm'].min()>10
