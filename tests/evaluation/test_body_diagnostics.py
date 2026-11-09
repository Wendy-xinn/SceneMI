import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scenemi.body_diagnostic_metrics import (motion_from_rotations, rotation_error_deg,
    standardized_points, diagnostics)
from scenemi.body_diagnostics import oracle_rest
from experiments.offline_camera_retrain_v1.supervision import PARENTS, forward_kinematics


def synthetic():
    rest=torch.zeros(1,22,3,dtype=torch.float64)
    for j in range(1,22):rest[:,j]=rest[:,PARENTS[j]]+torch.tensor([.02,.08,.01])
    rotations=torch.eye(3,dtype=torch.float64).expand(1,8,22,3,3).clone()
    root=torch.zeros(1,8,3,dtype=torch.float64)
    return rest,rotations,root


def test_rotation_metric_handles_identity_and_pi():
    eye=torch.eye(3,dtype=torch.float64)
    turn=torch.tensor(Rotation.from_rotvec([0,np.pi,0]).as_matrix())
    assert rotation_error_deg(eye,eye)==0
    assert abs(float(rotation_error_deg(eye,turn))-180)<1e-9


def test_same_pose_different_bones_disappears_on_standard_skeleton():
    rest,rotations,root=synthetic()
    other=rest*1.3
    a=motion_from_rotations(rotations,root,rest)
    b=motion_from_rotations(rotations,root,other)
    assert float((forward_kinematics(a,rest)-forward_kinematics(b,other)).norm(dim=-1).mean())>.01
    torch.testing.assert_close(standardized_points(a,rest),standardized_points(b,rest))


def test_root_orientation_and_articulation_are_separated():
    rest,rotations,root=synthetic()
    a=motion_from_rotations(rotations,root,rest)
    rotations[:,:,0]=torch.tensor(Rotation.from_euler('y',90,degrees=True).as_matrix())
    b=motion_from_rotations(rotations,root,rest)
    points=forward_kinematics(b,rest);gt=forward_kinematics(a,rest)
    row=diagnostics(points,b,rest,gt,a,rest,rest)[0]
    assert abs(row['metrics']['root_orientation_deg']-90)<1e-9
    assert row['metrics']['local_joint_rotation_deg']==0
    assert row['metrics']['standard_bone_rootrelative_mm']>10
    assert row['metrics']['standard_bone_rootaligned_mm']==0
    assert row['metrics']['mpjve_mm_s']==0
    assert row['metrics']['pred_jitter_100m_s3']==0
    assert row['metrics']['gt_foot_support_proxy_mpjpe_mm'] is None
    assert row['metrics']['gt_no_foot_support_proxy_frames']==7


def test_oracle_does_not_persist_or_change_normal_output():
    class Shape(torch.nn.Module):
        def forward(self,x):return dict(rest=x,head_camera_translation=x[:,0])
    class Model:pass
    model=Model();model.native_body=Shape()
    original=torch.zeros(1,22,3);target=torch.ones_like(original);mount=torch.full((1,3),2.)
    with oracle_rest(model,dict(target_rest=target),mount):
        result=model.native_body(original)
        torch.testing.assert_close(result['rest'],target)
        torch.testing.assert_close(result['head_camera_translation'],mount)
    result=model.native_body(original)
    torch.testing.assert_close(result['rest'],original)
    torch.testing.assert_close(result['head_camera_translation'],original[:,0])
