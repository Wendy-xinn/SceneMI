import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from protected_residual import protected_residual
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


class ProtectionTest(unittest.TestCase):
    def test_available_positions_unchanged_while_hidden_branches_can_move(self):
        torch.set_num_threads(2);torch.manual_seed(9)
        rest=torch.randn(2,22,3)*.1
        motion=reconcile_motion(torch.randn(2,64,201)*.1,rest)
        available=torch.rand(2,64,22)>.5
        details=dict(rotation=torch.randn(2,64,22,3)*.4,translation=torch.randn(2,64,3)*.2)
        pred=protected_residual(motion,details,available,rest)
        a=forward_kinematics(motion,rest);b=forward_kinematics(pred,rest)
        torch.testing.assert_close(a[available],b[available],atol=1e-6,rtol=1e-6)
        self.assertGreater(float((a[~available]-b[~available]).abs().sum()),0)

    def test_all_observed_and_no_observed_boundary_conditions(self):
        rest=torch.randn(1,22,3)*.1
        motion=reconcile_motion(torch.zeros(1,64,201),rest)
        details=dict(rotation=torch.randn(1,64,22,3)*.4,translation=torch.randn(1,64,3)*.2)
        all_on=torch.ones(1,64,22,dtype=torch.bool)
        a=protected_residual(motion,details,all_on,rest)
        torch.testing.assert_close(forward_kinematics(a,rest),forward_kinematics(motion,rest),atol=1e-7,rtol=1e-6)
        b=protected_residual(motion,details,~all_on,rest)
        target=reconcile_motion(motion,rest,details['translation'],details['rotation'])
        torch.testing.assert_close(b,target,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
