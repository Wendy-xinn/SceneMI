"""Counterexamples for proposed confidence semantics and local-foot target."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from candidates import conservative_root_reference, body_local_foot_position
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, PARENTS
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


class Candidates(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def known(self):
        trajectory=torch.zeros(1,8,22,9)
        trajectory[:,:,15,3:]=torch.tensor(IDENTITY_6D)
        meta=torch.zeros(1,8,22,5);meta[:,:,15,:4]=1
        return dict(trajectory=trajectory,observation_meta=meta,
            torso_anchor_confidence=torch.zeros(1,8),
            reference_confidence=torch.ones(1,8,22,2),
            reference_motion=torch.zeros(1,8,201))

    def test_head_only_guess_is_low_confidence_without_editing_head_or_motion(self):
        known=self.known();result=conservative_root_reference(known)
        self.assertTrue((result['reference_confidence'][:,:,0,1]==.05).all())
        self.assertTrue((result['reference_confidence'][:,:,15,1]==1).all())
        self.assertTrue((known['reference_confidence']==1).all())
        self.assertIs(result['reference_motion'],known['reference_motion'])

    def test_observable_torso_and_direct_root_remain_confident(self):
        known=self.known();known['torso_anchor_confidence'][:,0]=.8
        known['trajectory'][:,1,0,3:]=torch.tensor(IDENTITY_6D)
        known['observation_meta'][:,1,0,1]=known['observation_meta'][:,1,0,3]=1
        confidence=conservative_root_reference(known)['reference_confidence']
        self.assertEqual(float(confidence[0,0,0,1]),1.)
        self.assertEqual(float(confidence[0,1,0,1]),1.)
        self.assertEqual(float(confidence[0,2,0,1]),float(torch.tensor(.05)))

    def fixture(self):
        rest=torch.zeros(1,22,3)
        for j in range(1,22):rest[:,j]=rest[:,PARENTS[j]]+torch.tensor([.02,-.1,.03])
        truth=reconcile_motion(torch.zeros(1,48,201),rest)
        pred=truth.clone();pred[:,:,3+4*6+1]=.2
        return pred,truth,dict(rest=rest)

    def test_constant_leg_posture_error_is_penalized_with_leg_gradient(self):
        pred,truth,batch=self.fixture();pred.requires_grad_();truth.requires_grad_()
        loss=body_local_foot_position(pred,truth,batch,torch.ones(1))
        self.assertGreater(float(loss.detach()),0);loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad[:,:,3+4*6:3+5*6].norm()),0)
        self.assertIsNone(truth.grad)

    def test_common_time_varying_root_motion_cannot_fake_leg_improvement(self):
        pred,truth,batch=self.fixture()
        angle=torch.zeros(1,48,22,3);angle[:,:,0,1]=torch.linspace(-.8,.8,48)
        offset=torch.linspace(0,1,48)[None,:,None]*torch.tensor([.3,.1,.2])
        transformed=lambda x:reconcile_motion(x,batch['rest'],offset,angle)
        a=body_local_foot_position(pred,truth,batch,torch.ones(1))
        b=body_local_foot_position(transformed(pred),transformed(truth),batch,torch.ones(1))
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)

    def test_zero_signal_or_missing_labels_do_not_create_supervision(self):
        pred,truth,batch=self.fixture()
        self.assertEqual(float(body_local_foot_position(pred,truth,batch,torch.zeros(1))),0)
        batch['gt_position_weight']=torch.zeros(1,48,22)
        self.assertEqual(float(body_local_foot_position(pred,truth,batch,torch.ones(1))),0)


if __name__=='__main__':
    unittest.main()
