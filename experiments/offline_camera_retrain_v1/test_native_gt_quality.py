import unittest

import numpy as np

from experiments.offline_camera_retrain_v1.audit_native_gt_quality import screen
from experiments.offline_camera_retrain_v1.native30_input_noise import apply_input_noise,default_input_noise
from experiments.offline_camera_retrain_v1.native30_gt_reliability import quality_weights,DEFAULT_GT_RELIABILITY,derivative_valid,soft_mean


class CandidateScreenTests(unittest.TestCase):
    def sample(self,angles,ids=None,transl=None):
        pose=np.zeros((len(angles),22,3));pose[:,19,2]=np.radians(angles)
        return screen(pose,np.zeros((len(angles),3)) if transl is None else transl,
                      np.zeros((22,3)),np.arange(len(angles)) if ids is None else np.asarray(ids))

    def test_axis_angle_wrap_is_not_motion_jump(self):
        result=self.sample([179,-179,-177])
        self.assertFalse(result['large'].any());self.assertEqual(int(result['wrap'].sum()),1)
        self.assertAlmostEqual(result['step'][0,19],2.,places=5)

    def test_fast_monotonic_motion_is_not_isolated_flip(self):
        result=self.sample([0,70,140])
        self.assertTrue(result['large'][:,19].all());self.assertFalse(result['isolated'].any())

    def test_isolated_rotation_and_position_excursion(self):
        result=self.sample([0,120,0],transl=np.array([[0,0,0],[.1,0,0],[0,0,0]]))
        self.assertTrue(result['isolated'][1,19]);self.assertTrue(result['pulse'][1].all())

    def test_missing_frame_does_not_create_velocity_or_rotation_alert(self):
        result=self.sample([0,120,120],ids=[0,2,3])
        self.assertFalse(result['large'].any());self.assertFalse(result['isolated'].any())


class ForcedGap:
    def normal(self,loc,scale,size):return np.full(size,scale)
    def random(self):return 0.
    def choice(self,items):return items[-1]
    def integers(self,limit):return min(2,limit-1)


class ObservationNoiseTests(unittest.TestCase):
    def test_disabled_ego_wrists_keep_position_rotation_confidence_and_availability(self):
        trajectory=np.ones((32,22,9),np.float32);meta=np.ones((32,22,5),np.float32)
        meta[:,20,2]=.7;meta[3,21,:4]=0  # preserve existing low confidence / absence
        old=trajectory[:,(20,21)].copy();old_meta=meta[:,(20,21)].copy()
        cam=np.ones((32,9),np.float32);ca=np.ones((32,2),np.float32);cc=ca.copy()
        config=default_input_noise();config['clean_probability']=0.;config['gap_probability']=1.;config['ego_wrists']['enabled']=False
        apply_input_noise(trajectory,meta,cam,ca,cc,np.zeros((32,22,2)),'ego',(15,20,21),ForcedGap(),config)
        np.testing.assert_array_equal(trajectory[:,(20,21)],old)
        np.testing.assert_array_equal(meta[:,(20,21)],old_meta)
        self.assertTrue((cam[:,0]>1).all());self.assertFalse(ca[2:20].any())

    def test_exo_position_noise_and_gap_remain_active(self):
        trajectory=np.ones((32,22,9),np.float32);meta=np.ones((32,22,5),np.float32)
        cam=np.ones((32,9),np.float32);ca=np.ones((32,2),np.float32);cc=ca.copy()
        config=default_input_noise();config['clean_probability']=0.;config['gap_probability']=1.
        apply_input_noise(trajectory,meta,cam,ca,cc,np.zeros((32,22,2)),'exo',tuple(range(22)),ForcedGap(),config)
        self.assertTrue((trajectory[:,:,:3]>1).all());self.assertFalse(meta[2:20,21,:4].any())
        np.testing.assert_array_equal(cam,np.ones_like(cam));self.assertTrue(ca.all())


class ConfiguredNoiseTests(unittest.TestCase):
    def arrays(self):
        trajectory=np.zeros((32,22,9),np.float32);trajectory[...,3:]=[1,0,0,0,1,0]
        return trajectory,np.ones((32,22,5),np.float32),trajectory[:,15].copy(),np.ones((32,2),np.float32),np.ones((32,2),np.float32),np.zeros((32,22,2),np.float32)

    def test_clean_branch_keeps_all_observations_bitwise(self):
        arrays=self.arrays();before=[x.copy() for x in arrays];config=default_input_noise();config['clean_probability']=1.
        self.assertFalse(apply_input_noise(*arrays,'ego',(15,20,21),np.random.default_rng(1),config))
        for a,b in zip(arrays,before):np.testing.assert_array_equal(a,b)

    def test_only_observed_wrists_perturbed_and_can_disable(self):
        config=default_input_noise();config['clean_probability']=0.;config['gap_probability']=0.
        arrays=self.arrays();before=[x.copy() for x in arrays]
        self.assertTrue(apply_input_noise(*arrays,'ego',(15,20),np.random.default_rng(1),config))
        self.assertFalse(np.array_equal(arrays[0][:,20],before[0][:,20]))
        np.testing.assert_array_equal(arrays[0][:,21],before[0][:,21])
        for f in range(32):
            six=arrays[0][f,20,3:];r=np.stack((six[:3],six[3:],np.cross(six[:3],six[3:])),axis=-1)
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-6)
        config['ego_wrists']['enabled']=False;arrays=self.arrays();old=arrays[0].copy()
        apply_input_noise(*arrays,'ego',(15,20,21),np.random.default_rng(1),config)
        np.testing.assert_array_equal(arrays[0][:,(20,21)],old[:,(20,21)])

    def test_exo_has_no_invented_rotation_and_camera_not_perturbed(self):
        config=default_input_noise();config['clean_probability']=0.;config['gap_probability']=0.
        arrays=self.arrays();arrays[0][...,3:]=0;arrays[1][...,1]=0;arrays[1][...,3]=0
        camera=arrays[2].copy();apply_input_noise(*arrays,'exo',tuple(range(22)),np.random.default_rng(1),config)
        self.assertEqual(float(np.abs(arrays[0][...,3:]).sum()),0.)
        self.assertEqual(float(arrays[1][...,1].sum()),0.);np.testing.assert_array_equal(arrays[2],camera)


class ReliabilityTests(unittest.TestCase):
    def motion(self,angles,joint):
        from scipy.spatial.transform import Rotation
        from experiments.offline_camera_retrain_v1.data import rotation6d
        motion=np.zeros((len(angles),201),np.float32);rot=np.broadcast_to(np.eye(3),(len(angles),22,3,3)).copy()
        rot[:,joint]=Rotation.from_euler('z',angles,degrees=True).as_matrix()
        motion[:,3:135]=(rotation6d(rot)-np.array([1,0,0,0,1,0])).reshape(len(angles),132)
        return motion

    def test_leaf_rotation_only_downweights_leaf_rotation(self):
        result=quality_weights(self.motion([0,120,0],20),np.zeros((22,3)),DEFAULT_GT_RELIABILITY)
        self.assertEqual(result['gt_local_rotation_weight'][1,20],.25)
        self.assertTrue((result['gt_position_weight']==1).all())
        self.assertEqual(int((result['gt_global_rotation_weight']<1).sum()),1)

    def test_elbow_rotation_propagates_to_wrist_position_not_entire_body(self):
        result=quality_weights(self.motion([0,120,0],19),np.zeros((22,3)),DEFAULT_GT_RELIABILITY)
        self.assertEqual(result['gt_position_weight'][1,21],.25)
        self.assertEqual(result['gt_position_weight'][1,19],1.)
        self.assertEqual(result['gt_global_rotation_weight'][1,21],.25)
        self.assertTrue((result['gt_position_weight'][:,15]==1).all())

    def test_derivative_stencil_includes_interior_candidates_and_soft_weights_do_not_cancel(self):
        import torch
        weight=torch.ones(1,8,22);weight[:,3,20]=.25
        valid=derivative_valid(weight,3)
        self.assertFalse(valid[0,:4,20].any());self.assertEqual(float(valid[0,4,20]),1.)
        value=torch.ones(1,3,22,requires_grad=True)
        loss=soft_mean(value,torch.full_like(value,.25));self.assertAlmostEqual(float(loss.detach()),.25)
        loss.backward();self.assertTrue(torch.isfinite(value.grad).all())

    def test_clean_fast_motion_not_downweighted_and_policy_disable(self):
        motion=self.motion([0,70,140],19)
        result=quality_weights(motion,np.zeros((22,3)),DEFAULT_GT_RELIABILITY)
        self.assertTrue(all((w==1).all() for w in result.values()))
        config=dict(DEFAULT_GT_RELIABILITY,enabled=False)
        result=quality_weights(self.motion([0,120,0],19),np.zeros((22,3)),config)
        self.assertTrue(all((w==1).all() for w in result.values()))

    def test_base_losses_skip_target_derivatives_touching_candidate(self):
        import torch
        from experiments.offline_camera_retrain_v1.native30_losses import base_losses
        prediction=torch.zeros(1,8,201,requires_grad=True);truth=torch.zeros_like(prediction)
        truth[:,3,0]=.1;batch=dict(rest=torch.zeros(1,22,3))
        baseline=base_losses(prediction,truth,batch,torch.ones(1))
        self.assertGreater(float(baseline['fk_velocity_m2_s2'].detach()),0.)
        batch['gt_position_weight']=torch.ones(1,8,22);batch['gt_position_weight'][:,3]=.25
        result=base_losses(prediction,truth,batch,torch.ones(1))
        self.assertEqual(float(result['fk_velocity_m2_s2'].detach()),0.)
        self.assertEqual(float(result['fk_acceleration_m2_s4'].detach()),0.)
        self.assertAlmostEqual(float(result['x0'].detach()),float(baseline['x0'].detach())*.25,places=7)
        result['total'].backward();self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_original_GT_validation_reports_candidates_instead_of_hiding_them(self):
        import torch
        from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
        pred=torch.zeros(1,8,201);pred[:,:,0]=.05
        batch=dict(motion=torch.zeros_like(pred),rest=torch.zeros(1,22,3),gt_position_weight=torch.full((1,8,22),.25))
        metrics=raw_metrics(pred,batch,torch.zeros(1,8,22,dtype=torch.bool))
        self.assertAlmostEqual(metrics['mpjpe_mm']['value'],100.,places=4)
        self.assertAlmostEqual(metrics['candidate_position_mpjpe_mm']['value'],100.,places=4)
        self.assertEqual(metrics['candidate_position_mpjpe_mm']['count'],176)
        self.assertIsNone(metrics['regular_position_mpjpe_mm']['value'])


if __name__=='__main__':unittest.main()
