import unittest

import numpy as np

from experiments.offline_camera_retrain_v1.audit_native_gt_quality import screen
from experiments.offline_camera_retrain_v1.native30_observations import perturb_input_tracks


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
    def test_ego_wrists_keep_position_rotation_confidence_and_availability(self):
        trajectory=np.ones((32,22,9),np.float32);meta=np.ones((32,22,5),np.float32)
        meta[:,20,2]=.7;meta[3,21,:4]=0  # preserve existing low confidence / absence
        old=trajectory[:,(20,21)].copy();old_meta=meta[:,(20,21)].copy()
        cam=np.ones((32,9),np.float32);ca=np.ones((32,2),np.float32);cc=ca.copy()
        perturb_input_tracks(trajectory,meta,cam,ca,cc,'ego',(15,20,21),ForcedGap())
        np.testing.assert_array_equal(trajectory[:,(20,21)],old)
        np.testing.assert_array_equal(meta[:,(20,21)],old_meta)
        self.assertTrue((cam[:,0]>1).all());self.assertFalse(ca[2:20].any())
        np.testing.assert_allclose(trajectory[:,15,:3],cam[:,:3])

    def test_exo_position_noise_and_gap_remain_active(self):
        trajectory=np.ones((32,22,9),np.float32);meta=np.ones((32,22,5),np.float32)
        cam=np.ones((32,9),np.float32);ca=np.ones((32,2),np.float32);cc=ca.copy()
        perturb_input_tracks(trajectory,meta,cam,ca,cc,'exo',tuple(range(22)),ForcedGap())
        self.assertTrue((trajectory[:,:,:3]>1).all());self.assertFalse(meta[2:20,21,:4].any())
        np.testing.assert_array_equal(cam,np.ones_like(cam));self.assertTrue(ca.all())


if __name__=='__main__':unittest.main()
