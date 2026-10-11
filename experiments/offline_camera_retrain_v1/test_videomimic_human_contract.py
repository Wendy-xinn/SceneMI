"""Timing/source-topology checks needed before touching training data."""
import tempfile, unittest
from pathlib import Path
import h5py, numpy as np
from experiments.offline_camera_retrain_v1.videomimic_human_contract import time_queries,validate_rotations,import_smpl,Z_TO_Y

class ContractTests(unittest.TestCase):
    def test_original60_direct20_and_native15_rejected(self):
        q,a=time_queries(np.arange(301),60);np.testing.assert_array_equal(q,np.arange(0,301,3));self.assertEqual(a['output_frames'],101)
        with self.assertRaisesRegex(ValueError,'below20Hz'):time_queries(np.arange(0,301,4),60)
        with self.assertRaisesRegex(ValueError,'gap'):time_queries([0,1,2,30,31],60)
    def test_rotation_reflection_rejected(self):
        with self.assertRaisesRegex(ValueError,'reflected'):validate_rotations(np.diag([1,1,-1]),'world')
    def test_robot_only_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'robot.h5'
            with h5py.File(p,'w') as f:f.create_dataset('joints',data=np.zeros((2,23)))
            with self.assertRaisesRegex(ValueError,'robot-only'):import_smpl(p,'1',source_fps=60,gender='male',source_up_axis='z')
    def test_native_smpl_pelvis_world_conversion_and_eye_mount(self):
        # Construct a human-H5 protocol fixture, not a VideoMimic result.
        # Cross-check uses the installed full SMPL model versus shared22 FK.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'human_fixture.h5';n=96
            with h5py.File(p,'w') as f:
                h=f.create_group('our_pred_humans_smplx_params/1');h.create_dataset('body_pose',data=np.broadcast_to(np.eye(3),(n,23,3,3)));h.create_dataset('global_orient',data=np.broadcast_to(Z_TO_Y.T,(n,1,3,3)));h.create_dataset('betas',data=np.zeros(10));h.create_dataset('root_transl',data=np.column_stack((np.arange(n)*.01,np.zeros(n),np.ones(n))))
                f.create_dataset('person_frame_info_list/1',data=np.array([f'cam01/{i:05d}.jpg'.encode() for i in range(n)]))
            arrays,audit=import_smpl(p,'1',source_fps=60,gender='male',source_up_axis='z');self.assertLess(audit['native_FK_max_error_m'],1e-4);self.assertEqual(len(arrays['source_frame_ids']),32);np.testing.assert_allclose(arrays['joints_world'][:,0,1],1,atol=1e-5);self.assertTrue(np.isfinite(arrays['camera_position_scenemi_yup']).all())

if __name__=='__main__':unittest.main()
