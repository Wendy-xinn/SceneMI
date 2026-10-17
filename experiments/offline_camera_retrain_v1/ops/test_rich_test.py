"""CPU checks for independent TEST identities, missing labels and metrics."""
import hashlib
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from .rich_test import source_inventory, read_test_fits, metrics_for
from experiments.offline_camera_retrain_v1.native30_window_index import valid_window_grid
from experiments.offline_camera_retrain_v1.native30_memory import observer_stream_key


class RichTestContract(unittest.TestCase):
    def test_all_targets_keep_absolute_clock_and_missing_occluder_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for frame in range(5,505):
                folder=root/'LectureHall_009_021_reparingprojector1'/f'{frame:05d}'
                folder.mkdir(parents=True)
                (folder/'009.pkl').touch()
                if frame not in range(300,305):
                    (folder/'021.pkl').touch()
            rows=source_inventory(root)
            self.assertEqual(len(rows),2)
            self.assertEqual({r['subject_id'] for r in rows},{'009','021'})
            for row in rows:
                self.assertEqual((row['raw_first'],row['raw_last_exclusive']),(5,505))
                self.assertEqual(row['missing_any_subject_frame_ids'],list(range(300,305)))
                self.assertEqual(row['split'],'test')
                self.assertEqual(row['scan_filename'],'scan_yoga_scene_camcoord.ply')
                valid=np.ones(500,bool);valid[295:300]=False
                self.assertEqual(valid_window_grid(valid,192,64).tolist(),[0,64])
            a,b=[dict(row,world_frame_id='same') for row in rows]
            self.assertNotEqual(observer_stream_key(a,'ego'),observer_stream_key(b,'ego'))

    def test_missing_fit_is_masked_not_interpolated_or_clock_compressed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/f'{i}.pkl' for i in range(3)]
            fit=dict(betas=np.ones((1,10),np.float32),transl=np.array([[1.,2.,3.]],np.float32))
            for path in (paths[0],paths[2]):
                path.write_bytes(pickle.dumps(fit))
            values,present=read_test_fits(paths,hashlib.sha256())
            self.assertEqual(len(values),3)
            self.assertEqual(present.tolist(),[True,False,True])
            np.testing.assert_array_equal(values[1]['transl'],np.zeros((1,3)))
            np.testing.assert_array_equal(values[1]['betas'],fit['betas'])
            np.testing.assert_array_equal(values[2]['transl'],fit['transl'])

    def test_world_metric_detects_known_translation_and_pa_removes_it(self):
        motion=np.zeros((192,201),np.float32)
        rest=np.array([[i%3*.03,i*.04,i%5*.02] for i in range(22)],np.float32)
        prediction=motion.copy();prediction[:,0]=.05  # encoded position is m/2
        observed=np.zeros((192,22,9),np.float32)
        meta=np.zeros((192,22,5),np.float32)
        result=metrics_for(dict(prediction=prediction,truth=motion,rest=rest,observations=observed,observation_meta=meta))
        self.assertAlmostEqual(result['w_mpjpe_mm'],100.,places=3)
        self.assertAlmostEqual(result['pa_mpjpe_mm'],0.,places=3)
        self.assertAlmostEqual(result['foot_joint_horizontal_speed_gt_low_m_s'],0.,places=6)
        self.assertEqual(result['gt_low_speed_foot_joint_samples'],382)


if __name__=='__main__':
    unittest.main()
