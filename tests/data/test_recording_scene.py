import numpy as np
import unittest

from scenemi.data.recording_scene import observed_recording, bounded_scene_points


class RecordingSceneTests(unittest.TestCase):
    def test_later_static_observations_without_dynamic_invalid_or_nan(self):
        points=np.array([[[0.,0,0],[1,0,0],[np.nan,0,0]],
                         [[0.01,0,0],[2,0,0],[3,0,0]],
                         [[4.,0,0],[5,0,0],[6,0,0]]],np.float32)
        owners=np.array([[0,1,0],[0,0,-1],[0,0,0]])
        actual=observed_recording(points,owners,[True,True,False],.05)
        # Include a later static hit, unlike a prefix ending at frame zero.
        # Keep the first actual point, rather than inventing a voxel centroid.
        np.testing.assert_array_equal(actual,np.array([[0,0,0],[2,0,0]],np.float32))


    def test_empty_map_and_stable_budget(self):
        empty=observed_recording(np.zeros((2,3,3)),np.full((2,3),-1),[True,True],.05)
        self.assertEqual(empty.shape,(0,3))
        points=np.arange(30,dtype=np.float32).reshape(10,3)
        actual=bounded_scene_points(points,4)
        self.assertEqual(len(actual),4)
        self.assertTrue(all(any(np.array_equal(row,p) for p in points) for row in actual))
        np.testing.assert_array_equal(actual,bounded_scene_points(points,4))


    def test_different_observation_clocks_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'different clocks'):
            observed_recording(np.zeros((2,3,3)),np.zeros((2,3)),[True],.05)


if __name__=='__main__':unittest.main()
