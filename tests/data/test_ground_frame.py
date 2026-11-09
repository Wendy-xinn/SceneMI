import unittest
import numpy as np
from scenemi.geometry.ground_frame import observed_ground_height,translate_sample_ground


class GroundFrameTest(unittest.TestCase):
    def test_lowest_supported_surface_and_vertical_equivariance(self):
        x,z=np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
        floor=np.stack((x.ravel(),np.full(x.size,-1.55),z.ravel()),-1)
        table=floor+np.array([0,.8,0])
        points=np.concatenate((floor,table,table))
        normals=np.tile([0.,1.,0.],(len(points),1));valid=np.ones(len(points),bool)
        height,_=observed_ground_height(points,normals,valid)
        shifted,_=observed_ground_height(points+[0,.137,0],normals,valid)
        self.assertAlmostEqual(height,-1.55)
        self.assertAlmostEqual(shifted-height,.137)
        with self.assertRaisesRegex(ValueError,'ground evidence'):
            observed_ground_height(points,normals,np.zeros(len(points),bool))

    def test_scene_camera_and_target_translate_together_without_changing_rest(self):
        sample=dict(camera=np.zeros((128,9),np.float32),motion=np.zeros((128,201),np.float32),
            static_points=np.zeros((2,3),np.float32),static_valid=np.array([True,False]),
            frame_scene_points=np.zeros((128,2,3),np.float32),
            frame_scene_owner=np.tile([0,-1],(128,1)),target_rest=np.ones((22,3)))
        result=translate_sample_ground(sample,-1.5)
        np.testing.assert_equal(result['camera'][:,1],.75)
        np.testing.assert_equal(result['motion'][:,1],.75)
        np.testing.assert_equal(result['motion'][:,135:].reshape(128,22,3)[...,1],.75)
        np.testing.assert_equal(result['static_points'][:,1],[1.5,0])
        np.testing.assert_equal(result['target_rest'],sample['target_rest'])
        np.testing.assert_equal(sample['motion'],0)


if __name__=='__main__':unittest.main()
