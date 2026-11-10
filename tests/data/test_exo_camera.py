"""Physical camera and occlusion regressions for moving EXO inputs."""
import unittest

import numpy as np
import yaml

from scenemi.data.exo_camera import camera_trajectory, render_frame, projection_error
from scenemi.native_visualization import ROOT


class MovingCameraTests(unittest.TestCase):
    def setUp(self):
        self.policy = yaml.safe_load((ROOT/'configs/trumans_exo_moving.yaml').read_text())

    def test_optical_conversion_preserves_world_motion_and_has_proper_rotation(self):
        released = np.broadcast_to(np.eye(4), (3,4,4)).copy()
        released[:,0,3] = [0., .5, 1.]
        pose, camera = camera_trajectory(released, self.policy)
        np.testing.assert_allclose(pose[:,:3,3], released[:,:3,3])
        np.testing.assert_allclose(np.linalg.det(pose[:,:3,:3]), 1.)
        world = np.array([.1, .2, -3.])
        local = np.einsum('tji,tj->ti', pose[:,:3,:3], world-pose[:,:3,3])
        self.assertTrue((local[:,2] > 0).all())
        np.testing.assert_allclose(local[:,1], -.2)
        restored = np.einsum('tij,tj->ti', pose[:,:3,:3], local)+pose[:,:3,3]
        np.testing.assert_allclose(restored, np.broadcast_to(world, (3,3)), atol=1e-7)
        self.assertIs(camera['benchmark_camera_verified'], False)
        reflected = released.copy();reflected[:,0,0] = -1
        with self.assertRaises(ValueError):
            camera_trajectory(reflected, self.policy)

    def test_moving_observer_recasts_static_occlusion_and_masks_invalid_frames(self):
        from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
        # A small wall hides the entire target from x=0; moving sideways exposes it.
        wall = np.array([[-.35,-.8,2],[.35,-.8,2],[.35,.8,2],[-.35,.8,2]], np.float32)
        body = np.array([[-.3,-.4,3],[.3,-.4,3],[.3,.4,3],[-.3,.4,3]], np.float32)
        faces = np.array([[0,1,2],[0,2,3]], np.int32)
        scene = static_ray_scene(wall, faces)
        released = np.broadcast_to(np.diag([1,-1,-1,1]), (2,4,4)).copy().astype(float)
        released[1,0,3] = 1.2
        pose, camera = camera_trajectory(released, self.policy)
        results = []
        for t in range(2):
            hit, rays = render_frame(camera, pose[t], scene, [(body,faces)], np.zeros(2,np.int32), [], t)
            points, owner, _, _, _, _, reference, visible, blocked = hit
            np.testing.assert_array_equal(reference, visible+sum(blocked.values()))
            self.assertLess(projection_error(points, rays, camera, pose[t], owner), .02)
            results.append(hit)
        self.assertEqual(results[0][7].sum(), 0)
        self.assertGreater(results[1][7].sum(), 0)
        self.assertTrue(results[1][4][0])
        invalid, _ = render_frame(camera, pose[1], scene, [(body,faces)], np.zeros(2,np.int32), [], 1, False)
        self.assertFalse(invalid[4].any())
        self.assertFalse(invalid[5].any())


if __name__ == '__main__':
    unittest.main()
