import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scene_visibility_v2 import render_scene, sample_scene_hits, interpolate_transforms, temporal_bps, static_ray_scene


def plane(z, owner, x=0, size=10):
    return (np.array([[x-size,-size,z],[x+size,-size,z],
                      [x+size,size,z],[x-size,size,z]], np.float32),
            np.array([[0,1,2],[0,2,3]], np.int32), owner)


class VisibilityTests(unittest.TestCase):
    def render(self, meshes):
        return render_scene(meshes, np.zeros(3), np.eye(3), width=16, height=12)

    def test_self_limb_blocks_scene(self):
        r = self.render([plane(2, 0), plane(1, 100)])
        self.assertTrue((r['owner'] == 100).all())
        self.assertTrue((sample_scene_hits(r)[1] == -1).all())

    def test_other_person_nearest_wins(self):
        r = self.render([plane(3,0), plane(2,1), plane(1,101)])
        self.assertTrue((r['owner'] == 101).all())

    def test_dynamic_occludes_static_and_moves(self):
        a = self.render([plane(2,0),plane(1,1)])
        b = self.render([plane(2,0),plane(1,1,x=30)])
        self.assertTrue((a['owner']==1).all())
        self.assertTrue((b['owner']==0).all())

    def test_near_clipping_does_not_disable_far_occlusion(self):
        r = self.render([plane(.02,100), plane(1,100), plane(2,0)])
        self.assertTrue(np.allclose(r['depth'],1))

    def test_camera_world_roundtrip(self):
        r = self.render([plane(2,0)])
        self.assertTrue(np.allclose(r['points'][...,2],2))

    def test_cached_static_matches_joint_scene(self):
        v,f,_=plane(2,0)
        cached=static_ray_scene(v,f)
        a=self.render([plane(2,0),plane(1,100,size=.2)])
        b=render_scene([plane(1,100,size=.2)],np.zeros(3),np.eye(3),width=16,height=12,static_scene=cached)
        self.assertTrue(np.array_equal(a['owner'],b['owner']))
        self.assertTrue(np.allclose(a['depth'],b['depth']))
        c=render_scene([],np.zeros(3),np.eye(3),width=16,height=12,static_scene=cached)
        self.assertTrue((c['owner']==0).all())

    def test_world_rotation_equivariance(self):
        q=Rotation.from_euler('xyz',[20,35,-40],degrees=True).as_matrix()
        shift=np.array([1,-3,2]);v,f,owner=plane(2,0)
        a=self.render([(v,f,owner)])
        b=render_scene([(v@q.T+shift,f,owner)],shift,q,width=16,height=12)
        self.assertTrue(np.allclose(a['depth'],b['depth'],atol=1e-5))
        self.assertTrue(np.allclose(a['points']@q.T+shift,b['points'],atol=1e-5))

    def test_30_to_20hz_and_rotation_wrap(self):
        rot = Rotation.from_euler('y',[179,-179],degrees=True).as_matrix()
        r,p = interpolate_transforms([0,1],rot,[[0,0,0],[2,0,0]],[.5])
        self.assertTrue(np.allclose(p,[[1,0,0]]))
        self.assertLess(r[0,2,2],-.999)
        with self.assertRaises(ValueError):
            interpolate_transforms([0,1],rot,[[0,0,0],[2,0,0]],[2])

    def test_bps_has_no_old_dynamic_trail_or_future(self):
        static = [np.zeros((0,3)),np.zeros((0,3))]
        dynamic = [np.array([[.1,0,0]]),np.array([[.9,0,0]])]
        bps=temporal_bps(static,dynamic,np.zeros((2,3)),np.tile(np.eye(3),(2,1,1)),np.zeros((1,3)))
        self.assertTrue(np.allclose(bps[:,0,0],[.05,.45]))

    def test_unobserved_bps_is_masked_not_contact(self):
        empty=[np.zeros((0,3))]
        bps,valid=temporal_bps(empty,empty,np.zeros((1,3)),np.eye(3)[None],np.zeros((1,3)),return_valid=True)
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(bps).all())


if __name__ == '__main__':
    unittest.main()
