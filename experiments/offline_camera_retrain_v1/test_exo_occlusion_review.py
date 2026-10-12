"""Physical first-hit occlusion and exact input-mask/gap tests."""
import unittest

import numpy as np
import trimesh

from experiments.offline_camera_retrain_v1.exo_occlusion_review import (
    first_hit, rays_for, regional_observations, sampled_environment)
from experiments.offline_camera_retrain_v1.scene_visibility_v2 import static_ray_scene
from experiments.offline_camera_retrain_v1.view_exo_occlusion_review import trajectory_segments, playback_next
from experiments.offline_camera_retrain_v1.audit_exo_followup import recover_world_to_camera
from scipy.spatial.transform import Rotation


class ExoOcclusionReviewTests(unittest.TestCase):
    def setUp(self):
        self.camera=dict(size=[64,64],source_size=[64,64],K=np.array([[60.,0,32],[0,60.,32],[0,0,1]]),
                         distortion=np.zeros(8),position=np.zeros(3),rotation=np.eye(3),near=.1,far=12.)
        self.rays=rays_for(self.camera)
        self.target=trimesh.creation.box(extents=[1.,1.,.4]);self.target.vertices += [0,0,3.]
        self.wall=trimesh.creation.box(extents=[4.,4.,.3]);self.wall.vertices += [0,0,5.]
        self.static=static_ray_scene(self.wall.vertices,self.wall.faces)
        self.blocker=trimesh.creation.box(extents=[1.,1.,.3]);self.blocker.vertices += [0,0,2.]
        self.labels=np.zeros(len(self.target.faces),np.int32)

    def hit(self,people=None,objects=None,static=None):
        bodies=[(self.target.vertices,self.target.faces)]+(people or [])
        return first_hit(self.camera,self.rays,static or self.static,bodies,self.labels,objects or [],0)

    def test_person_removes_fully_occluded_joint(self):
        _,visible,parts,_=self.hit();before=regional_observations(parts,visible)
        _,owner,parts,_=self.hit(people=[(self.blocker.vertices,self.blocker.faces)])
        after=regional_observations(parts,owner)
        self.assertTrue(before['high'][0]);self.assertFalse(after['available'][0])
        self.assertEqual(after['person'][0],after['reference'][0]);self.assertEqual(after['confidence'][0],0.)

    def test_static_and_dynamic_objects_use_nearest_hit(self):
        blocker=static_ray_scene(self.blocker.vertices,self.blocker.faces)
        _,owner,parts,_=self.hit(static=blocker);stats=regional_observations(parts,owner)
        self.assertEqual(stats['static'][0],stats['reference'][0]);self.assertFalse(stats['available'][0])
        obj=(blocker,np.eye(3)[None],np.zeros((1,3)),1,np.array([True]))
        _,owner,parts,_=self.hit(objects=[obj]);stats=regional_observations(parts,owner)
        self.assertEqual(stats['rigid'][0],stats['reference'][0]);self.assertFalse(stats['high'][0])
        # Geometry farther than the target does not overwrite its visible skin.
        farther=self.blocker.copy();farther.vertices += [0,0,2.]
        obj=(static_ray_scene(farther.vertices,farther.faces),np.eye(3)[None],np.zeros((1,3)),1,np.array([True]))
        _,owner,parts,_=self.hit(objects=[obj]);self.assertTrue(regional_observations(parts,owner)['high'][0])

    def test_invalid_source_and_environment_target_exclusion(self):
        depth,owner,parts,_=self.hit();stats=regional_observations(parts,owner,valid=False)
        self.assertFalse(stats['available'].any());self.assertFalse(stats['confidence'].any())
        points=np.zeros((len(owner),3),np.float32);mask=owner>=0
        points[mask]=self.rays[mask,:3]+depth[mask,None]*self.rays[mask,3:]
        _,ids=sampled_environment(points,owner);self.assertFalse((ids==100).any())

    def test_confidence_is_not_the_current_availability_threshold(self):
        parts=np.zeros(100,np.int32);owner=np.full(100,101,np.int32);owner[:20]=100
        stats=regional_observations(parts,owner)
        self.assertTrue(stats['available'][0]);self.assertFalse(stats['high'][0])
        self.assertAlmostEqual(stats['confidence'][0],.2,places=6)

    def test_trajectory_does_not_bridge_unobserved_frames(self):
        p=np.arange(12,dtype=np.float32).reshape(4,1,3);mask=np.array([[True],[False],[True],[True]])
        segments=trajectory_segments(p,mask,3)
        np.testing.assert_array_equal(segments,np.stack((p[2,0],p[3,0]))[None])

    def test_noncyclic_motion_stops_without_false_boundary_velocity(self):
        self.assertEqual(playback_next(190,192),(191,True))
        self.assertEqual(playback_next(191,192),(191,False))
        self.assertEqual(playback_next(191,192,loop=True),(0,True))

    def test_paired_body_camera_recovery_respects_nonzero_root_pivot(self):
        rw=Rotation.from_euler('xyz',[[.3,.1,-.2],[-.1,.5,.2]]).as_matrix()
        known_r=Rotation.from_euler('xyz',[[.2,-.6,.1],[.1,-.4,-.2]]).as_matrix()
        known_t=np.array([[1.,2.,3.],[-2.,1.,4.]])
        tw=np.array([[.2,.3,.4],[-.3,.4,.2]]);rest=np.array([.01,-.3,.02])
        tc=np.einsum('tij,tj->ti',known_r,tw+rest)+known_t-rest
        r,t,centre=recover_world_to_camera(Rotation.from_matrix(rw).as_rotvec(),
                    Rotation.from_matrix(known_r@rw).as_rotvec(),tw,tc,rest)
        np.testing.assert_allclose(r,known_r,atol=1e-12)
        np.testing.assert_allclose(t,known_t,atol=1e-12)
        np.testing.assert_allclose(np.einsum('tij,tj->ti',r,centre)+t,0.,atol=1e-12)
        naive=tc-np.einsum('tij,tj->ti',r,tw)
        np.testing.assert_allclose(naive-known_t,np.einsum('tij,j->ti',r,rest)-rest,atol=1e-12)
        self.assertGreater(float(np.linalg.norm(naive-known_t)),.01)


if __name__=='__main__':unittest.main()
