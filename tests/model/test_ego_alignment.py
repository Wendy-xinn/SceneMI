"""EGO placement geometry with rotating, nonzero mounts and unknown shape."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scenemi.geometry.body import encode_body, PARENTS
from scenemi.geometry.alignment import place_ego_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


class EgoAlignmentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        rest = np.zeros((22, 3))
        for i in range(1, 22):
            rest[i] = rest[PARENTS[i]]+rng.normal(size=3)*.1
        local = Rotation.from_rotvec(rng.normal(size=(8*22, 3))*.2).as_matrix().reshape(8,22,3,3)
        self.rest = torch.tensor(rest[None], dtype=torch.float64)
        self.motion = torch.tensor(encode_body(local, rng.normal(size=(8,3)), rest)[None], dtype=torch.float64)
        self.mount = torch.tensor([[.02, .04, .11]], dtype=torch.float64)
        rotation = Rotation.from_rotvec(rng.normal(size=(8,3))).as_matrix()
        position = rng.normal(size=(8,3))
        self.camera = torch.tensor(np.concatenate((position/2, rotation[..., :,0], rotation[..., :,1]),-1)[None], dtype=torch.float64)

    def camera_error(self, motion):
        joints = forward_kinematics(motion, self.rest)
        rotation = global_rotations(motion)[:, :,15]
        position = joints[:, :,15]+(rotation @ self.mount[:,None,:,None])[...,0]
        return (position-self.camera[...,:3]*2).abs().max()

    def test_rigid_placement_preserves_articulation_and_bones(self):
        out = place_ego_motion(self.motion, self.rest, self.mount, self.camera)
        self.assertLess(float(self.camera_error(out)), 1e-10)
        self.assertTrue(torch.equal(out[...,9:135],self.motion[...,9:135]))
        rot = global_rotations(out)[:,:,15]
        self.assertLess(float((rot[..., :,0]-self.camera[...,3:6]).abs().max()),1e-10)
        before, after = [forward_kinematics(x,self.rest) for x in (self.motion,out)]
        for j in range(1,22):
            self.assertTrue(torch.allclose((before[:,:,j]-before[:,:,PARENTS[j]]).norm(dim=-1),
                                          (after[:,:,j]-after[:,:,PARENTS[j]]).norm(dim=-1),atol=1e-10))

    def test_translation_only_matches_position_without_changing_rotations(self):
        out = place_ego_motion(self.motion,self.rest,self.mount,self.camera,rotation=False)
        self.assertLess(float(self.camera_error(out)),1e-10)
        self.assertTrue(torch.equal(out[...,3:135],self.motion[...,3:135]))

    def test_invalid_camera_cannot_silently_become_identity(self):
        camera = self.camera.clone()
        camera[...,3:]=0
        with self.assertRaises(ValueError):
            place_ego_motion(self.motion,self.rest,self.mount,camera)


if __name__=='__main__':
    unittest.main()
