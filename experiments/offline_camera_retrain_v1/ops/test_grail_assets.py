"""CPU checks that GRAIL scene-query skin uses the declared body template."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .grail_assets import GrailSkinAnchorCache, G1_TEMPLATE
from .grail_source import g1_template_assets


class GrailAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.profile = dict(model='smplx', gender='neutral', betas=[0.] * 10,
                           scale=1., template=G1_TEMPLATE,
                           template_sha256=g1_template_assets()['sha256'])
        cls.cache = GrailSkinAnchorCache(device='cpu')
        cls.asset = cls.cache.get(cls.profile)

    def test_articulated_skin_matches_native_decoder(self):
        from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
        rng = np.random.default_rng(846)
        poses = (rng.normal(size=(3, 22, 3)) * .12).astype(np.float32)
        rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(3, 22, 3, 3)
        motion = np.zeros((3, 201), np.float32)
        motion[:, :3] = np.array([[0., 0., 0.], [.2, -.1, .3], [-.1, .2, 0.]])
        motion[:, 3:135] = (np.concatenate((rotations[..., 0], rotations[..., 1]), -1)
                             - [1, 0, 0, 0, 1, 0]).reshape(3, 132)
        model = self.cache.caches[G1_TEMPLATE].models[('smplx', 'neutral')]
        with torch.no_grad():
            output = model(global_orient=torch.from_numpy(poses[:, 0]),
                           body_pose=torch.from_numpy(poses[:, 1:].reshape(3, 63)),
                           betas=torch.zeros(3, 10), transl=torch.from_numpy(motion[:, :3] * 2),
                           left_hand_pose=torch.zeros(3, 45), right_hand_pose=torch.zeros(3, 45),
                           expression=torch.zeros(3, 10), jaw_pose=torch.zeros(3, 3),
                           leye_pose=torch.zeros(3, 3), reye_pose=torch.zeros(3, 3))
            actual = skin_points(torch.from_numpy(motion), self.asset).numpy()
        expected = output.vertices[:, self.asset['vertex_ids']].numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)

    def test_rejects_double_scale_and_wrong_template_identity(self):
        for replacement in (dict(scale=.7633275), dict(template_sha256={}),
                            dict(template='smplx'), dict(betas=[1.] + [0.] * 9)):
            with self.assertRaises(ValueError):
                self.cache.get(dict(self.profile, **replacement))


if __name__ == '__main__':
    unittest.main()
