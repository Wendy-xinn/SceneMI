"""AMASS split, camera extrinsic and body-representation regression checks."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from scenemi.config import load_config
from scenemi.data.amass import exclusion_reason, split_definition, verify_upstream
from scenemi.geometry.body import AMASS_ZUP_TO_YUP, PARENTS, encode_body, smplh_profile
from scenemi.geometry.camera import camera_from_head, cpf_from_camera, head_from_camera, transform, virtual_mount


class ConfigTests(unittest.TestCase):
    def test_rejects_protocol_changes(self):
        original = load_config()
        mutations = [('data', 'frames', 192), ('scene', 'enabled', True),
                     ('scene', 'geometry_losses', True), ('model', 'checkpoint', 'mixed.pt'),
                     ('model', 'diffusion_target', 'reference_residual'),
                     ('observation', 'ground_truth_shape_as_input', True),
                     ('observation', 'exo_input', 'ground_truth_body'),
                     ('model', 'shape_betas', 10), ('model', 'diffusion_steps', 500),
                     ('data', 'representation', 'native_rest_root'),
                     ('data', 'num_betas', 10), ('data', 'exo_camera', 'ground_truth_camera'),
                     ('observation', 'input_noise', 'unvalidated_noise'),
                     ('training', 'batch_size', 0), ('optimizer', 'warmup_updates', 0),
                     ('data', 'splits', {'train': ['train', 'val'], 'validation': ['val'], 'test': ['test']}),
                     ('loss', 'supervise_hidden_joints', False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for section, key, value in mutations:
                config = copy.deepcopy(original)
                config[section][key] = value
                path.write_text(yaml.safe_dump(config))
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    load_config(path)

    def test_disjoint_corpus_split_and_upstream_exclusions(self):
        splits = split_definition()['splits']
        self.assertEqual([len(splits[k]) for k in ('train', 'val', 'test')], [13, 4, 2])
        self.assertEqual(set(splits['test']), {'Transitions_mocap', 'SSM_synced'})
        self.assertIsNotNone(exclusion_reason('BioMotionLab_NTroje/subject/normal_walk_poses.npz'))
        self.assertIsNotNone(exclusion_reason('MPI_HDM05/dg/HDM_dg_07-01_poses.npz'))
        self.assertIsNone(exclusion_reason('CMU/normal_walk_poses.npz'))

    def test_local_upstream_matches_pinned_definition(self):
        upstream = Path(__file__).resolve().parents[3] / 'egoallo'
        if not upstream.is_dir():
            self.skipTest('Optional local EgoAllo checkout')
        self.assertEqual(verify_upstream(upstream)['commit'], split_definition()['commit'])


class CameraTests(unittest.TestCase):
    def test_virtual_camera_nonidentity_face_rotation_and_inverse(self):
        rotations = Rotation.from_rotvec([[.2, -.3, .4], [-.4, .1, .3]]).as_matrix()
        positions = np.array([[1., 2., 3.], [2., -.1, .7]])
        face = Rotation.from_rotvec([.3, -.1, .2]).as_matrix()
        offset = np.array([.01, .08, .09])
        mount = virtual_mount(offset, face)
        head = transform(rotations, positions)
        camera = camera_from_head(head, mount)
        expected_rotations = rotations @ face
        expected_positions = positions + (rotations @ offset[:, None])[..., 0] + .02 * expected_rotations[..., :, 2]
        np.testing.assert_allclose(camera[..., :3, :3], expected_rotations, atol=1e-12)
        np.testing.assert_allclose(camera[..., :3, 3], expected_positions, atol=1e-12)
        np.testing.assert_allclose(head_from_camera(camera, mount), head, atol=1e-12)
        self.assertGreater(np.linalg.norm(camera[0, :3, 3] - positions[0] - (rotations[0] @ face) @ offset), .005)

    def test_cpf_conversion_preserves_observation_and_changes_optical_center(self):
        head = transform(Rotation.from_rotvec([.1, .2, -.3]).as_matrix(), [1., 2., 3.])
        camera_mount = virtual_mount([0., .05, .08])
        cpf_mount = transform(np.eye(3), [0., .05, .08])
        camera = camera_from_head(head, camera_mount)
        cpf = cpf_from_camera(camera, camera_mount, cpf_mount)
        np.testing.assert_allclose(cpf, head @ cpf_mount, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(camera[:3, 3] - cpf[:3, 3]), .02)
        with self.assertRaises(ValueError):
            transform(np.diag([1., 1., -1.]), [0., 0., 0.])


class BodyTests(unittest.TestCase):
    def test_world_conversion_and_encoding_match_existing_fk(self):
        import torch
        from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
        rng = np.random.default_rng(8)
        rest = rng.normal(size=(22, 3)) * .1
        rest[0] = [.03, .08, -.01]
        local = Rotation.from_rotvec(rng.normal(size=(7*22, 3))*.2).as_matrix().reshape(7, 22, 3, 3)
        root = rng.normal(size=(7, 3))
        raw = encode_body(local, root, rest)
        local_yup = local.copy()
        local_yup[:, 0] = AMASS_ZUP_TO_YUP @ local[:, 0]
        converted = encode_body(local_yup, root @ AMASS_ZUP_TO_YUP.T, rest)
        expected = raw[:, 135:].reshape(7, 22, 3) @ AMASS_ZUP_TO_YUP.T
        np.testing.assert_allclose(converted[:, 135:].reshape(7, 22, 3), expected, atol=2e-7)
        fk = forward_kinematics(torch.from_numpy(converted[None]), torch.from_numpy(rest[None].astype(np.float32)))
        np.testing.assert_allclose(fk.numpy()[0], converted[:, 135:].reshape(7, 22, 3)*2, atol=5e-7)
        np.testing.assert_allclose(converted[:, :3]*2 + rest[0], root @ AMASS_ZUP_TO_YUP.T, atol=2e-7)

    def test_native_smplh_shape_and_distinct_mounts(self):
        path = Path('/mnt/e/Datasets/human/AGORA/smplh/neutral/model.npz')
        if not path.is_file():
            self.skipTest('Optional licensed SMPL-H model')
        profile = smplh_profile(path, np.zeros(16))
        self.assertEqual(profile['rest'].shape, (22, 3))
        self.assertEqual(len(profile['betas']), 16)
        self.assertEqual(len(PARENTS), 22)
        self.assertGreater(np.linalg.norm(profile['head_camera'][:3, 3] - profile['head_cpf'][:3, 3]), .015)
        with self.assertRaisesRegex(ValueError, 'truncate'):
            smplh_profile(path, np.zeros(10))


if __name__ == '__main__':
    unittest.main()
