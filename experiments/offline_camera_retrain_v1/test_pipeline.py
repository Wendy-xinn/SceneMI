"""Fast correctness gates for the offline SceneMI retraining adapter."""
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from experiments.offline_camera_retrain_v1.control import HEAD, sample_control_masks
from experiments.offline_camera_retrain_v1.data import HERE, OfflineSceneMIData, collate
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


class OfflineSceneMIPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation = OfflineSceneMIData('validation', seed=777)

    def test_zero_residual_is_identity_rotation(self):
        residual = torch.zeros(4, 22, 6)
        matrix = rotation_from_6d(residual)
        identity = torch.eye(3).expand_as(matrix)
        torch.testing.assert_close(matrix, identity, atol=1e-6, rtol=0.)

    def test_clean_targets_close_under_forward_kinematics(self):
        # TRUMANS joints and fitted SMPL-X have a known ~2.4 cm source-domain
        # discrepancy; EgoBody should be substantially tighter.
        thresholds_m = {'trumans': .08, 'camera_wearer': .02, 'interactee': .02}
        for group, threshold in thresholds_m.items():
            with self.subTest(group=group):
                sample, _ = self.validation.sample(128, group)
                batch = collate([sample])
                predicted = forward_kinematics(batch['motion'], batch['rest'])
                error = (predicted - batch['joints'] * 2).norm(dim=-1).mean()
                self.assertLess(float(error), threshold)

    def test_trajectory_is_full_body_and_head_is_camera(self):
        sample, _ = self.validation.sample(64, 'camera_wearer')
        self.assertEqual(sample['trajectory'].shape, (64, 22, 9))
        torch.testing.assert_close(torch.from_numpy(sample['trajectory'][:, HEAD]),
                                   torch.from_numpy(sample['camera']))

    def test_control_mixture_is_head_dominant_and_non_head_capable(self):
        import numpy as np
        rng = np.random.default_rng(2026)
        mask, modes = sample_control_masks(2000, 64, rng, 'cpu')
        head_fraction = float(mask[:, :, HEAD].any(dim=1).float().mean())
        self.assertGreater(head_fraction, .8)
        self.assertLess(head_fraction, .92)
        self.assertTrue(any(mode == 'pelvis' for mode in modes))
        self.assertTrue(any(mode == 'sparse_wrists' for mode in modes))

    def test_corrected_trumans_template_matches_raw_joint_targets(self):
        corrected = OfflineSceneMIData('validation', seed=777, skeleton_profile='trumans_male_v2')
        archived = OfflineSceneMIData('validation', seed=777)
        for _ in range(4):
            sample, identity = corrected.sample(128, 'trumans')
            old, old_identity = archived.sample(128, 'trumans')
            self.assertEqual(identity, old_identity)
            for key in ('joints', 'trajectory', 'camera', 'bps', 'occupancy'):
                torch.testing.assert_close(torch.from_numpy(sample[key]), torch.from_numpy(old[key]), atol=0, rtol=0)
            batch = collate([sample])
            error = (forward_kinematics(batch['motion'], batch['rest']) - batch['joints'] * 2).norm(dim=-1)
            self.assertLess(float(error.mean()), .001)
            self.assertLess(float(error[:, :, (10, 11)].mean()), .001)

    def test_native_rich_uses_original_motion_and_approved_camera(self):
        roots = [HERE / 'data/scene_visibility_v2_oct05' /
                 f'rich_{split}_smpl_native20_faceout_oct07'
                 for split in ('train', 'val')]
        metadata = [p for root in roots for p in root.glob('*/metadata.json')]
        self.assertEqual(len(metadata), 90)
        for path in metadata:
            with self.subTest(sequence=path.parent.name):
                meta = json.loads(path.read_text())
                self.assertEqual(meta['source_fps'], 30.)
                self.assertEqual(meta['source_stride_for_motion'], 1)
                self.assertEqual(meta['output_fps'], 20.)
                self.assertIn('identity face mount', meta['camera_protocol'])
                self.assertLess(meta['fk_p95_m'], .005)
                self.assertLess(meta['approved_camera_angle_max_deg'], .1)
                self.assertGreater(meta['face_normal_dot'], .5)
                self.assertLess(meta['full_wearer_first_hit_fraction'], .5)
                source_ids = np.load(path.parent / 'source_frame_ids.npy', mmap_mode='r')
                self.assertTrue(np.allclose(np.diff(source_ids), 1.5))

        native = OfflineSceneMIData('validation', seed=777,
                                    skeleton_profile='canonical_smpl',
                                    rich_source='native20_faceout_oct07')
        sample, _ = native.sample(128, 'rich')
        batch = collate([sample])
        error = (forward_kinematics(batch['motion'], batch['rest'])
                 - batch['joints'] * 2).norm(dim=-1).mean()
        self.assertLess(float(error), .001)


if __name__ == '__main__':
    unittest.main()
