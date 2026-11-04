"""Tracking outages: clocks, observation-only anchoring and hidden-data isolation."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scenemi.config import ROOT, load_config, load_paths
from scenemi.data.amass_dataset import AmassDataset
from scenemi.data.observation_gaps import mask_observations, sample_valid_frames, validate_policy
from scenemi.models import build_model, motion_loss
from scenemi.observations import public_observations, body_condition
from scenemi.training import input_isolation_check
from tests.model.test_amass_model import public_batch, make_shape_fixture
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_observations import observation_derivatives


class GapPolicyTests(unittest.TestCase):
    def test_duration_boundaries_clean_examples_and_reproducible_rng(self):
        policy = load_config()['observation']['temporal_dropout']
        rng = np.random.default_rng(124)
        placements, clean = set(), 0
        for mode in ('ego', 'exo'):
            for _ in range(200):
                valid, record = sample_valid_frames(rng, mode, 128, policy)
                self.assertGreaterEqual(valid.sum(), 96)
                self.assertEqual(valid.shape, (128,))
                if mode == 'ego':
                    self.assertTrue(valid.all(), 'EGO camera trajectories must remain continuous')
                if valid.all():
                    clean += 1
                    continue
                placement = record['placement']
                placements.add(placement)
                a, b = record['frame_ranges'][0]
                self.assertGreaterEqual(b-a+1, 8)
                self.assertLessEqual(b-a+1, 32)
                self.assertEqual(np.flatnonzero(~valid).tolist(), list(range(a, b+1)))
                self.assertEqual(a == 0, placement == 'prefix')
                self.assertEqual(b == 127, placement == 'suffix')
        self.assertEqual(placements, {'prefix', 'interior', 'suffix'})
        self.assertGreater(clean, 0)
        state = copy.deepcopy(rng.bit_generator.state)
        first, identity = sample_valid_frames(rng, 'exo', 128, policy)
        restored = np.random.default_rng()
        restored.bit_generator.state = state
        after, replay = sample_valid_frames(restored, 'exo', 128, policy)
        np.testing.assert_array_equal(first, after)
        self.assertEqual(identity, replay)

    def test_invalid_extreme_augmentation_cannot_hide_entire_window(self):
        policy = load_config()['observation']['temporal_dropout']
        for changes in (dict(gap_frames=[8, 128]), dict(gap_frames=[0, 8]),
                        dict(probabilities={'ego': float('nan'), 'exo': .5}),
                        dict(probabilities={'ego': -1., 'exo': .5}),
                        dict(minimum_observed_frames=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_policy(dict(policy, **changes))

    def test_masking_keeps_labels_and_clock_and_does_not_interpolate(self):
        batch = public_batch()
        sample = {key: value[0].numpy().copy() for key, value in batch.items()}
        sample['motion'] = np.arange(128*201, dtype=np.float32).reshape(128, 201)
        sample['target_betas'] = np.arange(16, dtype=np.float32)
        truth = sample['motion'].copy()
        shape = sample['target_betas'].copy()
        valid = np.ones(128, bool)
        valid[31:59] = False
        mask_observations(sample, valid)
        np.testing.assert_array_equal(sample['motion'], truth)
        np.testing.assert_array_equal(sample['target_betas'], shape)
        self.assertFalse(sample['camera'][~valid].any())
        self.assertFalse(sample['camera_available'][~valid].any())
        self.assertFalse(sample['observation_meta'][~valid, :, :4].any())
        self.assertFalse(sample['trajectory'][~valid].any())


class GapModelTests(unittest.TestCase):
    def test_missing_frames_block_head_features_derivatives_and_poisoned_inputs(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)/'model.npz'
            make_shape_fixture(model_path)
            model = build_model(config, model_path)
            batch = public_batch()
            for i in (0, 1):
                single = {key: value[i].numpy().copy() for key, value in batch.items()}
                valid = np.ones(128, bool)
                valid[0:24] = False
                mask_observations(single, valid)
                for key, value in single.items():
                    batch[key][i] = torch.from_numpy(value)
            shape = model.shape(torch.zeros(2, 16))
            batch.update(target_rest=shape['rest'].detach(), target_betas=torch.zeros(2, 16))
            batch['motion'] = reconcile_motion(torch.zeros(2, 128, 201), batch['target_rest'])
            safe = public_observations(batch)
            conditioned = body_condition(safe, shape)
            self.assertFalse(conditioned['observation_meta'][:, :24, :, :4].any())
            self.assertFalse(conditioned['trajectory'][:, :24].any())
            _, _, _, derivative_mask = observation_derivatives(conditioned['trajectory'], conditioned['observation_meta'])
            self.assertFalse(derivative_mask[:, :25].any())
            self.assertTrue(input_isolation_check(model, batch)['masked_observation_poison_invariant'])
            noisy = torch.randn(2, 128, 201)
            predicted, details = model(noisy, torch.tensor([50, 200]), batch, return_details=True)
            terms = motion_loss(predicted, batch['motion'], batch, torch.ones(2), details=details, config=config)
            terms['total'].backward()
            self.assertTrue(all(torch.isfinite(term) for term in terms.values()))
            self.assertGreater(float(model.shape_encoder.readout[-1].weight.grad.norm()), 0.)


class RealWindowGapTests(unittest.TestCase):
    def test_prefix_anchor_uses_first_available_observation_in_each_mode(self):
        paths_file = ROOT/'configs/paths.local.yaml'
        index = ROOT/'runs/amass128/index/complete.json'
        if not paths_file.is_file() or not index.is_file():
            self.skipTest('Optional licensed, fully verified local AMASS')
        data = AmassDataset(index, load_paths(paths_file))
        policy = copy.deepcopy(load_config()['observation']['temporal_dropout'])
        policy['probabilities'] = dict(ego=1., exo=1.)
        for mode in ('ego', 'exo'):
            for seed in range(30):
                sample, identity = data.sample(np.random.default_rng(seed), mode,
                    member=0, start=0, reflected=False, yaw=0., temporal_dropout=policy)
                if identity['observation_gaps']['placement'] == 'prefix':
                    break
            else:
                self.fail('Did not exercise a missing-prefix window')
            first = identity['anchor_frame']
            self.assertGreater(first, 0)
            self.assertEqual(sample['motion'].shape, (128, 201))
            self.assertTrue(np.isfinite(sample['motion']).all())
            self.assertFalse(sample['camera_available'][:first].any())
            self.assertFalse(sample['observation_meta'][:first, :, :4].any())
            if mode == 'ego':
                np.testing.assert_allclose(sample['camera'][first, [0, 2]], 0., atol=1e-7)
                forward = np.cross(sample['camera'][first, 3:6], sample['camera'][first, 6:9])
                self.assertAlmostEqual(float(forward[0]), 0., places=6)
                self.assertGreater(float(forward[2]), 0.)
            else:
                valid = sample['observation_meta'][first, :, 0] > 0
                np.testing.assert_allclose(sample['trajectory'][first, valid, :3].mean(0)[[0, 2]], 0., atol=2e-6)
            clean, original = data.sample(np.random.default_rng(seed), mode,
                member=0, start=0, reflected=False, yaw=0.)
            for body, reference in ((sample, identity), (clean, original)):
                points = body['motion'][:, 135:].reshape(128, 22, 3)*2
                world = points @ np.asarray(reference['anchor']) + reference['origin']
                if body is sample:
                    gap_world = world
                else:
                    np.testing.assert_allclose(gap_world, world, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
