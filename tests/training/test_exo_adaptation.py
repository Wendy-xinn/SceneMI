"""EGO preservation, input isolation and EXO observation corruption."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scenemi.data.exo_corruption import corrupt_exo
from scenemi.exo_model import ExoAdaptedPrior
from scenemi.models import AmassMotionPrior
from tests.model.test_amass_model import make_shape_fixture, public_batch


class ExoTests(unittest.TestCase):
    def test_corruption_preserves_labels_missing_slots_and_is_reproducible(self):
        batch = public_batch()
        sample = {k:v[1].numpy().copy() for k,v in batch.items()}
        sample['motion'] = np.ones((128, 201), np.float32)
        sample['target_betas'] = np.ones(16, np.float32)
        before, repeat = copy.deepcopy(sample), copy.deepcopy(sample)
        policy = dict(clean_probability=.25, joint_sigma_m=[.005, .03], joint_rho=.85,
                      root_drift_sigma_m=[.01, .06], root_rho=.98)
        recipe = corrupt_exo(sample, np.random.default_rng(73), policy, force=True)
        other = corrupt_exo(repeat, np.random.default_rng(73), policy, force=True)
        self.assertEqual(recipe, other)
        for key in sample:
            np.testing.assert_array_equal(sample[key], repeat[key])
        for key in ('motion', 'target_betas', 'camera', 'camera_available', 'camera_context'):
            np.testing.assert_array_equal(sample[key], before[key])
        missing = sample['observation_meta'][..., 0] == 0
        self.assertTrue((sample['trajectory'][missing] == 0).all())
        self.assertTrue((sample['observation_uncertainty'][missing] == 0).all())
        self.assertFalse(np.array_equal(sample['trajectory'], before['trajectory']))
        ego = {k:v[0].numpy().copy() for k,v in batch.items()}
        with self.assertRaises(ValueError):
            corrupt_exo(ego, np.random.default_rng(73), policy)

    def test_exo_update_cannot_change_ego_and_mixed_dispatch_ignores_labels(self):
        torch.set_num_threads(2)
        torch.manual_seed(81)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'shape.npz'
            make_shape_fixture(path)
            model = ExoAdaptedPrior(AmassMotionPrior(path), hidden=32)
            batch = public_batch()
            noisy, times = torch.randn(2, 128, 201), torch.tensor([10, 700])
            ego = {k:v[:1] for k,v in batch.items()}
            exo = {k:v[1:] for k,v in batch.items()}
            with torch.no_grad():
                original = model.base(noisy[:1], times[:1], ego)
                self.assertTrue(torch.equal(original, model(noisy[:1], times[:1], ego)))
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
            prediction, details = model(noisy[1:], times[1:], exo, return_details=True)
            (prediction.square().mean()+details['predicted_betas'].square().mean()).backward()
            optimizer.step()
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(original, model(noisy[:1], times[:1], ego)))
                joint = model(noisy, times, batch)
                separate = model(noisy[1:], times[1:], exo)
                self.assertTrue(torch.equal(original, joint[:1]))
                self.assertTrue(torch.equal(separate, joint[1:]))
                poisoned = dict(batch, target_betas=torch.full((2,16), float('nan')),
                    target_rest=torch.full((2,22,3), float('nan')), motion=torch.full_like(noisy, float('nan')))
                self.assertTrue(torch.equal(joint, model(noisy, times, poisoned)))
            self.assertFalse(model.base.training)


if __name__ == '__main__':
    unittest.main()
