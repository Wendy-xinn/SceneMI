"""Predicted shape, body-label isolation and 128-frame unified conditioning."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scenemi.config import load_config
from scenemi.geometry.body import PARENTS, smplh_profile
from scenemi.geometry.shape import SmplhShape, load_shape_basis, numpy_profile
from scenemi.models import build_model, motion_loss
from scenemi.observations import public_observations, body_condition
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def make_shape_fixture(path):
    """Synthetic, non-licensed topology; compare regression against full mesh."""
    rng = np.random.default_rng(71)
    vertices = rng.normal(size=(6890, 3)).astype(np.float32)*.1
    directions = rng.normal(size=(6890, 3, 16)).astype(np.float32)*.005
    regressor = np.zeros((22, 6890), np.float32)
    for joint in range(22):
        regressor[joint, joint*2:joint*2+2] = [.3, .7]
    hierarchy = np.array([PARENTS, tuple(range(22))], np.int64)
    np.savez(path, v_template=vertices, shapedirs=directions,
             J_regressor=regressor, kintree_table=hierarchy)


def public_batch():
    b, t = 2, 128
    trajectory = torch.zeros(b, t, 22, 9)
    trajectory[1, :, (0, 12, 20), :3] = .2
    meta = torch.zeros(b, t, 22, 5)
    meta[1, :, (0, 12, 20), 0] = 1
    meta[1, :, (0, 12, 20), 2] = 1
    camera = torch.zeros(b, t, 9)
    camera[0, :, :3] = torch.tensor([.1, .8, -.2])
    camera[0, :, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.])
    available = torch.zeros(b, t, 2)
    available[0] = 1
    return dict(camera=camera, camera_available=available, camera_confidence=available.clone(),
                camera_context=torch.tensor([[1., 0., 0., 1., 0., -1.], [0., 1., 0., 1., 0., -1.]]),
                camera_intrinsics=torch.zeros(b, t, 4), camera_intrinsics_valid=torch.zeros(b, t),
                trajectory=trajectory, observation_meta=meta,
                observation_uncertainty=torch.zeros(b, t, 22, 2))


class ShapeTests(unittest.TestCase):
    def test_regressed_shape_matches_full_native_mesh_and_has_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'
            make_shape_fixture(path)
            shape = SmplhShape(path)
            beta = torch.linspace(-2, 2, 32).reshape(2, 16).requires_grad_()
            result = shape(beta)
            basis = load_shape_basis(path)
            for i in range(2):
                native = smplh_profile(path, beta[i].detach().numpy())
                rest, mount, native_rest = numpy_profile(basis, beta[i].detach().numpy())
                np.testing.assert_allclose(result['native_rest'][i].detach(), native['rest'], atol=1e-7)
                np.testing.assert_allclose(result['rest'][i].detach(), rest, atol=1e-7)
                np.testing.assert_allclose(native_rest, native['rest'], atol=1e-7)
                np.testing.assert_allclose(result['head_camera_translation'][i].detach(),
                                           native['head_camera'][:3, 3], atol=1e-7)
                np.testing.assert_allclose(mount, native['head_camera'][:3, 3], atol=1e-7)
                np.testing.assert_allclose(result['head_cpf_translation'][i].detach(),
                                           native['head_cpf'][:3, 3], atol=1e-7)
            self.assertTrue(torch.equal(result['rest'][0, 0], result['rest'][1, 0]))
            self.assertFalse(torch.equal(result['native_rest'][0, 0], result['native_rest'][1, 0]))
            sum(v.square().sum() for v in result.values()).backward()
            self.assertTrue(torch.isfinite(beta.grad).all())
            self.assertGreater(float(beta.grad.norm()), 0)

    def test_missing_camera_rotation_never_recovers_head_position(self):
        batch = public_batch()
        batch['camera_available'][0, :, 1] = 0
        safe = public_observations(batch)
        shape = dict(rest=torch.zeros(2, 22, 3), head_camera_translation=torch.ones(2, 3)*.1)
        conditioned = body_condition(safe, shape)
        self.assertTrue((conditioned['observation_meta'][0, :, 15, :4] == 0).all())
        self.assertTrue((conditioned['trajectory'][0, :, 15] == 0).all())


class AmassModelTests(unittest.TestCase):
    def test_mixed_mode_shape_gradients_fk_and_forbidden_label_isolation(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'
            make_shape_fixture(path)
            model = build_model(config, path)
            batch = public_batch()
            labels = model.shape(torch.ones(2, 16))
            batch.update(target_rest=labels['rest'].detach(), target_betas=torch.ones(2, 16))
            target = reconcile_motion(torch.zeros(2, 128, 201), batch['target_rest'])
            noisy = torch.randn_like(target)
            time = torch.tensor([50, 500])
            prediction, details = model(noisy, time, batch, return_details=True)
            terms = motion_loss(prediction, target, batch, torch.tensor([.9, .5]), details=details, config=config)
            terms['total'].backward()
            self.assertTrue(torch.isfinite(terms['total']))
            self.assertGreater(float(model.shape_encoder.readout[-1].weight.grad.norm()), 0)
            self.assertGreater(float(model.observation_projection.weight.grad.norm()), 0)
            self.assertGreater(float(model.core.unet.final_conv[-1].weight.grad.norm()), 0)
            self.assertTrue(all(p.grad is None for p in model.core.scene_embedding.parameters()))
            self.assertTrue(all(p.grad is None for p in model.patch_projection.parameters()))
            torch.testing.assert_close(forward_kinematics(prediction, details['rest']),
                                       prediction[..., 135:].reshape(2, 128, 22, 3)*2)
            model.eval()
            with torch.no_grad():
                clean, before = model(noisy, time, batch, return_details=True)
                poisoned = dict(batch, rest=torch.full((2, 22, 3), float('nan')),
                                target_rest=torch.full((2, 22, 3), float('nan')),
                                target_betas=torch.full((2, 16), float('nan')),
                                occupancy=torch.full((2, 24, 48, 48), float('nan')),
                                mount_translation_m=torch.full((2, 3), float('nan')),
                                trajectory=batch['trajectory'].clone(),
                                observation_meta=batch['observation_meta'].clone())
                poisoned['trajectory'][0] = float('nan')
                poisoned['observation_meta'][0] = 1
                hidden = poisoned['observation_meta'][1, ..., 0] == 0
                poisoned['trajectory'][1][hidden] = 100
                after, extra = model(noisy, time, poisoned, return_details=True)
                self.assertTrue(torch.equal(clean, after))
                self.assertTrue(torch.equal(before['predicted_betas'], extra['predicted_betas']))
            with self.assertRaisesRegex(ValueError, 'without scene'):
                model(noisy, time, batch, use_scene=True)

    def test_target_fk_uses_target_shape_even_when_local_poses_match(self):
        predicted_rest = torch.zeros(2, 22, 3)
        target_rest = predicted_rest.clone()
        target_rest[:, 18, 0] = .25
        prediction = reconcile_motion(torch.zeros(2, 128, 201), predicted_rest)
        target = reconcile_motion(torch.zeros_like(prediction), target_rest)
        observations = public_observations(public_batch())
        details = dict(rest=predicted_rest, raw_motion=prediction, predicted_betas=torch.zeros(2, 16),
                       observations=observations, head_camera_translation=torch.zeros(2, 3))
        labels = dict(target_rest=target_rest, target_betas=torch.zeros(2, 16))
        terms = motion_loss(prediction, target, labels, torch.ones(2), details=details, config=load_config())
        expected = (prediction[..., 135:]-target[..., 135:]).square().mean()*4
        self.assertGreater(float(expected), 0)
        torch.testing.assert_close(terms['position'], expected)
        self.assertEqual(float(terms['rotation']), 0)


if __name__ == '__main__':
    unittest.main()
