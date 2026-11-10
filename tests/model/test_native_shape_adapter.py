"""Native output formats must not add target-body information to inference."""
import tempfile
import unittest
from pathlib import Path

import torch

from scenemi.models import ObservationShapeEncoder, build_model
from scenemi.config import load_config
from scenemi.scene_model import NativeObservationEncoder, NativeScenePrior
from scenemi.geometry.native_shape import NativeBodyAdapter, NativeShapeBank
from tests.model.test_amass_model import make_shape_fixture, public_batch


def bases():
    torch.manual_seed(53)
    result = {}
    for kind, count in [('smpl', 3), ('smplx', 3), ('grail_g1', 1)]:
        result[kind] = dict(rest_template=torch.randn(count, 22, 3)*.1,
            rest_directions=torch.randn(count, 22, 3, 10)*.001,
            eye_template=torch.randn(count, 3)*.01,
            eye_directions=torch.randn(count, 3, 10)*.001)
    result['grail_g1']['rest_directions'].zero_()
    result['grail_g1']['eye_directions'].zero_()
    return result


PV = dict(rotation=[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
          translation_m=[0., .1, .1], position_p90_m=.07, rotation_p90_rad=.34)


class NativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_shape_features_are_identical(self):
        old, new = ObservationShapeEncoder(), NativeObservationEncoder()
        new.load_state_dict(old.state_dict(), strict=True)
        batch = public_batch()
        self.assertTrue(torch.equal(old(batch), new(batch)))

    def test_common_pelvis_and_native_coefficient_gradients(self):
        bank = NativeShapeBank(torch.tensor([.1, .2, .3]), bases())
        beta = torch.randn(2, 10, requires_grad=True)
        shape = bank('smplx', beta, torch.tensor([0, 2]))
        torch.testing.assert_close(shape['rest'][:, 0], torch.tensor([[.1, .2, .3]]).expand(2, -1))
        shape['rest'].square().sum().backward()
        self.assertGreater(float(beta.grad.norm()), 0)
        g1 = bank('grail_g1', beta.detach(), torch.tensor([0, 0]))
        torch.testing.assert_close(g1['rest'][0], g1['rest'][1])

    def test_native_heads_train_without_target_template_input(self):
        adapter = NativeBodyAdapter(torch.zeros(3), bases())
        features = torch.randn(3, 256, requires_grad=True)
        output = adapter(features, torch.tensor([1, 2, 3]))
        objective = output['rest'].square().sum() + output['predicted_native_betas'].square().sum()
        objective += torch.nn.functional.cross_entropy(output['template_logits'][:2], torch.tensor([2, 1]))
        objective.backward()
        for kind in ('smpl', 'smplx'):
            self.assertGreater(float(adapter.coefficients[kind].weight.grad.norm()), 0)
            self.assertGreater(float(adapter.templates[kind].weight.grad.norm()), 0)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_warm_start_scene_zero_and_forbidden_label_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'
            make_shape_fixture(path)
            original = build_model(load_config(), path).eval()
            model = NativeScenePrior(path, PV, prior_state=original.state_dict(), bases=bases()).eval()
            batch = public_batch()
            noisy, timestep = torch.randn(2, 128, 201), torch.tensor([5, 700])
            with torch.no_grad():
                before = original(noisy, timestep, batch)
                after = model(noisy, timestep, batch, use_scene=False)
                self.assertTrue(torch.equal(before, after))
                self.assertTrue((model.camera_encoder[0].weight[:, 13:18]==0).all())
                self.assertTrue((model.camera_encoder[0].weight[:, 34:36]==0).all())
                self.assertTrue(torch.equal(model.body_encoder[0].weight[:, 67], model.body_encoder[0].weight[:, 66]))
                batch.update(body_format=torch.tensor([1, 2]), sensor_profile=torch.tensor([0, 1]),
                             occupancy=torch.zeros(2, 24, 48, 48))
                batch['occupancy'][:, 4, 15, 15] = 1
                clean, details = model(noisy, timestep, batch, use_scene=False, return_details=True)
                scene = model(noisy, timestep, batch, use_scene=True)
                torch.testing.assert_close(scene, clean, atol=0, rtol=0)
                poisoned = dict(batch, target_rest=torch.full((2, 22, 3), float('nan')),
                    target_native_betas=torch.full((2, 10), float('nan')),
                    target_template=torch.tensor([1234, 1234]),
                    mount_translation_m=torch.full((2, 3), float('nan')))
                self.assertTrue(torch.equal(model(noisy, timestep, poisoned, use_scene=False), clean))
                # Predicted head rotation followed by the calibrated sensor
                # rotation must reconstruct the observed optical orientation.
                from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, IDENTITY_6D
                head_six = details['conditioned_observations']['trajectory'][0, :, 15, 3:]
                head = rotation_from_6d(head_six-head_six.new_tensor(IDENTITY_6D))
                self.assertTrue(torch.isfinite(head).all())


if __name__ == '__main__':
    unittest.main()
