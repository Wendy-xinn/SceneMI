"""CPU counterfactuals for the isolated confidence and body-local candidates."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from pytorch3d.transforms import axis_angle_to_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import PARENTS, IDENTITY_6D

SPEC = importlib.util.spec_from_file_location('native30_isolated_candidate', Path(__file__).with_name('candidate.py'))
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def motion_example():
    # Nondegenerate articulated legs.  Pelvis offset exercises root conventions.
    rest = torch.zeros(1, 22, 3)
    rest[:, 0] = torch.tensor([.03, 1., -.02])
    for j in range(1, 22):
        rest[:, j] = rest[:, PARENTS[j]] + torch.tensor([.035 * (-1 if j % 2 else 1), -.12, .02])
    truth = reconcile_motion(torch.zeros(1, 16, 201), rest)
    batch = dict(rest=rest, gt_position_weight=torch.ones(1, 16, 22),
                 gt_global_rotation_weight=torch.ones(1, 16, 22))
    return truth, batch


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def known(self):
        trajectory = torch.zeros(1, 5, 22, 9)
        trajectory[..., 3:] = torch.tensor(IDENTITY_6D)
        return dict(reference_motion=torch.randn(1, 5, 201),
                    reference_confidence=torch.ones(1, 5, 22, 2),
                    trajectory=trajectory,
                    observation_meta=torch.zeros(1, 5, 22, 5),
                    torso_anchor_confidence=torch.zeros(1, 5))

    def test_head_guess_is_capped_without_changing_motion_or_original(self):
        known = self.known()
        known['observation_meta'][:, :, 15, 1:4:2] = 1
        fixed = candidate.conservative_root_reference(known)
        torch.testing.assert_close(fixed['reference_confidence'][:, :, 0, 1], torch.full((1, 5), .05))
        self.assertIs(fixed['reference_motion'], known['reference_motion'])
        self.assertTrue(torch.all(known['reference_confidence'] == 1))
        torch.testing.assert_close(fixed['reference_confidence'][:, :, 1:], known['reference_confidence'][:, :, 1:])

    def test_direct_root_and_observable_torso_keep_confidence(self):
        for direct in (True, False):
            known = self.known()
            if direct:
                known['observation_meta'][:, :, 0, 1] = 1
                known['observation_meta'][:, :, 0, 3] = .8
            else:
                known['torso_anchor_confidence'].fill_(.8)
            fixed = candidate.conservative_root_reference(known)
            torch.testing.assert_close(fixed['reference_confidence'], known['reference_confidence'])

    def test_missing_labels_and_zero_signal_produce_no_pose_gradient(self):
        truth, batch = motion_example()
        wrong = truth.clone()
        wrong[:, :, 3 + 4 * 6 + 1] += .2
        for missing in (True, False):
            active = copy.deepcopy(batch)
            if missing:
                active['gt_position_weight'].zero_()
            pred = wrong.clone().requires_grad_()
            loss = candidate.body_local_foot_position(pred, truth, active,
                torch.ones(1) if missing else torch.zeros(1))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_leg_error_and_mean_offset_have_gradient_without_target_gradient(self):
        truth, batch = motion_example()
        target = truth.clone().requires_grad_()
        wrong = truth.clone()
        # Constant articulation error survives centering/temporal differencing.
        wrong[:, :, 3 + 4 * 6 + 1] += .25
        pred = wrong.requires_grad_()
        loss = candidate.body_local_foot_position(pred, target, batch, torch.ones(1))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(pred.grad[:, :, 3:135].abs().sum().item(), 0)
        self.assertIsNone(target.grad)

    def test_root_orientation_reliability_and_soft_weights_attenuate(self):
        truth, batch = motion_example()
        pred = truth.clone()
        pred[:, :, 3 + 4 * 6 + 1] += .2
        baseline = candidate.body_local_foot_position(pred, truth, batch, torch.ones(1))
        batch['gt_global_rotation_weight'][:, :, 0] = .25
        reduced = candidate.body_local_foot_position(pred, truth, batch, torch.ones(1))
        torch.testing.assert_close(reduced, baseline * .25)
        batch['gt_global_rotation_weight'][:, :, 0] = 0
        self.assertEqual(candidate.body_local_foot_position(pred, truth, batch, torch.ones(1)).item(), 0)

    def test_common_rigid_world_transform_does_not_change_loss(self):
        truth, batch = motion_example()
        pred = truth.clone()
        pred[:, :, 3 + 4 * 6 + 1] += .2
        rotation = axis_angle_to_matrix(torch.tensor([.2, .7, -.3]))
        shift = torch.tensor([.4, -.2, .7])
        def transform(motion):
            moved = motion.clone()
            root = rotation @ global_rotations(motion)[:, :, 0]
            moved[:, :, :3] = ((motion[:, :, :3] * 2 + batch['rest'][:, None, 0]) @ rotation.T
                               + shift - batch['rest'][:, None, 0]) / 2
            moved[:, :, 3:9] = torch.cat((root[..., :, 0], root[..., :, 1]), -1) - motion.new_tensor(IDENTITY_6D)
            return reconcile_motion(moved, batch['rest'])
        before = candidate.body_local_foot_position(pred, truth, batch, torch.ones(1))
        after = candidate.body_local_foot_position(transform(pred), transform(truth), batch, torch.ones(1))
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-7)

    def test_wrapper_adds_final_and_raw_once_and_preserves_existing_terms(self):
        truth, batch = motion_example()
        raw = truth.clone(); raw[:, :, 3 + 4 * 6 + 1] += .2
        raw.requires_grad_()
        original_terms = dict(base=dict(total=torch.tensor(2.), unchanged=torch.tensor(7.)),
                              observation=dict(position=torch.tensor(3.)))
        original = mock.Mock(return_value=(torch.tensor(5.), original_terms))
        total, terms = candidate.objective_with_body_local(original, truth, truth, batch,
            torch.ones(1), {'raw_motion': raw}, {}, {}, coefficient=2., raw_weight=.5)
        self.assertEqual(original.call_count, 1)
        raw_loss = candidate.body_local_foot_position(raw, truth, batch, torch.ones(1))
        torch.testing.assert_close(total, 5 + raw_loss)
        self.assertEqual(terms['base']['unchanged'].item(), 7.)
        self.assertEqual(original_terms['base']['total'].item(), 2.)
        self.assertIs(terms['observation'], original_terms['observation'])
        total.backward(); self.assertGreater(raw.grad.norm().item(), 0)

    def test_all_off_preserves_original_functions_and_exact_outputs(self):
        from experiments.offline_camera_retrain_v1 import train_native30 as trainer
        from experiments.offline_camera_retrain_v1 import native30_validation as validation
        sentinel = object()
        input_original = mock.Mock(return_value=sentinel)
        validation_original = mock.Mock(return_value=sentinel)
        objective_original = mock.Mock(return_value=sentinel)
        with mock.patch.object(trainer, 'model_inputs', input_original), \
             mock.patch.object(validation, 'model_inputs', validation_original), \
             mock.patch.object(trainer, 'objective', objective_original):
            candidate.install_hooks({})
            self.assertIs(trainer.model_inputs, input_original)
            self.assertIs(validation.model_inputs, validation_original)
            self.assertIs(trainer.objective, objective_original)
            self.assertIs(trainer.model_inputs({}, {}), sentinel)
            self.assertIs(trainer.objective(), sentinel)

    def test_repeated_install_does_not_stack_and_validation_matches_train(self):
        from experiments.offline_camera_retrain_v1 import train_native30 as trainer
        from experiments.offline_camera_retrain_v1 import native30_validation as validation
        known = self.known()
        input_original = mock.Mock(return_value=known)
        validation_original = mock.Mock(return_value=known)
        objective_original = lambda *args: None
        enabled = {'candidate': {'root_reference_cap_enabled': True}}
        with mock.patch.object(trainer, 'model_inputs', input_original), \
             mock.patch.object(validation, 'model_inputs', validation_original), \
             mock.patch.object(trainer, 'objective', objective_original):
            candidate.install_hooks(enabled); candidate.install_hooks(enabled)
            train_known = trainer.model_inputs(enabled, {})
            val_known = validation.model_inputs(enabled, {})
            self.assertEqual(input_original.call_count, 1)
            self.assertEqual(validation_original.call_count, 1)
            torch.testing.assert_close(train_known['reference_confidence'], val_known['reference_confidence'])
            candidate.install_hooks({})
            self.assertIs(trainer.model_inputs, input_original)
            self.assertIs(validation.model_inputs, validation_original)
            self.assertIs(trainer.objective, objective_original)

    def test_repeated_enabled_objective_install_adds_loss_once(self):
        from experiments.offline_camera_retrain_v1 import train_native30 as trainer
        from experiments.offline_camera_retrain_v1 import native30_validation as validation
        truth, batch = motion_example()
        pred = truth.clone(); pred[:, :, 3 + 4 * 6 + 1] += .2
        original = mock.Mock(return_value=(torch.tensor(2.), {'base': {'total': torch.tensor(2.)}}))
        enabled = {'candidate': {'body_local_foot_enabled': True}}
        with mock.patch.object(trainer, 'model_inputs', lambda *args: {}), \
             mock.patch.object(validation, 'model_inputs', lambda *args: {}), \
             mock.patch.object(trainer, 'objective', original):
            candidate.install_hooks(enabled); candidate.install_hooks(enabled)
            total, _ = trainer.objective(pred, truth, batch, torch.ones(1),
                {'raw_motion': pred}, enabled, {})
            loss = candidate.body_local_foot_position(pred, truth, batch, torch.ones(1))
            torch.testing.assert_close(total, 2 + 1.5 * loss)
            self.assertEqual(original.call_count, 1)

    def test_hook_preserves_a_supplied_frontend_reference_confidence(self):
        from experiments.offline_camera_retrain_v1 import train_native30 as trainer
        from experiments.offline_camera_retrain_v1 import native30_validation as validation
        known = self.known()
        enabled = {'candidate': {'root_reference_cap_enabled': True}}
        with mock.patch.object(trainer, 'model_inputs', lambda *args: known), \
             mock.patch.object(validation, 'model_inputs', lambda *args: known), \
             mock.patch.object(trainer, 'objective', lambda *args: None):
            candidate.install_hooks(enabled)
            batch = {'reference_motion': known['reference_motion']}
            self.assertIs(trainer.model_inputs(enabled, batch), known)
            self.assertIs(validation.model_inputs(enabled, batch), known)

    def test_zero_coefficient_returns_original_objects(self):
        total = torch.tensor(2.)
        terms = {'base': {'total': total}}
        original = lambda *args: (total, terms)
        result, logged = candidate.objective_with_body_local(original,
            None, None, None, None, None, None, None, coefficient=0)
        self.assertIs(result, total)
        self.assertIs(logged, terms)


if __name__ == '__main__':
    unittest.main()
