"""Unknown geometry and temporal/official-label contracts for weak contacts."""
import unittest
import numpy as np
import torch
from experiments.offline_camera_retrain_v1.native30_scene_support import support_targets, region_contact_loss


def fixture():
    vertices = np.zeros((8, 3, 3), np.float32); vertices[..., 1] = .025
    shape = (8, 3, 3, 8)
    query = dict(points=np.zeros((*shape, 3), np.float32),
                 normals=np.zeros((*shape, 3), np.float32),
                 valid=np.ones(shape, bool), normal_reliable=np.ones(shape, bool),
                 owners=np.zeros(shape, np.int64), velocities=np.zeros((*shape, 3), np.float32),
                 velocity_valid=np.ones(shape, bool))
    query['normals'][..., 1] = 1
    return vertices, np.full(3, 10), query


class SupportTests(unittest.TestCase):
    def test_stable_observed_support_creates_weak_labels(self):
        result = support_targets(*fixture())
        self.assertTrue(result['target'][1:, 10].all())
        self.assertTrue((result['weight'][1:, 10] == .25).all())
        self.assertFalse(result['valid'][0, 10])

    def test_unseen_and_unreliable_surfaces_remain_unknown(self):
        for key in ('valid', 'normal_reliable'):
            v, labels, q = fixture(); q[key].fill(False)
            self.assertFalse(support_targets(v, labels, q)['valid'].any())

    def test_fast_or_isolated_near_contact_is_not_a_negative(self):
        v, labels, q = fixture(); v[:, :, 0] = np.arange(8)[:, None] * .02
        self.assertFalse(support_targets(v, labels, q)['target'].any())
        v, labels, q = fixture(); q['valid'].fill(False); q['valid'][4] = True
        self.assertFalse(support_targets(v, labels, q)['valid'].any())

    def test_other_person_without_velocity_is_unknown(self):
        v, labels, q = fixture(); q['owners'].fill(101); q['velocity_valid'].fill(False)
        self.assertFalse(support_targets(v, labels, q)['valid'].any())

    def test_nearby_wall_does_not_create_foot_support(self):
        v, labels, q = fixture(); q['normals'][..., 1] = 0; q['normals'][..., 0] = 1
        self.assertFalse(support_targets(v, labels, q)['target'].any())

    def test_official_annotations_override_weak_targets(self):
        v, labels, q = fixture(); official = np.zeros((8, 22), np.float32)
        mask = np.zeros_like(official, bool); mask[:, 10] = True
        result = support_targets(v, labels, q, official_target=official, official_valid=mask)
        self.assertFalse(result['target'][:, 10].any())
        self.assertTrue((result['weight'][:, 10] == 1).all())

    def test_region_negative_requires_all_anchors_known(self):
        v, labels, q = fixture(); v[..., 1] = .15
        self.assertTrue(support_targets(v, labels, q)['valid'][:, 10].all())
        q['valid'][:, 0] = False
        self.assertFalse(support_targets(v, labels, q)['valid'].any())

    def test_perfect_gt_contact_has_zero_loss_for_sparse_markers(self):
        distances = torch.tensor([[[.025, .09, .18]]])
        target = torch.zeros(1, 1, 22); target[..., 10] = 1
        valid = target.bool(); mask = torch.ones_like(distances, dtype=torch.bool)
        loss = region_contact_loss(distances, distances, [10, 10, 10], mask, target, valid)
        self.assertEqual(float(loss), 0.)
        self.assertGreater(float((distances - .02).clamp_min(0).square().mean()), 0.)

    def test_contact_force_only_affects_supported_marker(self):
        gt = torch.tensor([[[.025, .09, .18]]]); prediction = (gt + .1).requires_grad_()
        target = torch.zeros(1, 1, 22); target[..., 10] = 1
        loss = region_contact_loss(prediction, gt, [10, 10, 10], torch.ones_like(gt, dtype=torch.bool), target, target.bool())
        loss.backward()
        self.assertGreater(float(prediction.grad[..., 0]), 0.)
        self.assertEqual(float(prediction.grad[..., 1:].abs().sum()), 0.)

    def test_contact_loss_ignores_unrepresented_contact(self):
        gt = torch.full((1, 1, 3), .20); prediction = (gt + .1).requires_grad_()
        target = torch.zeros(1, 1, 22); target[..., 10] = 1
        loss = region_contact_loss(prediction, gt, [10, 10, 10], torch.ones_like(gt, dtype=torch.bool), target, target.bool())
        loss.backward(); self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(prediction.grad.abs().sum()), 0.)


if __name__ == '__main__': unittest.main()
