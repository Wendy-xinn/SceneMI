"""Selection and instrumentation invariants for the reusable EGO review."""
import unittest
from unittest.mock import patch

import torch

from . import ego_scene_review as review
from experiments.offline_camera_retrain_v1 import native30_physical_guidance as physics


class ReviewTests(unittest.TestCase):
    def test_identity_prefix_deduplicates_and_separates_panels(self):
        entries = []
        for dataset in review.DATASETS:
            for index in range(5):
                for window in range(2):
                    entries.append(dict(dataset=dataset, recording=f'{dataset}{index}',
                        case=len(entries), start=window * 64))
        selected = review.select_entries(entries, 4)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({(r['dataset'], r['recording']) for r in selected}), 12)
        self.assertTrue(all(r['start'] == 0 for r in selected))
        self.assertEqual(sum(r['review_panel'] == 'dev' for r in selected), 6)
        self.assertEqual(sum(r['review_panel'] == 'confirmation' for r in selected), 6)
        self.assertEqual(selected, review.select_entries(entries, 4))

    def test_missing_recordings_cannot_silently_reduce_panel(self):
        entries = [dict(dataset=dataset, recording=dataset, case=index)
            for index, dataset in enumerate(review.DATASETS)]
        with self.assertRaisesRegex(ValueError, 'Insufficient distinct'):
            review.select_entries(entries, 4)

    def test_diagnostic_distinguishes_near_and_far_context(self):
        frames, anchors = 3, 67
        labels = torch.arange(anchors) % 22
        valid = torch.zeros(frames, anchors, 3, 8, dtype=torch.bool)
        valid[0, 0, 0, 0] = True
        relation = torch.zeros(frames, 22, dtype=torch.bool)
        relation[:, 5] = True
        details = dict(geometry=[dict(labels=labels, valid=valid, relation_valid=relation)],
            root_delta_m=torch.zeros(1, frames, 3), rotation_delta_rad=torch.zeros(1, frames, 22, 3),
            contact_logits=torch.zeros(1, frames, 22), correspondence_logits=torch.zeros(1, frames, 22))
        result = review.stage_diagnostic(torch.tensor([100]), details)
        self.assertAlmostEqual(result['near_anchor_fraction'], 1 / (frames * anchors), places=7)
        self.assertAlmostEqual(result['near_region_fraction'], 1 / (frames * 22), places=7)
        self.assertAlmostEqual(result['far_only_region_fraction'], 1 / 22, places=7)
        self.assertEqual(result['contact_probability_mean'], .5)
        self.assertEqual(result['physical_accepted_steps'], 0)

    def test_no_scene_diagnostic_does_not_claim_no_scene_evidence(self):
        result = review.stage_diagnostic(torch.tensor([999]), dict(geometry=[]))
        self.assertFalse(result['geometry_queried'])
        self.assertIsNone(result['near_anchor_fraction'])
        self.assertIsNone(result['far_only_region_fraction'])

    def test_arm_control_preserves_outputs_and_restores_guidance(self):
        class Stub:
            guidance_enabled = True
            def scene_tokens(self, *args):
                return None
            def local_geometry(self, *args):
                return None
            def forward(self, value, timestep, known, use_scene=True, return_details=False):
                details = dict(geometry=[])
                if use_scene:
                    self.scene_tokens(known)
                    self.local_geometry(known)
                if use_scene and self.guidance_enabled:
                    value, _ = physics.guide_motion(value, timestep, known, details, {}, None)
                    details['physical_accepted_steps'] = torch.tensor([1.])
                return (value, details) if return_details else value
        model = Stub()
        value, timestep = torch.ones(1, 3, 201), torch.tensor([200])
        original_forward = model.forward
        with patch.object(physics, 'guide_motion', lambda *args: (args[0], {})):
            for arm in review.ARMS:
                with review.instrumented_arm(model, arm) as (stages, calls, guards):
                    result = model.forward(value, timestep, {}, use_scene=arm != 'none')
                self.assertTrue(torch.equal(result, value))
                self.assertEqual(len(stages), 1)
                self.assertEqual(calls['guide_motion'], int(arm == 'full'))
                self.assertTrue(model.guidance_enabled)
                self.assertEqual(model.forward, original_forward)
                if arm == 'none':
                    self.assertEqual(guards, dict(scene_tokens=0, local_geometry=0, guide_motion=0))


if __name__ == '__main__':
    unittest.main()
