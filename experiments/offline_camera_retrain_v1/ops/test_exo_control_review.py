"""Frozen EXO review: identity selection, clocks, and disabled physics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from . import exo_control_review as review
from .dependencies import sha256


class ReviewTests(unittest.TestCase):
    def entries(self):
        result = []
        for dataset in review.DATASETS:
            for index in range(5):
                result.append(dict(case=len(result), dataset=dataset, recording=f'{dataset}{index}',
                    sequence_id=f'{dataset}{index}', panel='new55', start=64, case_seed=123 + len(result)))
        result.append(dict(case=len(result), dataset='egobody', recording='regression', panel='old9',
            sequence_id='recording_20220215_S21_S22_01_interactee_1901-4901_master_full', start=1984, case_seed=56))
        return result

    def test_identity_selection_preserves_seeds_and_does_not_rank_scores(self):
        entries = self.entries()
        for index, entry in enumerate(entries):
            entry['irrelevant_error'] = -index
        selected = review.select_entries(entries, 'preflight13')
        self.assertEqual(len(selected), 13)
        self.assertEqual([r['case'] for r in selected], [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15])
        self.assertEqual([r['case_seed'] for r in selected],
            [entries[i]['case_seed'] for i in (0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15)])
        self.assertEqual(selected[-1]['review_panel'], 'regression')
        self.assertNotIn('review_panel', entries[0])

    def test_duplicate_recordings_fail_before_sampling(self):
        entries = self.entries()
        entries.append(dict(entries[0], case=999, start=128))
        with self.assertRaisesRegex(ValueError, 'repeats a recording'):
            review.select_entries(entries)

    def test_control_parent_and_stage_clocks_are_both_required(self):
        config = dict(frames=192, fps=30)
        saved = dict(config=config, step=3000, parent_step=41000, architecture='original', guidance_enabled=False)
        declaration = dict(expected_step=3000, expected_parent_step=41000, architecture='original',
            expected_guidance_enabled=False)
        metadata = review.checkpoint_metadata(saved, declaration, config)
        self.assertEqual(metadata['stage_step'], 3000)
        self.assertEqual(metadata['parent_step'], 41000)
        for changed in (dict(step=44000), dict(parent_step=55000), dict(guidance_enabled=True),
                dict(architecture='scene_cross_attention'), dict(config=dict(frames=192, fps=60))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                review.checkpoint_metadata(dict(saved, **changed), declaration, config)

    def test_historical_base_needs_no_stage_parent_but_exact_source_config(self):
        config = dict(frames=192)
        metadata = review.checkpoint_metadata(dict(config=config, step=41000), dict(expected_step=41000), config)
        self.assertIsNone(metadata['parent_step'])
        with self.assertRaisesRegex(ValueError, 'schemas disagree'):
            review.checkpoint_metadata(dict(config=config, effective_config={'config': {'frames': 1}}, step=41000),
                dict(expected_step=41000), config)

    def test_physics_sentinel_and_switch_restore_on_failure(self):
        from experiments.offline_camera_retrain_v1 import native30_physical_guidance as physics
        class Stub:
            guidance_enabled = True
        model = Stub()
        original = physics.guide_motion
        with self.assertRaisesRegex(RuntimeError, 'unexpectedly called'):
            with review.physics_off(model):
                self.assertFalse(model.guidance_enabled)
                physics.guide_motion(None)
        self.assertTrue(model.guidance_enabled)
        self.assertIs(physics.guide_motion, original)

    def test_resumed_rows_reject_changed_seed_source_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            artifact = out / 'prediction.npz'
            artifact.write_bytes(b'fixture')
            entry = dict(case=1, dataset='rich', recording='r', sequence_id='s', start=64, source_first=65, case_seed=7)
            model = dict(checkpoint_sha256='a' * 64, expected_step=3000)
            plan = dict(entries=[entry], model_order=['control3000'], models={'control3000': model},
                regimes=['clean'], reference_policy=review.reference_temporal.POLICY)
            row = dict(entry, regime='clean', model='control3000', seed=7,
                policy=plan['reference_policy'], analytical_physics=False, use_scene=True,
                checkpoint_sha256=model['checkpoint_sha256'], checkpoint_step=3000,
                artifact=str(artifact), artifact_sha256=sha256(artifact),
                input_sha256='paired', counts={'visible_joint_frames': 100})
            def saved(value):
                (out / 'rows.jsonl').write_text(json.dumps(value) + '\n')
            saved(row)
            self.assertEqual(len(review.rows_for(out, plan)), 1)
            for change in (dict(seed=8), dict(source_first=66), dict(checkpoint_step=41000),
                    dict(analytical_physics=True), dict(checkpoint_sha256='b' * 64)):
                saved(dict(row, **change))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    review.rows_for(out, plan)
            saved(row)
            artifact.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                review.rows_for(out, plan)

    def test_resume_rejects_different_recursive_inputs_and_missing_masks_within_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            artifact = out / 'prediction.npz'
            artifact.write_bytes(b'fixture')
            entry = dict(case=1, dataset='rich', recording='r', sequence_id='s', start=64, source_first=65, case_seed=7)
            declaration = dict(checkpoint_sha256='a' * 64, expected_step=3000)
            plan = dict(entries=[entry], model_order=['control3000'], models={'control3000': declaration},
                regimes=['clean'], reference_policy=review.reference_temporal.POLICY)
            common = dict(entry, regime='clean', seed=7, policy=plan['reference_policy'],
                analytical_physics=False, use_scene=True, artifact=str(artifact), artifact_sha256=sha256(artifact),
                input_sha256='paired', counts={'visible_joint_frames': 100})
            first = dict(common, model=review.REFERENCE)
            second = dict(common, model='control3000', checkpoint_sha256='a' * 64, checkpoint_step=3000)
            for change in ({}, dict(input_sha256='different'), dict(counts={'visible_joint_frames': 99})):
                (out / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in (first, dict(second, **change))) + '\n')
                if not change:
                    self.assertEqual(len(review.rows_for(out, plan)), 2)
                else:
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Paired'):
                        review.rows_for(out, plan)


if __name__ == '__main__':
    unittest.main()
