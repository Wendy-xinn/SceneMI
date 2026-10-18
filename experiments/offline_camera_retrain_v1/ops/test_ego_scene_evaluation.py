"""CPU checks for recording coverage, held-out panels and independent seeds."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import ego_scene_evaluation as evaluation


def source_entries():
    entries = []
    for dataset, count in [('rich', 27), ('trumans', 10), ('egobody', 10)]:
        for i in range(count):
            group = ('camera_wearer' if i % 2 == 0 else 'interactee') if dataset == 'egobody' else dataset
            base = dict(dataset=dataset, recording=f'{dataset}_{i}', group=group,
                sequence_id=f'{dataset}_{i}_role', source_first=64 * i,
                path=f'/validation/{dataset}_{i}', start=64 * i,
                case=len(entries), case_seed=1000 + len(entries), mode='ego', joints=[15])
            entries.append(base)
            entries.append(dict(base, start=base['start'] + 64, source_first=base['source_first'] + 64,
                                case=len(entries), case_seed=2000 + len(entries)))
    return entries


class EvaluationTests(unittest.TestCase):
    def test_recording_coverage_and_panels_are_identity_only(self):
        source = source_entries(); original = copy.deepcopy(source)
        selected = evaluation.select_cohort(source)
        self.assertEqual(source, original)
        self.assertEqual(len(selected), 43)
        self.assertEqual({d: sum(e['dataset'] == d for e in selected)
            for d in ['rich', 'trumans', 'egobody']}, {'rich': 27, 'trumans': 8, 'egobody': 8})
        self.assertEqual(sum(e['review_panel'] == 'dev' for e in selected), 12)
        self.assertEqual(len({evaluation.recording_key(e) for e in selected}), 43)
        self.assertEqual(selected, evaluation.select_cohort(source))
        for e in selected:
            self.assertEqual(e['case'], next(x['case'] for x in source if x['recording'] == e['recording']))

    def test_seed_indices_change_diffusion_but_preserve_observations(self):
        cohort = dict(entries=evaluation.select_cohort(source_entries()))
        first, second = [evaluation.entries_for(cohort, 'all', i) for i in [0, 1]]
        self.assertEqual([e['observation_seed'] for e in first], [e['observation_seed'] for e in second])
        self.assertTrue(all(a['case_seed'] != b['case_seed'] for a, b in zip(first, second)))
        self.assertEqual(len(evaluation.entries_for(cohort, 'dev', 0)), 12)
        self.assertEqual(len(evaluation.entries_for(cohort, 'confirmation', 0)), 31)
        self.assertTrue(all(0 <= e['case_seed'] < 2**32 for e in first + second))

    def test_observation_builder_receives_fixed_observation_seed(self):
        selected = evaluation.entries_for(dict(entries=evaluation.select_cohort(source_entries())), 'dev', 1)[0]
        with patch.object(evaluation.review.common, 'source_observation', side_effect=lambda d, b, e, r: e) as original:
            before = evaluation.review.common.source_observation
            with evaluation.evaluator():
                observed = evaluation.review.common.source_observation(None, None, selected, 'noisy')
                self.assertEqual(observed['case_seed'], selected['observation_seed'])
            self.assertIs(evaluation.review.common.source_observation, before)
            self.assertEqual(original.call_count, 1)

    def test_split_audit_records_short_rich_exclusion(self):
        selected = evaluation.select_cohort(source_entries())
        manifest = [dict(e, split='validation', frames=300) for e in selected]
        manifest.append(dict(dataset='rich', recording='short', sequence_id='short', group='rich', path='/short', split='validation', frames=165))
        manifest.append(dict(dataset='egobody', recording='train_only', sequence_id='train_role', group='interactee', path='/train', split='train', frames=300))
        proof = evaluation.split_proof(manifest, selected, [e['recording'] for e in selected if e['dataset'] == 'rich'])
        self.assertEqual(proof['rich_exclusions'], [dict(recording='short', maximum_source_frames=165, reason='shorter_than_L192')])
        self.assertEqual(proof['TRAIN_VAL_recording_overlap'], [])

    def test_cross_role_train_validation_leak_is_rejected(self):
        selected = evaluation.select_cohort(source_entries())
        manifest = [dict(e, split='validation', frames=300) for e in selected]
        shared = next(e for e in selected if e['dataset'] == 'egobody')
        manifest.append(dict(shared, group='interactee', sequence_id='another_role', path='/train', split='train', frames=300))
        with self.assertRaisesRegex(ValueError, 'share recordings'):
            evaluation.split_proof(manifest, selected, [e['recording'] for e in selected if e['dataset'] == 'rich'])

    def test_retired_attention_checkpoints_and_architecture_fail_before_loading(self):
        for declaration in [dict(architecture='scene_cross_attention', checkpoint='/not-opened.pt'),
                dict(architecture='original', checkpoint='/runs/native30_ego_scene_attention_trial/control/training/last.pt')]:
            with self.assertRaises(ValueError): evaluation.check_declaration(declaration)
        evaluation.check_declaration(dict(architecture='scene_fusion_v2', checkpoint='/new/fusion.pt'))

    def test_fusion_dimension_and_options_must_match_saved_metadata(self):
        options = dict(token_dim=64, heads=4, fusion_dim=96)
        saved = dict(fusion_options=options, fusion_spec=dict(options, include_global=True, include_local=True))
        self.assertEqual(evaluation.fusion_options(saved)['fusion_dim'], 96)
        changed = copy.deepcopy(saved); changed['fusion_options']['fusion_dim'] = 128
        with self.assertRaisesRegex(ValueError, 'disagree'):
            evaluation.fusion_options(changed)
        changed = copy.deepcopy(saved); del changed['fusion_options']['fusion_dim']
        with self.assertRaisesRegex(ValueError, 'explicit'):
            evaluation.fusion_options(changed)

    def test_atomic_last_replacement_keeps_pinned_inode_and_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); source = folder / 'last.pt'; source.write_bytes(b'checkpoint1000')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            declaration = dict(architecture='original', checkpoint=str(source), checkpoint_sha256=digest)
            pinned = evaluation.pin_models(folder / 'evaluation', dict(control=declaration))['control']
            target = Path(pinned['checkpoint'])
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
            next_file = folder / 'next.tmp'; next_file.write_bytes(b'checkpoint3000'); next_file.replace(source)
            self.assertEqual(target.read_bytes(), b'checkpoint1000')
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(pinned['source_checkpoint_sha256'], digest)

    def test_recording_paired_bootstrap_is_deterministic(self):
        result = evaluation.paired_bootstrap([8., 18., 28.], [10., 20., 30.])
        self.assertEqual(result['confidence_interval_95pct_mm'], [-2., -2.])
        self.assertEqual(result['mean_delta_mm'], -2.)
        self.assertEqual(result, evaluation.paired_bootstrap([8., 18., 28.], [10., 20., 30.]))

    def test_final_decision_requires_both_comparators_and_head_guard(self):
        entries = evaluation.select_cohort(source_entries())
        models = dict(base41k=dict(architecture='original', expected_step=41000),
            control=dict(architecture='original', expected_step=3000, stage_plan_sha256='paired'),
            fusion=dict(architecture='scene_fusion_v2', expected_step=3000, stage_plan_sha256='paired'))
        plan = dict(entries=entries, evaluation_panel='all', models=models, regimes=['clean', 'noisy'])
        def rows(head_error):
            output = []
            for e in entries:
                for regime in plan['regimes']:
                    for label, value in [('base41k', 100.), ('control', 90.), ('fusion', 85.)]:
                        metrics = dict(w_mpjpe_mm=value, pa_mpjpe_mm=value,
                            foot_joint_horizontal_speed_gt_low_m_s=value / 1000,
                            foot_joint_y_abs_error_mm=value, acceleration_error_mm_s2=value * 10,
                            observed_head_position_error_mm=head_error if label == 'fusion' else .1)
                        output.append(dict(case=e['case'], regime=regime, model=label, arm='learned_only',
                            input_sha256='identical', counts={'same': 1}, calls={'guide_motion': 0}, metrics=metrics))
            return output
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); acceptance = folder / 'ACCEPTANCE.json'
            original = evaluation.read(evaluation.review.BASE / 'runs/native30_ego_scene_fusion_v2/ACCEPTANCE.json')
            original['fusion_acceptance']['observed_head_position_error_maximum_regression_mm'] = 1.
            acceptance.write_text(json.dumps(original))
            plan['preregistered_acceptance'] = dict(sha256=hashlib.sha256(acceptance.read_bytes()).hexdigest())
            directories = [folder / 'seed0', folder / 'seed1']
            for directory in directories:
                directory.mkdir()
                for name in ['PLAN.json', 'rows.jsonl', 'CPU_RECHECK.json']: (directory / name).write_text('{}')
            for head, expected in [(.2, 'pass'), (2., 'reject')]:
                sources = [(plan, rows(head), directory) for directory in directories]
                with patch.object(evaluation, 'verified_seed_outputs', return_value=sources):
                    result = evaluation.decision(folder / expected, directories, acceptance)
                self.assertEqual(result['status'], expected)
                if head == 2.:
                    self.assertTrue(any('observed_head' in name
                        for name in result['comparisons']['control']['failed_checks']))


if __name__ == '__main__': unittest.main()
