"""Decision invariants: no confirmation leakage, natural weighting, guards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import scene_policy_review as review


class PolicyReviewTests(unittest.TestCase):
    def rows(self):
        result = []
        for label in ('original', 'patch', 'both'):
            for case in range(12):
                dataset = review.DATASETS[case // 4]
                for regime in review.REGIMES:
                    value = 500. if label == 'original' and case == 1 else 100.
                    if label == 'patch':
                        value = 20. if dataset == 'rich' else 100.
                    if label == 'both':
                        value = 70. if dataset == 'rich' else 90.
                    metrics = dict(hidden_mpjpe_mm=value, visible_mpjpe_mm=2., mpjpe_mm=value,
                        foot_joint_speed_gt_low_m_s=.1, mpjve_mm_s=20., acceleration_error_mm_s2=1000.)
                    result.append(dict(model=label, case=case, dataset=dataset, regime=regime,
                        review_panel='dev', sequence_id='Pavallion_006_yoga1' if case == 1 else str(case),
                        start=640 if case == 1 else 0, metrics=metrics,
                        reference_metrics=dict(metrics, hidden_mpjpe_mm=80. if case == 1 else 100.)))
        return result

    def assess(self, rows, label):
        checks = [dict(model=label, case=case, regime=regime, exact=True)
                  for case in (100, 101, 102) for regime in review.REGIMES]
        return review.assess(rows, dict(rich=.1, trumans=.4, egobody=.5),
                             'original', label, 'dev', ego_checks=checks)

    def test_natural_primary_can_choose_opposite_equal_dataset_macro(self):
        rows = self.rows()
        results = {label: self.assess(rows, label) for label in ('patch', 'both')}
        self.assertTrue(all(item['passed'] for item in results.values()))
        self.assertLess(results['patch']['candidate_summary']['equal_dataset_secondary']['combined'],
                        results['both']['candidate_summary']['equal_dataset_secondary']['combined'])
        self.assertEqual(review.choose(results, 'patch'), 'both')

    def test_dev_reader_skips_confirmation_before_json_metrics_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            good = dict(model='original', review_panel='dev', scene_variant='scene_on', metrics={})
            poison = '{"model":"both","review_panel":"confirmation","metrics":DO_NOT_DECODE}\n'
            (out / 'rows.jsonl').write_text(json.dumps(good) + '\n' + poison)
            self.assertEqual(review.phase_rows(out, 'dev', ['original', 'both']), [good])

    def test_visible_regression_and_missing_ego_fail_even_when_hidden_improves(self):
        rows = self.rows()
        for row in rows:
            if row['model'] == 'patch' and row['dataset'] == 'trumans':
                row['metrics']['visible_mpjpe_mm'] = 4.
        result = self.assess(rows, 'patch')
        self.assertFalse(result['passed'])
        self.assertTrue(any('visible_mpjpe_mm' in c['name'] for c in result['failed_checks']))
        result = review.assess(self.rows(), dict(rich=.1, trumans=.4, egobody=.5),
                               'original', 'patch', 'dev', ego_checks=[])
        self.assertFalse(result['passed'])
        self.assertTrue(any(c['name'] == 'ego6_exact' for c in result['failed_checks']))

    def test_no_eligible_candidate_never_selects_runner_up(self):
        results = {label: self.assess(self.rows(), label) for label in ('patch', 'both')}
        for item in results.values():
            item['passed'] = False
        self.assertIsNone(review.choose(results, 'patch'))

    def test_small_score_tie_prefers_simple_patch_policy(self):
        results = {label: self.assess(self.rows(), label) for label in ('patch', 'both')}
        results['patch']['candidate_summary']['natural_primary']['combined'] = 80.5
        results['both']['candidate_summary']['natural_primary']['combined'] = 80.
        self.assertEqual(review.choose(results, 'patch'), 'patch')

    def test_confirmation_refuses_read_before_dev_choice(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Freeze dev choice'):
                review.confirmation_report(folder)

    def test_freeze_supports_a_single_candidate_without_legacy_patch_label(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for filename in ('SELECTION.json', 'ACCEPTANCE.json'):
                (out / filename).write_text('{}\n')
            labels = dict(original='original', exo_patch_second_pass='exo_patch_second_pass')
            rules = dict(baseline_policy='original', candidate_policies=['exo_patch_second_pass'])
            result = dict(passed=True, candidate_summary=dict(natural_primary=dict(combined=80.)))
            with patch.object(review, 'binding', return_value=({}, rules, {}, labels)), \
                 patch.object(review, 'phase_rows', return_value=[]), \
                 patch.object(review, 'expected_rows'), \
                 patch.object(review, 'cpu_verify', return_value=dict(EGO_retention=[])), \
                 patch.object(review, 'assess', return_value=result):
                choice = review.freeze_dev_choice(out)
            self.assertEqual(choice['selected_policy'], 'exo_patch_second_pass')
            self.assertEqual(choice['selected_label'], 'exo_patch_second_pass')
            self.assertFalse(choice['confirmation_metrics_read'])


if __name__ == '__main__':
    unittest.main()
