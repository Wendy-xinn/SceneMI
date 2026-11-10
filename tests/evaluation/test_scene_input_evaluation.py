"""Scene interventions must preserve camera/GT and avoid same-room donors."""
import unittest
import json
import tempfile
from pathlib import Path

import torch

from scenemi.scene_input_evaluation import CONDITIONS, condition_payload, donor_plan, scene_identity, summarize, recover_prefix


class SceneInputEvaluationTest(unittest.TestCase):
    def test_donors_exclude_same_room_and_family_and_do_not_depend_on_batching(self):
        cases = [dict(case_id=0, dataset='amass', family='a')]
        cases += [dict(case_id=i+1, dataset='egobody', family=f'recording_{i}') for i in range(4)]
        scenes = {1: 'room_a', 2: 'room_a', 3: 'room_b', 4: 'room_c'}
        plan = donor_plan(cases, scenes, 20261104)
        self.assertIsNone(plan[0])
        self.assertEqual(plan, donor_plan(cases, scenes, 20261104))
        for case in cases[1:]:
            donor = plan[case['case_id']]
            self.assertEqual(donor['dataset'], case['dataset'])
            self.assertNotEqual(scenes[donor['case_id']], scenes[case['case_id']])
            self.assertNotEqual(donor['family'], case['family'])
        with self.assertRaises(ValueError):
            donor_plan(cases[:3], scenes, 0)

    def test_only_scene_channels_change_and_invalid_owners_mark_absence(self):
        public = dict(camera=torch.randn(2, 128, 9), body_format=torch.ones(2, dtype=torch.long),
            occupancy=torch.ones(2, 24, 48, 48), frame_scene_points=torch.randn(2, 128, 4, 3),
            frame_scene_owner=torch.zeros(2, 128, 4, dtype=torch.long))
        donor = {key: value + 1 for key, value in public.items() if key.startswith(('occupancy', 'frame_scene'))}
        for condition in ('matched', 'disabled', 'empty', 'mismatched'):
            value = condition_payload(public, condition, donor)
            self.assertIs(value['camera'], public['camera'])
            self.assertIs(value['body_format'], public['body_format'])
        empty = condition_payload(public, 'empty')
        self.assertTrue((empty['frame_scene_owner'] == -1).all())
        self.assertEqual(empty['occupancy'].count_nonzero(), 0)
        self.assertEqual(empty['frame_scene_points'].count_nonzero(), 0)
        self.assertTrue((public['frame_scene_owner'] == 0).all())
        self.assertEqual(set(condition_payload(public, 'disabled')), {'camera', 'body_format'})
        self.assertIs(condition_payload(public, 'mismatched', donor)['occupancy'], donor['occupancy'])

    def test_physical_scene_identity_requires_source_metadata(self):
        self.assertEqual(scene_identity({'dataset': 'egobody'}, {'scene_id': 'cab_e'}), 'egobody:cab_e')
        self.assertEqual(scene_identity({'dataset': 'grail'}, {'object_family_id': 'chair_a'}), 'grail:chair_a')
        with self.assertRaises(ValueError):
            scene_identity({'dataset': 'trumans'}, {})

    def test_summary_uses_paired_equal_dataset_means_and_rejects_missing_windows(self):
        rows = {key: [] for key in ('matched', 'disabled', 'empty', 'mismatched')}
        for index, dataset in enumerate(('amass', 'egobody', 'trumans', 'grail')):
            for condition in rows:
                rows[condition].append(dict(case=dict(case_id=index),
                    identity=dict(dataset=dataset, family='family'),
                    metrics=dict(mpjpe_mm=10. if condition == 'matched' else 11. + index),
                    prediction_change_from_matched_mm=2.))
        value = summarize(rows, 10, 0)
        self.assertEqual(value['delta_from_matched']['mismatched']['dataset_macro_mean']['mpjpe_mm'], 2.5)
        self.assertEqual(value['delta_from_matched']['mismatched']['scene_dataset_macro_mean']['mpjpe_mm'], 3.)
        rows['disabled'].pop()
        with self.assertRaises(ValueError):
            summarize(rows, 10, 0)

    def test_recovery_uses_common_durable_batches_after_reordered_disk_writes(self):
        cases = [dict(case_id=index) for index in range(96)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'STATUS.json').write_text(json.dumps(dict(completed_windows=64)))
            for condition, count in zip(CONDITIONS, (65, 64, 33, 64)):
                (root/f'{condition}.jsonl').write_text(''.join(
                    json.dumps(dict(case=case))+'\n' for case in cases[:count]) + '{"case":')
            count, rows, tails = recover_prefix(root, cases, 32)
            self.assertEqual(count, 32)
            self.assertTrue(all(len(values) == 32 for values in rows.values()))
            self.assertTrue(all(tail.endswith('{"case":') for tail in tails.values()))
            self.assertEqual(rows['matched'][-1]['case'], cases[31])


if __name__ == '__main__':
    unittest.main()
