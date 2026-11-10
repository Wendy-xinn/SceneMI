"""Paired scene diagnostics must preserve observations, identity and weighting."""
import unittest
from unittest.mock import Mock, patch

from scenemi.data.local_scene import LOCAL_INPUT_KEYS
from scenemi.scene_evaluation import check_parity, paired_summary, predict, without_scene


class SceneEvaluationTests(unittest.TestCase):
    def test_off_keeps_camera_and_does_not_mutate_original_payload(self):
        camera = object()
        payload = dict(camera=camera, camera_intrinsics=object(), body_format=object(),
                       occupancy=object(), frame_scene_points=object(), frame_scene_owner=object(),
                       local_history_points=object(), local_history_valid=object(),
                       local_frame_points=object(), local_frame_owner=object())
        off = without_scene(payload)
        self.assertIs(off['camera'], camera)
        self.assertEqual(set(off), {'camera', 'camera_intrinsics', 'body_format'})
        self.assertIn('occupancy', payload)

    def test_family_and_dataset_weights_do_not_depend_on_window_count(self):
        def row(dataset, family, value):
            return dict(identity=dict(dataset=dataset, corpus=dataset, family=family), metrics=dict(mpjpe_mm=value))
        on = [row('egobody', 'a', 10)]*3+[row('egobody', 'b', 30),
              row('trumans', 't', 50), row('grail', 'g', 80)]
        off = [dict(identity=r['identity'], metrics=dict(mpjpe_mm=r['metrics']['mpjpe_mm']+12)) for r in on]
        summary = paired_summary(on, off)
        self.assertEqual(summary['per_dataset']['egobody']['scene_on']['family_macro_mean']['mpjpe_mm'], 20)
        self.assertEqual(summary['dataset_macro']['scene_on']['mpjpe_mm'], 50)
        self.assertEqual(summary['dataset_macro']['off_minus_on']['mpjpe_mm'], 12)

    def test_reordered_or_missing_pairs_are_rejected(self):
        a = dict(identity=dict(dataset='egobody', corpus='egobody', family='a'), metrics=dict(mpjpe_mm=1))
        b = dict(identity=dict(dataset='trumans', corpus='trumans', family='b'), metrics=dict(mpjpe_mm=1))
        with self.assertRaises(ValueError):
            paired_summary([a, b], [b, a])
        with self.assertRaises(ValueError):
            paired_summary([a], [])

    def test_saved_scene_on_metrics_require_finite_reproduction(self):
        check_parity(dict(mpjpe_mm=123.000001), dict(mpjpe_mm=123))
        for value in (124, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                check_parity(dict(mpjpe_mm=value), dict(mpjpe_mm=123))

    def test_local_summary_preserves_predeclared_action_weights(self):
        def row(dataset, family, value, category=None):
            return dict(identity=dict(dataset=dataset, corpus=dataset, family=family,
                                      category=category or dataset), metrics=dict(mpjpe_mm=value))
        on = [row('egobody', 'e', 10), row('trumans', 't', 20)]
        on += [row('grail', 'p1_a', 10, 'stair_p1')]*9
        on += [row('grail', 'p1_b', 10, 'stair_p1')]
        on += [row('grail', cat, value, cat) for cat,value in
               [('stair_p2', 100), ('sitting', 40), ('curb', 50), ('slope', 60)]]
        off = [dict(identity=r['identity'], metrics=dict(mpjpe_mm=r['metrics']['mpjpe_mm']+2)) for r in on]
        result = paired_summary(on, off, category_balanced=True)
        self.assertAlmostEqual(result['grail_category_macro']['scene_on']['mpjpe_mm'], 52)
        self.assertAlmostEqual(result['dataset_macro']['scene_on']['mpjpe_mm'], (10+20+52)/3)
        self.assertAlmostEqual(result['dataset_macro']['off_minus_on']['mpjpe_mm'], 2)
        self.assertEqual(result['grail_per_category']['stair_p1']['scene_on']['windows'], 10)

    def test_local_scene_off_omits_geometry_and_labels_in_both_calls(self):
        batch = {key:object() for key in LOCAL_INPUT_KEYS}
        batch.update(target_rest=object(), target_template=object(), target_native_betas=object())
        model = Mock(return_value=(None, dict(rest=object())))
        with patch('scenemi.scene_evaluation.sample_native30', return_value=object()) as sample, \
             patch('scenemi.scene_evaluation.forward_kinematics', return_value=('joints',)):
            result = predict(model, batch, use_scene=False, seed=777, steps=20, device='cpu', local=True)
        self.assertEqual(result, 'joints')
        for payload,kwargs in [(sample.call_args.args[1],sample.call_args.kwargs),
                               (model.call_args.args[2],model.call_args.kwargs)]:
            self.assertFalse(kwargs['use_scene'])
            self.assertIs(payload['camera'],batch['camera'])
            self.assertNotIn('target_rest',payload)
            self.assertNotIn('target_native_betas',payload)
            self.assertNotIn('occupancy',payload)
            self.assertFalse(any(key.startswith('local_') for key in payload))


if __name__ == '__main__':
    unittest.main()
