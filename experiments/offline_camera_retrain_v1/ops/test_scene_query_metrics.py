"""Synthetic counterexamples for query diagnostics, without model/data access."""
import copy
import json
import unittest

import numpy as np

from .scene_query_metrics import (SCHEMA, aggregate_query_steps, alpha_bar_bin,
                                  summarize_query_step)


class SceneQueryMetricsTests(unittest.TestCase):
    def inputs(self, frames=4, anchors=2):
        proposal = np.zeros((frames, 22, 3), np.float64)
        centres = np.zeros((frames, anchors, 3), np.float64)
        points = np.zeros((frames, anchors, 3, 2, 3), np.float64)
        points[..., 1] = -.1
        return dict(timestep=100, alpha_bar=.9, proposal_joints_m=proposal,
                    query_centres_m=centres, surface_points_m=points,
                    surface_valid=np.ones(points.shape[:-1], bool),
                    surface_owners=np.zeros(points.shape[:-1], np.int64),
                    surface_normal_reliable=np.ones(points.shape[:-1], bool),
                    window_id='one/clean')

    def test_common_translation_is_removed_from_point_centre_residual(self):
        inputs = self.inputs()
        motion = np.arange(4) * .01
        inputs['query_centres_m'][..., 0] = motion[:, None]
        inputs['surface_points_m'][..., 0] = motion[:, None, None, None]
        inputs['proposal_joints_m'][..., 0] = motion[:, None]
        result = summarize_query_step(**inputs)
        metrics = result['metrics']
        self.assertAlmostEqual(metrics['selected_point_displacement_m']['mean'], .01)
        self.assertAlmostEqual(metrics['query_centre_speed_m_s']['mean'], .3)
        self.assertEqual(metrics['selected_point_minus_centre_displacement_m']['max'], 0.)
        self.assertEqual(metrics['selected_point_minus_centre_displacement_m']['count'], 6)
        self.assertAlmostEqual(metrics['proposal_foot_horizontal_speed_m_s']['mean'], .3)
        self.assertEqual(metrics['proposal_joint_speed_m_s']['count'], 3 * 22)

    def test_fixed_surface_can_have_nonzero_compensated_motion_without_switch(self):
        inputs = self.inputs()
        inputs['query_centres_m'][..., 0] = (np.arange(4) * .01)[:, None]
        result = summarize_query_step(**inputs)
        self.assertEqual(result['metrics']['selected_point_displacement_m']['mean'], 0.)
        self.assertAlmostEqual(result['metrics']['selected_point_minus_centre_displacement_m']['mean'], .01)
        self.assertEqual(result['fractions']['selected_owner_changed_adjacent_frames']['fraction'], 0.)
        self.assertIn('not a contact-switch detector', result['interpretation']['compensated'])

    def test_default_scale_and_neighbor_match_native_closest_slot(self):
        inputs = self.inputs(anchors=67)
        inputs['surface_points_m'][..., 1] = 0.
        inputs['surface_points_m'][:, :, 2, 0, 0] = .6
        inputs['surface_points_m'][:, :, 0, 0, 0] = .08
        inputs['surface_points_m'][:, :, 2, 1, 0] = .7
        result = summarize_query_step(**inputs)
        self.assertEqual(result['metrics']['selected_neighbor_distance_m']['count'], 4 * 67)
        self.assertAlmostEqual(result['metrics']['selected_neighbor_distance_m']['mean'], .6)
        self.assertEqual(result['metrics']['proposal_joint_speed_m_s']['count'], 3 * 22)
        self.assertEqual(result['shapes']['query_centres'], [4, 67, 3])
        self.assertEqual(result['selected_surface_slot']['scale_index'], 2)

    def test_rank_zero_invalid_does_not_substitute_another_neighbor(self):
        inputs = self.inputs()
        inputs['surface_valid'][:, :, 2, 0] = False
        inputs['surface_points_m'][:, :, 2, 0] = np.nan
        result = summarize_query_step(**inputs)
        self.assertEqual(result['metrics']['selected_neighbor_distance_m']['count'], 0)
        self.assertIsNone(result['metrics']['selected_neighbor_distance_m']['mean'])
        self.assertEqual(result['fractions']['invalid_selected_neighbor']['fraction'], 1.)
        self.assertGreater(result['metrics']['all_returned_neighbor_distance_m']['count'], 0)

    def test_empty_invalid_nan_padding_and_single_frame_have_none_metrics(self):
        inputs = self.inputs(frames=1)
        inputs['surface_valid'][:] = False
        inputs['surface_points_m'][:] = np.nan
        result = summarize_query_step(**inputs)
        self.assertIsNone(result['metrics']['selected_neighbor_distance_m']['mean'])
        self.assertEqual(result['fractions']['invalid_neighbor_slots']['fraction'], 1.)
        self.assertIsNone(result['fractions']['reliable_normal_among_valid_neighbors']['fraction'])
        self.assertEqual(result['metrics']['proposal_joint_speed_m_s']['count'], 0)
        self.assertIsNone(result['metrics']['proposal_joint_speed_m_s']['mean'])
        self.assertIsNone(result['fractions']['selected_owner_changed_adjacent_frames']['fraction'])
        json.dumps(result, allow_nan=False)

    def test_masks_exclude_invalid_endpoints_from_temporal_denominators(self):
        inputs = self.inputs()
        inputs['query_valid'] = np.ones((4, 2), bool)
        inputs['query_valid'][1, 0] = False
        inputs['query_centres_m'][1, 0] = np.nan
        inputs['surface_valid'][2, 1, 2, 0] = False
        inputs['surface_normal_reliable'][:, :, 2, 0] = False
        inputs['surface_normal_reliable'][0, 0, 2, 0] = True
        result = summarize_query_step(**inputs)
        self.assertEqual(result['counts']['valid_query_frames'], 7)
        self.assertEqual(result['counts']['valid_adjacent_query_pairs'], 4)
        self.assertEqual(result['counts']['valid_adjacent_selected_point_pairs'], 2)
        normal = result['fractions']['reliable_normal_among_valid_selected_neighbors']
        self.assertEqual((normal['numerator'], normal['denominator']), (1, 6))
        self.assertEqual(result['fractions']['invalid_selected_neighbor']['denominator'], 7)

    def test_owner_transitions_are_descriptive_and_static_pairs_separate(self):
        inputs = self.inputs(anchors=1)
        inputs['surface_owners'][2:, :, 2, 0] = 4
        result = summarize_query_step(**inputs)
        change = result['fractions']['selected_owner_changed_adjacent_frames']
        self.assertEqual((change['numerator'], change['denominator']), (1, 3))
        self.assertEqual(result['metrics']['selected_point_minus_centre_displacement_same_owner_m']['count'], 2)
        self.assertEqual(result['metrics']['selected_point_minus_centre_displacement_static_owner_m']['count'], 1)

    def test_gt_only_adds_posthoc_error_and_low_speed_masks(self):
        inputs = self.inputs()
        inputs['proposal_joints_m'][..., 0] = (np.arange(4) * .01)[:, None]
        before = copy.deepcopy(inputs)
        without = summarize_query_step(**inputs)
        with_gt = summarize_query_step(**inputs, gt_joints_m=np.zeros((4, 22, 3)))
        for name, metric in without['metrics'].items():
            self.assertEqual(metric, with_gt['metrics'][name])
        self.assertEqual(without['fractions'], with_gt['fractions'])
        self.assertAlmostEqual(with_gt['metrics']['proposal_joint_error_m']['mean'], .015)
        self.assertAlmostEqual(with_gt['metrics']['proposal_foot_horizontal_speed_gt_low_m_s']['mean'], .3)
        self.assertEqual(with_gt['metrics']['proposal_foot_horizontal_speed_gt_low_m_s']['count'], 6)
        for name, original in before.items():
            if isinstance(original, np.ndarray): np.testing.assert_array_equal(original, inputs[name])

    def test_gt_moving_feet_have_no_low_speed_opportunities(self):
        inputs = self.inputs()
        gt = np.zeros((4, 22, 3))
        gt[..., 1] = np.arange(4)[:, None]  # GT 3D, not horizontal, controls mask.
        result = summarize_query_step(**inputs, gt_joints_m=gt)
        self.assertIsNone(result['metrics']['proposal_foot_horizontal_speed_gt_low_m_s']['mean'])
        self.assertEqual(result['metrics']['proposal_foot_horizontal_speed_gt_low_m_s']['count'], 0)

    def test_selected_scale_and_selected_point_layouts_agree(self):
        inputs = self.inputs()
        full = summarize_query_step(**inputs)
        for name in ('surface_points_m', 'surface_valid', 'surface_owners', 'surface_normal_reliable'):
            inputs[name] = inputs[name][:, :, 2]
        scale = summarize_query_step(**inputs)
        for name in ('surface_points_m', 'surface_valid', 'surface_owners', 'surface_normal_reliable'):
            inputs[name] = inputs[name][:, :, 0]
        point = summarize_query_step(**inputs)
        metric = 'selected_neighbor_distance_m'
        self.assertEqual(full['metrics'][metric], scale['metrics'][metric])
        self.assertEqual(full['metrics'][metric], point['metrics'][metric])
        self.assertTrue(point['selected_surface_slot']['supplied_already_selected'])
        self.assertIsNone(point['selected_surface_slot']['neighbor_index'])

    def test_optional_features_respect_same_ordered_query_mask(self):
        inputs = self.inputs()
        features = np.repeat(np.arange(4)[:, None, None], 2, axis=1) * np.ones((1, 1, 4))
        mask = np.ones((4, 2), bool)
        mask[1, 0] = False
        features[1, 0] = np.nan
        result = summarize_query_step(**inputs, feature_values=features, feature_valid=mask)
        self.assertEqual(result['metrics']['feature_adjacent_frame_l2_difference']['mean'], 2.)
        self.assertEqual(result['metrics']['feature_adjacent_frame_rms_difference']['mean'], 1.)
        self.assertEqual(result['metrics']['feature_adjacent_frame_rms_difference']['count'], 4)

    def test_alpha_bins_include_endpoints_and_reject_bad_edges(self):
        self.assertEqual([alpha_bar_bin(x)['index'] for x in (0., .0999, .1, .5, .9, 1.)], [0, 0, 1, 2, 3, 3])
        self.assertTrue(alpha_bar_bin(1.)['upper_inclusive'])
        for value in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError): alpha_bar_bin(value)
        for edges in ((.1, 1.), (0., .5, .5, 1.), (0., np.nan, 1.)):
            with self.assertRaises(ValueError): alpha_bar_bin(.5, edges)

    def test_aggregation_weights_opportunities_not_steps_or_quantiles(self):
        first = self.inputs(frames=2, anchors=1)
        first['surface_points_m'][..., 1] = -.2
        second = self.inputs(frames=4, anchors=1)
        second['surface_points_m'][..., 1] = -.6
        rows = [summarize_query_step(**first), summarize_query_step(**second)]
        result = aggregate_query_steps(rows)
        group = result['groups'][0]
        metric = group['metrics']['selected_neighbor_distance_m']
        self.assertEqual(metric['count'], 6)
        self.assertAlmostEqual(metric['mean'], (2 * .2 + 4 * .6) / 6)
        self.assertEqual(group['steps'], 2)
        self.assertEqual(group['unique_named_windows'], 1)
        self.assertNotIn('p90', metric)
        self.assertEqual(result['schema'], SCHEMA)
        json.dumps(result, allow_nan=False)

    def test_invalid_shapes_valid_nans_and_wrong_owners_fail_closed(self):
        modifications = [dict(query_centres_m=np.zeros((3, 2, 3))),
                         dict(surface_valid=np.ones((4, 2), bool)),
                         dict(surface_owners=np.zeros((4, 2, 3, 2), float)),
                         dict(scale_index=3), dict(neighbor_index=2),
                         dict(foot_indices=(10, 10)), dict(fps=0),
                         dict(feature_valid=np.ones((4, 2), bool))]
        for modification in modifications:
            with self.subTest(modification=tuple(modification)):
                inputs = self.inputs(); inputs.update(modification)
                with self.assertRaises(ValueError): summarize_query_step(**inputs)
        inputs = self.inputs(); inputs['surface_points_m'][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): summarize_query_step(**inputs)


if __name__ == '__main__':
    unittest.main()
