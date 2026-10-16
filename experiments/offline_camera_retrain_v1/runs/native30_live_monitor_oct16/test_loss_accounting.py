import unittest

from loss_accounting import native_contributions


class LossAccountingTests(unittest.TestCase):
    def test_three_weighted_groups_sum_to_total_without_readding_subtotals(self):
        record = {'loss': 4.1, 'terms': {
            'base': {'total': 2., 'gait_total': 100., 'gait_raw_total': 100.},
            'observation': {'position': 3., 'rotation': 4.},
            'geometry': {'contact': .5},
        }}
        values = native_contributions(record, {'position': .1, 'rotation': .2}, {'contact': 2.})
        self.assertEqual(values['loss/weighted/base_body_and_gait'], 2.)
        self.assertAlmostEqual(values['loss/weighted/observation'], 1.1)
        self.assertEqual(values['loss/weighted/scene_geometry'], 1.)
        self.assertAlmostEqual(values['loss/audit/weighted_sum_minus_total'], 0.)
        self.assertEqual(values['loss/details/weighted_observation/position'], .1 * 3.)

    def test_missing_or_unexpected_coefficient_fails_instead_of_guessing(self):
        record = {'loss': 1., 'terms': {'base': {'total': 1.}, 'observation': {'position': 1.}, 'geometry': {}}}
        with self.assertRaisesRegex(ValueError, 'Observation log terms'):
            native_contributions(record, {}, {})


if __name__ == '__main__':
    unittest.main()
