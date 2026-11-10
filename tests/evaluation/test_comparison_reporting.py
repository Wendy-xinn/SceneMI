import unittest

from scenemi.replay.comparison_reporting import family_delta


def row(family, a, b):
    return dict(job=dict(case=dict(family=family)), rows=[
        dict(variant='no_scene', metrics=dict(mpjpe_mm=a)),
        dict(variant='scene_1000', metrics=dict(mpjpe_mm=b))])


class FamilyBootstrapTests(unittest.TestCase):
    def test_family_weight_does_not_depend_on_window_repetition(self):
        rows = [row('a', 10, 9)]*20+[row('b', 10, 13)]
        actual = family_delta(rows, 'mpjpe_mm')
        self.assertEqual(actual['scene_minus_baseline'], 1)
        self.assertEqual(actual['independent_families'], 2)
        self.assertEqual(actual['contributing_windows'], 21)
        self.assertEqual(actual['fraction_families_improved'], .5)
        self.assertEqual(actual, family_delta(rows[::-1], 'mpjpe_mm'))

    def test_missing_region_is_not_silently_replaced_with_zero(self):
        self.assertIsNone(family_delta([row('a', 10, 9)], 'missing_mpjpe_mm'))


if __name__ == '__main__':
    unittest.main()
