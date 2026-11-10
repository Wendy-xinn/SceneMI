"""Synthetic context must preserve causal observations and surface scale."""
import unittest

import numpy as np

from scenemi.data.grail_augmentation import observed_prefix, sample_chair


class ChairAugmentationTest(unittest.TestCase):
    def test_prefix_excludes_future_body_and_invalid_samples(self):
        points = np.array([[[1., 0, 0], [2, 0, 0], [3, 0, 0]],
                           [[1, 0, 0], [4, 0, 0], [5, 0, 0]],
                           [[100, 0, 0], [101, 0, 0], [102, 0, 0]]])
        owners = np.array([[0, 100, -1], [0, 0, 100], [0, 0, 0]])
        old = points.copy()
        result = observed_prefix(points, owners, 1, .02)
        np.testing.assert_array_equal(result, [[1, 0, 0], [4, 0, 0]])
        np.testing.assert_array_equal(points, old)
        points[2] = np.nan  # Future data cannot affect the prefix.
        np.testing.assert_array_equal(observed_prefix(points, owners, 1, .02), result)

    def test_area_weighted_surface_sampling_and_seed(self):
        # Two disjoint triangles with areas in a 1:4 ratio.
        v = np.array([[0., 0, 0], [1, 0, 0], [0, 0, 1],
                      [10, 0, 0], [12, 0, 0], [10, 0, 2]])
        f = np.array([[0, 1, 2], [3, 4, 5]])
        kw = dict(count=20000, retention=1, jitter_m=0, seed=123)
        points = sample_chair(v, f, **kw)
        np.testing.assert_array_equal(points, sample_chair(v, f, **kw))
        self.assertLess(abs((points[:, 0] < 5).mean()-.2), .015)
        self.assertTrue((points[:, 1] == 0).all())
        first = points[points[:, 0] < 5]
        self.assertTrue(((first[:, 0]+first[:, 2]) <= 1.000001).all())
        with self.assertRaises(ValueError):
            sample_chair(v, f, **dict(kw, jitter_m=-1))


if __name__ == '__main__':
    unittest.main()
