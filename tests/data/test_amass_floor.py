"""Equivalence to sklearn, including seed-dependent border assignment."""
import unittest
import numpy as np
from sklearn.cluster import DBSCAN
from scenemi.data.amass_floor import HeightDBSCAN


class HeightClusteringTests(unittest.TestCase):
    def compare(self, x, eps=.005, min_samples=3):
        x = np.asarray(x).reshape(-1, 1)
        expected = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        actual = HeightDBSCAN(eps=eps, min_samples=min_samples).fit(x)
        np.testing.assert_array_equal(actual.labels_, expected.labels_)
        np.testing.assert_array_equal(actual.core_sample_indices_, expected.core_sample_indices_)

    def test_random_and_duplicate_heights(self):
        rng = np.random.default_rng(20261022)
        for dtype in (np.float32, np.float64):
            for n in (1, 2, 10, 100, 1000):
                for _ in range(4):
                    x = rng.normal(0, .02, n).astype(dtype)
                    self.compare(x)
                    self.compare(np.round(x, 2))

    def test_border_ties_and_original_seed_order(self):
        rng = np.random.default_rng(42)
        # At min_samples=4, 0 touches two separate core components but is
        # not itself core. Its label depends on the first component's seed.
        x = np.array([-.018, -.017, -.016, -.009, 0, .009, .016, .017, .018])
        for _ in range(40):
            self.compare(rng.permutation(x), eps=.01, min_samples=4)

    def test_closed_epsilon_boundary(self):
        e = .005
        values = [0, e, -e, np.nextafter(e, 0), np.nextafter(e, np.inf),
                  np.nextafter(-e, 0), np.nextafter(-e, -np.inf), 1, 1+e]
        rng = np.random.default_rng(7)
        for m in (1, 2, 3, 4):
            for _ in range(10):
                self.compare(rng.permutation(values), min_samples=m)

    def test_dense_long_heights(self):
        result = HeightDBSCAN(eps=.005, min_samples=3).fit(np.zeros((100000, 1)))
        np.testing.assert_array_equal(result.labels_, np.zeros(100000, np.intp))
        self.assertEqual(len(result.core_sample_indices_), 100000)


if __name__ == '__main__':
    unittest.main()
