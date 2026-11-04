"""Regression checks for tread support and causal pre-sit observations."""
import json
from pathlib import Path
import unittest

import numpy as np

from scenemi.data.grail_chair import full_frustum, ready_windows
from scenemi.data.grail_contact import intersect_windows
from scenemi.data.grail_support import sole_measure, support_events, support_faces

ROOT = Path(__file__).resolve().parents[2]


class SoleSupportTest(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT/'configs/grail_ego_support_v2.json').read_text())

    def test_riser_nearness_is_not_tread_support(self):
        # Upper tread ends at x=0; lower tread is 17cm below the sole.
        v = np.array([[-1, .5, -1], [0, .5, -1], [0, .5, 1], [-1, .5, 1],
                      [0, .2, -1], [1, .2, -1], [1, .2, 1], [0, .2, 1]], np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [1, 4, 7], [1, 7, 2]])
        foot = np.array([[.02, .37, 0], [.02, .37, .02], [-.02, .52, 0],
                         [-.04, .56, 0], [-.02, .58, 0], [-.02, .59, 0], [-.02, .6, 0]], np.float32)
        points = np.broadcast_to(np.concatenate([foot, foot]), (20, 14, 3)).copy()
        arrays = sole_measure(points, [np.arange(7), np.arange(7, 14)], v, f, self.policy)
        np.testing.assert_allclose(arrays['sole_gap_q10_m'], .17, atol=1e-6)
        self.assertFalse(arrays['sole_supported_fraction'].any())
        record, bad = support_events(arrays, np.ones((20, 2)), 'stair_p2', self.policy)
        self.assertTrue(bad.all())
        self.assertIn('sole_estimated_stance_unsupported_0', record['events'])
        self.assertEqual(intersect_windows([[0, 0]], np.pad(bad, (60, 48)))[1], 0)
        # Winding must not change whether a triangle can support a foot.
        self.assertEqual(len(support_faces(v, f[:, ::-1], .7)), 4)

    def test_swing_foot_and_short_transitions_are_allowed(self):
        arrays = dict(sole_supported_fraction=np.ones((30, 2)),
                      sole_gap_q10_m=np.zeros((30, 2)), foot_speed_mps=np.zeros((30, 2)))
        arrays['sole_supported_fraction'][:, 0] = 0
        prob = np.zeros((30, 2))
        self.assertFalse(support_events(arrays, prob, 'curb', self.policy)[1].any())
        prob[10:17, 0] = 1  # Seven frames do not meet eight-frame duration.
        self.assertFalse(support_events(arrays, prob, 'curb', self.policy)[1].any())
        prob[10:18, 0] = 1
        bad = support_events(arrays, prob, 'curb', self.policy)[1]
        self.assertEqual(np.flatnonzero(bad).tolist(), list(range(4, 24)))
        self.assertFalse(support_events(arrays, np.ones((30, 2)), 'sitting', self.policy)[1].any())


class ChairHistoryTest(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT/'configs/grail_ego_support_v2.json').read_text())

    def test_future_view_cannot_make_pre_sit_ready(self):
        good = np.zeros(180, bool)
        good[100:120] = True
        ready, windows = ready_windows(good, 90, self.policy)
        self.assertIsNone(ready)
        self.assertFalse(windows.any())
        good[20:28] = True
        ready, windows = ready_windows(good, 90, self.policy)
        self.assertEqual(ready, 27)
        self.assertTrue(windows.all())  # Prefix history may precede the window.
        self.assertTrue(windows[40])
        self.assertIsNone(ready_windows(good, None, self.policy)[0])

    def test_whole_chair_not_only_center_must_fit(self):
        from itertools import product
        v = np.array(list(product([-.2, .2], [-.2, .2], [1.8, 2.2])))
        p = np.array([[0., 0, 0], [0, 2., 0]])
        r = np.broadcast_to(np.eye(3), (2, 3, 3))
        camera = dict(hfov_deg=66.56, vfov_deg=40.49, near_m=.05, far_m=4)
        np.testing.assert_array_equal(full_frustum(v, p, r, camera), [True, False])


if __name__ == '__main__':
    unittest.main()
