"""Geometry and temporal checks for contact-based training selection."""
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scenemi.data.grail_contact import (
    DEFAULT_CONFIG, intersect_windows, measure, sustained, with_context,
)
from scenemi.data.grail_visibility import sha256


class GrailContactTest(unittest.TestCase):
    def test_duration_context_and_window_overlap(self):
        mask = np.zeros(301, bool)
        mask[20:25] = True  # Short transition is not a persistent event.
        mask[204:225] = True
        event = with_context(sustained(mask, 8), 6)
        ranges, count = intersect_windows([[0, 173]], event)
        self.assertEqual(ranges, [[0, 70]])
        self.assertEqual(count, 71)
        self.assertFalse(event[:198].any())
        with self.assertRaises(ValueError):
            intersect_windows([[0, 174]], event)

    def fixture(self, root, category='stair_p2', closed_box=False, armrest=False):
        import trimesh
        policy = json.loads(DEFAULT_CONFIG.read_text())
        n = 20
        if closed_box:
            box = trimesh.creation.box(extents=[1, .6, 1])
            v = box.vertices + [0, .3, 0]
            f = box.faces
        else:
            v = np.array([[-.5, .5, -.5], [.5, .5, -.5], [.5, .5, .5], [-.5, .5, .5]], np.float32)
            f = np.array([[0, 1, 2], [0, 2, 3]])
            if armrest:
                v = np.concatenate([v, v+[0, .2, 0]])
                f = np.concatenate([f, f+4])
        gv = np.array([[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]], np.float32)
        gf = np.array([[0, 1, 2], [0, 2, 3]])
        meta = dict(path=str(root), category=category, source_fps=24, fps=30,
                    source_files={'recon': str(root/'recon.pkl')}, output_sha256={})
        for name, array in [('scene_object_vertices', v), ('scene_object_faces', f),
                            ('scene_ground_vertices', gv), ('scene_ground_faces', gf)]:
            path = root/(name+'.npy')
            np.save(path, array)
            meta['output_sha256'][path.name] = sha256(path)
        with (root/'recon.pkl').open('wb') as stream:
            pickle.dump({'human_data': {'foot_contact_probs': np.zeros((17, 4), np.float32)}}, stream)
        meta['source_hashes'] = {'recon': sha256(root/'recon.pkl')}
        body = SimpleNamespace(feet=[np.array([0, 1]), np.array([2, 3])],
            patches=[[np.array([0]), np.array([1])], [np.array([2]), np.array([3])]],
            seat=np.array([4, 5, 6]))
        vertices = np.broadcast_to(np.array([[-.1, 0, 0], [-.1, 0, .1], [.1, 0, 0], [.1, 0, .1],
                                            [-.1, .5, 0], [.1, .5, 0], [0, .5, .1]], np.float32), (n, 7, 3)).copy()
        joints = np.zeros((n, 22, 3), np.float32)
        joints[:, 0] = [0, .7, 0]
        joints[:, [1, 2], 1] = .6
        joints[:, [4, 5], 1] = .6
        joints[:, [4, 5], 2] = .3
        return meta, vertices, joints, body, policy

    def test_swing_foot_is_allowed_and_persistent_both_feet_float_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory))
            meta, v, joints, body, policy = fixture
            v[..., 0] += 2  # Ground beside the object, no overhead surface.
            v[:, [2, 3], 1] = .2
            result, _ = measure(*fixture)
            self.assertFalse(result['events'])
            v[:, :4, 1] = .2
            result, _ = measure(*fixture)
            self.assertIn('unsupported_both_feet', result['events'])

    def test_foot_inside_closed_geometry_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), closed_box=True)
            fixture[1][:, :4, 1] = .3
            result, _ = measure(*fixture)
            self.assertTrue(result['object_watertight'])
            self.assertIn('deep_foot_volume_overlap_0', result['events'])

    def test_overhead_armrest_does_not_create_seat_penetration(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory), category='sitting', armrest=True)
            result, _ = measure(*fixture)
            self.assertFalse(result['events'])
            self.assertAlmostEqual(result['seat_gap_during_stable_pose_m']['median'], 0, places=5)


if __name__ == '__main__':
    unittest.main()
