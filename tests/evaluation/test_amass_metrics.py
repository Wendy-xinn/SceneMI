"""Analytical errors distinguish standard PA from released-code behavior."""
import unittest
import ast
from pathlib import Path
import numpy as np
import torch

from scenemi.evaluation import test_windows, summarize
from scenemi.metrics import motion_metrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gt = torch.tensor(np.random.default_rng(42).normal(size=(8, 22, 3)), dtype=torch.float64)

    def test_identity_and_global_translation(self):
        same = motion_metrics(self.gt, self.gt)
        self.assertLess(same['mpjpe_mm'], 1e-8)
        self.assertLess(same['pa_mpjpe_mm'], 1e-8)
        pred = self.gt+torch.tensor([.1, 0., 0.])
        row = motion_metrics(pred, self.gt)
        self.assertAlmostEqual(row['mpjpe_mm'], 100., places=4)
        self.assertLess(row['pa_mpjpe_mm'], 1e-8)
        self.assertAlmostEqual(row['released_pa_compat_mm'], 100., places=4)

    def test_similarity_rotation_scale_and_proper_reflections(self):
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        pred = 1.4*(self.gt @ rotation.T)+.2
        self.assertLess(motion_metrics(pred, self.gt)['pa_mpjpe_mm'], 1e-8)
        reflected = self.gt.clone()
        reflected[..., 0] *= -1
        self.assertGreater(motion_metrics(reflected, self.gt)['pa_mpjpe_mm'], 100.)

    def test_articulation_error_cannot_disappear_under_standard_pa(self):
        pred = self.gt.clone()
        pred[:, 18, :] += torch.tensor([.3, .2, .1])
        self.assertGreater(motion_metrics(pred, self.gt)['pa_mpjpe_mm'], 10.)

    def test_parity_with_optional_pinned_upstream_functions(self):
        path = Path(__file__).resolve().parents[3]/'egoallo/src/egoallo/metrics_helpers.py'
        if not path.exists():
            self.skipTest('Optional pinned EgoAllo checkout')
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('compute_mpjpe', 'procrustes_align') and not node.decorator_list]
        scope = dict(torch=torch, np=np)
        exec('from __future__ import annotations\n'+ '\n'.join(ast.unparse(node) for node in nodes), scope)
        pred = self.gt.clone()+.1
        pred[:, 18] += .2
        def transforms(joints):
            out = torch.zeros((*joints.shape[:-1], 7), dtype=joints.dtype)
            out[..., 0] = 1
            out[..., 4:] = joints
            return out
        # The upstream routine constructs float32 matrices internally.
        pred, gt = pred.float(), self.gt.float()
        p, g = transforms(pred), transforms(gt)
        for aligned, key in [(False, 'mpjpe_mm'), (True, 'released_pa_compat_mm')]:
            upstream = scope['compute_mpjpe'](g[:, 0], g[:, 1:], p[None, :, 0], p[None, :, 1:], aligned)[0]
            self.assertAlmostEqual(upstream, motion_metrics(pred, gt)[key], delta=.001)

    def test_manifest_requires_previous_frame_and_never_crosses_source(self):
        families = [dict(split='test', source='a', family='a', corpus='SSM_synced', variants=[
            dict(path='a', sha256='a', reflected=False, frames_after_relative_pose_drop=129),
            dict(path='a_r', sha256='b', reflected=True, frames_after_relative_pose_drop=400)]),
            dict(split='train', source='never', variants=[])]
        cases = test_windows(dict(families=families))
        self.assertEqual(len(cases), 529//129)
        for case in cases:
            self.assertGreaterEqual(case['start_after_0b'], 1)
            self.assertLessEqual(case['start_after_0b']+128, case['frames_after_0b'])

    def test_uncertainty_groups_reflection_and_repeated_windows(self):
        rows = [dict(identity=dict(family=f, corpus='a'), metrics=dict(m=x))
                for f, x in [('a', 1.), ('a', 3.), ('b', 6.)]]
        result = summarize(rows)
        self.assertEqual(result['independent_source_families'], 2)
        self.assertEqual(result['family_macro_mean']['m'], 4.)


if __name__ == '__main__':
    unittest.main()
