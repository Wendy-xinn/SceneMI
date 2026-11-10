import json
from pathlib import Path
import tempfile
import unittest

import torch

from scenemi.replay.scene_comparison import ranked_cases, extra_metrics, report


class PairedComparisonTests(unittest.TestCase):
    def test_selection_is_order_independent_and_family_balanced(self):
        cases = [dict(family=f'f{i}', path=f'/val/{i}/{j}', start=j*128)
                 for i in range(10) for j in range(i+1)]
        selected = ranked_cases(cases, 6, 71)
        self.assertEqual(selected, ranked_cases(cases[::-1], 6, 71))
        self.assertEqual(len({c['family'] for c in selected}), 6)
        with self.assertRaises(ValueError):
            ranked_cases(cases, 11, 71)

    def test_constant_offset_retains_global_error_without_temporal_error(self):
        truth = torch.zeros(8, 22, 3)
        truth[..., 0] = torch.arange(8)[:, None]*.02
        pred = truth+torch.tensor([0., .25, 0.])
        actual = extra_metrics(pred, truth)
        self.assertAlmostEqual(actual['root_worst_frame_mm'], 250)
        self.assertAlmostEqual(actual['height_bias_mm'], 250)
        self.assertAlmostEqual(actual['tracked_acceleration_m_s2'], 0, places=4)
        self.assertAlmostEqual(actual['tracked_velocity_error_mm_s'], 0, places=4)
        self.assertIsNone(actual['missing_root_mean_mm'])

    def test_outage_boundary_does_not_contaminate_tracked_jitter(self):
        truth = torch.zeros(12, 22, 3)
        pred = truth.clone(); pred[:4, :, 0] = 1.
        available = torch.ones(12, dtype=torch.bool); available[:4] = False
        actual = extra_metrics(pred, truth, available)
        self.assertEqual(actual['missing_root_mean_mm'], 1000)
        self.assertEqual(actual['tracked_acceleration_m_s2'], 0)
        self.assertEqual(actual['tracked_jerk_m_s3'], 0)
        self.assertEqual(actual['tracked_velocity_error_mm_s'], 0)

    def test_summary_excludes_diagnostic_cases_and_verifies_assets(self):
        from scenemi.data.amass_prepare import atomic_json, sha256
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary); (out/'rows').mkdir(); (out/'joints').mkdir()
            protocol = dict(jobs=[dict(job_id=0), dict(job_id=1)])
            atomic_json(out/'PROTOCOL.json', protocol)
            for index, scope in [(0, 'balanced'), (1, 'diagnostic')]:
                file = out/'joints'/f'{index:04d}.npz'; file.write_bytes(b'proof')
                rows = []
                for variant, v in [('no_scene', 10), ('scene_1000', 8)]:
                    metrics = dict(mpjpe_mm=v if scope == 'balanced' else 999,
                                   root_error_mm=v, mpjve_mm_s=v)
                    rows.append(dict(mode='ego', condition='ego', source='egobody', family='f',
                                     scope=scope, variant=variant, metrics=metrics))
                atomic_json(out/'rows'/f'{index:04d}.json', dict(job=protocol['jobs'][index],
                    protocol_sha256=sha256(out/'PROTOCOL.json'), joints_file=str(file.relative_to(out)),
                    joints_sha256=sha256(file), rows=rows))
            result = report(out, protocol)
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['per_dataset']['ego_ego_egobody']['scene_minus_no_scene']['mpjpe_mm'], -2)
            self.assertEqual(len(result['diagnostics']), 2)
            (out/'joints/0000.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                report(out, protocol)


if __name__ == '__main__':
    unittest.main()
