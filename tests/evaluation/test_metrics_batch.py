import unittest
import torch
from scenemi.metrics import motion_metrics
from scenemi.metrics_batch import metric_rows,repair_rows


class BatchedMetricTests(unittest.TestCase):
    def test_every_metric_matches_retained_implementation(self):
        torch.manual_seed(821)
        truth=torch.randn(5,128,22,3);prediction=truth+.05*torch.randn_like(truth)
        truth[0,...,1]+=8;prediction[0,...,1]+=8
        prediction[1]=truth[1]+torch.tensor([.2,0.,0.])
        prediction[2,:,:,0]*=-1
        for actual,p,t in zip(metric_rows(prediction,truth),prediction,truth):
            expected=motion_metrics(p,t)
            self.assertEqual(actual.keys(),expected.keys())
            for key in expected:self.assertAlmostEqual(actual[key],expected[key],delta=1e-7)

    def test_no_contact_in_one_window_is_not_overridden_by_another(self):
        truth=torch.randn(2,128,22,3);truth[...,1]=5
        truth[1,20,7,1]=0.
        rows=metric_rows(truth,truth)
        self.assertEqual([r['egoallo_gnd_any_contact'] for r in rows],[0.,1.])

    def test_repair_masks_preserve_per_window_region_means(self):
        truth=torch.randn(3,128,22,3);points=truth+.1*torch.randn_like(truth)
        visible=torch.ones(3,128,22,dtype=torch.bool);visible[:,20:42,16:]=False
        rows=repair_rows(points,truth,truth,visible,'occlusion');error=(points-truth).norm(dim=-1)*1000
        for i,row in enumerate(rows):
            self.assertAlmostEqual(row['repair_region_mpjpe_mm'],float(error[i][~visible[i]].mean()),delta=5e-5)
            self.assertAlmostEqual(row['tracked_visible_mpjpe_mm'],float(error[i][visible[i]].mean()),delta=5e-5)


if __name__=='__main__':unittest.main()
