"""Test the unedited released helper separately from SceneMI's metric code."""
from pathlib import Path
import unittest
import torch
from scenemi.metric_audit import released_functions,analytical_checks,original_metrics


class ReleasedMetricAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[3]/'EgoGVAE_release/utils/from_egoallo/metrics_helpers.py'
        if not path.exists():raise unittest.SkipTest('Optional released EgoGVAE checkout')
        cls.functions=released_functions(path)

    def test_translation_exposes_difference_in_released_error_formula(self):
        report=analytical_checks(self.functions)
        self.assertAlmostEqual(report['released_function']['released_pa_compat_mm'],100,places=3)
        self.assertLess(report['scenemi_standard_pa_mm'],.001)

    def test_correct_alignment_removes_similarity_but_preserves_pose_errors(self):
        gt=torch.randn((1,8,22,3),generator=torch.Generator().manual_seed(9))*.3
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        similar=1.4*(gt@rotation.T)+torch.tensor([.1,.2,.3])
        self.assertLess(original_metrics(self.functions,similar,gt)[0]['pa_mpjpe_mm'],.001)
        changed=gt.clone();changed[:,:,18]+=torch.tensor([.3,.2,.1])
        self.assertGreater(original_metrics(self.functions,changed,gt)[0]['pa_mpjpe_mm'],10)


if __name__=='__main__':unittest.main()
