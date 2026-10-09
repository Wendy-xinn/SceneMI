import unittest
import numpy as np
from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics

class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gt=np.random.default_rng(7).normal(size=(6,22,3))
    def test_similarity_and_reflection(self):
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        pred=2*self.gt@r+np.array([3.,4.,5.])
        self.assertLess(standard_motion_metrics(pred,self.gt)['pa_mpjpe_mm'],1e-9)
        reflected=self.gt*np.array([-1.,1.,1.])
        self.assertGreater(standard_motion_metrics(reflected,self.gt)['pa_mpjpe_mm'],100)
    def test_translation_and_derivative_units(self):
        t=np.arange(6)/20
        pred=self.gt.copy();pred[:,:,0]+=t[:,None]**2
        m=standard_motion_metrics(pred,self.gt)
        self.assertLess(m['root_relative_mpjpe_mm'],1e-9)
        self.assertAlmostEqual(m['acceleration_error_mm_s2'],2000,places=7)
        self.assertAlmostEqual(m['mpjve_mm_s'],250,places=7)
    def test_identity_and_degenerate(self):
        for x in [self.gt,np.zeros_like(self.gt)]:
            self.assertTrue(all(abs(v)<1e-9 for v in standard_motion_metrics(x,x).values()))
if __name__=='__main__':unittest.main()
