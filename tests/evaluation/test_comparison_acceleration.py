import unittest

import torch

from scenemi.replay.comparison_acceleration import batch_noise, EgoDispatch, grouped_jobs
from scenemi.replay.validation_sampling import sample_ego
from experiments.offline_camera_retrain_v1.scene_model import ddim_sample


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.bias=torch.nn.Parameter(torch.zeros(()))

    def forward(self,x,t,batch,**kwargs):
        return .2*x+batch['camera'][...,:1]+self.bias


class AccelerationTests(unittest.TestCase):
    def test_batch_repeats_exact_original_cpu_noise(self):
        expected=torch.randn((1,128,201),generator=torch.Generator().manual_seed(777))
        actual=batch_noise(4,777,'cpu')
        self.assertTrue(all(torch.equal(actual[i:i+1],expected) for i in range(4)))

    def test_explicit_noise_sampler_matches_original_and_packs_windows(self):
        model=TinyModel().eval()
        camera=torch.arange(4).float()[:,None,None].expand(4,128,9)
        expected=torch.cat([ddim_sample(model,dict(camera=camera[i:i+1]),128,steps=20,seed=777,use_scene=False)
                            for i in range(4)])
        actual=sample_ego(EgoDispatch(model,False),dict(camera=camera),batch_noise(4,777,'cpu'),steps=20)
        self.assertTrue(torch.equal(expected,actual))

    def test_groups_do_not_mix_modes_sources_conditions(self):
        jobs=[dict(case=dict(mode=m,source=s),condition=c,scope='validation')
              for m,s,c in [('ego','grail','ego')]*5+[('exo','grail','gap')]*3]
        result=grouped_jobs(dict(jobs=jobs),4)
        self.assertEqual([len(g) for g in result],[4,1,3])

    def test_numerically_sensitive_group_uses_declared_serial_batch(self):
        jobs=[dict(case=dict(mode=m,source='grail'),condition=c,scope='validation')
              for m,c in [('ego','ego')]*8+[('exo','gap')]*3]
        result=grouped_jobs(dict(jobs=jobs),8,{'exo_grail':1})
        self.assertEqual([len(g) for g in result],[8,1,1,1])


if __name__=='__main__':unittest.main()
