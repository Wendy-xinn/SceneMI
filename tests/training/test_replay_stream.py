import copy
import unittest
from collections import Counter
import numpy as np
import torch
from scenemi.replay.stream import SOURCE_WEIGHTS,add_noise,stream_requests
from scenemi.replay.optimization import optimizer_factor,source_total


class StreamTests(unittest.TestCase):
    def test_batch32_and128_have_identical_ordered_examples(self):
        entire=stream_requests(42,0,256)
        small=sum([stream_requests(42,i,32) for i in range(0,256,32)],[])
        big=sum([stream_requests(42,i,128) for i in range(0,256,128)],[])
        self.assertEqual(entire,small);self.assertEqual(small,big)
        self.assertEqual([v['mode'] for v in entire],['ego','exo']*128)
        counts=Counter(v['dataset'] for v in entire[:128])
        self.assertEqual(counts,dict(amass=32,egobody=16,trumans=36,grail=44))
        for start in range(0,256,32):
            self.assertEqual(len({(v['dataset'],v['mode']) for v in small[start:start+32]}),8)

    def test_diffusion_noise_is_per_example_and_restart_stable(self):
        requests=stream_requests(42,0,128)
        first=[add_noise(({'motion':np.zeros((128,201))},{}),r,42) for r in requests]
        second=[add_noise(({'motion':np.zeros((128,201))},{}),r,42) for r in stream_requests(42,32,32)]
        for a,b in zip(first[32:64],second):
            self.assertTrue(np.array_equal(a[0]['diffusion_noise'],b[0]['diffusion_noise']))
            self.assertEqual(a[0]['diffusion_t'],b[0]['diffusion_t'])

    def test_schedule_matches_at_equal_sample_exposure(self):
        c=dict(training=dict(batch_size=32,sampled_windows=256000),optimizer=dict(warmup_windows=2400))
        d=copy.deepcopy(c);d['training']['batch_size']=128
        for step in (1,8,19,64,512):self.assertEqual(optimizer_factor(c,step*4),optimizer_factor(d,step))

    def test_source_weights_scale_gradients_once(self):
        x=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        total=source_total(list(x.square()),SOURCE_WEIGHTS);total.backward()
        self.assertTrue(torch.allclose(x.grad,2*x.detach()*torch.tensor(SOURCE_WEIGHTS)))
        with self.assertRaises(ValueError):source_total(list(x),[.5]*4)

    def test_group_learning_rates_do_not_change_sampling(self):
        from scenemi.replay.pilot import candidate_config
        c=dict(training={},optimizer=dict(backbone_lr=1e-5,native_shape_lr=5e-5,exo_lr=3e-5),source_weights=list(SOURCE_WEIGHTS))
        original=copy.deepcopy(c)
        settings=dict(formal_training=dict(sampled_windows=256000,warmup_windows=2400))
        candidate=dict(batch=128,lr_multiplier=1.,group_lr_multipliers=dict(backbone=.25,native_shape=4.,exo=1.),
            source_loss_weights=[.5,1/12,.1875,11/48])
        result=candidate_config(c,settings,candidate)
        self.assertEqual(c,original)
        self.assertAlmostEqual(result['optimizer']['backbone_lr'],2.5e-6)
        self.assertAlmostEqual(result['optimizer']['native_shape_lr'],2e-4)
        self.assertAlmostEqual(result['optimizer']['exo_lr'],3e-5)
        self.assertEqual(result['training']['updates'],2000)
        self.assertEqual(result['source_weights'],candidate['source_loss_weights'])
        self.assertEqual(SOURCE_WEIGHTS,(.25,.125,.28125,.34375))


if __name__=='__main__':unittest.main()
