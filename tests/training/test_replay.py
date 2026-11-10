import tempfile
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scenemi.replay.data import ArrayCache,CORPORA,Prefetch,normalize,requests
from scenemi.replay.runtime import budget
from scenemi.replay.optimization import subset


class DummySampler:
    seed=123
    def sample(self,r):
        rng=np.random.default_rng(r['seed'])
        return {'motion':rng.normal(size=(128,201)).astype(np.float32)},r


class ReplayTests(unittest.TestCase):
    def test_exact_source_mode_counts_and_counter_resume(self):
        for batch in (8,32,64,128):
            rows=requests(123,900,batch)
            self.assertEqual([r['mode'] for r in rows],['ego','exo']*(batch//2))
            self.assertEqual(Counter((r['dataset'],r['mode']) for r in rows),
                Counter({(s,m):batch//8 for s in CORPORA for m in ('ego','exo')}))
            self.assertEqual(rows,requests(123,900,batch))
            self.assertNotEqual(rows,requests(123,901,batch))
        with self.assertRaises(ValueError):requests(1,1,12)

    def test_prefetch_is_worker_and_restart_independent(self):
        outputs=[]
        for workers in (1,4):
            loader=Prefetch(DummySampler(),32,workers)
            try:
                loader.get(5,next_step=6)
                outputs.append(loader.get(6))
            finally:loader.close()
        self.assertEqual(outputs[0][2],outputs[1][2])
        for a,b in zip(outputs[0][:2],outputs[1][:2]):self.assertTrue(torch.equal(a['motion'],b['motion']))
        loader=Prefetch(DummySampler(),32,4)
        try:restarted=loader.get(6)
        finally:loader.close()
        self.assertTrue(torch.equal(outputs[0][0]['motion'],restarted[0]['motion']))

    def test_cache_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rest.npy';np.save(path,np.ones((22,3),np.float32));calls=[]
            def checked(meta,name):calls.append(name);return np.load(path,mmap_mode='r')
            cache=ArrayCache(checked);meta={'path':folder,'output_sha256':{'rest.npy':'checked'}}
            first=cache(meta,'rest');self.assertIs(cache(meta,'rest'),first);self.assertEqual(len(calls),1)
            np.save(path,np.zeros((22,3),np.float32))
            with self.assertRaises(ValueError):cache(meta,'rest')

    def test_distinct_native_and_smplh_supervision(self):
        beta=np.arange(16,dtype=np.float32)
        amass,_=normalize({'target_betas':beta},{},'amass','ego')
        self.assertTrue(np.array_equal(amass['target_betas'],beta));self.assertEqual(amass['body_format'],0)
        native,_=normalize({'target_betas':beta[:10]},{'native_format':'smpl'},'egobody','exo')
        self.assertEqual(native['body_format'],1);self.assertEqual(native['target_betas'].shape,(16,))
        self.assertTrue(np.array_equal(native['target_native_betas'],beta[:10]))

    def test_mapped_cache_does_not_accumulate_unbounded_descriptors(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=ArrayCache(lambda m,n:np.load(m['path'],mmap_mode='r'),size=4)
            for i in range(32):
                path=Path(folder)/f'{i}.npy';np.save(path,np.ones(10))
                cache({'path':str(path),'output_sha256':{'a.npy':str(i)}},'a')
                self.assertLessEqual(len(cache.cache),4)

    def test_budget_is_actual_free_memory_with_spare(self):
        self.assertEqual(budget(24*1024),19.)
        self.assertEqual(budget(20.7*1024),15.5)
        self.assertLessEqual(budget(24455,4)*1024+5*1024,24455)

    def test_detail_slicing_retains_gradients(self):
        data=torch.randn(16,5,requires_grad=True);mask=torch.arange(16)%4==0
        value=subset({'nested':{'x':data},'constant':torch.ones(3)},mask)
        self.assertEqual(value['nested']['x'].shape,(4,5));self.assertEqual(value['constant'].shape,(3,))
        value['nested']['x'].sum().backward();self.assertTrue(torch.equal(data.grad[:,0],mask.float()))


if __name__=='__main__':unittest.main()
