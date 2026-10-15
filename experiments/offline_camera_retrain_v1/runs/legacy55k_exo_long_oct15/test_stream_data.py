import gc
import hashlib
from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
import numpy as np
from temporal_guard import PILOT
from stream_data import StreamData


class CacheTest(unittest.TestCase):
    def test_private_cache_eviction_preserves_sources_and_verified_values(self):
        with tempfile.TemporaryDirectory(dir='/mnt/e/Projects/SceneMI/cache',prefix='exo_cache_test_') as source_dir, tempfile.TemporaryDirectory() as cache_dir:
            folder=Path(source_dir);expected={}
            for i,key in enumerate(('a','b')):
                np.save(folder/(key+'.npy'),np.full((32,3),i+1,np.float32))
                expected[key+'.npy']=hashlib.sha256((folder/(key+'.npy')).read_bytes()).hexdigest()
            data=StreamData.__new__(StreamData);data.task_read_cache=Path(cache_dir)
            data.task_read_limit=(folder/'a.npy').stat().st_size+20;data.read_lru=OrderedDict()
            data.source_hashes={str(folder):expected};data.arrays={}
            a=data.load(folder,'a');np.testing.assert_array_equal(a,np.ones((32,3),np.float32));del a;gc.collect()
            b=data.load(folder,'b');np.testing.assert_array_equal(b,np.full((32,3),2,np.float32))
            self.assertFalse((data.task_read_cache/(expected['a.npy']+'.npy')).exists())
            for key,digest in expected.items():self.assertEqual(hashlib.sha256((folder/key).read_bytes()).hexdigest(),digest)
            self.assertLessEqual(sum(p.stat().st_size for p in data.task_read_cache.glob('*.npy')),data.task_read_limit)


if __name__=='__main__':unittest.main()
