import tempfile
from pathlib import Path
import unittest

import numpy as np

from scenemi.data.mapped_arrays import map_numeric_npy,numeric_mmap_backend
from scenemi.data.scene_prefetch import ScenePrefetch


class Samples:
    def __init__(self,mode):self.mode=mode
    def sample(self,rng,**kwargs):
        length=int(rng.integers(1,12))
        return rng.normal(size=length),dict(mode=self.mode,length=length)


class SceneExecutionTests(unittest.TestCase):
    def test_numeric_mapping_matches_numpy_bytes_layout_and_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            for i,value in enumerate([np.arange(48,dtype=np.float32).reshape(2,8,3),
                np.asfortranarray(np.arange(12,dtype=np.float64).reshape(3,4)),
                np.array([True,False]),np.arange(5,dtype='>i4'),np.empty((0,3),np.float32)]):
                path=Path(folder)/f'{i}.npy';np.save(path,value)
                actual=map_numeric_npy(path)
                self.assertTrue(np.array_equal(actual,value));self.assertEqual(actual.dtype,value.dtype)
                self.assertFalse(actual.flags.writeable)
                with numeric_mmap_backend():self.assertTrue(np.array_equal(np.load(path,mmap_mode='r'),value))

    def test_truncated_payload_and_object_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.npy';np.save(path,np.arange(8));path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'length'):map_numeric_npy(path)
            np.save(path,np.array([{}],dtype=object))
            with self.assertRaisesRegex(ValueError,'numeric'):map_numeric_npy(path)

    def test_prefetch_preserves_variable_draw_sample_order_and_committed_rng(self):
        import copy
        data={m:Samples(m) for m in ('ego','exo')};counts=dict(ego=4,exo=4)
        expected_rng=np.random.default_rng(23);actual_rng=np.random.default_rng(23)
        prefetch=ScenePrefetch(data,actual_rng,counts,{})
        try:
            for _ in range(3):
                for mode in ('ego','exo'):
                    for _ in range(4):
                        expected=data[mode].sample(expected_rng);actual=prefetch.datasets[mode].sample(actual_rng)
                        self.assertTrue(np.array_equal(expected[0],actual[0]));self.assertEqual(expected[1],actual[1])
                    self.assertEqual(expected_rng.bit_generator.state,actual_rng.bit_generator.state)
                saved=copy.deepcopy(actual_rng.bit_generator.state)
            self.assertEqual(expected_rng.bit_generator.state,saved)
        finally:prefetch.close()


if __name__=='__main__':unittest.main()
