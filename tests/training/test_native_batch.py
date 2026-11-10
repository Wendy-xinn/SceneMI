import numpy as np
import torch
import unittest

from scenemi.native_batch import batch_for
from scenemi.native_joint_training import batch_for as original_batch_for


class Sampler:
    groups = {name: {'family': [i]} for i, name in enumerate(('egobody','trumans','grail'))}

    def sample(self, rng, *args, **kwargs):
        value = int(rng.integers(100000))
        return {'value': np.array([value], dtype=np.float32)}, dict(value=value, **kwargs)


class NativeBatchTests(unittest.TestCase):
    def test_batch_eight_preserves_original_sampling_and_rng(self):
        data = {'ego': Sampler(), 'exo': Sampler()}
        config = {'training': {'sample_order': ['ego','exo']*4}, 'frontend': {}}
        for forced in (False,True):
            rng_a = np.random.default_rng(41); rng_b = np.random.default_rng(41)
            actual = batch_for(data,rng_a,config,'cpu',forced)
            expected = original_batch_for(data,rng_b,config,'cpu',forced)
            assert torch.equal(actual[0]['value'], expected[0]['value'])
            assert torch.equal(actual[1]['value'], expected[1]['value'])
            assert actual[2] == expected[2]
            assert rng_a.bit_generator.state == rng_b.bit_generator.state


    def test_larger_batch_reaches_both_forward_inputs_and_all_corpora(self):
        data = {'ego': Sampler(), 'exo': Sampler()}
        for count in (16,32):
            config = {'training': {'sample_order': ['ego','exo']*(count//2)}, 'frontend': {}}
            ego,exo,identities = batch_for(data,np.random.default_rng(43),config,'cpu',True)
            assert len(ego['value']) == len(exo['value']) == count//2
            assert len(identities) == count
            assert {v['member'] for v in identities} == {0,1,2}


    class NativeBatchTests(unittest.TestCase):
        def test_legacy_sampling_compatibility(self):
            test_batch_eight_preserves_original_sampling_and_rng()

        def test_actual_larger_forward_inputs(self):
            test_larger_batch_reaches_both_forward_inputs_and_all_corpora()


    if __name__ == '__main__':
        unittest.main()
