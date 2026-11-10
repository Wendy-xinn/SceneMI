import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scenemi.continuation import restore_parent
from scenemi.training import rng_state


class ContinuationTests(unittest.TestCase):
    def parent(self,lr=5e-6):
        model=nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
        model(torch.ones(1,2)).sum().backward();optimizer.step()
        rng=np.random.default_rng(42)
        state=dict(optimizer=optimizer.state_dict(),rng=rng_state(rng,'cpu'))
        c=dict(seed=9,optimizer=dict(lr=1e-4,weight_decay=.01),
            continuation=dict(parent_checkpoint='unused.pt',learning_rate=5e-6))
        return model,state,c,rng

    def test_parent_moments_and_low_lr_are_preserved(self):
        model,state,c,_=self.parent()
        with patch('scenemi.continuation.torch.load',return_value=state):optimizer,_=restore_parent(model,c,'cpu')
        self.assertEqual(optimizer.param_groups[0]['lr'],5e-6)
        for saved,restored in zip(state['optimizer']['state'].values(),optimizer.state.values()):
            self.assertTrue(torch.equal(saved['exp_avg'],restored['exp_avg']))
            self.assertTrue(torch.equal(saved['step'],restored['step']))

    def test_parent_sampler_state_is_restored(self):
        model,state,c,rng=self.parent();expected=rng.random(5)
        with patch('scenemi.continuation.torch.load',return_value=state):_,restored=restore_parent(model,c,'cpu')
        np.testing.assert_array_equal(expected,restored.random(5))

    def test_unexpected_parent_lr_is_rejected(self):
        model,state,c,_=self.parent(lr=1e-4)
        with patch('scenemi.continuation.torch.load',return_value=state):
            with self.assertRaises(ValueError):restore_parent(model,c,'cpu')


if __name__=='__main__':unittest.main()
