"""Safety properties of scene adaptation around an immutable camera prior."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from scenemi.scene_components import component_flags
from scenemi.scene_residual import FrozenSceneResidual, bounded_translation
from scenemi.scene_repair import reconcile_journal


class ResidualTests(unittest.TestCase):
    def test_translation_bound_is_smooth_at_zero_and_finite_for_large_errors(self):
        x = torch.tensor([[0.,0.,0.],[3.,4.,0.]],requires_grad=True)
        y = bounded_translation(x,.08)
        self.assertTrue(torch.isfinite(y).all())
        self.assertLessEqual(float(y.norm(dim=-1).max()),.080001)
        self.assertTrue(torch.equal(y[0],torch.zeros(3)))
        y[0].sum().backward()
        self.assertTrue(torch.equal(x.grad[0],torch.ones(3)))
        self.assertTrue(torch.isfinite(x.grad).all())

    @staticmethod
    def toy():
        model = FrozenSceneResidual.__new__(FrozenSceneResidual)
        nn.Module.__init__(model)
        model.core = nn.Sequential(nn.Dropout(.8),nn.Linear(3,3))
        model.adapter = nn.Linear(3,3)
        model.core.requires_grad_(False)
        model._adapter_modules = ('adapter',)
        return model

    def test_training_never_enables_backbone_dropout(self):
        model = self.toy().train()
        self.assertTrue(model.training)
        self.assertTrue(model.adapter.training)
        self.assertFalse(model.core.training)
        self.assertFalse(model.core[0].training)
        model.eval()
        self.assertFalse(model.adapter.training)

    def test_compact_state_cannot_overwrite_frozen_backbone(self):
        model = self.toy()
        frozen = model.core[1].weight.detach().clone()
        state = model.adapter_state()
        self.assertEqual(set(state),{'adapter.weight','adapter.bias'})
        state['adapter.weight'].zero_()
        model.load_adapter_state(state)
        self.assertTrue(torch.equal(model.core[1].weight,frozen))
        self.assertEqual(float(model.adapter.weight.detach().abs().max()),0)
        state['core.1.weight'] = frozen
        with self.assertRaises(ValueError):
            model.load_adapter_state(state)

    def test_diagnostic_flags_restore_after_failure(self):
        model = self.toy()
        model.local_enabled = model.feedback_enabled = True
        with self.assertRaises(RuntimeError):
            with component_flags(model,'local_without_feedback'):
                self.assertTrue(model.local_enabled)
                self.assertFalse(model.feedback_enabled)
                raise RuntimeError('simulated inference failure')
        self.assertTrue(model.local_enabled)
        self.assertTrue(model.feedback_enabled)

    def test_crash_tail_is_preserved_without_duplicate_resumed_updates(self):
        with TemporaryDirectory() as folder:
            folder=Path(folder); path=folder/'updates.jsonl'
            path.write_text(''.join(json.dumps(dict(step=i))+'\n' for i in (1,2,3))+'{"step":')
            reconcile_journal(folder,2)
            self.assertEqual([json.loads(s)['step'] for s in path.read_text().splitlines()],[1,2])
            self.assertEqual(len(list(folder.glob('interrupted_tail_*.jsonl'))),1)
            reconcile_journal(folder,2)
            self.assertEqual(len(list(folder.glob('interrupted_tail_*.jsonl'))),1)


if __name__=='__main__':
    unittest.main()
