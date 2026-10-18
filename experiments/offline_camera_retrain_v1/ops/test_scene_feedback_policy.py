"""CPU policy tests: change feedback permission, never its evidence/weights."""
import copy
import unittest

import torch
from torch import nn

from . import scene_feedback_policy as policy


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.,2.]))
        self.feedback_enabled = True
        self.context = torch.randn(1, 2, 22, 64)
        self.bps = torch.randn(1, 2, 67, 3)
        self.valid = torch.zeros(1,2,67,dtype=torch.bool)
        hits = torch.zeros(2,67,3,8,dtype=torch.bool)
        hits[0,4,2,0] = True
        labels=torch.cat((torch.arange(22).repeat_interleave(3),torch.tensor([15])))
        self.geometry = [dict(labels=labels, valid=hits,
            region_valid=torch.ones(2,22,dtype=torch.bool),
            relation_valid=torch.ones(2,22,dtype=torch.bool))]

    def local_geometry(self, *args, **kwargs):
        return self.context, self.bps, self.valid, self.geometry


class FeedbackPolicyTests(unittest.TestCase):
    def test_near_gate_preserves_conditions_and_original_dictionaries(self):
        model = _Model()
        weights = copy.deepcopy(model.state_dict())
        metadata = policy.apply_scene_feedback_policy(model,'near_only')
        context,bps,valid,geometry = model.local_geometry()
        self.assertIs(context,model.context)
        self.assertIs(bps,model.bps)
        self.assertIs(valid,model.valid)
        expected = torch.zeros(2,22,dtype=torch.bool); expected[0,1]=True
        self.assertTrue(torch.equal(geometry[0]['region_valid'],expected))
        self.assertTrue(model.geometry[0]['region_valid'].all())
        self.assertIs(geometry[0]['valid'],model.geometry[0]['valid'])
        self.assertIs(geometry[0]['relation_valid'],model.geometry[0]['relation_valid'])
        self.assertTrue(model.feedback_enabled)
        self.assertTrue(all(torch.equal(model.state_dict()[name],value) for name,value in weights.items()))
        self.assertEqual(metadata['feedback_policy'],'near_only')

    def test_feedback_off_changes_only_the_feedback_switch(self):
        model=_Model(); state=copy.deepcopy(model.state_dict())
        original=model.local_geometry()
        policy.apply_scene_feedback_policy(model,'feedback_off')
        self.assertFalse(model.feedback_enabled)
        self.assertTrue(all(a is b for a,b in zip(original,model.local_geometry())))
        self.assertTrue(all(torch.equal(model.state_dict()[name],value) for name,value in state.items()))

    def test_original_policy_leaves_the_gate_unchanged(self):
        model=_Model(); original=model.local_geometry()
        policy.apply_scene_feedback_policy(model,'original')
        self.assertTrue(model.feedback_enabled)
        self.assertTrue(all(a is b for a,b in zip(original,model.local_geometry())))

    def test_rejects_attention_architecture_and_duplicate_install(self):
        model=_Model()
        with self.assertRaises((ValueError,AssertionError)):
            policy.apply_scene_feedback_policy(model,'near_only',architecture='scene_cross_attention')
        policy.apply_scene_feedback_policy(model,'near_only')
        with self.assertRaises((ValueError,AssertionError)):
            policy.apply_scene_feedback_policy(model,'feedback_off')

    def test_scoped_evaluator_restores_original_functions_and_protocol(self):
        functions=(policy.review.load_model,policy.review.closure,policy.review.PROTOCOL)
        with policy.evaluator():
            self.assertEqual(policy.review.PROTOCOL,policy.PROTOCOL)
            self.assertIsNot(policy.review.load_model,functions[0])
        self.assertEqual((policy.review.load_model,policy.review.closure,policy.review.PROTOCOL),functions)


if __name__=='__main__':
    unittest.main()
