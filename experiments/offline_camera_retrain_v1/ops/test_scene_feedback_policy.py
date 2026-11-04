"""CPU policy tests: change feedback permission, never its evidence/weights."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from . import scene_feedback_policy as policy


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.,2.]))
        self.feedback_enabled = True
        self.patch_enabled = True
        self.local_enabled = True
        self._sampling_scene = None
        self.dynamic_condition = nn.Linear(3, 4)
        self.patch_condition = nn.Linear(3, 4)
        self.summary = torch.randn(1, 8)
        self.patch_tokens = torch.randn(1, 16, 4)
        self.patch_valid = torch.ones(1, 16, dtype=torch.bool)
        self.dynamic = [(torch.randn(2, 1, 4), torch.ones(2, 1, dtype=torch.bool))]
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

    def scene_tokens(self, batch):
        return self.summary, self.patch_tokens, self.patch_valid, self.dynamic

    def _core_pass(self, noisy, timestep, batch, scene=None, control_mask=None,
                   use_scene=True, use_control=True, query_motion=None):
        if not use_scene:
            return noisy
        dynamic = self.dynamic_condition(noisy)
        return (self.patch_condition(noisy) if self.patch_enabled else torch.zeros_like(dynamic)) + dynamic

    def forward(self, noisy_motion, timestep, batch):
        return self._core_pass(noisy_motion, timestep, batch)


class _TwoPassModel(_Model):
    """Match native30's proposal/refinement call signature and no-grad scope."""
    def __init__(self):
        super().__init__()
        self.patch_condition = nn.Linear(3, 3)
        self.dynamic_condition = nn.Linear(3, 3)

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True):
        with torch.no_grad():
            proposal = self._core_pass(noisy_motion, timestep, batch,
                None, None, use_scene, True)
        refined = self._core_pass(noisy_motion, timestep, batch,
            None, None, use_scene, True, proposal.detach())
        return proposal, refined


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

    def test_cls_off_preserves_patch_dynamic_and_local_evidence_and_restores(self):
        model = _Model(); weights = copy.deepcopy(model.state_dict())
        with policy.scene_policy(model, 'cls_off'):
            summary, tokens, valid, dynamic = model.scene_tokens({})
            self.assertEqual(float(summary.abs().sum()), 0.)
            self.assertIs(tokens, model.patch_tokens)
            self.assertIs(valid, model.patch_valid)
            self.assertIs(dynamic, model.dynamic)
            self.assertTrue(model.patch_enabled and model.local_enabled and model.feedback_enabled)
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items()))
        self.assertIs(model.scene_tokens({})[0], model.summary)
        self.assertFalse(hasattr(model, '_scene_feedback_policy'))

    def test_patch_off_preserves_cls_dynamic_and_local_then_restores_after_exception(self):
        model = _Model(); x = torch.randn(2, 3); dynamic = model.dynamic_condition(x)
        with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
            with policy.scene_policy(model, 'patch_off'):
                self.assertFalse(model.patch_enabled)
                self.assertIs(model.scene_tokens({})[0], model.summary)
                self.assertTrue(torch.equal(model.dynamic_condition(x), dynamic))
                self.assertTrue(model.local_enabled and model.feedback_enabled)
                raise RuntimeError('sampling failed')
        self.assertTrue(model.patch_enabled)

    def test_dynamic_off_has_no_effect_on_scene_geometry_or_weights_and_restores(self):
        model = _Model(); x = torch.randn(2, 3)
        expected = model.dynamic_condition(x); weights = copy.deepcopy(model.state_dict())
        with policy.scene_policy(model, 'dynamic_off'):
            self.assertEqual(float(model.dynamic_condition(x).abs().sum()), 0.)
            self.assertIs(model.scene_tokens({})[3], model.dynamic)
            self.assertTrue(model.patch_enabled and model.local_enabled and model.feedback_enabled)
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items()))
        self.assertTrue(torch.equal(model.dynamic_condition(x), expected))
        self.assertEqual(len(model.dynamic_condition._forward_hooks), 0)

    def test_rejects_policy_install_with_stale_sampling_cache(self):
        model = _Model(); model._sampling_scene = object()
        with self.assertRaises((ValueError, AssertionError)):
            policy.apply_scene_feedback_policy(model, 'cls_off')
        self.assertFalse(hasattr(model, '_scene_feedback_policy'))

    def test_exo_loader_arm_is_policy_identity_and_scoped_patches_restore(self):
        from . import refinement_evaluation as paired, exo_longrun
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            policy.write(out / 'PLAN.json', {'models': {'cls_off': {'feedback_policy': 'cls_off'}}})
            model = _Model()
            original_protocol, original_summary = paired.PROTOCOL, paired.summarize
            metadata = {'stage_step': 10000, 'checkpoint_sha256': 'same-weight-hash'}
            with patch.object(exo_longrun, 'load_inference_model', return_value=(model, metadata)) as load:
                with policy.exo_evaluator(out, 'cls_off'):
                    loaded, identity = exo_longrun.load_inference_model('same.pt', device='cpu')
                    self.assertIs(loaded, model)
                    self.assertEqual(identity['arm'], 'cls_off')
                    self.assertTrue(identity['arm_is_inference_policy_label'])
                    self.assertEqual(identity['stage_step'], 10000)
                    self.assertEqual(identity['checkpoint_sha256'], 'same-weight-hash')
                    self.assertEqual(float(loaded.scene_tokens({})[0].abs().sum()), 0.)
                self.assertIs(exo_longrun.load_inference_model, load)
            self.assertEqual(paired.PROTOCOL, original_protocol)
            self.assertIs(paired.summarize, original_summary)
            self.assertIs(model.scene_tokens({})[0], model.summary)

    def test_exo_only_patch_policies_preserve_mixed_ego_rows_and_weights(self):
        for name in ('exo_patch_off', 'exo_static_global_off'):
            for training in (False, True):
                model = _Model(); model.train(training)
                model.summary = torch.randn(2, 8)
                x = torch.randn(2, 3, 3, requires_grad=training)
                batch = {'mode': ['ego', 'exo']}
                original = model._core_pass(x, None, batch)
                dynamic = model.dynamic_condition(x)
                weights = copy.deepcopy(model.state_dict())
                with policy.scene_policy(model, name):
                    actual = model._core_pass(x, None, batch)
                    self.assertTrue(torch.equal(actual[0], original[0]))
                    self.assertTrue(torch.equal(actual[1], dynamic[1]))
                    summary, tokens, valid, owners = model.scene_tokens(batch)
                    self.assertTrue(torch.equal(summary[0], model.summary[0]))
                    if name == 'exo_static_global_off':
                        self.assertEqual(float(summary[1].abs().sum()), 0.)
                    else:
                        self.assertIs(summary, model.summary)
                    self.assertIs(tokens, model.patch_tokens)
                    self.assertIs(owners, model.dynamic)
                    self.assertTrue(model.patch_enabled and model.local_enabled and model.feedback_enabled)
                    self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items()))
                    if training:
                        actual.sum().backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.equal(model._core_pass(x, None, batch), original))
                self.assertEqual(len(model.patch_condition._forward_hooks), 0)

    def test_exo_repair_bypasses_ego_only_and_scene_off_exactly(self):
        for name in ('exo_patch_off', 'exo_static_global_off'):
            model = _Model(); x = torch.randn(1, 2, 3)
            batch = {'mode': ['ego']}
            original = model._core_pass(x, None, batch)
            with policy.scene_policy(model, name):
                self.assertTrue(torch.equal(model._core_pass(x, None, batch), original))
                self.assertIs(model.scene_tokens(batch)[0], model.summary)
                self.assertIs(model._core_pass(x, None, {'mode': ['exo']}, use_scene=False), x)

    def test_exo_repair_rejects_missing_mode_and_does_not_leak_context(self):
        model = _Model(); x = torch.randn(1, 2, 3)
        original = model._core_pass(x, None, {'mode': ['ego']})
        with policy.scene_policy(model, 'exo_patch_off'):
            with self.assertRaises((ValueError, AssertionError)):
                model._core_pass(x, None, {})
            self.assertTrue(torch.equal(model._core_pass(x, None, {'mode': ['ego']}), original))

    def test_exo_policy_runs_after_real_lora_routing_and_matches_legacy_patch_off(self):
        from .adapter_model import install_adapters
        for name in ('exo_patch_off', 'exo_static_global_off'):
            model = _Model()
            install_adapters(model, {'rank': 2, 'alpha': 2., 'targets': ['patch_condition']})
            with torch.no_grad():
                model.exo_adapters['a0000'].up.weight.fill_(.4)
            model.eval(); x = torch.randn(2, 3, 3); batch = {'mode': ['ego', 'exo']}
            with torch.no_grad():
                baseline = model(x, None, batch)
                model.patch_enabled = False
                legacy = model(x, None, batch)
                model.patch_enabled = True
                with policy.scene_policy(model, name):
                    actual = model(x, None, batch)
                self.assertTrue(torch.equal(actual[0], baseline[0]))
                self.assertTrue(torch.equal(actual[1], legacy[1]))
            self.assertIsNone(model._exo_adapter_routing.active.get())

    def test_second_pass_policy_distinguishes_query_argument_in_no_grad(self):
        model = _Model(); x = torch.randn(2, 5, 3); batch = {'mode': ['ego', 'exo']}
        with torch.no_grad():
            original = model._core_pass(x, None, batch)
            dynamic = model.dynamic_condition(x)
            with policy.scene_policy(model, 'exo_patch_second_pass'):
                # Both calls are no_grad; only query_motion distinguishes them.
                first = model._core_pass(x, None, batch)
                explicit_first = model._core_pass(x, None, batch, query_motion=None)
                positional_second = model._core_pass(x, None, batch, None, None, True, True, x)
                keyword_second = model._core_pass(noisy=x, timestep=None, batch=batch, query_motion=x)
                self.assertTrue(torch.equal(first[0], original[0]))
                self.assertTrue(torch.equal(first[1], dynamic[1]))
                self.assertTrue(torch.equal(first, explicit_first))
                self.assertTrue(torch.equal(positional_second, original))
                self.assertTrue(torch.equal(keyword_second, original))
                self.assertIs(model.scene_tokens(batch)[0], model.summary)
                with self.assertRaises((ValueError, AssertionError)):
                    model._core_pass(x, None, batch, query_motion='not a motion tensor')
            self.assertTrue(torch.equal(model._core_pass(x, None, batch), original))

    def test_second_pass_policy_gates_complete_real_lora_after_its_hook(self):
        from .adapter_model import install_adapters
        model = _TwoPassModel()
        install_adapters(model, {'rank': 2, 'alpha': 2., 'targets': ['patch_condition']})
        with torch.no_grad():
            model.exo_adapters['a0000'].down.weight.fill_(.2)
            model.exo_adapters['a0000'].up.weight.fill_(.4)
        model.eval(); x = torch.arange(30.).reshape(2, 5, 3) / 30.
        batch = {'mode': ['ego', 'exo']}; state = copy.deepcopy(model.state_dict())
        original_hooks = tuple(model.patch_condition._forward_hooks)
        with torch.no_grad():
            original = model(x, None, batch)
            dynamic = model.dynamic_condition(x)
            base_patch = model.patch_condition(x)  # Outside adapter context.
            self.assertGreater(float((original[1][1] - dynamic[1] - base_patch[1]).abs().max()), .01)
            with policy.scene_policy(model, 'exo_patch_second_pass'):
                self.assertEqual(tuple(model.patch_condition._forward_hooks)[:-1], original_hooks)
                proposal, refined = model(x, None, batch)
                self.assertTrue(torch.equal(proposal[0], original[0][0]))
                self.assertTrue(torch.equal(proposal[1], dynamic[1]))
                self.assertTrue(torch.equal(refined, original[1]))
                self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in state.items()))
            restored = model(x, None, batch)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(restored, original)))
        self.assertEqual(tuple(model.patch_condition._forward_hooks), original_hooks)
        self.assertIsNone(model._exo_adapter_routing.active.get())

    def test_second_pass_policy_keeps_train_second_pass_adapter_gradient(self):
        from .adapter_model import install_adapters
        model = _TwoPassModel()
        install_adapters(model, {'rank': 2, 'alpha': 2., 'targets': ['patch_condition']})
        model.train(True); x = torch.ones(2, 5, 3, requires_grad=True)
        with policy.scene_policy(model, 'exo_patch_second_pass'):
            proposal, refined = model(x, None, {'mode': ['ego', 'exo']})
            self.assertFalse(proposal.requires_grad)
            self.assertTrue(refined.requires_grad)
            refined.sum().backward()
        up = model.exo_adapters['a0000'].up.weight
        self.assertIsNotNone(up.grad)
        self.assertTrue(torch.isfinite(up.grad).all())
        self.assertGreater(float(up.grad.abs().sum()), 0.)
        self.assertIsNone(model.patch_condition.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNone(model._exo_adapter_routing.active.get())

    def test_second_pass_policy_ego_and_scene_off_are_exact_for_both_passes(self):
        model = _TwoPassModel(); x = torch.randn(1, 5, 3); ego = {'mode': ['ego']}
        original = model(x, None, ego)
        with policy.scene_policy(model, 'exo_patch_second_pass'):
            actual = model(x, None, ego)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, original)))
            off = model(x, None, {'mode': ['exo']}, use_scene=False)
            self.assertTrue(all(value is x for value in off))
            self.assertTrue(model.patch_enabled and model.local_enabled and model.feedback_enabled)
        self.assertEqual(len(model.patch_condition._forward_hooks), 0)

    def test_second_pass_context_restores_after_core_exception_and_removal(self):
        model = _Model(); x = torch.randn(1, 5, 3); batch = {'mode': ['exo']}
        original_core = model._core_pass
        def fail(module, arguments, output):
            raise RuntimeError('core patch failure')
        with policy.scene_policy(model, 'exo_patch_second_pass'):
            failure = model.patch_condition.register_forward_hook(fail)
            try:
                with self.assertRaisesRegex(RuntimeError, 'core patch failure'):
                    model._core_pass(x, None, batch)
            finally:
                failure.remove()
            # A leaked context would let this out-of-core call proceed.
            with self.assertRaisesRegex((ValueError, AssertionError), 'inside an explicit-mode'):
                model.patch_condition(x)
            with self.assertRaises((ValueError, AssertionError)):
                model._core_pass(x, None, {})
            second = model._core_pass(x, None, batch, query_motion=x)
            # A subsequent EGO call must retain its ordinary patch contribution.
            ego = model._core_pass(x, None, {'mode': ['ego']})
            self.assertTrue(torch.equal(second, ego))
        self.assertEqual(model._core_pass, original_core)
        self.assertEqual(len(model.patch_condition._forward_hooks), 0)

    def test_second_pass_registration_and_metadata_do_not_claim_retrained_weights(self):
        self.assertIn('exo_patch_second_pass', policy.POLICIES)
        self.assertIn('exo_patch_second_pass', policy.EXO_REPAIR_POLICIES)
        model = _Model()
        with policy.scene_policy(model, 'exo_patch_second_pass') as metadata:
            self.assertEqual(metadata['patch_pass_policy'],
                dict(proposal='EXO_zero', refinement='unchanged'))
            self.assertTrue(metadata['rule_identical_in_training_and_inference'])
            self.assertFalse(metadata['checkpoint_trained_with_this_policy'])
            self.assertFalse(metadata['training_and_inference_same_policy'])


if __name__=='__main__':
    unittest.main()
