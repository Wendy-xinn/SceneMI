"""Targeted checks for regional scene fusion; no data or training job needed."""
import io
import os
import unittest

import torch
from torch import nn

from .scene_fusion import (RegionalSceneFusion, attach_scene_fusion,
                           scene_fusion_metadata, temporal_pool)


class _NativeRouting(nn.Module):
    """Two native-shaped passes sharing one temporal convolution trunk."""
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.latent_dim = 8
        self.core.dim_mults = (1,)
        self.core.attention = False
        self.core.unet = nn.Module()
        self.core.unet.mid_block2 = nn.Conv1d(8, 8, 3, padding=1)
        self.input_projection = nn.Conv1d(201, 8, 1)
        self.output_projection = nn.Conv1d(8, 201, 1)
        self.register_buffer('alphas', torch.linspace(1., 0., 1000), persistent=False)
        self.geometry_calls = 0
        self.noisy_inputs = []

    def _core_pass(self, noisy, timestep, batch, scene, control_mask, use_scene,
                   use_control, query_motion=None):
        self.noisy_inputs.append(noisy)
        hidden = self.input_projection(noisy.transpose(1, 2))
        return self.output_projection(self.core.unet.mid_block2(hidden)).transpose(1, 2)

    def local_geometry(self, proposal, timestep, batch):
        self.geometry_calls += 1
        return batch['local_tokens'], None, None, [dict(region_valid=v) for v in batch['local_valid']]

    def forward(self, noisy, timestep, batch, *, use_scene=True):
        scene = batch['scene']
        with torch.no_grad():
            proposal = self._core_pass(noisy, timestep, batch, scene, None, use_scene, True)
        if use_scene:
            self.local_geometry(proposal, timestep, batch)
        return self._core_pass(noisy, timestep, batch, scene, None, use_scene, True,
                               query_motion=proposal)


class SceneFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.activation = torch.randn(2, 8, 4)
        self.proposal = torch.randn(2, 8, 201)
        self.alpha = torch.tensor([.2, .9])
        self.times = torch.tensor([150, 800])
        self.global_tokens = torch.randn(2, 16, 64)
        self.global_valid = torch.ones(2, 16, dtype=torch.bool)
        self.local_tokens = torch.randn(2, 8, 22, 64)
        self.local_valid = torch.ones(2, 8, 22, dtype=torch.bool)

    def batch(self):
        return dict(scene=(None, self.global_tokens, self.global_valid, None),
                    local_tokens=self.local_tokens, local_valid=self.local_valid)

    def nonzero(self, module):
        with torch.no_grad():
            module.output_projection.weight.normal_(std=.05)

    def call(self, module, **changes):
        inputs = dict(activation=self.activation, global_tokens=self.global_tokens,
                      global_valid=self.global_valid, proposal=self.proposal,
                      alpha_bar=self.alpha, local_tokens=self.local_tokens,
                      local_valid=self.local_valid)
        inputs.update(changes)
        return module(**inputs)

    def test_attachment_is_exact_and_preserves_rng_parameters_and_flags(self):
        model = _NativeRouting().eval()
        next(model.parameters()).requires_grad_(False)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        flags = {name: value.requires_grad for name, value in model.named_parameters()}
        before = {scene: model(self.proposal, self.times, self.batch(), use_scene=scene)
                  for scene in (False, True)}
        rng = torch.random.get_rng_state().clone()
        attach_scene_fusion(model)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in state.items()))
        self.assertTrue(all(value.requires_grad == flags[name] for name, value in
                            model.named_parameters() if name in flags))
        for scene in (False, True):
            self.assertTrue(torch.equal(before[scene], model(self.proposal, self.times,
                                                            self.batch(), use_scene=scene)))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        metadata = scene_fusion_metadata(model)
        self.assertEqual(metadata['architecture'], 'scene_fusion_v2')
        self.assertEqual(metadata['zero_output_state_keys'], ['scene_fusion_v2.output_projection.weight'])
        self.assertIsNone(model.scene_fusion_v2.output_projection.bias)

    def test_first_pass_and_scene_off_bypass_without_extra_queries(self):
        model = _NativeRouting().eval()
        scene = self.batch()['scene']
        first = model._core_pass(self.proposal, self.times, self.batch(), scene,
                                 None, True, True)
        off = model(self.proposal, self.times, self.batch(), use_scene=False)
        attach_scene_fusion(model)
        self.nonzero(model.scene_fusion_v2)
        calls = []
        handle = model.scene_fusion_v2.register_forward_hook(lambda *_: calls.append(True))
        self.assertTrue(torch.equal(first, model._core_pass(self.proposal, self.times,
                            self.batch(), scene, None, True, True)))
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(off, model(self.proposal, self.times, self.batch(), use_scene=False)))
        self.assertEqual(calls, [])
        model.geometry_calls = 0
        model.noisy_inputs = []
        model(self.proposal, self.times, self.batch(), use_scene=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(model.geometry_calls, 1)
        self.assertEqual(len(model.noisy_inputs), 2)
        self.assertTrue(all(value is self.proposal for value in model.noisy_inputs))
        self.assertIsNone(model._scene_fusion_routing.local.get())
        self.assertIsNone(model._scene_fusion_routing.active.get())
        handle.remove()

    def test_two_scene_memories_and_proposal_can_change_the_result(self):
        module = RegionalSceneFusion(8).eval()
        self.nonzero(module)
        expected = self.call(module)
        global_tokens = self.global_tokens.clone(); global_tokens[:, 0, 0] += 5
        local_tokens = self.local_tokens.clone(); local_tokens[:, :, 3, 1] -= 5
        proposal = self.proposal.clone(); proposal[..., 135] += 1
        for changes in (dict(global_tokens=global_tokens), dict(local_tokens=local_tokens),
                        dict(proposal=proposal)):
            self.assertGreater(float((self.call(module, **changes)-expected).abs().max().detach()), 1e-7)

    def test_local_gate_scales_final_delta_and_global_is_not_gated(self):
        module = RegionalSceneFusion(8).eval()
        self.nonzero(module)
        no_global = torch.zeros_like(self.global_valid)
        full = self.call(module, global_valid=no_global, alpha_bar=torch.ones(2))
        quarter = self.call(module, global_valid=no_global, alpha_bar=torch.full((2,), .5))
        zero = self.call(module, global_valid=no_global, alpha_bar=torch.zeros(2))
        self.assertTrue(torch.equal(zero, self.activation))
        self.assertGreater(float((full-self.activation).abs().max().detach()), 1e-5)
        self.assertTrue(torch.allclose(quarter-self.activation, .25*(full-self.activation), atol=2e-7, rtol=2e-5))
        no_local = torch.zeros_like(self.local_valid)
        global_zero = self.call(module, local_valid=no_local, alpha_bar=torch.zeros(2))
        global_full = self.call(module, local_valid=no_local, alpha_bar=torch.ones(2))
        self.assertTrue(torch.equal(global_zero, global_full))
        self.assertFalse(torch.equal(global_zero, self.activation))

    def test_empty_and_nan_padding_are_finite_exact_identity_after_training(self):
        module = RegionalSceneFusion(8).eval()
        self.nonzero(module)
        with torch.no_grad():
            module.region_fusion[0].bias.fill_(.7)
            module.global_key_norm.bias.fill_(.3)
            module.local_key_norm.bias.fill_(.4)
        gv = torch.zeros_like(self.global_valid); lv = torch.zeros_like(self.local_valid)
        global_tokens = torch.full_like(self.global_tokens, float('nan'))
        local_tokens = torch.full_like(self.local_tokens, float('nan'))
        result = self.call(module, global_tokens=global_tokens, global_valid=gv,
                           local_tokens=local_tokens, local_valid=lv)
        self.assertTrue(torch.equal(result, self.activation))
        gv[:, 0] = True; lv[:, 2:4, 3] = True
        global_tokens[:, 0] = self.global_tokens[:, 0]
        local_tokens[:, 2:4, 3] = self.local_tokens[:, 2:4, 3]
        padded = self.call(module, global_tokens=global_tokens, global_valid=gv,
                           local_tokens=local_tokens, local_valid=lv)
        zeros = self.call(module, global_tokens=torch.where(gv[..., None], global_tokens, 0.), global_valid=gv,
                         local_tokens=torch.where(lv[..., None], local_tokens, 0.), local_valid=lv)
        self.assertTrue(torch.isfinite(padded).all())
        self.assertTrue(torch.equal(padded, zeros))

    def test_region_identity_and_pose_layout_are_present_in_queries_and_local_keys(self):
        module = RegionalSceneFusion(8).eval()
        pose_inputs, local_calls, global_calls = [], [], []
        handles = [module.query_pose_projection.register_forward_pre_hook(
                       lambda _m, args: pose_inputs.append(args[0].detach().clone())),
                   module.local_attention.register_forward_pre_hook(
                       lambda _m, args: local_calls.append(tuple(x.detach().clone() for x in args))),
                   module.global_attention.register_forward_pre_hook(
                       lambda _m, args: global_calls.append(tuple(x.detach().clone() for x in args)))]
        proposal = torch.zeros_like(self.proposal)
        proposal[..., 135:] = 2.  # Native m/2 coordinates remain 2, not 4 metres.
        self.call(module, proposal=proposal, local_tokens=torch.zeros_like(self.local_tokens))
        pose = pose_inputs[0]
        self.assertTrue(torch.equal(pose[..., :3], torch.full_like(pose[..., :3], 2.)))
        identity = torch.tensor([1., 0., 0., 0., 1., 0.]).expand_as(pose[..., 3:])
        self.assertTrue(torch.equal(pose[..., 3:], identity))
        lq, lk, lv = local_calls[0]; gq, gk, _ = global_calls[0]
        self.assertEqual(tuple(lq.shape), (8, 22, 64))
        self.assertEqual(tuple(lk.shape), (8, 23, 64))
        self.assertEqual(tuple(gk.shape), (8, 17, 64))
        self.assertTrue(torch.equal(lq, gq))
        self.assertTrue(torch.equal(lk, lv))
        self.assertFalse(torch.equal(lq[:, 0], lq[:, 1]))
        self.assertFalse(torch.equal(lk[:, 0], lk[:, 1]))
        for handle in handles: handle.remove()

    def test_nondivisible_validity_pool_and_shorter_than_clock(self):
        values = torch.arange(7, dtype=torch.float32)[None, :, None, None].expand(2, 7, 22, 9).clone()
        valid = torch.ones(2, 7, 22, dtype=torch.bool); valid[:, 1:3, 0] = False
        values[:, 1:3, 0] = float('nan')
        pooled, exists = temporal_pool(values, 3, valid)
        self.assertTrue(torch.equal(pooled[0, :, 1, 0], torch.tensor([1., 3., 5.])))
        self.assertTrue(torch.equal(pooled[0, :, 0, 0], torch.tensor([0., 3.5, 5.])))
        self.assertTrue(exists.all())
        short, mask = temporal_pool(values[:, :3], 8, valid[:, :3])
        self.assertEqual(tuple(short.shape), (2, 8, 22, 9))
        self.assertTrue(torch.isfinite(short).all())
        self.assertTrue(torch.equal(short[~mask], torch.zeros_like(short[~mask])))

    def test_local_pose_uses_the_same_valid_evidence_frames_as_local_context(self):
        module = RegionalSceneFusion(8).eval()
        query_poses, key_poses = [], []
        handles = [module.query_pose_projection.register_forward_pre_hook(
                       lambda _m, args: query_poses.append(args[0].detach().clone())),
                   module.local_pose_projection.register_forward_pre_hook(
                       lambda _m, args: key_poses.append(args[0].detach().clone()))]
        proposal = torch.zeros_like(self.proposal); proposal[:, 1, 135] = 100.
        valid = self.local_valid.clone(); valid[:, 1, 0] = False
        self.call(module, proposal=proposal, local_valid=valid)
        self.assertEqual(float(query_poses[0][0, 0, 0, 0]), 50.)
        self.assertEqual(float(key_poses[0][0, 0, 0, 0]), 0.)
        for handle in handles: handle.remove()

    def test_swapping_region_contexts_changes_the_bound_geometric_evidence(self):
        module = RegionalSceneFusion(8).eval(); self.nonzero(module)
        tokens = self.local_tokens.clone()
        swapped = tokens.clone(); swapped[:, :, [0, 10]] = tokens[:, :, [10, 0]]
        no_global = torch.zeros_like(self.global_valid)
        original = self.call(module, local_tokens=tokens, global_valid=no_global)
        reassigned = self.call(module, local_tokens=swapped, global_valid=no_global)
        self.assertGreater(float((original-reassigned).abs().max().detach()), 1e-7)

    def test_zero_output_layer_has_gradient_and_upstream_layers_learn_next_update(self):
        module = RegionalSceneFusion(8).train()
        optimizer = torch.optim.SGD(module.parameters(), lr=.2)
        self.call(module).square().mean().backward()
        gradient = module.output_projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        self.call(module).square().mean().backward()
        for name in ('query_projection.weight', 'query_pose_projection.weight',
                     'local_pose_projection.weight', 'region_embedding.weight',
                     'global_attention.in_proj_weight', 'local_attention.in_proj_weight',
                     'region_fusion.0.weight'):
            gradient = dict(module.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0., name)

    def test_strict_state_round_trip_and_parameter_budget(self):
        model = attach_scene_fusion(_NativeRouting().eval())
        self.nonzero(model.scene_fusion_v2)
        expected = model(self.proposal, self.times, self.batch())
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = attach_scene_fusion(_NativeRouting().eval())
        restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
        self.assertTrue(torch.equal(expected, restored(self.proposal, self.times, self.batch())))
        self.assertEqual(scene_fusion_metadata(model), scene_fusion_metadata(restored))
        self.assertTrue(all(name.startswith('scene_fusion_v2.') for name in
                            set(model.state_dict()) - set(_NativeRouting().state_dict())))
        large = RegionalSceneFusion(1024)
        self.assertLess(sum(p.numel() for p in large.parameters()), 1_000_000)

    def test_real_temporal_unet_bottleneck_interface(self):
        from model.mdm_scene_unet import TemporalUnet

        class RealRouting(_NativeRouting):
            def __init__(self):
                super().__init__()
                self.core.dim_mults = (1, 2)
                self.core.unet = TemporalUnet(input_dim=201, cond_dim=8, dim=8,
                    dim_mults=self.core.dim_mults, attention=False, adagn=True)

            def _core_pass(self, noisy, timestep, batch, scene, control_mask,
                           use_scene, use_control, query_motion=None):
                value = self.core.unet(noisy.permute(1, 0, 2), noisy.new_zeros(len(noisy), 8))
                return value.permute(1, 0, 2)

        model = RealRouting().eval()
        baseline = model(self.proposal, self.times, self.batch())
        attach_scene_fusion(model)
        self.assertEqual(model.scene_fusion_v2.channels, 16)
        self.assertTrue(torch.equal(baseline, model(self.proposal, self.times, self.batch())))
        self.nonzero(model.scene_fusion_v2)
        candidate = model(self.proposal, self.times, self.batch())
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertFalse(torch.equal(baseline, candidate))

    def test_routing_is_cleared_on_exception_and_duplicate_attachment_is_rejected(self):
        model = attach_scene_fusion(_NativeRouting().eval())
        with self.assertRaises(ValueError): attach_scene_fusion(model)
        def fail(*_args): raise RuntimeError('deliberate injected failure')
        handle = model.scene_fusion_v2.register_forward_pre_hook(fail)
        with self.assertRaisesRegex(RuntimeError, 'deliberate injected failure'):
            model(self.proposal, self.times, self.batch())
        self.assertIsNone(model._scene_fusion_routing.local.get())
        self.assertIsNone(model._scene_fusion_routing.active.get())
        handle.remove()
        self.assertTrue(torch.isfinite(model(self.proposal, self.times, self.batch())).all())

    @unittest.skipUnless(os.environ.get('SCENEMI_SCENE_FUSION_GPU_TESTS') == '1',
                         'Set SCENEMI_SCENE_FUSION_GPU_TESTS=1 for optional CUDA regression')
    def test_cuda_forward_and_backward_are_deterministic(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for frames, clock in ((192, 48), (7, 3), (3, 8)):
                module = RegionalSceneFusion(32).cuda().train(); self.nonzero(module)
                inputs = [torch.randn(2, 32, clock, device='cuda'),
                          torch.randn(2, 16, 64, device='cuda'),
                          torch.randn(2, frames, 22, 64, device='cuda')]
                proposal = torch.randn(2, frames, 201, device='cuda')
                gv = torch.ones(2, 16, device='cuda', dtype=torch.bool)
                lv = torch.ones(2, frames, 22, device='cuda', dtype=torch.bool); lv[:, 1::3, 0] = False
                alpha = torch.tensor([.3, .8], device='cuda')
                results = []
                for _ in range(2):
                    module.zero_grad(set_to_none=True)
                    activation, global_tokens, local_tokens = [v.clone().requires_grad_() for v in inputs]
                    output = module(activation, global_tokens, gv, proposal, alpha, local_tokens, lv)
                    output.square().mean().backward()
                    gradients = [p.grad for p in module.parameters()]
                    gradients += [activation.grad, global_tokens.grad, local_tokens.grad]
                    self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
                    results.append([output.detach().cpu()] + [g.detach().cpu() for g in gradients])
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == '__main__':
    unittest.main()
