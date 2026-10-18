"""CPU checks for the isolated second-pass U-Net scene attention variant."""
import io
import os
import unittest

import torch
from torch import nn

from .scene_attention import (SceneCrossAttention, attach_scene_attention,
                              scene_attention_metadata)


class _NativeRouting(nn.Module):
    """Small native-shaped two-pass caller with a real temporal Conv1d trunk."""
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.latent_dim = 8
        self.core.dim_mults = (1,)
        self.core.attention = False
        self.core.unet = nn.Module()
        self.core.unet.mid_block2 = nn.Conv1d(8, 8, 3, padding=1)

    def _core_pass(self, noisy, timestep, batch, scene, control_mask, use_scene,
                   use_control, query_motion=None):
        return self.core.unet.mid_block2(noisy)

    def local_geometry(self, proposal, timestep, batch):
        valid = batch['local_valid']
        return batch['local_tokens'], None, None, [dict(region_valid=v) for v in valid]

    def forward(self, noisy, timestep, batch, *, use_scene=True):
        scene = batch['scene']
        with torch.no_grad():
            proposal = self._core_pass(noisy, timestep, batch, scene, None, use_scene, True)
        if use_scene:
            self.local_geometry(proposal, timestep, batch)
        return self._core_pass(noisy, timestep, batch, scene, None, use_scene, True,
                               query_motion=proposal)


class SceneAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.activation = torch.randn(2, 8, 4)
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
            module.output_projection.bias.fill_(.03)

    def test_zero_initialization_preserves_values_and_attachment_rng(self):
        model = _NativeRouting().eval()
        before = model(self.activation, None, self.batch())
        flags = {name: value.requires_grad for name, value in model.named_parameters()}
        state = torch.random.get_rng_state().clone()
        attach_scene_attention(model)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(before, model(self.activation, None, self.batch())))
        self.assertTrue(all(value.requires_grad == flags[name] for name, value in
                            model.named_parameters() if name in flags))
        self.assertEqual(scene_attention_metadata(model)['pass_policy'], 'second_core_pass_only')

    def test_scene_off_exact_and_second_pass_only(self):
        model = _NativeRouting().eval()
        control = model(self.activation, None, self.batch(), use_scene=False)
        attach_scene_attention(model)
        self.nonzero(model.scene_cross_attention)
        calls = []
        handle = model.scene_cross_attention.register_forward_hook(lambda *_: calls.append(True))
        result = model(self.activation, None, self.batch(), use_scene=False)
        self.assertTrue(torch.equal(control, result))
        self.assertEqual(calls, [])
        on = model(self.activation, None, self.batch(), use_scene=True)
        self.assertEqual(len(calls), 1)
        self.assertFalse(torch.equal(control, on))
        self.assertIsNone(model._scene_attention_routing.local.get())
        self.assertIsNone(model._scene_attention_routing.active.get())
        handle.remove()

    def test_nonzero_projection_reads_global_and_local_scene(self):
        module = SceneCrossAttention(8).eval()
        self.nonzero(module)
        output = module(self.activation, self.global_tokens, self.global_valid,
                        self.local_tokens, self.local_valid)
        changed_global = module(self.activation, self.global_tokens + 1, self.global_valid,
                                self.local_tokens, self.local_valid)
        changed_local = module(self.activation, self.global_tokens, self.global_valid,
                               self.local_tokens + 1, self.local_valid)
        self.assertFalse(torch.allclose(output, changed_global))
        self.assertFalse(torch.allclose(output, changed_local))

    def test_padding_and_empty_are_finite_and_empty_is_identity(self):
        module = SceneCrossAttention(8).eval()
        self.nonzero(module)
        gv = torch.zeros_like(self.global_valid)
        lv = torch.zeros_like(self.local_valid)
        nan_global = torch.full_like(self.global_tokens, float('nan'))
        nan_local = torch.full_like(self.local_tokens, float('nan'))
        result = module(self.activation, nan_global, gv, nan_local, lv)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result, self.activation))
        gv[:, 0] = True
        lv[:, 2:4, 0] = True
        nan_global[:, 0] = self.global_tokens[:, 0]
        nan_local[:, 2:4, 0] = self.local_tokens[:, 2:4, 0]
        padded = module(self.activation, nan_global, gv, nan_local, lv)
        zero_global = torch.where(gv[..., None], self.global_tokens, 0.)
        zero_local = torch.where(lv[..., None], self.local_tokens, 0.)
        unpadded = module(self.activation, zero_global, gv, zero_local, lv)
        self.assertTrue(torch.isfinite(padded).all())
        self.assertTrue(torch.equal(padded, unpadded))

    def test_initial_output_projection_receives_training_gradient(self):
        module = SceneCrossAttention(8)
        output = module(self.activation, self.global_tokens, self.global_valid,
                        self.local_tokens, self.local_valid)
        output.square().mean().backward()
        gradient = module.output_projection.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_nondivisible_pool_preserves_validity_weighted_adaptive_bins(self):
        module = SceneCrossAttention(8).eval()
        observed_keys = []
        handle = module.attention.register_forward_pre_hook(
            lambda _module, args: observed_keys.append(args[1].detach().clone()))
        local = torch.arange(7, dtype=torch.float32)[None, :, None, None].expand(2, 7, 22, 64).clone()
        valid = torch.ones(2, 7, 22, dtype=torch.bool)
        valid[:, 1, 0] = False
        module(self.activation[:, :, :3], self.global_tokens, self.global_valid, local, valid)
        keys = observed_keys[0].reshape(2, 3, 39, 64)
        expected = torch.tensor([1., 3., 5.])
        self.assertTrue(torch.equal(keys[0, :, 17, 0], expected))
        # First joint's first bin contains only clock positions 0 and 2.
        self.assertEqual(float(keys[0, 0, 16, 0]), 1.)
        handle.remove()

    def test_strict_state_round_trip(self):
        model = attach_scene_attention(_NativeRouting().eval())
        self.nonzero(model.scene_cross_attention)
        expected = model(self.activation, None, self.batch())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = attach_scene_attention(_NativeRouting().eval())
        restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
        self.assertTrue(torch.equal(expected, restored(self.activation, None, self.batch())))
        new_keys = [key for key in model.state_dict() if key.startswith('scene_cross_attention.')]
        self.assertTrue(new_keys)
        self.assertTrue(all(key.startswith(('core.', 'scene_cross_attention.')) for key in model.state_dict()))

    def test_real_temporal_unet_bottleneck_interface(self):
        from model.mdm_scene_unet import TemporalUnet

        class RealTemporalRouting(_NativeRouting):
            def __init__(self):
                super().__init__()
                self.core.dim_mults = (1, 2)
                self.core.unet = TemporalUnet(input_dim=8, cond_dim=8, dim=8,
                    dim_mults=self.core.dim_mults, attention=False, adagn=True)

            def _core_pass(self, noisy, timestep, batch, scene, control_mask,
                           use_scene, use_control, query_motion=None):
                value = self.core.unet(noisy.permute(2, 0, 1), noisy.new_zeros(len(noisy), 8))
                return value.permute(1, 2, 0)

        model = RealTemporalRouting().eval()
        baseline = model(self.activation, None, self.batch())
        attach_scene_attention(model)
        self.assertEqual(model.scene_cross_attention.channels, 16)
        self.assertTrue(torch.equal(baseline, model(self.activation, None, self.batch())))
        self.nonzero(model.scene_cross_attention)
        candidate = model(self.activation, None, self.batch())
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertFalse(torch.equal(baseline, candidate))

    @unittest.skipUnless(os.environ.get('SCENEMI_SCENE_ATTENTION_GPU_TESTS') == '1',
                         'Set SCENEMI_SCENE_ATTENTION_GPU_TESTS=1 for CUDA regression')
    def test_cuda_pool_backward_is_deterministic(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for frames, clock in ((192, 48), (7, 3), (3, 8)):
                module = SceneCrossAttention(1024).cuda().train()
                self.nonzero(module)
                inputs = [torch.randn(2, 1024, clock, device='cuda'),
                          torch.randn(2, 16, 64, device='cuda'),
                          torch.randn(2, frames, 22, 64, device='cuda')]
                gv = torch.ones(2, 16, device='cuda', dtype=torch.bool)
                lv = torch.ones(2, frames, 22, device='cuda', dtype=torch.bool)
                lv[:, 1::3, 0] = False
                results = []
                for _ in range(2):
                    module.zero_grad(set_to_none=True)
                    activation, global_tokens, local_tokens = [value.clone().requires_grad_() for value in inputs]
                    output = module(activation, global_tokens, gv, local_tokens, lv)
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
