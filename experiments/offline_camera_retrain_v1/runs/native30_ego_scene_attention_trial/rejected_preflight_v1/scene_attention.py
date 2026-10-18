"""Versioned scene cross-attention inside the shared temporal U-Net.

Attach after loading a historical native30 checkpoint.  Only the second
``_core_pass`` reads existing global patch and local body-region features at
``TemporalUnet.mid_block2``.  The proposal, scene queries and historical source
files are unchanged.  An exactly zero output projection preserves the loaded
model at initialization; ``use_scene=False`` bypasses the new module entirely.
Checkpoints use the single additional ``scene_cross_attention.`` key prefix.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


POLICY = 'native30-second-pass-bottleneck-scene-cross-attention-v1'


class SceneCrossAttention(nn.Module):
    """One full-body temporal feature queries 16 global and 22 local tokens.

    Local evidence is pooled to the bottleneck clock with validity weighting.
    Attention is over scene/body-region evidence at that clock, not between
    temporal queries.  A null key makes empty/padded contexts finite; an
    evidence gate makes the all-empty result exactly the original activation.
    """

    def __init__(self, channels, token_dim=64, heads=4,
                 include_global=True, include_local=True):
        super().__init__()
        if channels < 1 or token_dim != 64 or heads < 1 or token_dim % heads:
            raise ValueError('Positive channels and divisible 64-dimensional scene tokens required')
        if not include_global and not include_local:
            raise ValueError('At least one scene token source is required')
        self.channels, self.token_dim, self.heads = int(channels), int(token_dim), int(heads)
        self.include_global, self.include_local = bool(include_global), bool(include_local)
        self.query_projection = nn.Linear(self.channels, self.token_dim)
        self.attention = nn.MultiheadAttention(self.token_dim, self.heads,
                                               dropout=0., batch_first=True)
        self.output_projection = nn.Linear(self.token_dim, self.channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, activation, global_tokens, global_valid,
                local_tokens=None, local_valid=None):
        if activation.ndim != 3 or activation.shape[1] != self.channels:
            raise ValueError('Expected [batch,bottleneck_channels,time] activation')
        batch, _, clock = activation.shape
        tokens, masks = [], []
        if self.include_global:
            if global_tokens.shape != (batch, 16, self.token_dim) or global_valid.shape != (batch, 16):
                raise ValueError('Expected 16 existing global scene patch tokens')
            valid = global_valid.to(device=activation.device, dtype=torch.bool)
            values = global_tokens.to(activation)
            values = torch.where(valid[..., None], values, 0.)
            tokens.append(values[:, None].expand(-1, clock, -1, -1))
            masks.append(valid[:, None].expand(-1, clock, -1))
        if self.include_local:
            if local_tokens is None:
                local = activation.new_zeros(batch, clock, 22, self.token_dim)
                valid = torch.zeros(batch, clock, 22, device=activation.device, dtype=torch.bool)
            else:
                if local_tokens.ndim != 4 or local_tokens.shape[0] != batch or local_tokens.shape[2:] != (22, self.token_dim):
                    raise ValueError('Expected [batch,frames,22,64] local region tokens')
                if local_valid is None or local_valid.shape != local_tokens.shape[:3]:
                    raise ValueError('Local region tokens require explicit validity')
                valid = local_valid.to(device=activation.device, dtype=torch.bool)
                values = torch.where(valid[..., None], local_tokens.to(activation), 0.)
                frames = values.shape[1]
                numerator = F.adaptive_avg_pool1d(values.permute(0, 2, 3, 1).reshape(
                    batch, 22 * self.token_dim, frames), clock)
                denominator = F.adaptive_avg_pool1d(valid.transpose(1, 2).to(activation), clock)
                local = numerator.reshape(batch, 22, self.token_dim, clock).permute(0, 3, 1, 2)
                local = local / denominator.transpose(1, 2)[..., None].clamp_min(1e-8)
                valid = denominator.transpose(1, 2) > 0
                local = torch.where(valid[..., None], local, 0.)
            tokens.append(local)
            masks.append(valid)
        values = torch.cat(tokens, dim=2)
        valid = torch.cat(masks, dim=2)
        has_evidence = valid.any(-1)
        values = torch.cat((values, activation.new_zeros(batch, clock, 1, self.token_dim)), dim=2)
        padding = torch.cat((~valid, torch.zeros(batch, clock, 1,
                            device=activation.device, dtype=torch.bool)), dim=2)
        query = self.query_projection(activation.transpose(1, 2)).reshape(batch * clock, 1, self.token_dim)
        keys = values.reshape(batch * clock, -1, self.token_dim)
        attended, _ = self.attention(query, keys, keys,
            key_padding_mask=padding.reshape(batch * clock, -1), need_weights=False)
        delta = self.output_projection(attended[:, 0]).reshape(batch, clock, self.channels)
        delta = delta * has_evidence[..., None]
        return activation + delta.transpose(1, 2)


class _Routing:
    def __init__(self):
        self.local = ContextVar('native30_scene_attention_local', default=None)
        self.active = ContextVar('native30_scene_attention_active', default=None)


def attach_scene_attention(model, *, token_dim=64, heads=4,
                           include_global=True, include_local=True):
    """Attach in place; preserve original parameter values and trainable flags.

    For historical weights: strict-load first, then attach.  For this variant's
    checkpoint: construct base, attach with the saved settings, then strict-load.
    The existing native30 forwards already require serial calls on each model.
    """
    if hasattr(model, 'scene_cross_attention') or hasattr(model, '_scene_attention_routing'):
        raise ValueError('Scene cross-attention is already attached')
    core = model.core
    if not hasattr(core.unet, 'mid_block2') or getattr(core, 'attention', False):
        raise ValueError('An existing temporal U-Net with attention=False is required')
    channels = int(core.latent_dim * core.dim_mults[-1])
    parameter = next(model.parameters())
    # Initializing the new branch must not shift later training-noise or
    # dropout draws in a paired control/candidate experiment.
    with torch.random.fork_rng(devices=[]):
        module = SceneCrossAttention(channels, token_dim, heads, include_global, include_local)
    module.to(device=parameter.device, dtype=parameter.dtype)
    module.train(model.training)
    model.add_module('scene_cross_attention', module)
    routing = _Routing()
    model._scene_attention_routing = routing
    original_forward = model.forward
    original_core_pass = model._core_pass
    original_geometry = model.local_geometry

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        token = routing.local.set(None)
        try:
            return original_forward(*args, **kwargs)
        finally:
            routing.local.reset(token)

    @wraps(original_geometry)
    def local_geometry(self, *args, **kwargs):
        result = original_geometry(*args, **kwargs)
        context, _, _, geometry = result
        valid = torch.stack([details['region_valid'] for details in geometry])
        routing.local.set((context, valid))
        return result

    @wraps(original_core_pass)
    def core_pass(self, noisy, timestep, batch, scene, control_mask, use_scene,
                  use_control, query_motion=None):
        active = None
        # Native30's first call has no query_motion.  The second call supplies
        # the proposal, while retaining the original noisy diffusion input.
        if use_scene and query_motion is not None:
            local = routing.local.get()
            active = (scene[1], scene[2], *(local if local is not None else (None, None)))
        token = routing.active.set(active)
        try:
            return original_core_pass(noisy, timestep, batch, scene, control_mask,
                                      use_scene, use_control, query_motion=query_motion)
        finally:
            routing.active.reset(token)

    def inject(_layer, _inputs, output):
        active = routing.active.get()
        if active is None:
            return output
        return model.scene_cross_attention(output, *active)

    routing.hook = core.unet.mid_block2.register_forward_hook(inject)
    model.forward = MethodType(forward, model)
    model.local_geometry = MethodType(local_geometry, model)
    model._core_pass = MethodType(core_pass, model)
    return model


def scene_attention_metadata(model):
    module = getattr(model, 'scene_cross_attention', None)
    if not isinstance(module, SceneCrossAttention):
        raise ValueError('Scene cross-attention is not attached')
    return dict(policy=POLICY, state_prefix='scene_cross_attention.',
        hook='core.unet.mid_block2', pass_policy='second_core_pass_only',
        query='temporal bottleneck frame features',
        key_value='existing global patch tokens and validity-pooled local body-region tokens plus null',
        channels=module.channels, token_dim=module.token_dim, heads=module.heads,
        include_global=module.include_global, include_local=module.include_local,
        global_token_count=16, local_token_count=22, null_token_count=1,
        parameters=sum(parameter.numel() for parameter in module.parameters()),
        initialized_output='exactly zero linear projection', use_scene_gate=True,
        extra_scene_encoder_calls=0, extra_geometry_queries=0,
        main_unet_self_attention=False)
