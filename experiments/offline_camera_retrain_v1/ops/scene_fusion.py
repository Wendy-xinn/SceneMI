"""Regional scene fusion inside the second native30 temporal U-Net pass.

This opt-in module reuses the proposal, global scene tokens and local geometry
already computed by native30.  It adds no FK, scene encoding or surface query.
Historical source files and checkpoint keys remain untouched; new state lives
under ``scene_fusion_v2.``.  The last, bias-free projection starts exactly at
zero, preserving the historical output and attachment RNG state.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch
from torch import nn


POLICY = 'native30-second-pass-regional-scene-fusion-v2'
ARCHITECTURE = 'scene_fusion_v2'
STATE_PREFIX = 'scene_fusion_v2.'
IDENTITY_6D = (1., 0., 0., 0., 1., 0.)


def temporal_pool(values, clock, valid=None):
    """Deterministic validity-weighted bins, with no CUDA scatter/pooling op.

    ``values`` is [batch,frames,regions,features].  Nondivisible lengths use
    adaptive pooling's floor/ceil bins, including when clock exceeds frames.
    Invalid values are masked before reduction, so padded NaNs cannot leak.
    """
    if values.ndim != 4 or values.shape[1] < 1 or clock < 1:
        raise ValueError('Expected nonempty [batch,frames,regions,features] values')
    batch, frames, regions, features = values.shape
    if valid is None:
        valid = torch.ones(batch, frames, regions, device=values.device, dtype=torch.bool)
    elif valid.shape != values.shape[:3]:
        raise ValueError('Temporal values require matching region validity')
    valid = valid.to(device=values.device, dtype=torch.bool)
    values = torch.where(valid[..., None], values, 0.)
    if frames % clock == 0:
        width = frames // clock
        numerator = values.reshape(batch, clock, width, regions, features).sum(2)
        denominator = valid.reshape(batch, clock, width, regions).sum(2).to(values)
    else:
        bins = [(index * frames // clock,
                 ((index + 1) * frames + clock - 1) // clock) for index in range(clock)]
        numerator = torch.stack([values[:, lo:hi].sum(1) for lo, hi in bins], dim=1)
        denominator = torch.stack([valid[:, lo:hi].sum(1) for lo, hi in bins], dim=1).to(values)
    pooled = numerator / denominator[..., None].clamp_min(1)
    valid = denominator > 0
    return torch.where(valid[..., None], pooled, 0.), valid


class RegionalSceneFusion(nn.Module):
    """22 pose-aware queries read separate global/local scene memories.

    Regional attention outputs retain their ordered joint identity through a
    small shared fusion layer.  Local *hidden updates*, including its fusion
    bias, are multiplied by alpha_bar squared before the final linear map.
    Global updates are independent of that gate.  The last map has no bias:
    empty memories are an exact identity, even after training.
    """

    def __init__(self, channels, token_dim=64, heads=4, fusion_dim=128,
                 include_global=True, include_local=True):
        super().__init__()
        if channels < 1 or token_dim != 64 or heads < 1 or token_dim % heads or fusion_dim < 1:
            raise ValueError('Positive channels/fusion size and divisible 64-dimensional tokens required')
        if not include_global and not include_local:
            raise ValueError('At least one scene token source is required')
        self.channels, self.token_dim = int(channels), int(token_dim)
        self.heads, self.fusion_dim = int(heads), int(fusion_dim)
        self.include_global, self.include_local = bool(include_global), bool(include_local)
        self.query_projection = nn.Linear(self.channels, self.token_dim)
        self.region_embedding = nn.Embedding(22, self.token_dim)
        self.query_pose_projection = nn.Linear(9, self.token_dim)
        self.local_pose_projection = nn.Linear(9, self.token_dim)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.global_key_norm = nn.LayerNorm(self.token_dim)
        self.local_context_norm = nn.LayerNorm(self.token_dim)
        self.local_key_norm = nn.LayerNorm(self.token_dim)
        self.global_attention = nn.MultiheadAttention(self.token_dim, self.heads,
                                                     dropout=0., batch_first=True)
        self.local_attention = nn.MultiheadAttention(self.token_dim, self.heads,
                                                    dropout=0., batch_first=True)
        self.region_fusion = nn.Sequential(nn.Linear(22 * self.token_dim, self.fusion_dim), nn.SiLU())
        self.output_projection = nn.Linear(self.fusion_dim, self.channels, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def attend(self, attention, query, keys, valid):
        batch_clock = query.shape[0]
        keys = torch.where(valid[..., None], keys, 0.)
        keys = torch.cat((keys, keys.new_zeros(batch_clock, 1, self.token_dim)), dim=1)
        padding = torch.cat((~valid, torch.zeros(batch_clock, 1,
                              device=valid.device, dtype=torch.bool)), dim=1)
        attended, _ = attention(query, keys, keys, key_padding_mask=padding, need_weights=False)
        hidden = self.region_fusion(attended.flatten(1))
        return torch.where(valid.any(-1, keepdim=True), hidden, 0.)

    def forward(self, activation, global_tokens, global_valid, proposal, alpha_bar,
                local_tokens=None, local_valid=None):
        if activation.ndim != 3 or activation.shape[1] != self.channels:
            raise ValueError('Expected [batch,bottleneck_channels,time] activation')
        batch, _, clock = activation.shape
        if proposal.ndim != 3 or proposal.shape[0] != batch or proposal.shape[2] != 201:
            raise ValueError('Queries require the existing [batch,frames,201] native proposal')
        if alpha_bar.shape != (batch,):
            raise ValueError('One cumulative diffusion alpha_bar per window is required')
        # Native positions and rotation-6D are already present in reconciled
        # proposal.  Averaging them as query features requires no extra FK.
        proposal = proposal.detach().to(activation)
        rotations = (proposal[..., 3:135].reshape(batch, -1, 22, 6)
                     + proposal.new_tensor(IDENTITY_6D))
        pose_frames = torch.cat((proposal[..., 135:].reshape(batch, -1, 22, 3), rotations), dim=-1)
        pose, _ = temporal_pool(pose_frames, clock)
        region_ids = self.region_embedding.weight[None, None]
        query = (self.query_projection(activation.transpose(1, 2))[:, :, None]
                 + region_ids + self.query_pose_projection(pose))
        query = self.query_norm(query).reshape(batch * clock, 22, self.token_dim)
        global_hidden = activation.new_zeros(batch * clock, self.fusion_dim)
        local_hidden = torch.zeros_like(global_hidden)
        if self.include_global:
            if global_tokens.shape != (batch, 16, self.token_dim) or global_valid.shape != (batch, 16):
                raise ValueError('Expected 16 existing global patch tokens with validity')
            valid = global_valid.to(device=activation.device, dtype=torch.bool)
            values = torch.where(valid[..., None], global_tokens.to(activation), 0.)
            values = self.global_key_norm(values)
            values = values[:, None].expand(-1, clock, -1, -1).reshape(batch * clock, 16, self.token_dim)
            valid = valid[:, None].expand(-1, clock, -1).reshape(batch * clock, 16)
            global_hidden = self.attend(self.global_attention, query, values, valid)
        if self.include_local and local_tokens is not None:
            if local_tokens.ndim != 4 or local_tokens.shape[0] != batch or local_tokens.shape[2:] != (22, self.token_dim):
                raise ValueError('Expected [batch,frames,22,64] local region context')
            if local_tokens.shape[1] != proposal.shape[1] or local_valid is None:
                raise ValueError('Local context must match proposal frames and include validity')
            values, valid = temporal_pool(local_tokens.to(activation), clock, local_valid)
            local_pose, _ = temporal_pool(pose_frames, clock, local_valid)
            # Keep local K/V bound to their body-region identity and predicted
            # pose.  Invalid entries are masked again after these embeddings.
            values = self.local_key_norm(self.local_context_norm(values)
                                        + region_ids + self.local_pose_projection(local_pose))
            local_hidden = self.attend(self.local_attention, query,
                values.reshape(batch * clock, 22, self.token_dim),
                valid.reshape(batch * clock, 22))
        local_gate = alpha_bar.to(activation).square()[:, None, None].expand(-1, clock, -1)
        hidden = global_hidden + local_hidden * local_gate.reshape(batch * clock, 1)
        delta = self.output_projection(hidden).reshape(batch, clock, self.channels)
        return activation + delta.transpose(1, 2)


class _Routing:
    def __init__(self):
        self.local = ContextVar('native30_scene_fusion_local', default=None)
        self.active = ContextVar('native30_scene_fusion_active', default=None)


def attach_scene_fusion(model, *, token_dim=64, heads=4, fusion_dim=128,
                        include_global=True, include_local=True):
    """Attach after loading the parent, or before strict-loading v2 state.

    The original trainable flags remain unchanged.  Experiment runners choose
    their training scope and optimizer explicitly.  As in native30 itself,
    forward calls on a single model must be serial.
    """
    if any(hasattr(model, name) for name in
           ('scene_fusion_v2', '_scene_fusion_routing', '_scene_attention_routing')):
        raise ValueError('A scene-fusion/attention variant is already attached')
    core = model.core
    if not hasattr(core.unet, 'mid_block2') or getattr(core, 'attention', False):
        raise ValueError('An existing temporal U-Net with attention=False is required')
    if not hasattr(model, 'alphas') or model.alphas.ndim != 1:
        raise ValueError('The native cumulative diffusion alpha_bar schedule is required')
    channels = int(core.latent_dim * core.dim_mults[-1])
    parameter = next(model.parameters())
    with torch.random.fork_rng(devices=[]):
        module = RegionalSceneFusion(channels, token_dim, heads, fusion_dim,
                                     include_global, include_local)
    module.to(device=parameter.device, dtype=parameter.dtype)
    module.train(model.training)
    model.add_module('scene_fusion_v2', module)
    routing = _Routing()
    model._scene_fusion_routing = routing
    original_forward, original_geometry = model.forward, model.local_geometry
    original_core_pass = model._core_pass

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
        routing.local.set((context, torch.stack([item['region_valid'] for item in geometry])))
        return result

    @wraps(original_core_pass)
    def core_pass(self, noisy, timestep, batch, scene, control_mask, use_scene,
                  use_control, query_motion=None):
        active = None
        if use_scene and query_motion is not None:
            local = routing.local.get()
            active = (scene[1], scene[2], query_motion, self.alphas[timestep],
                      *(local if local is not None else (None, None)))
        token = routing.active.set(active)
        try:
            return original_core_pass(noisy, timestep, batch, scene, control_mask,
                                      use_scene, use_control, query_motion=query_motion)
        finally:
            routing.active.reset(token)

    def inject(_layer, _inputs, output):
        active = routing.active.get()
        return output if active is None else model.scene_fusion_v2(output, *active)

    routing.hook = core.unet.mid_block2.register_forward_hook(inject)
    model.forward = MethodType(forward, model)
    model.local_geometry = MethodType(local_geometry, model)
    model._core_pass = MethodType(core_pass, model)
    return model


def scene_fusion_metadata(model):
    module = getattr(model, 'scene_fusion_v2', None)
    if not isinstance(module, RegionalSceneFusion):
        raise ValueError('Regional scene fusion v2 is not attached')
    return dict(policy=POLICY, architecture=ARCHITECTURE, state_prefix=STATE_PREFIX,
        hook='core.unet.mid_block2', pass_policy='second_core_pass_only',
        channels=module.channels, token_dim=module.token_dim, heads=module.heads,
        fusion_dim=module.fusion_dim, include_global=module.include_global,
        include_local=module.include_local, query_count_per_bin=22,
        query='temporal bottleneck projection + region embedding + detached proposal position/rotation-6D',
        proposal_pose='existing window-world positions (m/2) and local rotation columns (residual-6D + identity); temporal bin mean; no FK',
        global_token_count=16, local_token_count=22, null_tokens_per_branch=1,
        global_local_attention='separate; region-ordered shared fusion; additive hidden updates',
        local_keys='normalized raw region context + region embedding + validity-pooled proposal position/rotation-6D projection',
        local_validity='existing region_valid (including enabled relation context)',
        local_gate='alpha_bar_squared_after_region_fusion_before_bias_free_output_projection',
        global_gate='validity only; independent of local alpha_bar_squared gate',
        empty_policy='per-branch evidence mask; bias-free final map gives exact identity',
        temporal_pool='deterministic validity-weighted floor/ceil bin reduction',
        query_pose_pool='uniform native-valid window frames; no GT frame weights',
        normalization='LayerNorm query/global K-V/raw local context/composed local K-V; affine defaults',
        attention_dropout=0., parameters=sum(p.numel() for p in module.parameters()),
        initialized_output='exactly zero bias-free linear projection',
        zero_output_state_keys=[STATE_PREFIX + 'output_projection.weight'],
        attachment_preserves_rng=True, use_scene_gate=True,
        extra_scene_encoder_calls=0, extra_geometry_queries=0, extra_fk_calls=0,
        main_unet_self_attention=False)
