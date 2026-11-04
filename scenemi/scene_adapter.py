"""Frozen motion prior with body-relative, observed-scene residual adapters.

The first denoiser pass locates predicted skin queries. Trainable, zero-ended
adapters inject relations into the bottleneck and both decoder resolutions of
the second pass. Original source files, parameters, shape and observations are
untouched. All variants share the same adapter parameterization.
"""
from contextlib import contextmanager

import torch
from torch import nn

from scenemi.geometry.scene_features import FEATURE_NAMES, ObservedNeighborhoods, relation_features
from scenemi.scene_residual import bounded_translation
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


VARIANTS = ('full', 'motion_only', 'points_only', 'late_only', 'ungated')


def resize_temporal(value, length):
    """Half-pixel linear reduction using disjoint slices, including on CUDA.

    The sealed U-Net uses 128/64/32 frames. Integer even reductions sample
    the two central values in each block. Unlike interpolate's CUDA backward,
    these slices have deterministic gradients without changing global policy.
    """
    source = value.shape[-1]
    if source == length:
        return value
    if source % length or (source//length) % 2:
        raise ValueError('Expected an even integer temporal reduction')
    stride = source//length
    left = stride//2-1
    return .5*value[..., left::stride]+.5*value[..., left+1::stride]


class RelationAdapter(nn.Module):
    def __init__(self, widths, hidden=128):
        super().__init__()
        self.anchor_ids = nn.Embedding(67, 8)
        self.token = nn.Sequential(nn.Linear(12+len(FEATURE_NAMES)+8+1, 64), nn.SiLU(),
                                   nn.Linear(64, 32), nn.SiLU())
        self.temporal = nn.Sequential(nn.Conv1d(22*32, hidden, 5, padding=2), nn.SiLU(),
                                     nn.Conv1d(hidden, hidden, 5, padding=2), nn.SiLU())
        self.projections = nn.ModuleList(nn.Conv1d(hidden, width, 1, bias=False) for width in widths)
        for layer in self.projections:
            nn.init.zeros_(layer.weight)

    def forward(self, body, geometry, gate, labels, signal):
        embedding = self.anchor_ids.weight[None, None].expand(*body.shape[:2], -1, -1)
        signal = signal[:, None, None, None].expand(*body.shape[:3], 1)
        tokens = self.token(torch.cat((body, geometry, embedding, signal), -1))*gate[..., None]
        regions = torch.stack([torch.stack([tokens[b, :, labels[b] == j].mean(1)
                            for j in range(22)], 1) for b in range(len(tokens))])
        hidden = self.temporal(regions.flatten(2).transpose(1, 2))
        return [layer(hidden) for layer in self.projections]


@contextmanager
def inject_residuals(unet, residuals, gate, *, late_only=False):
    """Ordinary scoped hooks avoid editing or copying the sealed U-Net."""
    modules = [unet.mid_block2] + [row[1] for row in unet.ups]
    handles = []

    def scaled(value, length):
        return resize_temporal(value, length)*resize_temporal(gate[:, None], length)

    try:
        if late_only:
            # Same three projections, all applied at the final hidden layer.
            # Channel pooling is fixed, not a separate trainable output head.
            def final_hook(module, inputs, output):
                width = output.shape[1]
                values = [scaled(v, output.shape[-1]).reshape(len(v), -1, width,
                          output.shape[-1]).mean(1) for v in residuals]
                return output+sum(values)
            handles.append(unet.final_conv[0].register_forward_hook(final_hook))
        else:
            for module, value in zip(modules, residuals):
                def hook(module, inputs, output, value=value):
                    return output+scaled(value, output.shape[-1])
                handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


class GeometryConditionedPrior(nn.Module):
    def __init__(self, base, policy, variant='full'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown conditioning variant')
        self.base = base.requires_grad_(False)
        self.policy, self.variant = dict(policy), variant
        unet = base.core.unet
        modules = [unet.mid_block2] + [row[1] for row in unet.ups]
        widths = [module.time_mlp[1].out_features//2 for module in modules]
        self.adapter = RelationAdapter(widths, policy['hidden'])
        self._sampling_scene = None
        self.train(False)

    @property
    def alphas(self):
        return self.base.alphas

    def train(self, mode=True):
        super().train(False)
        self.training = mode
        self.adapter.train(mode)
        return self

    def scene_tokens(self, batch):
        if self.variant == 'motion_only':
            return None
        return [ObservedNeighborhoods(batch['local_history_points'][i], batch['local_history_valid'][i],
            batch['local_frame_points'][i], batch['local_frame_owner'][i]) for i in range(len(batch['camera']))]

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True, use_control=True,
                control_mask=None, return_details=False):
        if noisy_motion.shape[1:] != (128, 201) or not torch.isfinite(noisy_motion).all():
            raise ValueError('Expected finite 128-frame native motion')
        with torch.no_grad():
            conditioned, observations, shape, formats = self.base.conditions(batch, use_control, control_mask)
            empty = (noisy_motion.new_zeros(len(noisy_motion), self.base.core.latent_dim), None, None, None)
            raw_base = self.base._core_pass(noisy_motion, timestep, conditioned, empty, control_mask, False, use_control)
            proposal = reconcile_motion(raw_base, shape['rest'])
        details = dict(raw_motion=raw_base, baseline_motion=proposal, observations=observations,
            conditioned_observations=conditioned, body_format=formats, **shape)
        if not use_scene:
            return (proposal, details) if return_details else proposal
        with torch.no_grad():
            anchors, labels = self.base.skin(proposal, shape, formats)
            rotations = global_rotations(proposal)
            rotations = torch.stack([rotations[i, :, labels[i]] for i in range(len(proposal))])
            velocity = torch.zeros_like(anchors)
            velocity[:, 1:] = torch.diff(anchors, dim=1)*30
            cp = observations['camera'][..., :3]*2
            body = torch.cat(((anchors-cp[:, :, None])/2, (velocity/.2).clamp(-20, 20),
                              rotations[..., :, 0], rotations[..., :, 1]), -1)
            if self.variant == 'motion_only':
                geometry = body.new_zeros(*body.shape[:-1], len(FEATURE_NAMES))
                gate = body.new_ones(body.shape[:-1])
                diagnostic = body.new_zeros(len(body), 5)
            else:
                cache = self._sampling_scene[1] if self._sampling_scene is not None else None
                geometry, gate, diagnostic = relation_features(anchors, rotations, batch,
                    self.policy['geometry'], cache)
                if self.variant == 'points_only':
                    geometry = geometry.clone()
                    geometry[..., 4:11] = 0.
                    gate = torch.maximum(.25*geometry[..., 12]*torch.exp(-geometry[..., 3]), geometry[..., 11])
                elif self.variant == 'ungated':
                    gate = torch.ones_like(gate)
        signal = self.alphas[timestep].sqrt()
        residuals = self.adapter(body, geometry, gate, labels, signal)
        frame_gate = gate.amax(-1)*self.alphas[timestep, None].pow(self.policy['noise_gate_power'])
        with inject_residuals(self.base.core.unet, residuals, frame_gate,
                              late_only=self.variant == 'late_only'):
            raw = self.base._core_pass(noisy_motion, timestep, conditioned, empty, control_mask, False, use_control)
        # Preserve pose validity and use identical correction bounds in every
        # trained variant. These are bounds on each x0 call, not on the final
        # sampled trajectory or anatomical rotation angles.
        bounded = raw.clone()
        root_delta = bounded_translation((raw[..., :3]-raw_base[..., :3])*2, self.policy['root_limit_m'])
        bounded[..., :3] = raw_base[..., :3]+root_delta/2
        limit = self.policy['rotation_6d_limit']
        bounded[..., 3:135] = raw_base[..., 3:135]+limit*torch.tanh((raw[..., 3:135]-raw_base[..., 3:135])/limit)
        prediction = reconcile_motion(bounded, shape['rest'])
        details.update(raw_motion=bounded, local_coverage=diagnostic[:, 0],
                       geometry_diagnostics=diagnostic, relation_features=geometry)
        return (prediction, details) if return_details else prediction

    def adapter_state(self):
        return {name:value.detach().cpu().clone() for name,value in self.adapter.state_dict().items()}

    def load_adapter_state(self, state):
        self.adapter.load_state_dict(state, strict=True)
