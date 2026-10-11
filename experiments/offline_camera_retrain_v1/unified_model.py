"""Warm-startable typed trajectories and generated skin-to-scene attention."""
import numpy as np
import torch
from torch import nn

from experiments.offline_camera_retrain_v1.scene_model import OfflineSceneMI
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
from experiments.offline_camera_retrain_v1.unified_observations import observation_derivatives


class UnifiedSceneMI(OfflineSceneMI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.core.sparse_control_process.lin0
        self.core.sparse_control_process.lin0 = nn.Linear(22 * 26, old.out_features)
        self.anchor_embedding = nn.Embedding(67, 64)
        self.point_embedding = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64))
        self.surface_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.surface_projection = nn.Linear(64, self.core.cond_latent_dim)
        self.surface_contact = nn.Linear(64, 1)
        nn.init.zeros_(self.surface_projection.weight); nn.init.zeros_(self.surface_projection.bias)
        nn.init.zeros_(self.surface_contact.weight); nn.init.zeros_(self.surface_contact.bias)
        self.surface_bps_weight = 0.
        self.surface_enabled = True

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state = dict(state_dict); key = 'core.sparse_control_process.lin0.weight'
        if state[key].shape[-1] == 220:
            old = state[key].reshape(-1, 22, 10)
            new = old.new_zeros(old.shape[0], 22, 26); new[..., :10] = old
            state[key] = new.flatten(1)
            for name, value in self.state_dict().items():
                if name not in state:
                    state[name] = value
        return super().load_state_dict(state, strict=strict, assign=assign)

    def _forward(self, noisy, timestep, batch, **kwargs):
        def expand(module, inputs):
            sparse = inputs[0].reshape(*batch['trajectory'].shape[:2], 22, 10)
            meta = batch['observation_meta']; control = sparse[..., 9]
            pm, rm = control * meta[..., 0], control * meta[..., 1]
            position = torch.where(pm[..., None] > 0, batch['trajectory'][..., :3], 0.)
            rotation = torch.where(rm[..., None] > 0, batch['trajectory'][..., 3:], 0.)
            v, a, w, masks = observation_derivatives(batch['trajectory'], meta)
            masks = masks.to(noisy) * control[..., None]
            derivatives = torch.cat((v / .15, a / 1., w / np.deg2rad(20)), -1)
            derivatives = derivatives * masks.repeat_interleave(3, -1)
            values = torch.cat((position, rotation, pm[..., None], rm[..., None],
                                meta[..., 2:4] * control[..., None], meta[..., 4:5] * control[..., None],
                                derivatives.clamp(-20, 20), masks), -1)
            return (values.flatten(2),)
        handle = self.core.sparse_control_process.register_forward_pre_hook(expand)
        try:
            # Sanitize before the legacy transport; NaN * zero is not a mask.
            safe = dict(batch, trajectory=torch.nan_to_num(batch['trajectory']))
            return super().forward(noisy, timestep, safe, **kwargs)
        finally:
            handle.remove()

    def surface_features(self, motion, batch):
        rotations = global_rotations(motion)
        all_bps, all_valid, embeddings, logits = [], [], [], []
        for b, (asset, scene) in enumerate(zip(batch['skin_assets'], batch['surface_windows'])):
            vertices = skin_points(motion[b], asset)
            rotation = rotations[b, :, asset['labels']]
            nearest, valid, moving = scene.query(vertices.detach().cpu().numpy(), rotation.detach().cpu().numpy())
            nearest = torch.as_tensor(nearest, device=motion.device, dtype=motion.dtype)
            valid = torch.as_tensor(valid, device=motion.device)
            moving = torch.as_tensor(moving, device=motion.device, dtype=motion.dtype)
            # Neighbour identity is discrete; displacement/LBS/rotation are
            # differentiable once those observed neighbours have been selected.
            delta = nearest - vertices[:, :, None]
            local = torch.einsum('tjik,tjpi->tjpk', rotation, delta)
            local = torch.where(valid[..., None], local, 0.)
            token = self.point_embedding(torch.cat((local, local.norm(dim=-1, keepdim=True), moving[..., None]), -1))
            frames = len(motion[b]); query = self.anchor_embedding.weight[None].expand(frames, -1, -1).reshape(-1, 1, 64)
            token = token.reshape(-1, 13, 64)
            # Explicit null token prevents all-unknown attention producing NaN.
            token = torch.cat((token, torch.zeros_like(token[:, :1])), 1)
            padding = torch.cat((~valid.reshape(-1, 13), torch.zeros((frames * 67, 1), device=motion.device, dtype=torch.bool)), 1)
            context, _ = self.surface_attention(query, token, token, key_padding_mask=padding, need_weights=False)
            context = context.reshape(frames, 67, 64) * valid.any(-1)[..., None]
            contact = self.surface_contact(context)[..., 0]
            region_logits = motion.new_zeros(frames, 22)
            for region in range(22):
                region_logits[:, region] = contact[:, asset['labels'] == region].mean(-1)
            # Predicted contact feeds the scene context back to the denoiser.
            pooled = (context * (.5 + contact.sigmoid()[..., None])).mean(1)
            embeddings.append(self.surface_projection(pooled))
            all_bps.append(local[:, :, 0].clamp(-2, 2) / 2)
            all_valid.append(valid[:, :, 0]); logits.append(region_logits)
        return torch.stack(all_bps), torch.stack(all_valid), torch.stack(embeddings), torch.stack(logits)

    def forward(self, noisy_motion, timestep, batch, *, control_mask=None, use_scene=True, use_control=True):
        kwargs = dict(control_mask=control_mask, use_scene=use_scene, use_control=use_control)
        if not self.surface_enabled or not use_scene:
            return self._forward(noisy_motion, timestep, batch, **kwargs)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                proposal = self._forward(noisy_motion, timestep, batch, **kwargs)
        finally:
            self.train(was_training)
        bps, valid, embedding, _ = self.surface_features(proposal.detach(), batch)
        weight = bps.new_tensor([1. if mode == 'exo' else self.surface_bps_weight for mode in batch['mode']])[:, None, None, None]
        final = dict(batch, bps=(1 - weight) * batch['bps'] + weight * bps,
                     bps_valid=((weight[..., 0] < 1) & batch['bps_valid']) |
                               ((weight[..., 0] > 0) & valid), body_local_embedding=embedding)
        return self._forward(noisy_motion, timestep, final, **kwargs)

    def contact_logits(self, motion, batch, *, use_scene=True):
        legacy = super().contact_logits(motion, batch, use_scene=use_scene)
        if not self.surface_enabled or not use_scene:
            return legacy
        _, _, _, logits = self.surface_features(motion, batch)
        return legacy + logits
