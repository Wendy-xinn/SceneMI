"""Mode-gated EXO observation adapters over an immutable EGO motion prior."""
import copy
from contextlib import contextmanager

import torch
from torch import nn

from scenemi.observations import public_observations, body_condition
from scenemi.scene_adapter import resize_temporal
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


class ExoObservationAdapter(nn.Module):
    def __init__(self, widths, hidden=128):
        super().__init__()
        self.joint_ids = nn.Embedding(22, 8)
        self.tokens = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        # The noisy root/rotations let observation corrections depend on the
        # current denoising state. No clean/hidden target enters this branch.
        self.temporal = nn.Sequential(nn.Conv1d(22*32+135+1, hidden, 5, padding=2), nn.SiLU(),
                                      nn.Conv1d(hidden, hidden, 5, padding=2), nn.SiLU())
        self.projections = nn.ModuleList(nn.Conv1d(hidden, width, 1) for width in widths)
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, observations, noisy, signal):
        meta = observations['observation_meta']
        ids = self.joint_ids.weight[None, None].expand(*noisy.shape[:2], -1, -1)
        values = torch.cat((observations['trajectory'][..., :3]*meta[..., 2:3],
                            meta[..., :4], observations['observation_uncertainty'], ids), -1)
        tokens = self.tokens(values)*(meta[..., :2].amax(-1) > 0)[..., None]
        x = torch.cat((tokens.flatten(2), noisy[..., :135],
                       signal[:, None, None].expand(-1, noisy.shape[1], 1)), -1)
        hidden = self.temporal(x.transpose(1, 2))
        return [projection(hidden) for projection in self.projections]


@contextmanager
def inject_exo(unet, residuals):
    modules = [unet.mid_block2]+[row[1] for row in unet.ups]
    handles = []
    try:
        for module, residual in zip(modules, residuals):
            def hook(module, inputs, output, residual=residual):
                return output+resize_temporal(residual, output.shape[-1])
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def merge_modes(ego_value, exo_value, ego_mask):
    if isinstance(ego_value, dict):
        return {key: merge_modes(ego_value[key], exo_value[key], ego_mask) for key in ego_value}
    value = ego_value.new_empty(len(ego_mask), *ego_value.shape[1:])
    value[ego_mask], value[~ego_mask] = ego_value, exo_value
    return value


class ExoAdaptedPrior(nn.Module):
    def __init__(self, base, hidden=128):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.exo_shape_encoder = copy.deepcopy(base.shape_encoder).requires_grad_(True)
        unet = base.core.unet
        modules = [unet.mid_block2]+[row[1] for row in unet.ups]
        widths = [module.time_mlp[1].out_features//2 for module in modules]
        self.adapter = ExoObservationAdapter(widths, hidden)
        self._sampling_scene = None
        self.train(False)

    @property
    def alphas(self):
        return self.base.alphas

    @property
    def shape_encoder(self):
        return self.exo_shape_encoder

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, noisy_motion, timestep, batch, *, use_scene=False, use_control=True,
                control_mask=None, return_details=False):
        if use_scene:
            raise ValueError('AMASS EXO adaptation has no scene input')
        if noisy_motion.shape[1:] != (128, 201) or not torch.isfinite(noisy_motion).all():
            raise ValueError('Expected finite [B,128,201] motion')
        ego = batch['camera_context'][:, 0] > 0
        if ego.all():
            # Exact original path: no new parameter is evaluated.
            return self.base(noisy_motion, timestep, batch, use_scene=False,
                             use_control=use_control, control_mask=control_mask,
                             return_details=return_details)
        if ego.any():
            def part(mask):
                payload = {k:v[mask] for k,v in batch.items() if torch.is_tensor(v)}
                return self(noisy_motion[mask], timestep[mask], payload, use_control=use_control,
                    control_mask=control_mask[mask] if control_mask is not None else None,
                    return_details=True)
            ep, ed = part(ego)
            xp, xd = part(~ego)
            prediction, details = merge_modes(ep, xp, ego), merge_modes(ed, xd, ego)
            return (prediction, details) if return_details else prediction
        observations = public_observations(batch, enabled=use_control)
        if control_mask is not None:
            observations['observation_meta'] = observations['observation_meta'].clone()
            observations['observation_meta'][..., :4] *= control_mask[..., None]
            observations = public_observations(observations, enabled=use_control)
        betas = self.exo_shape_encoder(observations)
        shape = self.base.shape(betas)
        conditioned = body_condition(observations, shape)
        empty = (noisy_motion.new_zeros(len(noisy_motion), self.base.core.latent_dim), None, None, None)
        residuals = self.adapter(observations, noisy_motion, self.alphas[timestep].sqrt())
        with inject_exo(self.base.core.unet, residuals):
            raw = self.base._core_pass(noisy_motion, timestep, conditioned, empty,
                                       control_mask, False, use_control)
        prediction = reconcile_motion(raw, shape['rest'])
        details = dict(raw_motion=raw, predicted_betas=betas, **shape,
                       observations=observations, conditioned_observations=conditioned)
        return (prediction, details) if return_details else prediction

    def adapter_state(self):
        return dict(adapter=self.adapter.state_dict(), shape=self.exo_shape_encoder.state_dict())

    def load_adapter_state(self, state):
        self.adapter.load_state_dict(state['adapter'], strict=True)
        self.exo_shape_encoder.load_state_dict(state['shape'], strict=True)
