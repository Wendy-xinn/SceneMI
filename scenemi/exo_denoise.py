"""EXO repair through internal adaptation of the pretrained denoiser.

All learned corrections pass through the frozen U-Net. A matched frozen
reference pass removes its absolute-generation bias, leaving a learned x0
residual around an estimated-motion anchor. EGO uses the original model.
"""
from contextlib import contextmanager

import torch
from torch import nn
from pytorch3d.transforms import matrix_to_axis_angle

from scenemi.exo_basis import TemporalBasisExoPrior
from scenemi.observations import public_observations, body_condition
from scenemi.scene_adapter import resize_temporal
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d, forward_kinematics


class DenoisingExoPrior(nn.Module):
    def __init__(self, base, *, hidden=128, root_limit_m=.5,
                 rotation_limit_rad=1.2, residual_gain=5.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.root_limit_m, self.rotation_limit_rad = root_limit_m, rotation_limit_rad
        self.residual_gain = residual_gain
        # Fixed, untrained temporal proposals; no weights from the 2k trial.
        self.geometry = TemporalBasisExoPrior(nn.Identity(), hidden=hidden,
            root_limit_m=.25, rotation_limit_rad=.6).requires_grad_(False)
        self.condition = nn.Sequential(
            nn.Conv1d(201+135+22+1+1+22+1, hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=16, dilation=8), nn.SiLU())
        blocks = self.injection_modules()
        widths = [block.time_mlp[1].out_features//2 for block in blocks]
        self.projections = nn.ModuleList(nn.Conv1d(hidden, width, 1) for width in widths)
        for layer in self.projections:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self._anchor_payload = self._anchor_value = None
        self.train(False)

    def injection_modules(self):
        unet = self.base.core.unet
        return ([row[1] for row in unet.downs] + [unet.mid_block1, unet.mid_block2]
                + [row[1] for row in unet.ups])

    @property
    def readout(self):
        # Compatibility with shared acceptance: this is an INTERNAL projection,
        # not a temporal output head bypassing the denoiser.
        return self.projections[-1]

    @property
    def alphas(self):
        return self.base.alphas

    def train(self, mode=True):
        super().train(mode); self.base.eval(); self.geometry.eval()
        return self

    def repair_state(self):
        return dict(condition=self.condition.state_dict(), projections=self.projections.state_dict())

    def load_repair_state(self, state):
        self.condition.load_state_dict(state['condition'], strict=True)
        self.projections.load_state_dict(state['projections'], strict=True)

    @torch.no_grad()
    def anchor(self, payload):
        if self._anchor_payload is payload:
            return self._anchor_value
        initial = payload['initial_motion']
        roots, angles = self.geometry.proposals(payload)
        weights = self.geometry.readout.bias.reshape(23, 6).softmax(-1)
        root = (weights[0, :, None]*roots).sum(2)*(1-payload['initial_root_reliability'][..., None])
        rotation = (weights[1:, :, None]*angles).sum(3)*(1-payload['initial_reliability'][..., None])
        root = root*(.25/root.norm(dim=-1, keepdim=True).clamp_min(.25))
        rotation = rotation*(.6/rotation.norm(dim=-1, keepdim=True).clamp_min(.6))
        moved = reconcile_motion(initial, payload['initial_rest'], translation_m=root,
                                  rotation_axis_angle=rotation)
        fixed = (payload['initial_root_reliability'] >= 1)&(payload['initial_reliability'] >= 1).all(-1)
        result = torch.where(fixed[..., None], initial, moved)
        self._anchor_payload, self._anchor_value = payload, result
        return result

    @contextmanager
    def inject(self, residuals):
        handles = []
        try:
            for module, residual in zip(self.injection_modules(), residuals):
                def hook(module, inputs, output, residual=residual):
                    return output+resize_temporal(residual, output.shape[-1])
                handles.append(module.register_forward_hook(hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, noisy, timestep, payload, *, return_details=False):
        ego = payload['camera_context'][:, 0] > 0
        if ego.all():
            return self.base(noisy, timestep, payload, use_scene=False, return_details=return_details)
        if ego.any():
            raise ValueError('This stage uses single-mode batches; split modes before dispatch')
        initial, rest = payload['initial_motion'], payload['initial_rest']
        if initial.shape != noisy.shape or initial.shape[1:] != (128, 201):
            raise ValueError('Expected finite complete estimated motion [B,128,201]')
        if not torch.isfinite(initial).all() or not torch.isfinite(rest).all():
            raise ValueError('Nonfinite estimated initialization')
        reliability = payload['initial_reliability'].clamp(0, 1)
        root_reliability = payload['initial_root_reliability'].clamp(0, 1)
        anchor = self.anchor(payload)
        observations = public_observations(payload)
        conditioned = body_condition(observations, dict(rest=rest,
            head_camera_translation=rest.new_zeros(len(rest), 3)))
        empty = (noisy.new_zeros(len(noisy), self.base.core.latent_dim), None, None, None)
        with torch.no_grad():
            reference = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
        signal = self.alphas[timestep].sqrt()[:, None, None].expand(-1, 128, 1)
        features = torch.cat((initial, noisy[..., :135], reliability,
            root_reliability[..., None], payload['estimate_available'][..., None],
            payload['joint_visible'], signal), -1)
        hidden = self.condition(features.transpose(1, 2))
        with self.inject([layer(hidden) for layer in self.projections]):
            adapted = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
        difference = (adapted[..., :135]-reference[..., :135])*self.residual_gain
        root = self.root_limit_m*torch.tanh(difference[..., :3]*2/self.root_limit_m)
        root = root/root.norm(dim=-1, keepdim=True).clamp_min(self.root_limit_m)*self.root_limit_m
        root = root*(1-root_reliability[..., None])
        delta_rotation = rotation_from_6d(difference[..., 3:].reshape(len(noisy), 128, 22, 6))
        angle = matrix_to_axis_angle(delta_rotation)
        rotation = self.rotation_limit_rad*torch.tanh(angle/self.rotation_limit_rad)
        rotation = rotation/rotation.norm(dim=-1, keepdim=True).clamp_min(self.rotation_limit_rad)*self.rotation_limit_rad
        rotation = rotation*(1-reliability[..., None])
        moved = reconcile_motion(anchor, rest, translation_m=root, rotation_axis_angle=rotation)
        fixed = (root_reliability >= 1)&(reliability >= 1).all(-1)
        prediction = torch.where(fixed[..., None], initial, moved)
        details = dict(rest=rest, root_delta_m=root, rotation_delta_rad=rotation,
            fixed_frames=fixed, frozen_prior_x0=reference, anchor_motion=anchor)
        return (prediction, details) if return_details else prediction


def channel_reliability(payload):
    root = payload['initial_root_reliability'][..., None].expand(-1, -1, 3)
    rotation = payload['initial_reliability'][..., None].expand(-1, -1, -1, 6).flatten(2)
    # Position channels are redundant FK outputs and carry global reliability.
    position = torch.minimum(payload['initial_reliability'],
        payload['initial_root_reliability'][..., None])[..., None].expand(-1, -1, -1, 3).flatten(2)
    return torch.cat((root, rotation, position), -1).clamp(0, 1)


@torch.no_grad()
def sample_denoising(model, payload, *, steps=8, start_timestep=200, seed=777):
    initial = payload['initial_motion']
    if ((payload['initial_root_reliability'] >= 1).all()
            and (payload['initial_reliability'] >= 1).all()):
        # Exact preservation is already enforced at every forward step. Avoid
        # 16 redundant frozen-network passes for the identity condition.
        return initial
    anchor = model.anchor(payload)
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    noise = torch.randn(initial.shape, device=initial.device, dtype=initial.dtype, generator=generator)
    schedule = torch.linspace(start_timestep, 0, steps, device=initial.device).round().long()
    alpha = model.alphas[start_timestep]
    state = alpha.sqrt()*anchor+(1-alpha).sqrt()*noise
    confidence = channel_reliability(payload)
    was_training = model.training; model.eval()
    try:
        for i, t in enumerate(schedule):
            prediction = model(state, t.expand(len(initial)), payload)
            if i == len(schedule)-1:
                return prediction
            following = schedule[i+1]; next_alpha = model.alphas[following]
            eps = (state-model.alphas[t].sqrt()*prediction)/(1-model.alphas[t]).sqrt().clamp_min(1e-6)
            state = next_alpha.sqrt()*prediction+(1-next_alpha).sqrt()*eps
            observed = next_alpha.sqrt()*anchor+(1-next_alpha).sqrt()*noise
            state = torch.lerp(state, observed, confidence)
    finally:
        model.train(was_training)


def denoising_loss(prediction, batch, details):
    points = forward_kinematics(prediction, batch['initial_rest'])
    truth = forward_kinematics(batch['motion'], batch['target_rest']).detach()
    initial = forward_kinematics(batch['initial_motion'], batch['initial_rest']).detach()
    error = (points-truth).square().mean(-1)
    hidden = 1-batch['joint_visible']*batch['estimate_available'][..., None]
    missing = 1-batch['estimate_available']
    rotation = rotation_from_6d(prediction[..., 3:135].reshape(len(prediction), 128, 22, 6))
    target_rotation = rotation_from_6d(batch['motion'][..., 3:135].reshape(len(prediction), 128, 22, 6)).detach()
    rotation_error = (rotation-target_rotation).square().mean((-1, -2))
    def masked(value, mask):
        return (value*mask).sum()/mask.sum().clamp_min(1)
    position, root, pose = error.mean(), error[..., 0].mean(), rotation_error.mean()
    hidden_position = masked(error, hidden)
    hidden_root = masked(error[..., 0], missing)
    hidden_pose = masked(rotation_error, hidden)
    velocity = ((torch.diff(points, dim=1)-torch.diff(truth, dim=1))*30).square().mean()
    reliability = batch['initial_reliability']*batch['initial_root_reliability'][..., None]
    preserve = ((points-initial).square().mean(-1)*reliability.square()).mean()
    total = (10*position+10*hidden_position+5*root+5*hidden_root+2*pose+4*hidden_pose+.01*velocity+preserve)
    return dict(total=total, position=position, root=root, pose=pose, hidden_position=hidden_position,
                hidden_root=hidden_root, hidden_pose=hidden_pose, velocity=velocity, preserve=preserve)
