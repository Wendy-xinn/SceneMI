"""Initialization-preserving EXO x0 repair using an immutable motion prior.

A zero-initialized temporal head predicts bounded root/SO(3) corrections.
The pretrained denoiser supplies frozen features, rather than replacing an
already good body with its unconditional absolute output. EGO bypasses repair.
"""
import torch
from torch import nn
from torch.nn import functional as F

from scenemi.observations import public_observations, body_condition
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


class InitializedExoPrior(nn.Module):
    def __init__(self, base, *, hidden=128, root_limit_m=.25, rotation_limit_rad=.6):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.root_limit_m, self.rotation_limit_rad = root_limit_m, rotation_limit_rad
        # Frozen prior x0 + initial x0 + xt + local/root confidence + presence
        # and image visibility + diffusion signal. No clean label is accepted.
        width = 201*3+22+1+1+22+1
        layers = []
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv1d(width, hidden, 5, padding=2*dilation, dilation=dilation), nn.SiLU()]
            width = hidden
        self.temporal = nn.Sequential(*layers)
        self.readout = nn.Conv1d(hidden, 69, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.train(False)

    @property
    def alphas(self):
        return self.base.alphas

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def repair_state(self):
        return dict(temporal=self.temporal.state_dict(), readout=self.readout.state_dict())

    def load_repair_state(self, state):
        self.temporal.load_state_dict(state['temporal'])
        self.readout.load_state_dict(state['readout'])

    def forward(self, noisy, timestep, payload, *, return_details=False):
        ego = payload['camera_context'][:, 0]>0
        if ego.all():
            return self.base(noisy, timestep, payload, use_scene=False, return_details=return_details)
        if ego.any():
            raise ValueError('Pilot batches are single-mode; mixed dispatch is not yet validated')
        initial, rest = payload['initial_motion'], payload['initial_rest']
        if initial.shape != noisy.shape or initial.shape[1:] != (128, 201) or rest.shape != (len(initial),22,3):
            raise ValueError('Expected a complete estimated 128-frame initializer and native estimated rest')
        if not torch.isfinite(initial).all() or not torch.isfinite(rest).all():
            raise ValueError('Nonfinite initialization; fill missing estimates from available context first')
        reliability = payload['initial_reliability'].clamp(0, 1)
        root_reliability = payload['initial_root_reliability'].clamp(0, 1)
        observations = public_observations(payload)
        with torch.no_grad():
            conditioned = body_condition(observations, dict(rest=rest,
                head_camera_translation=rest.new_zeros(len(rest),3)))
            empty = (noisy.new_zeros(len(noisy), self.base.core.latent_dim), None, None, None)
            prior = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
        signal = self.alphas[timestep].sqrt()[:, None, None].expand(-1,128,1)
        features = torch.cat((prior.clamp(-5,5), initial, noisy,
            reliability, root_reliability[..., None], payload['estimate_available'][..., None],
            payload['joint_visible'], signal), -1)
        correction = self.readout(self.temporal(features.transpose(1,2))).transpose(1,2)
        root = torch.tanh(correction[..., :3])
        root = root/root.norm(dim=-1,keepdim=True).clamp_min(1)
        root = root*self.root_limit_m*(1-root_reliability[..., None])
        rotation = torch.tanh(correction[..., 3:].reshape(len(noisy),128,22,3))
        rotation = rotation/rotation.norm(dim=-1,keepdim=True).clamp_min(1)
        rotation = rotation*self.rotation_limit_rad*(1-reliability[..., None])
        moved = reconcile_motion(initial,rest,translation_m=root,rotation_axis_angle=rotation)
        # Exact identity for fully reliable frames, including after optimization.
        fixed = (root_reliability>=1)&(reliability>=1).all(-1)
        prediction = torch.where(fixed[..., None],initial,moved)
        details = dict(rest=rest, root_delta_m=root, rotation_delta_rad=rotation,
            frozen_prior_x0=prior, fixed_frames=fixed)
        return (prediction, details) if return_details else prediction


@torch.no_grad()
def sample_initialized(model, payload, *, steps=8, start_timestep=200, seed=777):
    initial = payload['initial_motion']
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    alpha = model.alphas[start_timestep]
    noise = torch.randn(initial.shape,device=initial.device,dtype=initial.dtype,generator=generator)
    state = alpha.sqrt()*initial+(1-alpha).sqrt()*noise
    schedule = torch.linspace(start_timestep,0,steps,device=initial.device).round().long()
    was_training = model.training
    model.eval()
    try:
        for i,t in enumerate(schedule):
            prediction = model(state,t.expand(len(initial)),payload)
            if i==len(schedule)-1:
                return prediction
            following = schedule[i+1]
            eps = (state-model.alphas[t].sqrt()*prediction)/(1-model.alphas[t]).sqrt().clamp_min(1e-6)
            state = model.alphas[following].sqrt()*prediction+(1-model.alphas[following]).sqrt()*eps
    finally:
        model.train(was_training)


def repair_loss(prediction, batch, details):
    rest = batch['initial_rest']
    points = forward_kinematics(prediction,rest)
    truth = forward_kinematics(batch['motion'],batch['target_rest']).detach()
    initial_points = forward_kinematics(batch['initial_motion'],rest).detach()
    from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
    rotations = rotation_from_6d(prediction[...,3:135].reshape(len(prediction),128,22,6))
    target_rotations = rotation_from_6d(batch['motion'][...,3:135].reshape(len(prediction),128,22,6)).detach()
    # Metre and rotation-matrix objectives supervise clean motion, not noisy fit.
    position = F.mse_loss(points,truth)
    root = F.mse_loss(points[:,:,0],truth[:,:,0])
    pose = F.mse_loss(rotations,target_rotations)
    velocity = F.mse_loss(torch.diff(points,dim=1)*30,torch.diff(truth,dim=1)*30)
    reliability = batch['initial_reliability']*batch['initial_root_reliability'][...,None]
    preserve = ((points-initial_points).square().mean(-1)*reliability.square()).mean()
    total = 10*position+5*root+2*pose+.01*velocity+preserve
    return dict(total=total,position=position,root=root,pose=pose,velocity=velocity,preserve=preserve)
