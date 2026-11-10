"""AMASS prior: existing native30 backbone plus observation-only static shape."""
import torch
from torch import nn

from experiments.offline_camera_retrain_v1.native30_model import Native30SceneMI, reconcile_motion
from scenemi.geometry.shape import SmplhShape
from scenemi.observations import public_observations, body_condition


class ObservationShapeEncoder(nn.Module):
    """One native beta vector per window; no motion state or GT shape input."""
    def __init__(self):
        super().__init__()
        self.joints = nn.Sequential(nn.Linear(15, 32), nn.SiLU(), nn.Linear(32, 32))
        self.body = nn.Linear(22*32, 128, bias=False)
        self.camera = nn.Sequential(nn.Linear(13, 128), nn.SiLU(), nn.Linear(128, 128, bias=False))
        self.temporal = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.SiLU(),
                                      nn.Conv1d(128, 128, 5, padding=2), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 16))
        nn.init.normal_(self.readout[-1].weight, std=.001)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, observations):
        trajectory, meta = observations['trajectory'], observations['observation_meta']
        values = torch.cat((trajectory[..., :3]*meta[..., 2:3],
                            trajectory[..., 3:]*meta[..., 3:4], meta[..., :4],
                            observations['observation_uncertainty']), -1)
        active = meta[..., :2].amax(-1) > 0
        encoded = self.joints(values) * active[..., None]
        ca, cc = observations['camera_available'], observations['camera_confidence']
        camera = torch.cat((observations['camera'], ca, cc), -1)
        frame_valid = active.any(-1) | (ca > 0).any(-1)
        x = self.body(encoded.flatten(2)) + self.camera(camera)*(ca > 0).any(-1)[..., None]
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        weights = frame_valid[..., None].to(x)
        mean = (x*weights).sum(1)/weights.sum(1).clamp_min(1)
        maximum = x.masked_fill(~frame_valid[..., None], -1e9).amax(1)
        maximum = torch.where(frame_valid.any(1)[:, None], maximum, 0.)
        return self.readout(torch.cat((mean, maximum), -1))


class AmassMotionPrior(Native30SceneMI):
    def __init__(self, body_model_path, latent_dim=256, dim_mults=(1, 2, 4)):
        super().__init__(latent_dim=latent_dim, dim_mults=dim_mults)
        self.patch_enabled = self.local_enabled = self.feedback_enabled = False
        scene_modules = ('patch_projection', 'patch_position', 'motion_query', 'region_embedding',
                         'patch_attention', 'patch_condition', 'owner_embedding', 'dynamic_embedding',
                         'dynamic_attention', 'dynamic_condition', 'surface_embedding', 'anchor_embedding',
                         'anchor_query', 'surface_attention', 'region_condition', 'contact_readout',
                         'contact_pose', 'correspondence_readout', 'feedback_temporal',
                         'root_feedback', 'rotation_feedback')
        for name in scene_modules:
            getattr(self, name).requires_grad_(False)
        self.core.scene_embedding.requires_grad_(False)
        self.shape = SmplhShape(body_model_path)
        self.shape_encoder = ObservationShapeEncoder()

    def forward(self, noisy_motion, timestep, batch, *, use_scene=False, use_control=True,
                control_mask=None, return_details=False):
        if use_scene:
            raise ValueError('AMASS stage A must run without scene conditions')
        if noisy_motion.ndim != 3 or noisy_motion.shape[1:] != (128, 201) or not torch.isfinite(noisy_motion).all():
            raise ValueError('This experiment requires 128-frame windows')
        observations = public_observations(batch, enabled=use_control)
        if control_mask is not None:
            # Apply EXO joint selection BEFORE the shape encoder, too.
            observations['observation_meta'] = observations['observation_meta'].clone()
            observations['observation_meta'][..., :4] *= control_mask[..., None]
        betas = self.shape_encoder(observations)
        shape = self.shape(betas)
        conditioned = body_condition(observations, shape)
        scene = (noisy_motion.new_zeros(len(noisy_motion), self.core.latent_dim), None, None, None)
        # With all scene queries disabled, the legacy detached proposal is
        # unused. One denoiser pass is mathematically the same final pass.
        raw = self._core_pass(noisy_motion, timestep, conditioned, scene, control_mask, False, use_control)
        result = reconcile_motion(raw, shape['rest'])
        details = dict(raw_motion=raw, predicted_betas=betas, **shape,
                       observations=observations, conditioned_observations=conditioned)
        return (result, details) if return_details else result


def build_model(config, body_model_path):
    model = config['model']
    if config['scene']['enabled'] or model['initialization'] != 'from_scratch' or model.get('checkpoint') is not None:
        raise ValueError('Use a validated fresh, scene-free stage-A configuration')
    if config['observation']['ground_truth_shape_as_input']:
        raise ValueError('This model predicts shape from observations')
    return AmassMotionPrior(body_model_path, model['latent_dim'], tuple(model['dim_mults']))


def motion_loss(prediction, target, batch, signal, *, details, config):
    from scenemi.losses import motion_loss as objective
    return objective(prediction, target, batch, signal, details=details, config=config)
