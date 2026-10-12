"""Independent camera and typed observations; legacy scene and motion decoder."""
import numpy as np
import torch
from torch import nn

from experiments.offline_camera_retrain_v1.unified_model import UnifiedSceneMI
from experiments.offline_camera_retrain_v1.unified_observations import observation_derivatives

NEW_PREFIXES = ('camera_encoder.', 'camera_projection.', 'joint_embedding.',
                'sensor_adapter.', 'anatomical_adapter.', 'observation_projection.')


def camera_features(batch):
    """48 physical/masked features; camera availability is independent of body."""
    camera = batch['camera']
    available = batch['camera_available'].to(camera)
    confidence = torch.where(available > 0, batch['camera_confidence'].to(camera), 0.).clamp(0, 1)
    pm, rm = available.unbind(-1)
    pm = pm * torch.isfinite(camera[..., :3]).all(-1)
    rm = rm * torch.isfinite(camera[..., 3:]).all(-1)
    pose = torch.cat((torch.where(pm[..., None] > 0, camera[..., :3], 0.),
                      torch.where(rm[..., None] > 0, camera[..., 3:], 0.)), -1)
    meta = camera.new_zeros(*camera.shape[:2], 1, 5)
    meta[..., 0, 0] = pm; meta[..., 0, 1] = rm
    v, a, w, mask = observation_derivatives(camera[:, :, None], meta)
    derivatives = torch.cat((v / .15, a, w / np.deg2rad(20)), -1).squeeze(2).clamp(-20, 20)
    mask = mask.squeeze(2).to(camera)
    intr_valid = batch['camera_intrinsics_valid'].to(camera)
    intr = torch.where(intr_valid[..., None] > 0, batch['camera_intrinsics'], 0.)
    mount_valid = batch['mount_valid'].to(camera)
    mount = torch.cat((batch['mount_translation_m'] / .1,
                       batch['mount_rotation'][:, :, :2].transpose(1, 2).flatten(1)), -1)
    mount = torch.where(mount_valid[:, None] > 0, mount, 0.)
    uncertainty = torch.where(mount_valid[:, None] > 0, batch['mount_uncertainty'], 0.)
    static = torch.cat((batch['camera_context'], mount, mount_valid[:, None],
                        uncertainty / camera.new_tensor([.05, np.deg2rad(10)])), -1)
    static = static[:, None].expand(-1, camera.shape[1], -1)
    result = torch.cat((pose, torch.stack((pm, rm), -1), confidence * available,
                        derivatives, mask, intr, intr_valid[..., None], static), -1)
    if result.shape[-1] != 48 or not torch.isfinite(result).all():
        raise ValueError('Nonfinite or malformed camera features')
    return result


class CameraAwareSceneMI(UnifiedSceneMI):
    """Zero residual condition channels preserve legacy weights and output head.

    Additional conditions enter the existing 64D control feature, without
    changing U-Net widths, global/local scene modules, or the 201D decoder.
    Joint identity survives the ordered 22-slot projection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_encoder = nn.Sequential(nn.Linear(48, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.camera_projection = nn.Linear(64, 64)
        self.joint_embedding = nn.Embedding(22, 16)
        self.sensor_adapter = nn.Sequential(nn.Linear(42, 64), nn.SiLU(), nn.Linear(64, 32))
        self.anatomical_adapter = nn.Sequential(nn.Linear(42, 64), nn.SiLU(), nn.Linear(64, 32))
        self.observation_projection = nn.Linear(22 * 32, 64)
        for layer in (self.camera_projection, self.observation_projection):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.camera_enabled = True
        self.typed_observations_enabled = True

    def load_legacy_state_dict(self, state):
        if any(key.startswith(NEW_PREFIXES) for key in state):
            raise ValueError('Use strict load_state_dict for a camera-v2 checkpoint')
        expanded = dict(state)
        expanded.update({key: value for key, value in self.state_dict().items() if key.startswith(NEW_PREFIXES)})
        return super().load_state_dict(expanded)

    def _forward(self, noisy, timestep, batch, **kwargs):
        if batch['camera_context'].shape != (len(noisy), 6):
            raise ValueError('Expected self/other, real/synthetic, moving, image-y convention')
        meta = batch['observation_meta']
        other = batch['camera_context'][:, 1] > .5
        active_sensor = (meta[..., :2].amax(-1) > 0) & (meta[..., 4] < .5)
        if torch.any(active_sensor & other[:, None, None]):
            raise ValueError('An external observer camera cannot be a target body sensor')

        def condition(module, inputs, output):
            result = output
            if self.camera_enabled:
                valid = batch['camera_available'].amax(-1) > 0
                delta = self.camera_projection(self.camera_encoder(camera_features(batch)))
                result = result + delta * valid[..., None]
            if self.typed_observations_enabled:
                values = inputs[0].reshape(*noisy.shape[:2], 22, 26)
                active = values[..., 9:11].amax(-1) > 0
                ids = self.joint_embedding.weight[None, None].expand(*noisy.shape[:2], -1, -1)
                features = torch.cat((values, ids), -1)
                sensor = values[..., 13:14] < .5
                encoded = torch.where(sensor, self.sensor_adapter(features), self.anatomical_adapter(features))
                confidence = (values[..., 11:13] * values[..., 9:11]).sum(-1)
                confidence = confidence / values[..., 9:11].sum(-1).clamp_min(1)
                encoded = encoded * active[..., None] * confidence[..., None]
                delta = self.observation_projection(encoded.flatten(2))
                result = result + delta * active.any(-1)[..., None]
            return result

        hook = self.core.sparse_control_process.register_forward_hook(condition)
        try:
            return super()._forward(noisy, timestep, batch, **kwargs)
        finally:
            hook.remove()
