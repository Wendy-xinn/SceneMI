"""Public observation contract: labels and native truth are never consumed."""
import torch

from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, rotation_from_6d

INPUT_KEYS = ('camera', 'camera_available', 'camera_confidence', 'camera_context',
              'camera_intrinsics', 'camera_intrinsics_valid', 'trajectory',
              'observation_meta', 'observation_uncertainty')


def camera_supervision(batch, details):
    """Training-only clean reference, expressed with the same sensor mount.

    The separate reference never enters public inference observations, shape
    prediction or the denoiser. Unaugmented historical losses are unchanged.
    """
    if 'supervision_camera' not in batch:
        return details['observations']
    clean = public_observations(dict(batch, camera=batch['supervision_camera']))
    pv = batch.get('sensor_profile', torch.zeros(len(batch['camera']),
                   device=batch['camera'].device, dtype=torch.long)) == 1
    if pv.any():
        camera = clean['camera']; identity = camera.new_tensor(IDENTITY_6D)
        rotation = rotation_from_6d(torch.where(clean['camera_available'][..., 1:] > 0,
                                               camera[..., 3:], identity) - identity)
        mount = details['conditioned_observations']['mount_rotation']
        head = rotation @ mount[:, None].transpose(-1, -2)
        clean['camera'] = torch.cat((camera[..., :3], head[..., 0], head[..., 1]), -1)
    return clean


def public_observations(batch, *, enabled=True):
    """Sanitize before any feature extraction; EGO ignores every joint track."""
    result = {key: batch[key] for key in INPUT_KEYS}
    camera = batch['camera']
    if camera.ndim != 3 or camera.shape[1:] != (128, 9):
        raise ValueError('Expected a 128-frame optical camera stream')
    b, t = camera.shape[:2]
    if batch['trajectory'].shape != (b, t, 22, 9) or batch['observation_meta'].shape != (b, t, 22, 5):
        raise ValueError('Expected joint tracks [B,128,22,9] and metadata [...,5]')
    ego = batch['camera_context'][:, 0] > 0
    meta = torch.where(ego[:, None, None, None], 0., batch['observation_meta'])
    trajectory, meta = sanitize_observations(batch['trajectory'], meta)
    confidence = torch.nan_to_num(batch['camera_confidence']).clamp(0, 1)
    available = (batch['camera_available'] > 0) & (confidence > 0)
    available = available & torch.stack((torch.isfinite(camera[..., :3]).all(-1),
                                        torch.isfinite(camera[..., 3:]).all(-1)), -1)
    six = torch.nan_to_num(camera[..., 3:])
    valid_rotation = (six[..., :3].norm(dim=-1) > 1e-6) & (torch.cross(six[..., :3], six[..., 3:], dim=-1).norm(dim=-1) > 1e-6)
    available = torch.stack((available[..., 0], available[..., 1] & valid_rotation), -1)
    if not enabled:
        available = torch.zeros_like(available)
        trajectory, meta = torch.zeros_like(trajectory), torch.zeros_like(meta)
    safe = torch.cat((torch.where(available[..., :1], camera[..., :3], 0.),
                      torch.where(available[..., 1:], camera[..., 3:], 0.)), -1)
    uncertainty = torch.nan_to_num(batch['observation_uncertainty'], nan=0., posinf=0., neginf=0.).clamp_min(0)
    uncertainty = torch.where((~ego[:, None, None, None]) & (meta[..., :2].amax(-1) > 0)[..., None], uncertainty, 0.)
    result.update(camera=safe, camera_available=available.to(camera),
                  camera_confidence=confidence * available, trajectory=trajectory, observation_meta=meta,
                  observation_uncertainty=uncertainty)
    return result


def body_condition(observations, shape):
    """Convert the same optical stream to head using PREDICTED fixed mount."""
    camera = observations['camera']
    b, t = camera.shape[:2]
    ego = observations['camera_context'][:, 0] > 0
    available = observations['camera_available'] > 0
    confidence = observations['camera_confidence']
    identity = camera.new_tensor(IDENTITY_6D)
    six = torch.where(available[..., 1:], camera[..., 3:], identity)
    rotation = rotation_from_6d(six - identity)
    mount = shape['head_camera_translation']
    head_position = camera[..., :3]*2 - (rotation @ mount[:, None, :, None])[..., 0]
    head = torch.cat((head_position/2, six), -1)
    hm = camera.new_zeros(b, t, 5)
    hm[..., 0] = available.all(-1)
    hm[..., 1] = available[..., 1]
    hm[..., 2] = confidence.amin(-1) * hm[..., 0]
    hm[..., 3] = confidence[..., 1] * hm[..., 1]
    hm[..., 4] = 1
    head = torch.cat((torch.where(hm[..., :1] > 0, head[..., :3], 0.),
                      torch.where(hm[..., 1:2] > 0, head[..., 3:], 0.)), -1)
    trajectory = observations['trajectory'].clone()
    meta = observations['observation_meta'].clone()
    trajectory[:, :, 15] = torch.where(ego[:, None, None], head, trajectory[:, :, 15])
    meta[:, :, 15] = torch.where(ego[:, None, None], hm, meta[:, :, 15])
    result = dict(observations, trajectory=trajectory, observation_meta=meta,
                  rest=shape['rest'], body_type=camera.new_tensor([1., 0.]).expand(b, -1),
                  body_scale=camera.new_ones(b, 1),
                  mount_rotation=torch.eye(3, device=camera.device, dtype=camera.dtype).expand(b, 3, 3),
                  mount_translation_m=torch.where(ego[:, None], mount, 0.),
                  mount_valid=ego.to(camera), mount_uncertainty=camera.new_zeros(b, 2),
                  occupancy=camera.new_zeros(b, 24, 48, 48),
                  bps=camera.new_zeros(b, t, 67, 3),
                  bps_valid=torch.zeros(b, t, 67, device=camera.device, dtype=torch.bool))
    return result
