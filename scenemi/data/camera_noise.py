"""Synthetic EGO camera errors, clean supervision and observation-only anchoring.

Position amplitudes are per-axis metres; rotation amplitudes are per-axis
degrees in so(3). No target determines the corruption or the new coordinate
frame. Labels are only re-expressed in that same frame, never corrupted.
Availability/confidence remain the original deployable inputs.
"""
import hashlib
import json
import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scenemi.data.amass_dataset import camera_anchor
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, rotation_from_6d


def validate_policy(policy):
    if not isinstance(policy, dict) or policy.get('version') != 'correlated-camera-error-v1':
        raise ValueError('Expected correlated-camera-error-v1')
    required = {'version', 'probability', 'position_std_m', 'rotation_std_deg',
                'drift_position_std_m', 'drift_rotation_std_deg', 'rho', 'ramp_updates', 'seed'}
    if set(policy) != required:
        raise ValueError(f'Camera noise fields must be exactly {sorted(required)}')
    for key in ('probability', 'position_std_m', 'rotation_std_deg',
                'drift_position_std_m', 'drift_rotation_std_deg', 'rho'):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid camera noise {key}')
    if not 0 <= policy['probability'] <= 1 or not 0 <= policy['rho'] < 1:
        raise ValueError('Require probability in [0,1] and rho in [0,1)')
    if type(policy['ramp_updates']) is not int or policy['ramp_updates'] < 0 or type(policy['seed']) is not int:
        raise ValueError('Camera noise requires an integer seed and nonnegative ramp_updates')
    return policy


def case_seed(case, seed):
    # Original historical case_id values are not globally unique.
    key = json.dumps({k: case.get(k) for k in ('dataset', 'family', 'member', 'start')}, sort_keys=True)
    return int.from_bytes(hashlib.sha256(f'{seed}|{key}'.encode()).digest()[:8], 'little')


def _ar1(rng, frames, rho):
    values = rng.normal(size=(frames, 3))
    for t in range(1, frames):
        values[t] = rho * values[t - 1] + math.sqrt(1 - rho * rho) * values[t]
    return values


def augment_camera_batch(batch, policy, *, seed=0, seeds=None, strength=1., probability=None):
    """Return an independent augmented batch; the input is never mutated.

    A selected window is re-anchored from its noisy first camera. This keeps
    the first-camera convention valid at deployment. One fixed yaw/translation
    is also applied to the clean motion and clean camera reference.
    """
    validate_policy(policy)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Camera noise strength must be finite and nonnegative')
    probability = policy['probability'] if probability is None else probability
    if not 0 <= probability <= 1:
        raise ValueError('Invalid camera noise probability')
    if strength == 0 or probability == 0 or not any(policy[k] for k in
            ('position_std_m', 'rotation_std_deg', 'drift_position_std_m', 'drift_rotation_std_deg')):
        return batch
    if 'supervision_camera' in batch:
        raise ValueError('Camera augmentation must not be applied twice')
    if (batch['camera_context'][:, 0] <= 0).any():
        raise ValueError('Camera-error adaptation currently supports EGO-only batches')
    camera = batch['camera']
    b, frames, _ = camera.shape
    roots = np.random.default_rng(seed)
    seeds = roots.integers(0, 2**63 - 1, b) if seeds is None else seeds
    if len(seeds) != b:
        raise ValueError('One independent noise seed is required per window')
    eligible = ((batch['camera_available'] > 0) & (batch['camera_confidence'] > 0)).all((1, 2)).cpu().numpy()
    source = camera.detach().cpu().numpy()
    noisy = source.copy(); anchors = np.broadcast_to(np.eye(3), (b, 3, 3)).copy()
    origins = np.zeros((b, 3)); selected = np.zeros(b, bool)
    t = np.linspace(0., 1., frames)[:, None]
    smooth = t * t * (3 - 2 * t)
    for i, sample_seed in enumerate(seeds):
        rng = np.random.default_rng(int(sample_seed))
        if rng.random() >= probability or not eligible[i]:
            continue
        selected[i] = True
        pos_error = (_ar1(rng, frames, policy['rho']) * policy['position_std_m'] +
                     smooth * rng.normal(size=3) * policy['drift_position_std_m']) * strength
        rot_error = (_ar1(rng, frames, policy['rho']) * policy['rotation_std_deg'] +
                     smooth * rng.normal(size=3) * policy['drift_rotation_std_deg']) * (strength * np.pi / 180)
        six = source[i, :, 3:]
        rotation = np.stack((six[:, :3], six[:, 3:], np.cross(six[:, :3], six[:, 3:])), -1)
        rotation = Rotation.from_rotvec(rot_error).as_matrix() @ rotation
        position = source[i, :, :3] * 2 + pos_error
        anchor = camera_anchor(rotation[0]); origin = position[0].copy(); origin[1] = 0
        anchors[i], origins[i] = anchor, origin
        local_rotation = anchor @ rotation
        noisy[i, :, :3] = (position - origin) @ anchor.T / 2
        noisy[i, :, 3:] = np.concatenate((local_rotation[..., 0], local_rotation[..., 1]), -1)
    if not selected.any():
        return batch
    result = dict(batch, camera=torch.as_tensor(noisy, device=camera.device, dtype=camera.dtype))
    result['supervision_camera'] = camera.clone()
    result['camera_noise_selected'] = torch.as_tensor(selected, device=camera.device)
    idx = result['camera_noise_selected']
    a = torch.as_tensor(anchors[selected], device=camera.device, dtype=camera.dtype)
    o = torch.as_tensor(origins[selected], device=camera.device, dtype=camera.dtype)
    clean = camera[idx]
    clean_rotation = rotation_from_6d(clean[..., 3:] - clean.new_tensor(IDENTITY_6D))
    clean_rotation = a[:, None] @ clean_rotation
    result['supervision_camera'][idx] = torch.cat(
        (((clean[..., :3] * 2 - o[:, None]) @ a.transpose(-1, -2)) / 2,
         clean_rotation[..., 0], clean_rotation[..., 1]), -1)
    if 'motion' in batch:
        motion = batch['motion'][idx].clone()
        rest = batch['target_rest'][idx]
        root = motion[..., :3] * 2 + rest[:, None, 0]
        motion[..., :3] = ((root - o[:, None]) @ a.transpose(-1, -2) - rest[:, None, 0]) / 2
        root_rotation = a[:, None] @ rotation_from_6d(motion[..., 3:9])
        motion[..., 3:9] = torch.cat((root_rotation[..., 0], root_rotation[..., 1]), -1) - motion.new_tensor(IDENTITY_6D)
        joints = motion[..., 135:].reshape(-1, frames, 22, 3) * 2
        motion[..., 135:] = (((joints - o[:, None, None]) @ a[:, None].transpose(-1, -2)) / 2).flatten(2)
        result['motion'] = batch['motion'].clone(); result['motion'][idx] = motion
    return result
