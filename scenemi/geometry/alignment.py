"""Place generated EGO bodies from observed cameras and predicted calibration.

This changes the global root pose only. It does not use ground-truth head,
shape, floor, scene, or modify local articulation. EXO tracks need a separate
uncertainty-aware treatment and must not use this EGO-only operation.
"""
import torch

from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import (
    IDENTITY_6D, forward_kinematics, rotation_from_6d,
)


def place_ego_motion(motion, rest, mount, camera, *, rotation=True):
    """Match an entirely valid optical stream using the predicted fixed mount.

    `camera` uses position/2 + absolute column-6D rotation, as in training.
    If rotation=False, only camera position is matched, keeping orientation.
    The caller must require a valid, confident EGO pose at every frame.
    """
    b, t = motion.shape[:2]
    if (motion.shape != (b, t, 201) or rest.shape != (b, 22, 3)
            or mount.shape != (b, 3) or camera.shape != (b, t, 9)):
        raise ValueError('Expected motion, predicted rest/mount and matching optical camera')
    if not all(torch.isfinite(x).all() for x in (motion, rest, mount, camera)):
        raise ValueError('Fully finite EGO observations and body required')
    six = camera[..., 3:]
    if ((six[..., :3].norm(dim=-1) < 1e-6).any()
            or (torch.cross(six[..., :3], six[..., 3:], dim=-1).norm(dim=-1) < 1e-6).any()):
        raise ValueError('Camera rotation is unavailable or degenerate')
    joints, rotations = forward_kinematics(motion, rest), global_rotations(motion)
    identity = motion.new_tensor(IDENTITY_6D)
    head_rotation = rotation_from_6d(six-identity) if rotation else rotations[:, :, 15]
    delta = head_rotation @ rotations[:, :, 15].transpose(-1, -2)
    if not rotation:
        delta = torch.eye(3, device=motion.device, dtype=motion.dtype).expand(b, t, 3, 3)
    head_position = camera[..., :3]*2 - (head_rotation @ mount[:, None, :, None])[..., 0]
    root_position = head_position + (delta @ (joints[:, :, 0]-joints[:, :, 15])[..., None])[..., 0]
    out = motion.clone()
    out[..., :3] = (root_position-rest[:, None, 0])/2
    if rotation:
        root_rotation = delta @ rotations[:, :, 0]
        out[..., 3:9] = torch.cat((root_rotation[..., :, 0], root_rotation[..., :, 1]), -1)-identity
    out[..., 135:] = forward_kinematics(out, rest).flatten(2)/2
    return out
