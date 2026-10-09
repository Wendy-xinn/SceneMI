"""Camera-calibrated head rotation projection, with no GT motion inputs.

This solves joint 15's local rotation only: root, neck and feet stay fixed.
Position is deliberately not projected, since root translation would move feet.
Callers must supply a known rigid camera-to-head calibration; real PV without
calibration must keep the soft condition. This does not enforce neck limits.
"""
import torch
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, rotation_from_6d


def project_head_orientation(motion, camera, enabled, camera_from_head=None):
    if camera.shape != (*motion.shape[:2], 9):
        raise ValueError('Expected calibrated camera [B,T,9]')
    if enabled.shape != motion.shape[:2]:
        raise ValueError('Expected explicit per-frame calibration/control mask')
    identity = motion.new_tensor(IDENTITY_6D)
    camera_r = rotation_from_6d(camera[..., 3:9] - identity)
    mount = torch.eye(3, device=motion.device, dtype=motion.dtype) if camera_from_head is None else camera_from_head.to(motion)
    target = camera_r @ mount.transpose(-1, -2)
    parent = global_rotations(motion)[:, :, 12]
    local = parent.transpose(-1, -2) @ target
    six = torch.cat((local[..., :, 0], local[..., :, 1]), -1) - identity
    result = motion.clone()
    result[..., 93:99] = torch.where(enabled[..., None], six, motion[..., 93:99])
    return result
