"""Utilities for adapting ProtoMotions SMPL clips to CondMDI/HumanML.

The source clips use a right-handed Z-up frame.  HumanML uses a right-handed
Y-up frame.  We keep the complete canonicalization transform so generated
joints can be placed back in the source world frame later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# The original HumanML code still uses aliases removed in recent NumPy.
if "float" not in np.__dict__:
    np.float = np.float64  # type: ignore[attr-defined]
if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]

from data_loaders.humanml.common.quaternion import qbetween_np, qinv_np, qrot_np
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts.motion_process import (
    extract_features,
    recover_from_ric,
    uniform_skeleton,
)
from data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain,
    t2m_raw_offsets,
)


HML_JOINT_COUNT = 22
HML_FEATURE_COUNT = 263
HML_HEAD_INDEX = 15
HML_HEAD_POS_SLICE = slice(4 + (HML_HEAD_INDEX - 1) * 3,
                           4 + HML_HEAD_INDEX * 3)

# ProtoMotions' SMPL MJCF bodies are emitted in depth-first XML order, not in
# the canonical SMPL/HumanML joint order.  This maps HumanML's 22 joints to
# ProtoMotions indices.  Hands (Proto indices 18 and 23) are intentionally
# omitted because HumanML ends at the wrists.
PROTO_SMPL_TO_HML22 = np.asarray([
    0,   # pelvis
    1, 5, 9,            # left hip, right hip, torso
    2, 6, 10,           # knees, spine
    3, 7, 11,           # ankles, chest
    4, 8,               # toes
    12, 14, 19, 13,     # neck, left/right thorax, head
    15, 20, 16, 21, 17, 22,  # shoulders, elbows, wrists
], dtype=np.int64)


@dataclass
class CanonicalTransform:
    """Reversible world-frame part of HumanML canonicalization.

    Bone-length standardization is not exactly reversible. ``scale`` restores
    the source subject's approximate size while retaining HumanML proportions.
    """

    scale: float
    floor_height: float
    root_origin_xz: np.ndarray
    initial_rotation: np.ndarray

    def hml_world_to_canonical(self, joints: np.ndarray) -> np.ndarray:
        """Apply the fixed episode's global frame without re-estimating it."""
        points = np.asarray(joints, dtype=np.float64).copy() * self.scale
        points[..., 0] -= self.root_origin_xz[0]
        points[..., 2] -= self.root_origin_xz[2]
        points[..., 1] -= self.floor_height
        return qrot_np(np.broadcast_to(self.initial_rotation, points.shape[:-1] + (4,)),
                       points).astype(np.float32)

    def canonical_to_hml_world(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        inv_rotation = qinv_np(self.initial_rotation[None])[0]
        restored = qrot_np(
            np.broadcast_to(inv_rotation, joints.shape[:-1] + (4,)), joints
        )
        restored[..., 0] += self.root_origin_xz[0]
        restored[..., 2] += self.root_origin_xz[2]
        restored[..., 1] += self.floor_height
        restored /= self.scale
        return restored.astype(np.float32)

    def canonical_to_proto_world(self, joints: np.ndarray) -> np.ndarray:
        restored = self.canonical_to_hml_world(joints)
        # Inverse of Proto (x, y, z) -> HumanML (x, z, -y).
        return np.stack(
            [restored[..., 0], -restored[..., 2], restored[..., 1]], axis=-1
        ).astype(np.float32)


def proto_zup_to_hml_yup(joints: np.ndarray) -> np.ndarray:
    """Rotate ProtoMotions coordinates into HumanML coordinates."""

    joints = np.asarray(joints)
    return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)


def head_local_yaw_features(
    world_position: np.ndarray, world_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express a world-head trajectory in its first-frame yaw anchor.

    Gravity remains aligned with +Y.  This removes arbitrary room translation
    and heading without removing the measured head pitch/roll or relative yaw.
    The returned anchor is sufficient to place generated motion back in world.
    """

    position = np.asarray(world_position, dtype=np.float64)
    rotation = np.asarray(world_rotation, dtype=np.float64)
    if position.ndim != 3 or rotation.shape != position.shape[:2] + (3, 3):
        raise ValueError("Expected position [B,T,3] and rotation [B,T,3,3]")
    forward = rotation[:, 0, :, 2]
    horizontal_norm = np.linalg.norm(forward[:, [0, 2]], axis=-1)
    if np.any(horizontal_norm < 1e-5):
        raise ValueError("First-frame head forward axis is nearly vertical")
    anchor_yaw = np.arctan2(forward[:, 0], forward[:, 2])
    cosine, sine = np.cos(anchor_yaw), np.sin(anchor_yaw)
    anchor_rotation = np.zeros((len(position), 3, 3), dtype=np.float64)
    anchor_rotation[:, 0, 0] = cosine
    anchor_rotation[:, 0, 2] = sine
    anchor_rotation[:, 1, 1] = 1.0
    anchor_rotation[:, 2, 0] = -sine
    anchor_rotation[:, 2, 2] = cosine
    world_to_anchor = np.swapaxes(anchor_rotation, -1, -2)
    local_position = np.einsum(
        "bij,btj->bti", world_to_anchor, position - position[:, :1]
    )
    local_rotation = np.einsum("bij,btjk->btik", world_to_anchor, rotation)
    rotation_6d = np.concatenate(
        (local_rotation[..., 0], local_rotation[..., 1]), axis=-1
    )
    features = np.concatenate((local_position, rotation_6d), axis=-1)
    return (
        features.astype(np.float32),
        position[:, 0].astype(np.float32),
        anchor_rotation.astype(np.float32),
    )


def resample_positions(
    joints: np.ndarray, source_fps: float, target_fps: float
) -> np.ndarray:
    """Linearly resample joint positions while preserving both endpoints."""

    joints = np.asarray(joints, dtype=np.float64)
    duration = (len(joints) - 1) / source_fps
    target_frames = int(round(duration * target_fps)) + 1
    source_times = np.arange(len(joints), dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times[-1] = duration
    flat = joints.reshape(len(joints), -1)
    sampled = np.empty((target_frames, flat.shape[1]), dtype=np.float64)
    for feature_idx in range(flat.shape[1]):
        sampled[:, feature_idx] = np.interp(
            target_times, source_times, flat[:, feature_idx]
        )
    return sampled.reshape(target_frames, *joints.shape[1:]).astype(np.float32)


def centered_crop(joints: np.ndarray, frames: int) -> tuple[np.ndarray, int]:
    if len(joints) < frames:
        raise ValueError(f"Need at least {frames} frames, got {len(joints)}")
    start = (len(joints) - frames) // 2
    return joints[start:start + frames], start


def canonicalize_hml_joints(
    positions: np.ndarray, target_skeleton_path: str, floor_reference_frames: int | None = None
) -> tuple[np.ndarray, CanonicalTransform]:
    """Match HumanML bone lengths and canonical world frame."""

    positions = np.asarray(positions, dtype=np.float64)
    target_example = np.load(target_skeleton_path)
    if target_example.ndim != 3 or target_example.shape[-1] != 3:
        raise ValueError(f"Unexpected target skeleton shape: {target_example.shape}")
    target_example = target_example[:, :HML_JOINT_COUNT]
    raw_offsets = torch.from_numpy(t2m_raw_offsets)
    target_skeleton = Skeleton(raw_offsets, t2m_kinematic_chain, "cpu")
    target_offsets = target_skeleton.get_offsets_joints(
        torch.from_numpy(target_example[0])
    )

    source_skeleton = Skeleton(raw_offsets, t2m_kinematic_chain, "cpu")
    source_offsets = source_skeleton.get_offsets_joints(
        torch.from_numpy(positions[0])
    ).numpy()
    target_offsets_np = target_offsets.numpy()
    source_leg = np.abs(source_offsets[5]).max() + np.abs(source_offsets[8]).max()
    target_leg = np.abs(target_offsets_np[5]).max() + np.abs(target_offsets_np[8]).max()
    scale = float(target_leg / source_leg)

    standardized = uniform_skeleton(
        positions, target_offsets, raw_offsets, t2m_kinematic_chain
    )
    floor_height = float(standardized[:floor_reference_frames, ..., 1].min())
    standardized[..., 1] -= floor_height

    root_origin_xz = standardized[0, 0] * np.array([1.0, 0.0, 1.0])
    canonical = standardized - root_origin_xz

    # HumanML faces +Z. Across points from left side to right side.
    across = (
        standardized[0, 2] - standardized[0, 1]
        + standardized[0, 17] - standardized[0, 16]
    )
    across /= np.linalg.norm(across)
    forward = np.cross(np.array([0.0, 1.0, 0.0]), across)
    forward /= np.linalg.norm(forward)
    initial_rotation = qbetween_np(forward[None], np.array([[0.0, 0.0, 1.0]]))[0]
    canonical = qrot_np(
        np.broadcast_to(initial_rotation, canonical.shape[:-1] + (4,)), canonical
    )

    transform = CanonicalTransform(
        scale=scale,
        floor_height=floor_height,
        root_origin_xz=root_origin_xz.astype(np.float64),
        initial_rotation=initial_rotation.astype(np.float64),
    )
    return canonical.astype(np.float32), transform


def joints_to_hml_abs_features(canonical_joints: np.ndarray) -> np.ndarray:
    """Encode canonical joints using CondMDI's absolute-root HumanML format."""

    relative = extract_features(
        canonical_joints.copy(),
        feet_thre=0.002,
        n_raw_offsets=torch.from_numpy(t2m_raw_offsets),
        kinematic_chain=t2m_kinematic_chain,
        face_joint_indx=[2, 1, 17, 16],
        fid_r=[8, 11],
        fid_l=[7, 10],
    )
    # Do not integrate the legacy relative root velocity here.  HumanML stores
    # it as asin(delta_quaternion.y), which is only an approximation when a
    # clip contains a sharp turn.  CondMDI's abs_3d representation stores the
    # root state directly, so derive it from the same smoothed facing frame
    # used by extract_features instead.  This avoids centimetre-scale drift.
    skeleton = Skeleton(
        torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu"
    )
    root_quaternion = skeleton.inverse_kinematics_np(
        canonical_joints,
        [2, 1, 17, 16],
        smooth_forward=True,
    )[:, 0]
    root_angle = np.unwrap(
        np.arctan2(root_quaternion[:, 2], root_quaternion[:, 0]),
        period=np.pi,
    )
    absolute = relative.copy()
    absolute[:, 0] = root_angle[:-1]
    absolute[:, 1:3] = canonical_joints[:-1, 0][:, [0, 2]]
    return absolute.astype(np.float32)


def _recover_relative_root(data: torch.Tensor):
    """Local copy returning root angle without importing dataset machinery."""

    rotation_velocity = data[..., 0]
    root_angle = torch.zeros_like(rotation_velocity)
    root_angle[..., 1:] = rotation_velocity[..., :-1]
    root_angle = torch.cumsum(root_angle, dim=-1)
    root_quaternion = torch.zeros(data.shape[:-1] + (4,), dtype=data.dtype)
    root_quaternion[..., 0] = torch.cos(root_angle)
    root_quaternion[..., 2] = torch.sin(root_angle)

    root_position = torch.zeros(data.shape[:-1] + (3,), dtype=data.dtype)
    root_position[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    from data_loaders.humanml.common.quaternion import qinv, qrot

    root_position = qrot(qinv(root_quaternion), root_position)
    root_position = torch.cumsum(root_position, dim=-2)
    root_position[..., 1] = data[..., 3]
    return root_quaternion, root_position, root_angle


def recover_hml_abs_joints(features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features)).float()
    return recover_from_ric(tensor, HML_JOINT_COUNT, abs_3d=True).numpy()


def proto_body_positions_to_hml_state_prefix(
    body_positions_zup: np.ndarray,
    target_skeleton_path: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, CanonicalTransform]:
    """Encode three simulator poses as CondMDI's two-frame state prefix.

    CondMDI's 263-D feature at frame ``t`` describes the transition from
    pose ``t`` to pose ``t + 1``.  Therefore the two past feature frames used
    by the receding-horizon interface require three consecutive simulator
    poses.  The returned prefix is normalized and has shape ``[2, 263]``.
    This helper intentionally uses the same canonicalization as dataset
    preparation so the state has the training distribution's coordinates.
    """

    positions = np.asarray(body_positions_zup, dtype=np.float32)
    if positions.shape != (3, 24, 3):
        raise ValueError(
            "Expected exactly three ProtoMotions SMPL body positions with "
            f"shape [3, 24, 3], got {positions.shape}"
        )
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (HML_FEATURE_COUNT,) or std.shape != (HML_FEATURE_COUNT,):
        raise ValueError(f"Expected mean/std shape [263], got {mean.shape}/{std.shape}")
    if np.any(std <= 0):
        raise ValueError("Feature standard deviation must be positive")

    hml_yup = proto_zup_to_hml_yup(positions[:, PROTO_SMPL_TO_HML22])
    canonical, transform = canonicalize_hml_joints(
        hml_yup, target_skeleton_path
    )
    features = joints_to_hml_abs_features(canonical)
    if features.shape != (2, HML_FEATURE_COUNT):
        raise RuntimeError(f"Unexpected state feature shape: {features.shape}")
    return ((features - mean) / std).astype(np.float32), canonical, transform


def pelvis_local_head_feature_mask(frames: int) -> np.ndarray:
    """Observe absolute root pose and local head position only."""

    mask = np.zeros((HML_FEATURE_COUNT, 1, frames), dtype=bool)
    mask[0:4, 0, :] = True
    mask[HML_HEAD_POS_SLICE, 0, :] = True
    return mask
