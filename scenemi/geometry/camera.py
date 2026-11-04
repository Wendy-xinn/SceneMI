"""T_a_b maps column-vector coordinates from frame b into frame a.

Camera optical forward is +Z. Extrinsics are head-local and remain fixed for
one calibration profile; their world-space displacement rotates with the head.
"""
import numpy as np


def transform(rotation, translation):
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    if rotation.shape[-2:] != (3, 3) or translation.shape[-1:] != (3,):
        raise ValueError('Expected rotation [...,3,3] and translation [...,3]')
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Nonfinite camera transform')
    if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Camera rotation must belong to SO(3)')
    batch = np.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
    result = np.broadcast_to(np.eye(4), (*batch, 4, 4)).copy()
    result[..., :3, :3], result[..., :3, 3] = rotation, translation
    return result


def inverse(pose):
    pose = np.asarray(pose)
    if pose.shape[-2:] != (4, 4) or not np.allclose(pose[..., 3, :], [0, 0, 0, 1]):
        raise ValueError('Expected homogeneous SE(3) transform')
    rotation = pose[..., :3, :3].swapaxes(-1, -2)
    return transform(rotation, -(rotation @ pose[..., :3, 3, None])[..., 0])


def virtual_mount(eye_offset_head, face_rotation=None, clearance_m=0.02):
    """Match the existing virtual_head_camera convention, including clearance.

    eye_offset_head already uses the head basis and must not be rotated by
    face_rotation a second time. Store the returned total extrinsic in data.
    """
    rotation = np.eye(3) if face_rotation is None else np.asarray(face_rotation)
    if np.asarray(eye_offset_head).shape != (3,) or rotation.shape != (3, 3):
        raise ValueError('One fixed mount is required per calibration profile')
    return transform(rotation, np.asarray(eye_offset_head) + clearance_m * rotation[:, 2])


def camera_from_head(world_head, head_camera):
    return np.asarray(world_head) @ np.asarray(head_camera)


def head_from_camera(world_camera, head_camera):
    return np.asarray(world_camera) @ inverse(head_camera)


def cpf_from_camera(world_camera, head_camera, head_cpf):
    """Adapt the same camera observation to CPF without adding GT information."""
    return head_from_camera(world_camera, head_camera) @ np.asarray(head_cpf)
