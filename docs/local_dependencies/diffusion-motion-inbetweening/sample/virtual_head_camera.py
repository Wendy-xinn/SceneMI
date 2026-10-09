"""Shared virtual head-camera convention for SceneMI datasets.

The pose is expressed in the dataset world frame.  The head rotation at each
frame carries the camera through the motion; the fixed offset and face-frame
calibration are not re-estimated per frame or per sequence.
"""

from __future__ import annotations

import numpy as np


# Existing TRUMANS eye-line offset.  This is now the common SceneMI offset.
DEFAULT_CAMERA_OFFSET_HEAD = np.asarray(
    (-0.01031326, 0.04308906, 0.07903221), dtype=np.float32
)
CAMERA_FACE_CLEARANCE_M = 0.02
# EgoBody interactee virtual-camera example: PV intrinsics fx=60.9396,
# fy=81.3017, image size 80x60.
VIRTUAL_HORIZONTAL_FOV_DEG = 66.56
VIRTUAL_VERTICAL_FOV_DEG = 40.49

# EgoBody's interactee camera uses the SMPL-X head global rotation directly;
# the head-local +Z axis is the optical forward direction.  In particular, it
# does not use the downward-sloping eye-midpoint-to-nose vector as gaze.
DEFAULT_FACE_LOCAL_ROTATION = np.eye(3, dtype=np.float32)


def virtual_head_camera(head_positions, head_rotations,
                        offset_head=DEFAULT_CAMERA_OFFSET_HEAD,
                        face_local_rotation=DEFAULT_FACE_LOCAL_ROTATION):
    """Build positions and rotations for the shared virtual head camera.

    Frame zero is not reset: ``R_camera[0]`` is the action's frame-zero head
    rotation multiplied by the fixed face calibration.  Subsequent frames
    follow the corresponding head rotations.
    """
    head_positions = np.asarray(head_positions, dtype=np.float32)
    head_rotations = np.asarray(head_rotations, dtype=np.float32)
    offset_head = np.asarray(offset_head, dtype=np.float32)
    face_local_rotation = np.asarray(face_local_rotation, dtype=np.float32)
    # ``offset_head`` is expressed in the SMPL-X head-joint frame (this is how
    # TRUMANS calibrates its eye midpoint).  Do not apply the face calibration
    # to this offset a second time: that would rotate the eye displacement by
    # ``head_rotation @ face_local_rotation`` and can move the optical center
    # more than ten centimetres away from the eyes.  The small clearance is a
    # separate displacement along the final optical axis.
    camera_rotation = np.einsum("tij,jk->tik", head_rotations,
                                face_local_rotation).astype(np.float32)
    # The offset remains in the SMPL-X head-joint frame even when a dataset
    # needs a fixed face-axis calibration for the optical orientation.
    camera_position = (
        head_positions
        + np.einsum("tij,j->ti", head_rotations, offset_head)
        + CAMERA_FACE_CLEARANCE_M * camera_rotation[:, :, 2]
    ).astype(np.float32)
    return camera_position, camera_rotation
