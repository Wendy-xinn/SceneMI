"""Offline EXO reference gap filling in local joint rotation coordinates.

Only the observation-derived initializer changes. At a joint with no usable
rotation or parent-to-child position evidence, interpolate its local rotation
between evidence frames and hold the nearest endpoint outside that interval.
The parent is repaired first, so an occluded limb still follows its parent.
Neither motion labels nor full GT body pose are read. Confidence stays exactly
as recorded by the original initializer; interpolation is not new evidence.

This is an explicitly versioned process-local override. Existing sealed source
files/checkpoints stay unchanged, and callers must record this policy and the
module hash when preparing training or sampling payloads. It is noncausal over
the same offline window as the original reference interpolation.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from experiments.offline_camera_retrain_v1 import native30_refinement as refinement
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, PARENTS, forward_kinematics, rotation_from_6d


POLICY = "exo-local-so3-reference-gap-fill-v1"
_ORIGINAL_REFERENCE = refinement.reference_from_observations


def _exo_mask(batch, size, device):
    modes = batch.get("mode")
    if modes is None:
        # This override is deliberately opt-in by observer mode. Historical
        # callers without modes retain the exact original behavior.
        return torch.zeros(size, device=device, dtype=torch.bool)
    if len(modes) != size or any(mode not in ("ego", "exo") for mode in modes):
        raise ValueError("One explicit ego/exo mode is required per batch item")
    return torch.tensor([mode == "exo" for mode in modes], device=device)


def rotation_evidence(batch):
    """Identify actual per-joint evidence, without converting gaps to evidence.

    A non-root orientation can come from its measured rotation, or from at
    least one observed outgoing bone with nonzero confidence. Terminal joints
    without a measured rotation have no orientation evidence of their own.
    The root uses the original initializer and is never altered by this policy.
    """
    _, meta = sanitize_observations(batch["trajectory"], batch["observation_meta"])
    cp = meta[..., 0] * meta[..., 2]
    valid = (meta[..., 1] * meta[..., 3] > 0).clone()
    for j in range(1, 22):
        children = [k for k in range(j + 1, 22) if int(PARENTS[k]) == j]
        if children:
            weights = cp[:, :, children] * cp[:, :, j, None]
            valid[:, :, j] |= weights.sum(-1) > 0
    return valid


@torch.no_grad()
def reference_from_observations_temporal(batch):
    """Return the original reference/confidence with EXO rotation gaps filled.

    Original world orientations are retained at evidence frames. Local SO(3)
    interpolation fills only their missing frames, relative to the repaired
    parent. With no evidence anywhere for a joint, its original local rotation
    is retained. EGO items and root channels are bitwise unchanged.
    """
    reference, confidence = _ORIGINAL_REFERENCE(batch)
    exo = _exo_mask(batch, len(reference), reference.device)
    if not bool(exo.any()):
        return reference, confidence

    evidence = rotation_evidence(batch)
    original_global = global_rotations(reference)
    repaired_global = [original_global[:, :, 0]]
    six = reference[..., 3:135].reshape(*reference.shape[:2], 22, 6).clone()
    identity = reference.new_tensor(IDENTITY_6D)

    for j in range(1, 22):
        parent = repaired_global[int(PARENTS[j])]
        valid = evidence[:, :, j]
        # Keep the original orientation where evidence exists, even when the
        # parent had to be repaired at the same frame. Interpolate LOCAL rather
        # than world rotations, so missing limbs inherit parent movement.
        relative = parent.transpose(-1, -2) @ original_global[:, :, j]
        filled = refinement.interpolate_evidence(relative, valid, rotations=True)
        has_evidence = valid.any(1, keepdim=True)
        affected = exo[:, None] & has_evidence
        replacement = torch.cat((filled[..., :, 0], filled[..., :, 1]), -1) - identity
        six[:, :, j] = torch.where(affected[..., None], replacement, six[:, :, j])
        # Build the next parent from the actual returned six-dimensional value.
        # Reuse the standard decoder to preserve the model's rotation convention.
        local = rotation_from_6d(six[:, :, j])
        repaired_global.append(parent @ local)

    provisional = reference.clone()
    provisional[..., 3:135] = six.flatten(2)
    joints = forward_kinematics(provisional, batch["rest"])
    provisional[..., 135:] = joints.flatten(2) / 2
    # This explicit copy prevents tiny FK roundoff from changing EGO items in
    # mixed batches, an important shared-generator compatibility contract.
    result = torch.where(exo[:, None, None], provisional, reference)
    return result, confidence


def install():
    """Install before payload construction; never changes checkpoint tensors."""
    current = refinement.reference_from_observations
    if current not in (_ORIGINAL_REFERENCE, reference_from_observations_temporal):
        raise RuntimeError("Another process-local reference override is installed")
    refinement.reference_from_observations = reference_from_observations_temporal
    return uninstall


def uninstall():
    current = refinement.reference_from_observations
    if current is reference_from_observations_temporal:
        refinement.reference_from_observations = _ORIGINAL_REFERENCE
    elif current is not _ORIGINAL_REFERENCE:
        raise RuntimeError("Refusing to remove another reference override")


@contextmanager
def installed():
    """Scoped installation that preserves an existing outer installation."""
    already_installed = refinement.reference_from_observations is reference_from_observations_temporal
    install()
    try:
        yield
    finally:
        if not already_installed:
            uninstall()
