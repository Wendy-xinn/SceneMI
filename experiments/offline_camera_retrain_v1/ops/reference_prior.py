"""Process-local yaw-only fallback for an observation-derived ego reference.

This changes the initializer, not the measured head rotation or an x0 constraint.
Install the same policy before constructing training and evaluation payloads.
Original sealed trainer/model files and their checkpoint configuration stay intact;
the caller must record this override and this module's hash in its stage manifest.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from experiments.offline_camera_retrain_v1 import native30_refinement as refinement
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, PARENTS, rotation_from_6d


POLICY = 'ego-observation-reference-yaw-fallback-v1'
MIN_HORIZONTAL_NORM = .1
_ORIGINAL_REFERENCE = refinement.reference_from_observations


def _ego_mask(batch, size, device):
    modes = batch.get('mode')
    if modes is None:
        return torch.zeros(size, device=device, dtype=torch.bool)
    if len(modes) != size or any(mode not in ('ego', 'exo') for mode in modes):
        raise ValueError('One explicit ego/exo mode is required per batch item')
    return torch.tensor([mode == 'ego' for mode in modes], device=device)


def _yaw_rotations(rotations, evidence_valid):
    """Project +Z onto XZ; fill undefined headings in the offline window.

    Around vertical, the forward direction does not identify heading. Interpolate
    valid projected rotations and hold endpoints, using the existing SO(3) helper.
    With no usable heading anywhere, use identity. No GT or floor enters this.
    Crossing a vertical pose can still make physical heading ambiguous; this is
    an initializer convention rather than evidence of the true pelvis heading.
    """
    horizontal = rotations[..., (0, 2), 2]
    valid = evidence_valid & (horizontal.norm(dim=-1) > MIN_HORIZONTAL_NORM)
    angle = torch.atan2(horizontal[..., 0], horizontal[..., 1])
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    yaw = torch.stack((c, zero, s, zero, one, zero, -s, zero, c), -1)
    yaw = yaw.reshape(*rotations.shape[:-2], 3, 3)
    identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype).expand_as(yaw)
    yaw = torch.where(valid[..., None, None], yaw, identity)
    return refinement.interpolate_evidence(yaw, valid, rotations=True)


@torch.no_grad()
def _build_reference(batch, *, yaw_only):
    """Original reference equations, with one optional root fallback projection.

    The disabled path intentionally follows the sealed function operation for
    operation, allowing CPU tests to reject unrelated mathematical changes.
    """
    trajectory, meta = sanitize_observations(batch['trajectory'], batch['observation_meta'])
    pos = trajectory[..., :3] * 2
    cp = meta[..., 0] * meta[..., 2]
    rest = batch['rest']
    b, t = pos.shape[:2]
    eye = torch.eye(3, device=pos.device, dtype=pos.dtype).expand(b, t, 3, 3)
    observed_r = rotation_from_6d(trajectory[..., 3:] - trajectory.new_tensor(IDENTITY_6D))
    head_valid = meta[:, :, 15, 1] * meta[:, :, 15, 3] > 0
    fallback = torch.where(head_valid[..., None, None], observed_r[:, :, 15], eye)
    x = rest[:, None, :4].expand(b, t, 4, 3)
    r, s, cx, cy = refinement.fit_rotation(x, pos[:, :, :4], cp[:, :, :4], center=True)
    root_valid = ((cp[:, :, :4] > 0).sum(-1) >= 3) & (s[..., 1] > 1e-7)
    root_r = torch.where(root_valid[..., None, None], r, fallback)
    root_r = refinement.interpolate_evidence(root_r, root_valid | head_valid, rotations=True)

    if yaw_only:
        eligible = _ego_mask(batch, b, pos.device)[:, None] & ~root_valid
        if bool(eligible.any()):
            yaw = _yaw_rotations(root_r, root_valid | head_valid)
            root_r = torch.where(eligible[..., None, None], yaw, root_r)

    # Recompute translation from the new reference root orientation. The input
    # head remains unchanged; in head-only windows FK still hits its position.
    root_pos = cy - (root_r @ (cx - rest[:, None, 0])[..., None]).squeeze(-1)
    all_w = cp / cp.sum(-1, keepdim=True).clamp_min(1e-8)
    offsets = rest[:, None] - rest[:, None, :1]
    expected = (root_r[:, :, None] @ offsets[..., None]).squeeze(-1)
    fallback_pos = ((pos - expected) * all_w[..., None]).sum(2)
    root_pos = torch.where(root_valid[..., None], root_pos, fallback_pos)
    root_pos = refinement.interpolate_evidence(root_pos, cp.sum(-1) > 0)
    global_r = [root_r]
    local = [root_r]
    # Preserve the original scalar confidence: changing confidence is a separate
    # ablation. Observed head confidence does not become a new pelvis label.
    rotation_confidence = [torch.where(root_valid, cp[:, :, :4].amax(-1), meta[:, :, 15, 3])]
    for j in range(1, 22):
        parent = int(PARENTS[j])
        inherited = global_r[parent]
        children = [k for k in range(j + 1, 22) if int(PARENTS[k]) == j]
        candidate = inherited
        confidence = pos.new_full((b, t), .05)
        if children:
            weights = cp[:, :, children] * cp[:, :, j, None]
            bones = (rest[:, children] - rest[:, j, None])[:, None].expand(b, t, -1, -1)
            delta = pos[:, :, children] - pos[:, :, j, None]
            source = (inherited[:, :, None] @ bones[..., None]).squeeze(-1)
            weighted_source = (source * weights[..., None]).sum(2)
            weighted_delta = (delta * weights[..., None]).sum(2)
            candidate = refinement.align_vectors(weighted_source, weighted_delta) @ inherited
            if len(children) > 1:
                fitted, singular, _, _ = refinement.fit_rotation(bones, delta, weights)
                candidate = torch.where((singular[..., 1] > 1e-7)[..., None, None], fitted, candidate)
            confidence = torch.where(weights.sum(-1) > 0, weights.amax(-1).sqrt() * .5, confidence)
        observed_valid = meta[:, :, j, 1] * meta[:, :, j, 3] > 0
        candidate = torch.where(observed_valid[..., None, None], observed_r[:, :, j], candidate)
        confidence = torch.where(observed_valid, meta[:, :, j, 3], confidence)
        global_r.append(candidate)
        local.append(inherited.transpose(-1, -2) @ candidate)
        rotation_confidence.append(confidence)
    local = torch.stack(local, 2)
    six = torch.cat((local[..., :, 0], local[..., :, 1]), -1) - pos.new_tensor(IDENTITY_6D)
    reference = torch.cat(((root_pos - rest[:, None, 0]) / 2, six.flatten(2), pos.new_zeros(b, t, 66)), -1)
    reference = reconcile_motion(reference, rest)
    confidence = torch.stack((cp.clamp_min(.05), torch.stack(rotation_confidence, 2).clamp_min(.05)), -1)
    return reference, confidence


@torch.no_grad()
def yaw_only_reference_from_observations(batch):
    """Ego fallback only; EXO and supplied full-reference paths stay original."""
    modes = batch.get('mode', ())
    if 'reference_motion' in batch or 'ego' not in modes:
        return _ORIGINAL_REFERENCE(batch)
    return _build_reference(batch, yaw_only=True)


def install():
    """Install in this process; existing imported model_inputs aliases see it.

    Returns an uninstall callback. Call once in each training/evaluation worker,
    before building model inputs. No checkpoint tensor or measured input changes.
    """
    current = refinement.reference_from_observations
    if current not in (_ORIGINAL_REFERENCE, yaw_only_reference_from_observations):
        raise RuntimeError('Another process-local reference override is installed')
    refinement.reference_from_observations = yaw_only_reference_from_observations
    return uninstall


def uninstall():
    current = refinement.reference_from_observations
    if current is yaw_only_reference_from_observations:
        refinement.reference_from_observations = _ORIGINAL_REFERENCE
    elif current is not _ORIGINAL_REFERENCE:
        raise RuntimeError('Refusing to remove another reference override')


@contextmanager
def installed():
    """Scoped installation for CPU checks; preserve an outer installation."""
    already_installed = refinement.reference_from_observations is yaw_only_reference_from_observations
    install()
    try:
        yield
    finally:
        if not already_installed:
            uninstall()
