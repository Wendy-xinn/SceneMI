"""Small inference-only support candidates for controlled native30 ablations.

These functions read predicted motion, body calibration and observed surfaces
only. ``gravity_weight`` is a root-upright prior, NOT Newtonian gravity: gravity
does not itself orient a freely falling body. Local spine/hip/knee rotations
remain untouched. The independent support candidate uses finite observed static
patches, never a single global floor height, ground-truth contact or motion.

Both weights default to zero. Call AFTER the chosen observation constraint in
an instance ``constrain_motion`` hook; do not change a sealed training module.
The candidate deliberately does not reapply a head constraint afterwards,
which could immediately undo the support correction being measured.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from pytorch3d.transforms import matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d


DEFAULTS = {
    "gravity_weight": 0.0,
    "gravity_deadzone_rad": math.radians(10),
    "gravity_step_limit_rad": math.radians(2),
    "gravity_maximum_tilt_rad": math.radians(60),
    "minimum_signal": 0.5,
    "ego_only": True,
    "support_weight": 0.0,
    "support_step_limit_m": 0.01,
    "support_max_height_m": 0.25,
    "support_max_penetration_m": 0.05,
    "support_tangent_limit_m": 0.08,
    "support_min_normal_y": 0.9,
    "support_max_speed_m_s": 0.5,
    "support_clearance_m": 0.003,
}


def _configuration(config):
    if config is None:
        return dict(DEFAULTS)
    if not isinstance(config, Mapping):
        raise TypeError("Support configuration must be a mapping")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown support settings: " + ", ".join(sorted(unknown)))
    result = dict(DEFAULTS, **config)
    for key, value in result.items():
        if key == "ego_only":
            if not isinstance(value, bool):
                raise ValueError("ego_only must be boolean")
        elif isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
    for key in ("gravity_weight", "support_weight", "minimum_signal", "support_min_normal_y"):
        if result[key] > 1:
            raise ValueError(key + " must be in [0, 1]")
    if result["gravity_maximum_tilt_rad"] >= math.pi / 2:
        raise ValueError("Maximum gravity tilt must be less than pi/2")
    if result["gravity_deadzone_rad"] > result["gravity_maximum_tilt_rad"]:
        raise ValueError("Gravity deadzone exceeds maximum tilt")
    return result


def _gate(motion, batch, signal, config):
    signal = torch.as_tensor(signal, device=motion.device, dtype=motion.dtype)
    if signal.numel() not in (1, len(motion)):
        raise ValueError("signal must be scalar or have one value per example")
    signal = signal.reshape(-1).expand(len(motion))
    start = config["minimum_signal"]
    gate = ((signal - start) / max(1 - start, 1e-8)).clamp(0, 1)
    if config["ego_only"]:
        if len(batch["mode"]) != len(motion):
            raise ValueError("Mode batch size mismatch")
        gate = gate * motion.new_tensor([mode == "ego" for mode in batch["mode"]])
    return gate


def _yxz_matrix(yaw, pitch, roll):
    """R_y(yaw) @ R_x(pitch) @ R_z(roll), with +Y gravity up."""
    cy, sy = yaw.cos(), yaw.sin()
    cx, sx = pitch.cos(), pitch.sin()
    cz, sz = roll.cos(), roll.sin()
    return torch.stack((
        cy * cz + sy * sx * sz, -cy * sz + sy * sx * cz, sy * cx,
        cx * sz, cx * cz, -sx,
        -sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, cy * cx,
    ), -1).reshape(*yaw.shape, 3, 3)


def root_upright(motion, batch, signal, config=None):
    """Shrink moderate root pitch/roll, preserving horizontal forward heading.

    A 10-degree deadzone permits ordinary pelvic tilt. Deeply tilted, inverted,
    or forward-vertical roots are skipped: this is not a universal posture rule.
    A conservative L1 bound on Euler changes bounds each SO(3) update too.
    """
    cfg = _configuration(config)
    if cfg["gravity_weight"] == 0:
        return motion
    gate = _gate(motion, batch, signal, cfg)[:, None]
    if not bool((gate > 0).any()):
        return motion
    root = rotation_from_6d(motion[..., 3:9])
    horizontal = root[..., (0, 2), 2].square().sum(-1).sqrt()
    # clamp keeps angle derivatives finite at an exactly upright root.
    tilt = torch.acos(root[..., 1, 1].clamp(-1 + 1e-7, 1 - 1e-7))
    valid = (horizontal > 0.1) & (tilt < cfg["gravity_maximum_tilt_rad"])
    active = valid & (tilt > cfg["gravity_deadzone_rad"]) & (gate > 0)
    if not bool(active.any()):
        return motion
    yaw = torch.atan2(root[..., 0, 2], root[..., 2, 2])
    pitch = torch.atan2(-root[..., 1, 2], horizontal.clamp_min(1e-8))
    roll = torch.atan2(root[..., 1, 0], root[..., 1, 1])
    fraction = ((tilt - cfg["gravity_deadzone_rad"]).clamp_min(0)
                / tilt.clamp_min(1e-8)) * cfg["gravity_weight"] * gate * active
    dp, dr = -pitch * fraction, -roll * fraction
    bound = dp.abs() + dr.abs()
    limit = (cfg["gravity_step_limit_rad"] / bound.clamp_min(1e-8)).clamp_max(1)
    target = _yxz_matrix(yaw, pitch + dp * limit, roll + dr * limit)
    delta = matrix_to_axis_angle(target @ root.transpose(-1, -2))
    updates = motion.new_zeros(*motion.shape[:2], 22, 3)
    updates[..., 0, :] = delta * active[..., None]
    return reconcile_motion(motion, batch["rest"], rotation_axis_angle=updates)


def observed_support(motion, batch, signal, config=None, *, return_diagnostics=False):
    """Bounded root-Y support step from a persistent, slow predicted sole.

    Each foot queries its own nearby upward static patch. Missing evidence,
    walls, moving surfaces, fast swing/flight, and isolated one-frame evidence
    produce no force. Inference contact is heuristic and must be evaluated for
    false support during stairs, jumps and sitting; it is not contact truth.
    """
    cfg = _configuration(config)
    diagnostic = {"support_eligible_frames": 0, "support_total_frames": motion.shape[0] * motion.shape[1],
                  "support_root_delta_max_m": 0.0, "ground_truth_read": False}
    if cfg["support_weight"] == 0:
        return (motion, diagnostic) if return_diagnostics else motion
    gate = _gate(motion, batch, signal, cfg)
    if not bool((gate > 0).any()):
        return (motion, diagnostic) if return_diagnostics else motion
    from experiments.offline_camera_retrain_v1.unified_geometry import skin_points
    from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

    rotations = global_rotations(motion)
    shifts = motion.new_zeros(*motion.shape[:2], 3)
    for b, (asset, scene) in enumerate(zip(batch["skin_assets"], batch["surface_windows"])):
        if not bool(gate[b] > 0):
            continue
        labels = torch.as_tensor(asset["labels"], device=motion.device)
        sole_ids = [torch.isin(labels, labels.new_tensor(pair)).nonzero().flatten()
                    for pair in ((7, 10), (8, 11))]
        if any(len(ids) == 0 for ids in sole_ids):
            continue
        vertices = skin_points(motion[b], asset)
        # Lowest native skin anchor per foot, independent of future motion GT.
        ids = torch.stack([ids[vertices[:, ids, 1].argmin(-1)] for ids in sole_ids], -1)
        frame = torch.arange(motion.shape[1], device=motion.device)[:, None]
        soles = vertices[frame, ids]
        query_rotation = rotations[b, frame, labels[ids]]
        if motion.device.type == "cuda" and hasattr(scene, "query_wide_motion"):
            raw = scene.query_wide_motion(soles.detach().cpu().numpy(),
                                          query_rotation.detach().cpu().numpy(), motion.device)
        else:
            raw = getattr(scene, "query_wide", scene.query)(
                soles.detach().cpu().numpy(), query_rotation.detach().cpu().numpy())
        q = {key: torch.as_tensor(value, device=motion.device)[:, :, 2]
             for key, value in raw.items()}
        delta = soles.detach()[:, :, None] - q["points"]
        signed = (delta * q["normals"]).sum(-1)
        tangent = (delta - signed[..., None] * q["normals"]).norm(dim=-1)
        valid = (q["valid"] & q["normal_reliable"] & (q["owners"] == 0)
                 & (q["normals"][..., 1] > cfg["support_min_normal_y"])
                 & (tangent < cfg["support_tangent_limit_m"])
                 & (signed < cfg["support_max_height_m"])
                 & (signed > -cfg["support_max_penetration_m"]))
        distance = delta.norm(dim=-1)
        score, nearest = torch.where(valid, distance, torch.inf).min(-1)
        selected_signed = signed.gather(-1, nearest[..., None]).squeeze(-1)
        selected_y = q["normals"][..., 1].gather(-1, nearest[..., None]).squeeze(-1)
        velocity = torch.zeros_like(soles)
        if len(soles) > 1:
            velocity[1:] = soles.detach().diff(dim=0) * 30
            velocity[0] = velocity[1]
        supported = torch.isfinite(score) & (velocity.norm(dim=-1) < cfg["support_max_speed_m_s"])
        persistent = torch.zeros_like(supported)
        if len(soles) >= 3:
            persistent[1:-1] = supported[:-2] & supported[1:-1] & supported[2:]
        # Prefer the closest supported sole; do not pin a raised swing foot.
        chosen_score, foot = torch.where(persistent, score, torch.inf).min(-1)
        chosen_signed = selected_signed.gather(-1, foot[:, None]).squeeze(-1)
        chosen_y = selected_y.gather(-1, foot[:, None]).squeeze(-1)
        usable = torch.isfinite(chosen_score)
        correction = ((cfg["support_clearance_m"] - chosen_signed)
                      / chosen_y.clamp_min(cfg["support_min_normal_y"]))
        correction = correction * cfg["support_weight"] * gate[b]
        correction = correction.clamp(-cfg["support_step_limit_m"], cfg["support_step_limit_m"])
        shifts[b, :, 1] = torch.where(usable, correction, 0)
        diagnostic["support_eligible_frames"] += int(usable.sum())
    diagnostic["support_root_delta_max_m"] = float(shifts.norm(dim=-1).max().detach())
    result = reconcile_motion(motion, batch["rest"], translation_m=shifts)
    return (result, diagnostic) if return_diagnostics else result


def apply_support_constraints(motion, batch, signal, config=None):
    """Observation hook adapter: root-upright prior, then observed support step."""
    cfg = _configuration(config)
    if cfg["gravity_weight"] == 0 and cfg["support_weight"] == 0:
        return motion
    return observed_support(root_upright(motion, batch, signal, cfg), batch, signal, cfg)


constraint = apply_support_constraints
