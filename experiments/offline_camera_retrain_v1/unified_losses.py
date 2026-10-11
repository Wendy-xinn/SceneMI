"""Confidence-weighted per-track supervision with explicit units and masks."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_axis_angle

from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import (
    IDENTITY_6D, forward_kinematics, rotation_from_6d)
from experiments.offline_camera_retrain_v1.unified_observations import FPS, observation_derivatives

DEFAULT_WEIGHTS = dict(position=.02, rotation=.005, velocity=.003,
                       acceleration=.0002, angular_velocity=.001, body_turn_rate=.0005)
SCALES = dict(position=.05, rotation=float(np.deg2rad(10)), velocity=.15,
              acceleration=1., angular_velocity=float(np.deg2rad(20)),
              body_turn_rate=float(np.deg2rad(20)))
JOINT_WEIGHTS = [2., 1., 1., 1., 1., 1., 1., 1.5, 1.5, 1., 1.5, 1.5,
                 1., 1., 1., 2., 1., 1., 1., 1., 1.5, 1.5]


@torch.no_grad()
def fit_camera_mounts(data, seed=2026101137, samples_per_group=8):
    """TRAIN-only robust rigid-mount approximation, with measured uncertainty.

    Raw camera points are not anatomical head joints. Calibration is frozen
    for validation and never inferred from held-out body truth.
    """
    if data.split != 'train':
        raise ValueError('Camera mounts must be fitted on TRAIN only')
    output = {}
    for group in ('trumans', 'camera_wearer', 'interactee', 'rich'):
        rotations, translations, identities = [], [], []
        for _ in range(samples_per_group):
            sample, identity = data.sample(128, group)
            motion = torch.from_numpy(sample['motion'][None])
            head = global_rotations(motion)[0, ::8, 15].numpy()
            camera6 = torch.from_numpy(sample['camera'][::8, 3:])
            camera = rotation_from_6d(camera6 - camera6.new_tensor(IDENTITY_6D)).numpy()
            rotations.append(head.transpose(0, 2, 1) @ camera)
            delta = (sample['camera'][::8, :3] - sample['joints'][::8, 15]) * 2
            translations.append(np.einsum('tji,tj->ti', head, delta))
            identities.append(identity)
        rotations = np.concatenate(rotations); translations = np.concatenate(translations)
        mount = Rotation.from_matrix(rotations).mean().as_matrix()
        offset = np.median(translations, axis=0)
        rotation_error = Rotation.from_matrix(rotations @ mount.T).magnitude()
        output[group] = dict(rotation=mount.tolist(), translation_m=offset.tolist(),
                             position_p90_m=float(np.quantile(np.linalg.norm(translations - offset, axis=-1), .9)),
                             rotation_p90_rad=float(np.quantile(rotation_error, .9)),
                             identities=identities, fitting_split='train', seed=seed)
    return output


def track_losses(prediction, truth, batch, signal_weight, mounts, groups):
    meta = batch['observation_meta']
    position = forward_kinematics(prediction, batch['rest'])
    rotation = global_rotations(prediction)
    # Predict the sensor trajectory for camera-type conditions, while exo
    # anatomical observations supervise the native joints directly.
    position = position.clone(); rotation = rotation.clone()
    position_scale = prediction.new_full(meta.shape[:-1], SCALES['position'])
    rotation_scale = prediction.new_full(meta.shape[:-1], SCALES['rotation'])
    for b, group in enumerate(groups):
        camera = meta[b, :, 15, 4] < .5
        calibration = mounts[group]
        r = prediction.new_tensor(calibration['rotation'])
        t = prediction.new_tensor(calibration['translation_m'])
        rh = rotation[b, :, 15]
        camera_position = position[b, :, 15] + (rh @ t[:, None]).squeeze(-1)
        camera_rotation = rh @ r
        position[b, :, 15] = torch.where(camera[:, None], camera_position, position[b, :, 15])
        rotation[b, :, 15] = torch.where(camera[:, None, None], camera_rotation, rh)
        position_scale[b, :, 15] = torch.where(camera, max(.05, calibration['position_p90_m']), .05)
        rotation_scale[b, :, 15] = torch.where(camera, max(SCALES['rotation'], calibration['rotation_p90_rad']), SCALES['rotation'])
    observed = batch['trajectory']
    pm = (meta[..., 0] > 0) & torch.isfinite(observed[..., :3]).all(-1)
    rm = (meta[..., 1] > 0) & torch.isfinite(observed[..., 3:]).all(-1)
    target_position = torch.where(pm[..., None], observed[..., :3] * 2, 0.)
    target6 = torch.where(rm[..., None], observed[..., 3:], observed.new_tensor(IDENTITY_6D))
    target_rotation = rotation_from_6d(target6 - observed.new_tensor(IDENTITY_6D))
    target_v, target_a, target_w, derivative_mask = observation_derivatives(observed, meta)
    predicted_v = torch.zeros_like(position); predicted_v[:, 1:] = torch.diff(position, dim=1) * FPS
    predicted_a = torch.zeros_like(position); predicted_a[:, 2:] = torch.diff(predicted_v[:, 1:], dim=1) * FPS
    predicted_w = torch.zeros_like(position)
    predicted_w[:, 1:] = matrix_to_axis_angle(rotation[:, 1:] @ rotation[:, :-1].transpose(-1, -2)) * FPS
    importance = prediction.new_tensor(JOINT_WEIGHTS)[None, None]
    gate = signal_weight.detach().reshape(-1).square()

    def reduce(error, mask, confidence):
        weight = mask.to(prediction) * confidence.clamp(0, 1) * importance
        # Denominator excludes confidence and signal; those factors must
        # attenuate gradients, rather than cancelling during normalization.
        denominator = (mask.to(prediction) * importance).flatten(1).sum(1).clamp_min(1)
        per_example = (F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none') * weight).flatten(1).sum(1) / denominator
        return (per_example * gate).mean()

    pos_error = (position - target_position).norm(dim=-1) / position_scale
    rot_error = matrix_to_axis_angle(rotation @ target_rotation.transpose(-1, -2)).norm(dim=-1) / rotation_scale
    confidence_p = meta[..., 2]
    confidence_r = meta[..., 3]
    vc = torch.zeros_like(confidence_p); vc[:, 1:] = torch.minimum(confidence_p[:, 1:], confidence_p[:, :-1])
    ac = torch.zeros_like(confidence_p); ac[:, 2:] = torch.minimum(vc[:, 2:], vc[:, 1:-1])
    wc = torch.zeros_like(confidence_r); wc[:, 1:] = torch.minimum(confidence_r[:, 1:], confidence_r[:, :-1])
    terms = dict(position=reduce(pos_error, pm, confidence_p),
                 rotation=reduce(rot_error, rm, confidence_r),
                 velocity=reduce((predicted_v - target_v).norm(dim=-1) / SCALES['velocity'], derivative_mask[..., 0], vc),
                 acceleration=reduce((predicted_a - target_a).norm(dim=-1) / SCALES['acceleration'], derivative_mask[..., 1], ac),
                 angular_velocity=reduce((predicted_w - target_w).norm(dim=-1) / SCALES['angular_velocity'], derivative_mask[..., 2], wc))
    # Body angular velocity is training truth, independently labelled. It is
    # never reconstructed from the translation direction or raw camera yaw.
    pred_root = global_rotations(prediction)[:, :, 0]
    true_root = global_rotations(truth).detach()[:, :, 0]
    pr = matrix_to_axis_angle(pred_root[:, 1:] @ pred_root[:, :-1].transpose(-1, -2)) * FPS
    tr = matrix_to_axis_angle(true_root[:, 1:] @ true_root[:, :-1].transpose(-1, -2)) * FPS
    error = (pr - tr).norm(dim=-1) / SCALES['body_turn_rate']
    terms['body_turn_rate'] = (F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(1) * gate).mean()
    return terms


def calibrate_weights(base, terms, prediction, targets=None):
    """Fit weights on shared prediction gradients, using TRAIN examples only.

    Compare gradient magnitudes rather than raw losses in different units.
    This is a logged initial scale check, not a validation-driven search.
    """
    targets = targets or dict(position=.10, rotation=.04, velocity=.03,
                              acceleration=.005, angular_velocity=.015, body_turn_rate=.015)
    reference = torch.autograd.grad(base, prediction, retain_graph=True)[0].norm().item()
    weights, report = {}, {}
    for key, value in terms.items():
        norm = torch.autograd.grad(value, prediction, retain_graph=True)[0].norm().item()
        weight = min(DEFAULT_WEIGHTS[key] * 5,
                     reference * targets[key] / max(norm, 1e-12))
        weights[key] = weight
        report[key] = dict(loss=float(value.detach()), gradient_norm=norm, coefficient=weight,
                           weighted_gradient_ratio=weight * norm / max(reference, 1e-12),
                           target_ratio=targets[key], scale=SCALES[key])
    return weights, dict(reference_gradient_norm=reference, terms=report,
                         fitting_split='train', method='shared output gradient norm')
