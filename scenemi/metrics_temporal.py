"""Explicit temporal metrics for cross-model validation, in a Y-up world.

Jitter/100 and the XYZ skating reduction follow AvatarJLM test.py and
utils/utils_metric.py, cited by EgoGVAE. The additional horizontal statistic
and released EgoAllo cumulative skating value have distinct names/units.
No alignment is applied to prediction or truth before these measurements.
"""
import torch


def temporal_metrics(prediction, target, fps=30):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[-2:] != (22, 3):
        raise ValueError('Expected matching [B,T,22,3] predictions and targets')
    if prediction.shape[1] < 4 or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('At least four finite frames required')
    pred, gt = prediction.double(), target.double()
    feet = [7, 8, 10, 11]
    delta = torch.diff(pred[:, :, feet], dim=1)
    gt_delta = torch.diff(gt[:, :, feet], dim=1)
    contact = gt_delta.norm(dim=-1) <= .01
    # Exact AvatarJLM reduction: zero non-contact XYZ components, then mean
    # over all transitions, four feet and three components (not contact count).
    skating = (delta.abs() * contact[..., None]).flatten(1).mean(1) * 100
    horizontal = delta[..., [0, 2]].norm(dim=-1)
    counts = contact.flatten(1).sum(1)
    horizontal_mean = (horizontal * contact).flatten(1).sum(1) / counts.clamp_min(1)
    # Released EgoAllo drops the whole predicted window to its lowest joint.
    # This floor shift is confined to this diagnostic; it never changes MPJPE.
    height = pred[:, :, feet, 1] - pred[:, :, 1:, 1].flatten(1).amin(1)[:, None, None]
    thresholds = pred.new_tensor([.08, .08, .04, .04])
    ratio = height[:, 1:] / thresholds
    weight = (2 - torch.pow(2., ratio.clamp(max=1))) * (height[:, 1:] < thresholds)
    released_skate = (delta[..., [0, 2]].abs().sum(-1) * weight).flatten(1).sum(1)
    return {
        'jitter_100m_s3': torch.diff(pred, n=3, dim=1).norm(dim=-1).flatten(1).mean(1) * fps**3 / 100,
        'gt_jitter_100m_s3': torch.diff(gt, n=3, dim=1).norm(dim=-1).flatten(1).mean(1) * fps**3 / 100,
        'foot_sliding_avatarjlm_cm_frame': skating,
        'foot_sliding_horizontal_mm_contact_frame': horizontal_mean * 1000,
        'gt_foot_contact_transitions': counts.double(),
        'foot_sliding_horizontal_has_contact': (counts > 0).double(),
        'egoallo_cumulative_foot_skate_m': released_skate,
    }
