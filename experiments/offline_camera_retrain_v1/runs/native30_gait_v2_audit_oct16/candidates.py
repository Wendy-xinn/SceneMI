"""Isolated proposals for later matched TRAIN experiments, not active V2 code.

Neither helper changes the running experiment. This file is intentionally under
the review directory, outside the source hash set of the frozen training run.
"""
import torch
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.native30_gt_reliability import gt_weight
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d


def conservative_root_reference(known, inferred_rotation_confidence=.05):
    """Preserve the estimate, but distinguish head fallback from root evidence.

    A directly observed root or an observable torso fit keeps its confidence.
    A head-only guess does not inherit the head sensor's rotation confidence.
    No motion target, contact target, or other GT label is read.
    """
    if not 0 <= inferred_rotation_confidence <= 1:
        raise ValueError('Expected confidence in [0,1]')
    _, meta = sanitize_observations(known['trajectory'], known['observation_meta'])
    root_observed = (meta[:,:,0,1] > 0) & (meta[:,:,0,3] > 0)
    torso_fit = known['torso_anchor_confidence'] > 0
    confidence = known['reference_confidence'].clone()
    confidence[:,:,0,1] = torch.where(root_observed | torso_fit, confidence[:,:,0,1],
        confidence[:,:,0,1].clamp_max(inferred_rotation_confidence))
    return dict(known, reference_confidence=confidence)


def body_local_foot_position(prediction, truth, batch, signal):
    """Training-only per-foot position; retains mean posture and exact timing.

    This complements the existing bilateral difference/amplitude/relation terms.
    It is not a claim that the current complete loss lacks a position penalty:
    the active V1-derived pelvis-relative position term uses world axes.
    Coefficient choice and benefit require a matched training comparison.
    """
    def local(motion):
        joints = forward_kinematics(motion, batch['rest'])
        root = rotation_from_6d(motion[...,3:9])
        return torch.einsum('btji,btkj->btki',root,joints-joints[:,:,:1])[:,:,(10,11)]
    predicted, target = local(prediction), local(truth.detach())
    quality = gt_weight(batch,'gt_position_weight',prediction)
    rotation_quality = gt_weight(batch,'gt_global_rotation_weight',prediction)
    mask = torch.minimum(quality[:,:,(10,11)], quality[:,:,:1])
    mask = torch.minimum(mask, rotation_quality[:,:,:1])
    error = (predicted-target).square().sum(-1)*mask
    return (error.flatten(1).mean(1)*signal.detach().reshape(-1)).mean()
