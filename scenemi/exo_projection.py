"""Robust observed-joint translation constraints for EXO denoising.

This changes translation only: local rotations, shape and bone lengths remain
unchanged. An observation-free sequence returns zero correction. This is an
observable-coordinate correction, not recovery of unknowable global drift.
"""
import torch
from torch import nn

from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from scenemi.observations import public_observations


def translation_correction(predicted, observed, available, confidence, *, iterations=3,
                           huber_m=.08, temporal_weight=1., ridge=1e-5):
    if predicted.shape != observed.shape or predicted.shape[-2:] != (22, 3):
        raise ValueError('Expected world joint tracks [B,T,22,3] in matching units')
    valid = available.bool() & torch.isfinite(observed).all(-1) & torch.isfinite(predicted).all(-1)
    error = torch.where(valid[..., None], observed-predicted, 0.)
    weights = torch.nan_to_num(confidence).clamp(0, 1)*valid
    # Give the pelvis/torso most influence, while retaining non-torso evidence.
    anatomical = weights.new_full((22,), .25)
    anatomical[[0,1,2,3,6,9,12,15]] = 1.
    weights = weights*anatomical
    frames = predicted.shape[1]
    difference = torch.eye(frames, device=predicted.device, dtype=predicted.dtype)
    difference = difference[2:]-2*difference[1:-1]+difference[:-2]
    regularizer = temporal_weight*(difference.T@difference)+ridge*torch.eye(frames, device=predicted.device, dtype=predicted.dtype)
    delta = error.new_zeros(predicted.shape[0], frames, 3)
    for _ in range(iterations):
        distance = (error-delta[:, :, None]).norm(dim=-1)
        robust = (huber_m/distance.clamp_min(huber_m))
        w = weights*robust
        count = w.sum(-1)
        matrix = torch.diag_embed(count)+regularizer[None]
        rhs = (w[..., None]*error).sum(-2)
        delta = torch.linalg.solve(matrix, rhs)
    return delta


class AnchoredExoPrior(nn.Module):
    def __init__(self, prior, policy):
        super().__init__()
        self.prior = prior
        self.policy = dict(policy)
        self._sampling_scene = None

    @property
    def alphas(self):
        return self.prior.alphas

    def forward(self, noisy_motion, timestep, batch, *, use_scene=False, use_control=True,
                control_mask=None, return_details=False):
        prediction, details = self.prior(noisy_motion, timestep, batch,
            use_scene=use_scene, use_control=use_control, control_mask=control_mask, return_details=True)
        observations = public_observations(batch, enabled=use_control)
        meta = observations['observation_meta']
        if control_mask is not None:
            meta = meta.clone()
            meta[..., :4] *= control_mask[..., None]
        joints = forward_kinematics(prediction, details['rest'])
        delta = translation_correction(joints, observations['trajectory'][..., :3]*2,
            meta[..., 0]>0, meta[..., 2], **self.policy)
        # EGO never reaches an observed-body constraint; exact mode bypass.
        exo = observations['camera_context'][:, 0] == 0
        delta = delta*exo[:, None, None]
        if exo.any():
            moved = reconcile_motion(prediction, details['rest'], translation_m=delta)
            prediction = torch.where(exo[:, None, None], moved, prediction)
        details = dict(details, observation_translation_delta_m=delta)
        return (prediction, details) if return_details else prediction
