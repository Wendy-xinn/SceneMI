"""Differentiable translation anchoring; predictions and public camera only.

L2 reproduces the validated scheme B. Optional Huber IRLS is an ablation,
not a validated robustness claim. Missing observations are availability masks,
never ground-truth-derived camera confidence.
"""
import numpy as np
import torch
from torch import nn

from scenemi.geometry.head_anchor import camera_residual, correction_operator
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


class TemporalHeadAnchor(nn.Module):
    def __init__(self, policy, frames=128, fps=30):
        super().__init__()
        self.policy = dict(policy)
        if policy['data_loss'] not in ('l2', 'huber'):
            raise ValueError('Anchoring supports L2 or the explicit Huber ablation')
        method = dict(kind='quadratic', data_weight=policy['data_weight'],
            acceleration_weight=policy['acceleration_weight'], jerk_weight=policy['jerk_weight'])
        eye = np.eye(frames)
        d2, d3 = np.diff(eye, n=2, axis=0)*fps**2, np.diff(eye, n=3, axis=0)*fps**3
        penalty = method['acceleration_weight']*d2.T@d2 + method['jerk_weight']*d3.T@d3
        self.register_buffer('operator', torch.from_numpy(correction_operator(method, frames, fps)), persistent=False)
        self.register_buffer('penalty', torch.from_numpy(penalty), persistent=False)

    def correction(self, residual, availability=None):
        if residual.ndim != 3 or residual.shape[1:] != (len(self.operator), 3) or not torch.isfinite(residual).all():
            raise ValueError('Expected finite [B,128,3] camera residuals in meters')
        e = residual.double()
        weight = torch.ones_like(e[..., 0]) if availability is None else availability.to(e)
        if weight.shape != e.shape[:2] or not torch.isfinite(weight).all() or ((weight<0)|(weight>1)).any():
            raise ValueError('Observation availability must be finite [B,T] in [0,1]')
        if self.policy['data_loss']=='l2' and bool((weight==1).all()):
            return self.operator @ e
        base_weight = weight*self.policy['data_weight']
        delta = self.operator @ e
        iterations = self.policy['irls_iterations'] if self.policy['data_loss']=='huber' else 1
        for _ in range(iterations):
            effective = base_weight
            if self.policy['data_loss']=='huber':
                # Radial Huber: one robust weight per frame for its 3D residual.
                distance = (delta-e).norm(dim=-1).clamp_min(1e-12)
                effective = effective*(self.policy['huber_delta_m']/distance).clamp_max(1)
            matrix = torch.diag_embed(effective)+self.penalty
            # D2/D3 leave affine trajectories unconstrained in missing streams.
            # This small zero-correction prior makes that case well-posed.
            ridge = self.policy['missing_ridge']
            matrix = matrix + torch.eye(len(self.operator), device=e.device, dtype=e.dtype)*ridge
            delta = torch.linalg.solve(matrix, effective[..., None]*e)
        return delta

    def forward(self, motion, rest, mount, camera, availability=None, *, return_details=False):
        residual = camera_residual(motion, rest, mount, camera)
        correction = self.correction(residual, availability)
        # Concatenation avoids modifying FK views needed during backpropagation.
        root = motion[..., :3]+correction.to(motion)/2
        provisional = torch.cat((root, motion[..., 3:]), -1)
        joints = forward_kinematics(provisional, rest)
        result = torch.cat((root, motion[..., 3:135], joints.flatten(2)/2), -1)
        if return_details:
            return result, dict(correction_m=correction, camera_residual_before_m=residual,
                camera_residual_after_m=residual-correction)
        return result
