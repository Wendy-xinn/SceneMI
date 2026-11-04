"""One trainable AMASS denoiser for EGO generation and EXO repair.

The EXO reference and adapted passes share weights. When the shared prior is
trainable, BOTH passes participate in autograd: d(adapted-reference)/d(theta).
There is no frozen teacher and no second set of denoiser weights.
"""
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from scenemi.exo_completion import BoundaryEncoder
from scenemi.exo_denoise import DenoisingExoPrior
from scenemi.observations import public_observations, body_condition
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d


class JointMotionPrior(DenoisingExoPrior):
    def __init__(self, base, **settings):
        trainable = {name for name, p in base.named_parameters() if p.requires_grad}
        super().__init__(base, **settings)
        self.condition = BoundaryEncoder(self.condition, settings.get('hidden', 128))
        for name, parameter in self.base.named_parameters():
            parameter.requires_grad_(name in trainable)

    def train(self, mode=True):
        super().train(mode)
        self.base.train(mode)
        self.geometry.eval()
        return self

    def forward(self, noisy, timestep, payload, *, return_details=False):
        ego = payload['camera_context'][:, 0] > 0
        if ego.all():
            return self.base(noisy, timestep, payload, use_scene=False,
                             return_details=return_details)
        if ego.any():
            raise ValueError('Accumulate homogeneous EGO and EXO microbatches')
        initial, rest = payload['initial_motion'], payload['initial_rest']
        if initial.shape != noisy.shape or initial.shape[1:] != (128, 201):
            raise ValueError('Expected complete estimated motion [B,128,201]')
        if not torch.isfinite(initial).all() or not torch.isfinite(rest).all():
            raise ValueError('Nonfinite estimated initialization')
        reliability = payload['initial_reliability'].clamp(0, 1)
        root_reliability = payload['initial_root_reliability'].clamp(0, 1)
        anchor = self.anchor(payload)
        conditioned = body_condition(public_observations(payload), dict(
            rest=rest, head_camera_translation=rest.new_zeros(len(rest), 3)))
        empty = (noisy.new_zeros(len(noisy), self.base.core.latent_dim), None, None, None)
        # Deliberately no no_grad()/detach(): the reference is the CURRENT
        # shared model, not an immutable teacher or an asymmetric gradient.
        reference = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
        signal = self.alphas[timestep].sqrt()[:, None, None].expand(-1, 128, 1)
        features = torch.cat((initial, noisy[..., :135], reliability,
            root_reliability[..., None], payload['estimate_available'][..., None],
            payload['joint_visible'], signal), -1)
        hidden = self.condition(features.transpose(1, 2))
        with self.inject([layer(hidden) for layer in self.projections]):
            adapted = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
        difference = (adapted[..., :135]-reference[..., :135])*self.residual_gain
        root = self.root_limit_m*torch.tanh(difference[..., :3]*2/self.root_limit_m)
        root = root/root.norm(dim=-1, keepdim=True).clamp_min(self.root_limit_m)*self.root_limit_m
        root = root*(1-root_reliability[..., None])
        delta_rotation = rotation_from_6d(difference[..., 3:].reshape(len(noisy), 128, 22, 6))
        angle = matrix_to_axis_angle(delta_rotation)
        rotation = self.rotation_limit_rad*torch.tanh(angle/self.rotation_limit_rad)
        rotation = rotation/rotation.norm(dim=-1, keepdim=True).clamp_min(self.rotation_limit_rad)*self.rotation_limit_rad
        rotation = rotation*(1-reliability[..., None])
        moved = reconcile_motion(anchor, rest, translation_m=root, rotation_axis_angle=rotation)
        fixed = (root_reliability >= 1)&(reliability >= 1).all(-1)
        prediction = torch.where(fixed[..., None], initial, moved)
        details = dict(rest=rest, root_delta_m=root, rotation_delta_rad=rotation,
            fixed_frames=fixed, reference_x0=reference, anchor_motion=anchor,
            reference_gradient='full_difference_both_passes_same_trainable_weights')
        return (prediction, details) if return_details else prediction
