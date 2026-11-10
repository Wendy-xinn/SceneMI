"""Bounded scene adaptation around a frozen, independently trained prior.

The original backbone and shape decoder stay in evaluation mode. Scene paths
start at zero. Only observed-scene modules receive parameter updates; gradients
may pass through the frozen denoiser. This does not guarantee improved metrics.
"""
import torch

from scenemi.local_model import LocalScenePrior
from scenemi.scene_ablation import SCENE_MODULES
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion


def bounded_translation(delta, limit):
    """Smooth radial bound in physical metres, with identity derivative at zero."""
    length = delta.norm(dim=-1, keepdim=True)/limit
    scale = torch.where(length>1e-6, length.tanh()/length.clamp_min(1e-9), torch.ones_like(length))
    return delta*scale


class FrozenSceneResidual(LocalScenePrior):
    def set_adaptation(self, policy):
        self.residual_policy = dict(policy)
        self.requires_grad_(False)
        self._adapter_modules = tuple(SCENE_MODULES)+('core.scene_embedding',)
        for name in self._adapter_modules:
            self.get_submodule(name).requires_grad_(True)
        self.local_bps_gain.requires_grad_(True)
        self.patch_enabled = self.local_enabled = self.feedback_enabled = True
        self._baseline_raw = None
        self.train(False)
        return self

    def train(self, mode=True):
        # Never enable dropout or mutable training statistics in the base prior.
        super().train(False)
        self.training = mode
        for name in getattr(self, '_adapter_modules', ()):
            self.get_submodule(name).train(mode)
        return self

    def _core_pass(self, noisy, timestep, batch, scene, control_mask,
                   use_scene, use_control, query_motion=None):
        raw = super()._core_pass(noisy, timestep, batch, scene, control_mask,
                                use_scene, use_control, query_motion=query_motion)
        if not use_scene:
            self._baseline_raw = raw.detach()
        return raw

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True,
                use_control=True, control_mask=None, return_details=False):
        self._baseline_raw = None
        candidate, details = super().forward(noisy_motion, timestep, batch,
            use_scene=use_scene, use_control=use_control, control_mask=control_mask, return_details=True)
        raw = self._baseline_raw
        if raw is None:
            raise RuntimeError('The frozen scene-free proposal was not computed')
        baseline = reconcile_motion(raw, details['rest'])
        if use_scene:
            bounded = details['raw_motion'].clone()
            root_m = bounded_translation((candidate[..., :3]-baseline[..., :3])*2,
                                          self.residual_policy['root_limit_m'])
            bounded[..., :3] = baseline[..., :3]+root_m/2
            limit = self.residual_policy['rotation_6d_limit']
            bounded[..., 3:135] = raw[..., 3:135]+limit*torch.tanh(
                (candidate[..., 3:135]-baseline[..., 3:135])/limit)
            prediction = reconcile_motion(bounded, details['rest'])
            details['raw_motion'] = bounded
        else:
            prediction = candidate
        details['baseline_motion'] = baseline
        self._baseline_raw = None
        return (prediction, details) if return_details else prediction

    def adapter_state(self):
        return {name:p.detach().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def load_adapter_state(self, state):
        parameters = dict(self.named_parameters())
        if set(state) != {name for name,p in parameters.items() if p.requires_grad}:
            raise ValueError('Adapter checkpoint parameter set changed')
        with torch.no_grad():
            for name,value in state.items():
                if parameters[name].shape != value.shape:
                    raise ValueError('Adapter checkpoint shape changed: '+name)
                parameters[name].copy_(value)
