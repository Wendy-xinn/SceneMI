"""Scene adaptation with scheme-B query locations and a shared final readout.

The denoiser still returns RAW clean motion to the diffusion sampler. Call
``final_motion`` exactly once on the final sample, or on a training x0 for loss.
"""
import torch

from scenemi.scene_geometry_model import GeometryScenePrior
from scenemi.scene_adaptation import SCENE_MODULES
from scenemi.local_model import LocalScenePrior
from scenemi.geometry.temporal_head_anchor import TemporalHeadAnchor
from scenemi.geometry.surface_query import query_surface, plane_gap
from scenemi.data.scene_ego import SCENE_KEYS
from scenemi.data.scene_geometry_data import GEOMETRY_KEYS
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion

SCENE_INPUTS = SCENE_KEYS + GEOMETRY_KEYS
GEOMETRY_MODULES = ('geometry_encoder', 'geometry_joint', 'geometry_projection')


def mask_scene_inputs(public, keep):
    result = dict(public)
    for key in SCENE_INPUTS:
        value = public[key]
        fill = torch.full_like(value, -1) if key=='frame_scene_owner' else torch.zeros_like(value)
        result[key] = torch.where(keep.reshape(len(keep), *([1]*(value.ndim-1))), value, fill)
    return result


class AnchoredScenePrior(GeometryScenePrior):
    def __init__(self, *args, anchor_policy, adaptation, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor = TemporalHeadAnchor(anchor_policy)
        self.backbone_prefixes = tuple(adaptation['backbone_prefixes'])
        self.requires_grad_(False)
        for name in SCENE_MODULES + GEOMETRY_MODULES + self.backbone_prefixes:
            self.get_submodule(name).requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        for name in self.backbone_prefixes:
            self.get_submodule(name).train(mode)
        return self

    def final_motion(self, motion, details):
        observations = details['observations']
        return self.anchor(motion, details['rest'], details['head_camera_translation'],
            observations['camera'], observations['camera_available'][..., 0])

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True, use_control=True,
                control_mask=None, return_details=False):
        if not use_scene:
            return super().forward(noisy_motion, timestep, batch, use_scene=False,
                use_control=use_control, control_mask=control_mask, return_details=return_details)
        conditioned, observations, shape, formats = LocalScenePrior.conditions(self, batch, use_control, control_mask)
        empty = (noisy_motion.new_zeros(len(noisy_motion), self.core.latent_dim), None, None, None)
        with torch.no_grad():
            proposal = reconcile_motion(self._core_pass(noisy_motion, timestep, conditioned, empty,
                control_mask, False, use_control), shape['rest'])
            proposal = self.anchor(proposal, shape['rest'], shape['head_camera_translation'],
                observations['camera'], observations['camera_available'][..., 0])
            anchors, labels = self.skin_points(proposal, shape, formats)
            patch = query_surface(anchors, batch, self.geometry_policy)
        hit, valid = patch['hit'], patch['valid']; radius = self.geometry_policy['query_radius_m']
        delta = torch.where(hit[..., None], (patch['centre']-anchors)/radius, 0.)
        feature = torch.cat((delta, patch['normal'], plane_gap(anchors, patch)[..., None]/radius,
            patch['distance'][..., None]/radius, hit[..., None], valid[..., None],
            (patch['owner']==0)[..., None], anchors/2,
            self.alphas[timestep].sqrt()[:, None, None, None].expand(*anchors.shape[:-1], 1)), -1)
        gate = hit & (formats[:, None, None]>0)
        label_feature = self.geometry_joint(labels)[:, None].expand(-1, anchors.shape[1], -1, -1)
        context = self.geometry_encoder(torch.cat((feature, label_feature), -1))*gate[..., None]
        regions = context[:, :, :66].reshape(len(context), 128, 22, 3, 64).mean(3).clone()
        regions[:, :, 15] = (context[:, :, 45:48].sum(2)+context[:, :, 66])/4
        embedding = self.geometry_projection(regions.flatten(2))*self.alphas[timestep].square()[:, None, None]
        final = dict(conditioned, body_local_embedding=embedding)
        scene = self._sampling_scene[1] if self._sampling_scene is not None else self.scene_tokens(conditioned)
        raw = self._core_pass(noisy_motion, timestep, final, scene, control_mask, True, use_control, query_motion=proposal)
        result = reconcile_motion(raw, shape['rest'])
        details = dict(raw_motion=raw, **shape, observations=observations,
            conditioned_observations=conditioned, body_format=formats,
            surface_query_coverage=gate.float().mean(), surface_normal_coverage=valid.float().mean())
        return (result, details) if return_details else result
