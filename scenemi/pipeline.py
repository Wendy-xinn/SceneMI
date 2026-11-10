"""Top-level SceneMI motion pipeline.

The pipeline owns the order of operations: observation sanitization, body
geometry, camera-to-head conditioning, optional scene conditioning, temporal
denoising, and FK.  The component implementations live in ``observation_encoder``,
``body``, ``scene``, and ``unet``.  Existing parameter names are deliberately
unchanged so checkpoints from the former ``scene_model`` loader remain valid.
"""
import copy
import json
from pathlib import Path
import yaml

from scenemi.config import ROOT

import torch
from torch import nn
from torch.nn import functional as F

from scenemi.models import AmassMotionPrior
from scenemi.observations import body_condition, public_observations, camera_supervision
from scenemi.geometry.native_shape import NativeBodyAdapter
from scenemi.losses import motion_loss
from scenemi.observation_encoder import NativeObservationEncoder
from scenemi.body import BodyGeometryMixin
from scenemi.scene import SceneConditionerMixin
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.supervision import IDENTITY_6D, rotation_from_6d


class MotionPipeline(BodyGeometryMixin, SceneConditionerMixin, AmassMotionPrior):
    def __init__(self, body_model_path, pv_mount, *, prior_state=None, bases=None,
                 latent_dim=256, dim_mults=(1, 2, 4)):
        super().__init__(body_model_path, latent_dim=latent_dim, dim_mults=dim_mults)
        if prior_state is not None:
            self.load_state_dict(prior_state, strict=True)
        encoder = NativeObservationEncoder()
        encoder.load_state_dict(self.shape_encoder.state_dict(), strict=True)
        self.shape_encoder = encoder
        self.native_body = NativeBodyAdapter(self.shape.rest_template[:1], bases)
        self.register_buffer('pv_rotation', torch.tensor(pv_mount['rotation'], dtype=torch.float32))
        self.register_buffer('pv_translation', torch.tensor(pv_mount['translation_m'], dtype=torch.float32))
        self.register_buffer('pv_uncertainty', torch.tensor([pv_mount['position_p90_m'],
                             pv_mount['rotation_p90_rad']], dtype=torch.float32))
        # AMASS has no image intrinsics: these five feature columns never
        # received training. Zero-init their transfer, preserving the exact
        # prior on AMASS, rather than exposing random weights to pixel values.
        with torch.no_grad():
            self.camera_encoder[0].weight[:, 13:18].zero_()
            self.camera_encoder[0].weight[:, 34:36].zero_()
            # The SMPL-X tag was also absent in AMASS. Start its embedding
            # from the learned shared 22-joint SMPL-family tag.
            self.body_encoder[0].weight[:, 67].copy_(self.body_encoder[0].weight[:, 66])
        self.patch_enabled = True
        for name in ('patch_projection', 'patch_position', 'motion_query', 'region_embedding',
                     'patch_attention', 'patch_condition', 'owner_embedding', 'dynamic_embedding',
                     'dynamic_attention', 'dynamic_condition'):
            getattr(self, name).requires_grad_(True)
        self.core.scene_embedding.requires_grad_(True)
        nn.init.zeros_(self.patch_condition.weight)
        nn.init.zeros_(self.dynamic_condition.weight)
        nn.init.zeros_(self.core.scene_embedding.mlp_head.weight)
        nn.init.zeros_(self.core.scene_embedding.mlp_head.bias)

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True, use_control=True,
                control_mask=None, return_details=False):
        if noisy_motion.shape[1:] != (128, 201) or not torch.isfinite(noisy_motion).all():
            raise ValueError('Expected finite 128 x 201 motion')
        observations = public_observations(batch, enabled=use_control)
        if control_mask is not None:
            observations['observation_meta'] = observations['observation_meta'].clone()
            observations['observation_meta'][..., :4] *= control_mask[..., None]
        b = len(noisy_motion)
        formats = batch.get('body_format', torch.zeros(b, device=noisy_motion.device, dtype=torch.long))
        if formats.shape != (b,) or formats.dtype != torch.long or not ((formats>=0)&(formats<=3)).all():
            raise ValueError('A configured output format, without GT shape, is required')
        shape = self.predicted_body(observations, formats)
        pv = batch.get('sensor_profile', torch.zeros_like(formats)) == 1
        camera = observations['camera']
        mount_rotation = torch.eye(3, device=camera.device, dtype=camera.dtype).expand(b, 3, 3).clone()
        mount_rotation[pv] = self.pv_rotation
        shape['head_camera_translation'] = shape['head_camera_translation'].clone()
        shape['head_camera_translation'][pv] = self.pv_translation
        equivalent = dict(observations)
        if pv.any():
            identity = camera.new_tensor(IDENTITY_6D)
            rotation = rotation_from_6d(torch.where(observations['camera_available'][..., 1:] > 0,
                                                  camera[..., 3:], identity)-identity)
            head_rotation = rotation @ mount_rotation[:, None].transpose(-1, -2)
            six = torch.cat((head_rotation[..., 0], head_rotation[..., 1]), -1)
            equivalent['camera'] = torch.cat((camera[..., :3], six), -1)
        conditioned = body_condition(equivalent, shape)
        conditioned.update(camera=camera, mount_rotation=mount_rotation,
            mount_uncertainty=torch.where(pv[:, None], self.pv_uncertainty, 0.),
            body_type=F.one_hot((formats>=2).long(), 2).to(camera))
        empty = (noisy_motion.new_zeros(b, self.core.latent_dim), None, None, None)
        if use_scene:
            occupancy = batch['occupancy']
            if occupancy.shape != (b, 24, 48, 48) or not torch.isfinite(occupancy).all():
                raise ValueError('Expected a finite observed static occupancy grid')
            conditioned['occupancy'] = occupancy
            for key in ('frame_scene_points', 'frame_scene_owner'):
                if key in batch:
                    conditioned[key] = batch[key]
            with torch.no_grad():
                proposal = reconcile_motion(self._core_pass(noisy_motion, timestep, conditioned,
                    empty, control_mask, False, use_control), shape['rest'])
            scene = self._sampling_scene[1] if self._sampling_scene is not None else self.scene_tokens(conditioned)
            raw = self._core_pass(noisy_motion, timestep, conditioned, scene, control_mask,
                                  True, use_control, query_motion=proposal)
        else:
            raw = self._core_pass(noisy_motion, timestep, conditioned, empty, control_mask, False, use_control)
        result = reconcile_motion(raw, shape['rest'])
        details = dict(raw_motion=raw, **shape, observations=equivalent,
                       conditioned_observations=conditioned, body_format=formats)
        return (result, details) if return_details else result


def native_motion_loss(prediction, target, batch, signal, *, details, config):
    """Reuse the sealed motion terms; native betas/template supervision is separate."""
    base = copy.deepcopy(config)
    base['loss']['coefficients']['shape_betas'] = 0.
    # A zero-weight H coefficient term is retained only for the frozen loss API.
    labels = dict(batch, target_betas=torch.zeros_like(details['predicted_betas']))
    loss_details = dict(details, observations=camera_supervision(batch, details))
    terms = motion_loss(prediction, target, labels, signal, details=loss_details, config=base)
    human = (batch['body_format']==1) | (batch['body_format']==2)
    if human.any():
        coefficient = F.mse_loss(details['predicted_native_betas'][human], batch['target_native_betas'][human])
        template = F.cross_entropy(details['template_logits'][human], batch['target_template'][human])
    else:
        coefficient = template = prediction.new_zeros(())
    terms.update(native_shape_betas=coefficient, native_template=template)
    terms['total'] = terms['total'] + .05*coefficient + .01*template
    return terms


# Compatibility name retained for every existing trainer, evaluator, and
# checkpoint loader.  This is an alias rather than a subclass, so the module
# parameter tree is identical to the former scene_model implementation.
NativeScenePrior = MotionPipeline


def load_camera_mount(paths, scene_paths):
    spec = yaml.safe_load(Path(scene_paths).read_text())
    mount_file = Path(spec['pv_mount'])
    mount = json.loads(mount_file.read_text())['camera_wearer']
    if mount.get('fitting_split') != 'train':
        raise ValueError('Camera mount must be fit on train data only')
    return mount, mount_file


def build_ego_model(c, paths):
    mount, _ = load_camera_mount(paths, ROOT / c['scene_paths'])
    model = MotionPipeline(paths['smplh_models'] / 'neutral/model.npz', mount,
                          latent_dim=c['model']['latent_dim'], dim_mults=tuple(c['model']['dim_mults']))
    # No scene module is allowed to receive gradients or enter the forward
    # pass in this stage.  Native shape heads remain trainable.
    scene_names = ('patch_projection', 'patch_position', 'motion_query',
                   'region_embedding', 'patch_attention', 'patch_condition',
                   'owner_embedding', 'dynamic_embedding', 'dynamic_attention',
                   'dynamic_condition', 'surface_embedding', 'anchor_embedding',
                   'anchor_query', 'surface_attention', 'region_condition',
                   'contact_readout', 'contact_pose', 'correspondence_readout',
                   'feedback_temporal', 'root_feedback', 'rotation_feedback')
    for name in scene_names:
        if hasattr(model, name):
            getattr(model, name).requires_grad_(False)
    model.core.scene_embedding.requires_grad_(False)
    model.patch_enabled = model.local_enabled = model.feedback_enabled = False
    return model


