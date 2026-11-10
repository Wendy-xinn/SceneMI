"""Native scene denoising with predicted-skin conditions and gated feedback.

Keep two shared denoiser passes. Surface lookup sees only causal observed
history and the current frame, never full hidden meshes or target body labels.
New output projections start at zero to preserve the selected global model.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from scenemi.geometry.skin import PredictedSkin
from scenemi.observations import public_observations, body_condition
from scenemi.scene_model import NativeScenePrior
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d, IDENTITY_6D


def nearest_observed(anchors, history, history_valid, frame_points, frame_owner, *, k=8, radius=.6):
    """Detached discrete lookup; missing surface evidence has an explicit mask."""
    query = anchors.detach().cpu().numpy()
    hp = history.detach().cpu().numpy()
    hv = history_valid.detach().cpu().numpy()
    fp = frame_points.detach().cpu().numpy()
    fo = frame_owner.detach().cpu().numpy()
    t, a = query.shape[:2]
    def query_tree(points, owners, centres):
        shape = centres.shape[:-1]+(k,)
        neighbours = np.zeros(shape+(3,), np.float32)
        ids = np.zeros(shape, np.int64)
        distances = np.full(shape, np.inf)
        valid = np.isfinite(points).all(-1)&(owners>=0)&(owners<100)
        points, owners = points[valid], owners[valid]
        if len(points):
            n = min(k, len(points))
            d, index = cKDTree(points).query(centres, k=list(range(1, n+1)), distance_upper_bound=radius)
            found = np.isfinite(d)
            index = np.minimum(index, len(points)-1)
            neighbours[..., :n, :] = points[index]
            ids[..., :n] = owners[index]
            distances[..., :n] = d
            neighbours[..., :n, :][~found] = 0
        return neighbours, ids, distances
    static = query_tree(hp[hv], np.zeros(int(hv.sum()), np.int64), query)
    current = [query_tree(fp[i], fo[i], query[i]) for i in range(t)]
    combined = [np.concatenate((static[j], np.stack([v[j] for v in current])), axis=-2 if j==0 else -1)
                for j in range(3)]
    order = np.argsort(combined[2], axis=-1, kind='stable')[..., :k]
    points = np.take_along_axis(combined[0], order[..., None], axis=2)
    owners = np.take_along_axis(combined[1], order, axis=2)
    valid = np.isfinite(np.take_along_axis(combined[2], order, axis=2))
    return (torch.as_tensor(points, device=anchors.device, dtype=anchors.dtype),
            torch.as_tensor(owners, device=anchors.device), torch.as_tensor(valid, device=anchors.device))


class LocalScenePrior(NativeScenePrior):
    def __init__(self, body_model_path, pv_mount, *, local, prior_state=None,
                 scene_state=None, bases=None, skin_bases=None):
        if (prior_state is None)==(scene_state is None):
            raise ValueError('Provide exactly one AMASS prior or scene initializer')
        super().__init__(body_model_path, pv_mount, prior_state=prior_state, bases=bases)
        if scene_state is not None:
            self.load_state_dict(scene_state, strict=True)
        self.skin = PredictedSkin(skin_bases)
        self.local_config = dict(local)
        self.local_enabled = self.feedback_enabled = True
        self.local_bps_gain = nn.Parameter(torch.zeros(()))
        for name in ('surface_embedding', 'anchor_embedding', 'anchor_query', 'surface_attention',
                     'region_condition', 'contact_readout', 'contact_pose', 'correspondence_readout',
                     'feedback_temporal', 'root_feedback', 'rotation_feedback'):
            getattr(self, name).requires_grad_(True)
        nn.init.zeros_(self.region_condition.weight)
        for layer in (self.root_feedback, self.rotation_feedback):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.root_limit_m = local['root_limit_m']
        self.rotation_limit_rad = local['rotation_limit_rad']

    def conditions(self, batch, use_control, control_mask):
        observations = public_observations(batch, enabled=use_control)
        if control_mask is not None:
            observations['observation_meta'] = observations['observation_meta'].clone()
            observations['observation_meta'][..., :4] *= control_mask[..., None]
        camera = observations['camera']
        b = len(camera)
        formats = batch['body_format']
        shape = self.predicted_body(observations, formats)
        pv = batch['sensor_profile']==1
        mount = torch.eye(3, device=camera.device, dtype=camera.dtype).expand(b, 3, 3).clone()
        mount[pv] = self.pv_rotation
        shape['head_camera_translation'] = shape['head_camera_translation'].clone()
        shape['head_camera_translation'][pv] = self.pv_translation
        equivalent = dict(observations)
        if pv.any():
            identity = camera.new_tensor(IDENTITY_6D)
            rotation = rotation_from_6d(torch.where(observations['camera_available'][..., 1:]>0,
                                                  camera[..., 3:], identity)-identity)
            head = rotation@mount[:, None].transpose(-1, -2)
            equivalent['camera'] = torch.cat((camera[..., :3], head[..., :, 0], head[..., :, 1]), -1)
        conditioned = body_condition(equivalent, shape)
        conditioned.update(camera=camera, mount_rotation=mount,
            mount_uncertainty=torch.where(pv[:, None], self.pv_uncertainty, 0.),
            body_type=F.one_hot((formats>=2).long(), 2).to(camera), occupancy=batch['occupancy'],
            frame_scene_points=batch['frame_scene_points'], frame_scene_owner=batch['frame_scene_owner'])
        return conditioned, equivalent, shape, formats

    def local_geometry(self, proposal, timestep, batch, shape, formats):
        vertices, labels = self.skin(proposal, shape, formats)
        rotations = global_rotations(proposal)
        contexts, bps, masks, reliability, coverage = [], [], [], [], []
        velocity = torch.zeros_like(vertices)
        velocity[:, 1:] = torch.diff(vertices, dim=1)*30
        for i in range(len(proposal)):
            p, owner, valid = nearest_observed(vertices[i], batch['local_history_points'][i],
                batch['local_history_valid'][i], batch['local_frame_points'][i], batch['local_frame_owner'][i],
                k=self.local_config['neighbours'], radius=self.local_config['query_radius_m'])
            r = rotations[i, :, labels[i]]
            delta = p-vertices[i, :, :, None]
            local = torch.einsum('taik,tani->tank', r, delta)
            distance = local.norm(dim=-1, keepdim=True)
            # No reliable normal, signed distance or surface velocity was observed.
            zeros = torch.zeros_like(distance)
            feature = torch.cat((local, distance, torch.zeros_like(local), zeros, zeros,
                (owner>0)[..., None].to(local), torch.zeros_like(local), zeros,
                self.owner_embedding(owner.clamp(0, 127))), -1)
            feature = torch.where(valid[..., None], feature, 0.)
            tokens = self.surface_embedding(feature)
            t, a, k = valid.shape
            keys = torch.cat((tokens.reshape(t*a, k, 64), tokens.new_zeros(t*a, 1, 64)), 1)
            padding = torch.cat((~valid.reshape(t*a, k), valid.new_zeros(t*a, 1)), 1)
            pose = torch.cat((r[..., :, 0], r[..., :, 1]), -1)
            hit = valid.any(-1)
            query = torch.cat((vertices[i]/2, (velocity[i]/.2).clamp(-20, 20), pose,
                self.alphas[timestep[i]].sqrt().expand(t, a, 1), hit[..., None].to(local)), -1)
            query = self.anchor_embedding.weight[None]+self.anchor_query(query)
            context, _ = self.surface_attention(query.reshape(t*a, 1, 64), keys, keys,
                key_padding_mask=padding, need_weights=False)
            context = context.reshape(t, a, 64)*hit[..., None]
            near = (distance[..., 0]<=self.local_config['feedback_radius_m'])&valid
            contexts.append(torch.stack([context[:, labels[i]==j].mean(1) for j in range(22)], 1))
            reliability.append(torch.stack([near[:, labels[i]==j].any((1, 2)) for j in range(22)], 1))
            bps.append(torch.where(valid[..., :1], local[..., 0, :]/2, 0.))
            masks.append(valid[..., 0])
            coverage.append(hit.float().mean())
        return torch.stack(contexts), torch.stack(bps), torch.stack(masks), torch.stack(reliability), torch.stack(coverage)

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True, use_control=True,
                control_mask=None, return_details=False):
        if not use_scene or not self.local_enabled:
            return super().forward(noisy_motion, timestep, batch, use_scene=use_scene,
                use_control=use_control, control_mask=control_mask, return_details=return_details)
        if noisy_motion.shape[1:]!=(128, 201) or not torch.isfinite(noisy_motion).all():
            raise ValueError('Expected finite 128-frame native motion')
        conditioned, observations, shape, formats = self.conditions(batch, use_control, control_mask)
        empty = (noisy_motion.new_zeros(len(noisy_motion), self.core.latent_dim), None, None, None)
        with torch.no_grad():
            proposal = reconcile_motion(self._core_pass(noisy_motion, timestep, conditioned,
                empty, control_mask, False, use_control), shape['rest'])
        context, bps, valid, reliable, coverage = self.local_geometry(proposal.detach(), timestep, batch, shape, formats)
        fine = self.alphas[timestep].square()[:, None, None]
        final = dict(conditioned, bps=bps*fine[..., None]*self.local_bps_gain.tanh(), bps_valid=valid,
                     body_local_embedding=self.region_condition(context.flatten(2))*fine)
        scene = self._sampling_scene[1] if self._sampling_scene is not None else self.scene_tokens(conditioned)
        raw = self._core_pass(noisy_motion, timestep, final, scene, control_mask, True, use_control, query_motion=proposal)
        safe = reconcile_motion(raw, shape['rest'])
        positions = forward_kinematics(safe, shape['rest'])
        rotations = global_rotations(safe)
        velocity = torch.zeros_like(positions)
        velocity[:, 1:] = torch.diff(positions, dim=1)*30
        features = torch.cat((positions/2, (velocity/.2).clamp(-20, 20), rotations[..., :, 0],
            rotations[..., :, 1], self.alphas[timestep].sqrt()[:, None, None, None].expand(*positions.shape[:3], 1)), -1)
        contact = self.contact_readout(context+self.contact_pose(features))[..., 0]
        correspondence = self.correspondence_readout(context)[..., 0]
        support = contact.sigmoid()*correspondence.sigmoid()*reliable
        root_delta = raw.new_zeros(*raw.shape[:2], 3)
        rotation_delta = raw.new_zeros(*raw.shape[:2], 22, 3)
        if self.feedback_enabled:
            hidden = self.feedback_temporal((context*support[..., None]).flatten(2).transpose(1, 2))
            root = torch.tanh(self.root_feedback(hidden).transpose(1, 2))
            root = root/root.norm(dim=-1, keepdim=True).clamp_min(1)
            rotation = torch.tanh(self.rotation_feedback(hidden).transpose(1, 2)).reshape(*raw.shape[:2], 22, 3)
            rotation = rotation/rotation.norm(dim=-1, keepdim=True).clamp_min(1)
            root_delta = root*self.root_limit_m*fine*support.amax(-1, keepdim=True)
            rotation_delta = rotation*self.rotation_limit_rad*fine[..., None]*support[..., None]
        result = reconcile_motion(raw, shape['rest'], root_delta, rotation_delta)
        details = dict(raw_motion=raw, **shape, observations=observations, body_format=formats,
            conditioned_observations=conditioned, root_delta_m=root_delta, rotation_delta_rad=rotation_delta,
            contact_logits=contact, correspondence_logits=correspondence, local_coverage=coverage)
        return (result, details) if return_details else result
