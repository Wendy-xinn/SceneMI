"""Metric support relations for the shared residual body denoiser.

Only predicted bodies, observed tracks and observed scene samples enter queries.
Wide upward patches supply context; they never imply a completed floor or SDF.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from experiments.offline_camera_retrain_v1.native30_refinement import Native30Refinement
from experiments.offline_camera_retrain_v1.native30_observations import sanitize_observations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


def relation_query(scene, centres, *, neighbors=4, radius=1.5, support_radius=1.25):
    """Nearest visible samples plus broad, explicitly marked upward samples."""
    if hasattr(scene, 'base') and hasattr(scene, 'rotation'):
        result = relation_query(scene.base, centres @ scene.rotation, neighbors=neighbors,
                                radius=radius, support_radius=support_radius)
        for key in ('points', 'normals', 'velocities'): result[key] = result[key] @ scene.rotation.T
        return result
    frames, joints = centres.shape[:2]; k = neighbors
    shape = (frames, joints, 2 * k)
    result = {key: np.zeros((*shape, 3), np.float32) for key in ('points', 'normals', 'velocities')}
    result.update({key: np.zeros(shape, bool) for key in ('valid', 'normal_reliable', 'velocity_valid', 'wide')})
    result['owners'] = np.zeros(shape, np.int64)
    static = scene.static; count = len(static['points'])
    # Query static neighbours for the entire window at once.
    if count:
        sd, si = scene._static_tree.query(centres, k=list(range(1, k + 1)), distance_upper_bound=radius)
    else: sd = np.full((frames, joints, k), np.inf); si = np.zeros_like(sd, dtype=np.int64)
    for t in range(frames):
        dynamic = scene.dynamic[t]; dn = len(dynamic['points'])
        if dn:
            if scene._dynamic_trees[t] is None: scene._dynamic_trees[t] = cKDTree(dynamic['points'])
            dd, di = scene._dynamic_trees[t].query(centres[t], k=list(range(1, k + 1)), distance_upper_bound=radius)
        else: dd = np.full((joints, k), np.inf); di = np.zeros_like(dd, dtype=np.int64)
        order = np.argsort(np.concatenate((sd[t], dd), -1), axis=-1)[:, :k]
        distances = np.take_along_axis(np.concatenate((sd[t], dd), -1), order, -1)
        is_static = order < k
        ids = np.where(is_static, np.take_along_axis(si[t], order.clip(max=k-1), -1),
                       np.take_along_axis(di, (order-k).clip(min=0), -1))
        for key in ('points', 'normals', 'velocities', 'owners', 'velocity_valid'):
            values = np.zeros((*ids.shape, *static[key].shape[1:]), dtype=static[key].dtype)
            for mask, cloud in ((is_static, static), (~is_static, dynamic)):
                use = mask & np.isfinite(distances)
                if use.any(): values[use] = cloud[key][ids[use]]
            result[key][t, :, :k] = values
        reliability = np.zeros_like(is_static)
        for mask, cloud in ((is_static, static), (~is_static, dynamic)):
            use = mask & np.isfinite(distances)
            if use.any(): reliability[use] = cloud['reliable'][ids[use]]
        result['valid'][t, :, :k] = np.isfinite(distances)
        result['normal_reliable'][t, :, :k] = reliability
    if not hasattr(scene, '_relation_up_ids'):
        scene._relation_up_ids = np.flatnonzero(static['reliable'] & (static['normals'][:, 1] > .7))
        up = static['points'][scene._relation_up_ids]
        scene._relation_up_tree = cKDTree(up[:, (0, 2)]) if len(up) else None
    if scene._relation_up_tree is not None:
        distance, ids = scene._relation_up_tree.query(centres[..., (0, 2)], k=list(range(1, 2*k+1)),
                                                     distance_upper_bound=support_radius)
        safe = scene._relation_up_ids[ids.clip(max=len(scene._relation_up_ids)-1)]
        points = static['points'][safe]
        vertical = centres[..., None, 1] - points[..., 1]
        valid = np.isfinite(distance) & (vertical > -.35) & (vertical < 2.5)
        score = np.where(valid, distance + np.abs(vertical) * .15, np.inf)
        chosen = score.argsort(-1)[..., :k]
        selected = np.take_along_axis(safe, chosen, -1)
        valid = np.take_along_axis(valid, chosen, -1)
        for key in ('points', 'normals', 'velocities', 'owners', 'velocity_valid'):
            result[key][:, :, k:] = static[key][selected]
        result['valid'][..., k:] = valid; result['normal_reliable'][..., k:] = valid
        result['wide'][..., k:] = valid
    return result


class Native30SceneRelations(Native30Refinement):
    def __init__(self, *args, relations, guidance, **kwargs):
        super().__init__(*args, **kwargs)
        self.relations_config = relations; self.guidance_config = guidance
        self.guidance_enabled = bool(guidance.get('enabled', True))
        self.relation_encoder = nn.Sequential(nn.Linear(20, 64), nn.SiLU(), nn.Linear(64, 64))
        self.relation_query_encoder = nn.Linear(11, 64)
        self.relation_joint_embedding = nn.Embedding(22, 64)
        self.relation_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.relation_output = nn.Linear(64, 64, bias=False)
        nn.init.normal_(self.relation_output.weight, std=.001)

    def local_geometry(self, proposal, timestep, batch):
        context, bps, valid, geometry = super().local_geometry(proposal, timestep, batch)
        positions = forward_kinematics(proposal, batch['rest'])
        rotations = global_rotations(proposal)
        trajectory, meta = sanitize_observations(batch['trajectory'], batch['observation_meta'])
        alpha = self.alphas[timestep]
        uncertainty = batch['observation_uncertainty'][..., 0].clamp_min(0)
        confidence = meta[..., 0] * meta[..., 2] * .05**2 / (.05**2 + uncertainty.square())
        blend = (1 - alpha)[:, None, None, None] * confidence[..., None]
        centres = positions * (1 - blend) + trajectory[..., :3] * 2 * blend
        body_velocity = torch.zeros_like(positions); body_velocity[:, 1:] = positions.diff(dim=1) * 30
        contexts = []
        for b, scene in enumerate(batch['surface_windows']):
            query = relation_query(scene, centres[b].detach().cpu().numpy(),
                radius=self.relations_config['radius_m'], support_radius=self.relations_config['support_radius_m'])
            q = {k: torch.as_tensor(v, device=proposal.device) for k, v in query.items()}
            delta = q['points'] - positions[b, :, :, None]
            r = rotations[b]
            local = torch.einsum('tjik,tjni->tjnk', r, delta)
            normal = torch.einsum('tjik,tjni->tjnk', r, q['normals']) * q['normal_reliable'][..., None]
            velocity = torch.einsum('tjik,tjni->tjnk', r, q['velocities'] - body_velocity[b, :, :, None])
            velocity = velocity * q['velocity_valid'][...,None]
            distance = delta.norm(dim=-1, keepdim=True)
            features = torch.cat((local, distance, normal, (local*normal).sum(-1, keepdim=True),
                (velocity/.2).clamp(-20,20), q['normal_reliable'][...,None], q['velocity_valid'][...,None],
                (q['owners']>0)[...,None], (q['normals'][...,1]>.7)[...,None], q['wide'][...,None],
                confidence[b,:,:,None,None].expand_as(distance), alpha[b].expand_as(distance),
                delta[...,1:2], delta[...,(0,2)].norm(dim=-1,keepdim=True)), -1)
            features = torch.where(q['valid'][..., None], features, 0.)
            tokens = self.relation_encoder(features).flatten(0,1)
            keys = torch.cat((tokens, tokens.new_zeros(len(tokens),1,64)),1)
            padding = torch.cat((~q['valid'].flatten(0,1),q['valid'].new_zeros(len(tokens),1)),1)
            pose = torch.cat((positions[b]/2, r[..., :, 0], r[..., :, 1],
                              confidence[b,...,None], alpha[b].expand(*positions[b].shape[:2],1)), -1)
            queries = self.relation_query_encoder(pose) + self.relation_joint_embedding.weight[None]
            attended,_ = self.relation_attention(queries.flatten(0,1)[:,None], keys, keys,
                                                  key_padding_mask=padding, need_weights=False)
            exists = q['valid'].any(-1)
            added = self.relation_output(attended[:,0].reshape_as(context[b])) * exists[...,None]
            contexts.append(context[b] + added * self.relations_config['gain'])
            geometry[b]['region_valid'] = geometry[b]['region_valid'] | exists
            geometry[b]['relation_valid'] = exists
        return torch.stack(contexts), bps, valid, geometry

    def forward(self, noisy_motion, timestep, batch, *, use_scene=True, return_details=False, **kwargs):
        result, details = super().forward(noisy_motion, timestep, batch, use_scene=use_scene, return_details=True, **kwargs)
        if use_scene and self.guidance_enabled:
            from experiments.offline_camera_retrain_v1.native30_physical_guidance import guide_motion
            result, diagnostic = guide_motion(result, timestep, batch, details, self.guidance_config, self.constrain_motion)
            details.update(diagnostic)
        return (result, details) if return_details else result
