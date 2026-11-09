"""Fixed-neighbour multiscale scene refinement on a frozen motion prior.

This is a new input/model version. Historical 15-channel refiner checkpoints
remain unchanged and must not be interpreted as trained multiscale models.
"""
import numpy as np
import torch
from torch import nn

from scenemi.geometry.spatial_scene import SceneView, query_view
from scenemi.refinement.model import (ScenePoseRefiner, SceneRefinementPipeline,
    ANCHOR_JOINTS, apply_rotations)
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations


class SpatialPoseRefiner(ScenePoseRefiner):
    """Shared point encoding; separate near/context summaries per skin anchor."""
    def __init__(self, refiner_policy, spatial_policy):
        super().__init__(refiner_policy)
        self.spatial_policy = spatial_policy
        point_width = 32
        # Relative xyz, normal, distance, normal-valid, observation count,
        # static/dynamic ownership. Invalid/padded points are explicitly masked.
        self.point_encoder = nn.Sequential(nn.Linear(11, point_width), nn.SiLU(),
            nn.Linear(point_width, point_width), nn.SiLU())
        self.scene_feature_dim = point_width*4+8+10
        self.geometry = nn.Sequential(nn.Linear(self.scene_feature_dim, refiner_policy['width']),
            nn.SiLU(), nn.Linear(refiner_policy['width'], refiner_policy['width']))

    def encode(self, queries, anchors, rotations):
        convert = lambda value: torch.as_tensor(value, device=anchors.device, dtype=anchors.dtype)
        in_frame = lambda value: (rotations.transpose(-1, -2)[..., None, :, :] @ value[..., None])[..., 0]
        pooled, quality = [], []
        for name, radius in (('near', self.spatial_policy['query']['near_radius_m']),
                             ('context', self.spatial_policy['query']['context_radius_m'])):
            patch = queries[name]; valid = convert(patch['valid'])
            observations = convert(patch['observations'])
            count_cap = self.spatial_policy['query']['observation_count_cap']
            features = torch.cat((in_frame(convert(patch['points'])-anchors[..., None, :])/radius,
                in_frame(convert(patch['normals']))*convert(patch['normal_valid'])[..., None],
                convert(patch['distance'])[..., None]/radius,
                convert(patch['normal_valid'])[..., None],
                (torch.log1p(observations)/np.log1p(count_cap)).clamp_max(1)[..., None],
                (convert(patch['owner']) == 0)[..., None], (convert(patch['owner']) > 0)[..., None]), -1)
            value = self.point_encoder(features)*valid[..., None]
            mean = value.sum(-2)/valid.sum(-1, keepdim=True).clamp_min(1)
            maximum = value.masked_fill(~valid.bool()[..., None], -torch.inf).amax(-2)
            maximum = torch.where(valid.any(-1)[..., None], maximum, torch.zeros_like(maximum))
            pooled.extend((mean, maximum))
            nearest = convert(patch['nearest_distance'])
            quality.extend((valid.mean(-1), convert(patch['angular_coverage']),
                (torch.log1p(convert(patch['neighbour_count']))/np.log1p(128)).clamp_max(1),
                torch.where(torch.isfinite(nearest), (nearest/radius).clamp_max(4), 4.)))
        ground = queries['ground']; mask = convert(ground['valid'])
        normal = (rotations.transpose(-1, -2) @ convert(ground['normal'])[..., None])[..., 0]
        offset = (rotations.transpose(-1, -2) @ convert(ground['offset'])[..., None])[..., 0]
        # Ten explicit ground descriptors, without substituting a plane for
        # a missing measured surface. Ambiguous candidates remain masked out.
        ground_features = torch.cat((offset/self.spatial_policy['ground']['search_radius_m'], normal,
            convert(ground['confidence'])[..., None], mask[..., None],
            convert(ground['fit_error'])[..., None]/self.spatial_policy['ground']['maximum_fit_error_m'],
            convert(ground['ambiguous'])[..., None]), -1)
        ground_features = ground_features*mask[..., None]
        return torch.cat((*pooled, torch.stack(quality, -1), ground_features), -1)


class SpatialSceneRefinementPipeline(SceneRefinementPipeline):
    """Full frozen sampling -> B -> indexed queries -> rotation residual -> B."""
    def __init__(self, original_pipeline, refiner_policy, spatial_policy):
        nn.Module.__init__(self)
        self.base = original_pipeline.base.requires_grad_(False).eval()
        self.skin = original_pipeline.skin.requires_grad_(False).eval()
        self.anchor = original_pipeline.anchor
        self.geometry_policy = dict(original_pipeline.geometry_policy)
        self.spatial_policy = spatial_policy
        self.refiner = SpatialPoseRefiner(refiner_policy, spatial_policy)
        self.query_executor = None

    def close_queries(self):
        if self.query_executor is not None:
            self.query_executor.close()

    @staticmethod
    def check_public(public):
        SceneRefinementPipeline.check_public({k: v for k, v in public.items() if k != 'scene_views'})
        if 'scene_views' in public and not all(v is None or isinstance(v, SceneView) for v in public['scene_views']):
            raise ValueError('scene_views must contain public observed map indices, never labels')

    @torch.no_grad()
    def coarse(self, public, *, steps=20, seed=777):
        self.check_public(public)
        # CPU map handles are used only after the original frozen sampler.
        return super().coarse({k: v for k, v in public.items() if k != 'scene_views'}, steps=steps, seed=seed)

    def refine(self, raw, details, public, *, use_scene=True):
        self.check_public(public)
        placed = self.locate(raw, details)
        tensor_public = {k: v for k, v in public.items() if k != 'scene_views'}
        motion_features, _, _ = super().features(placed, details, tensor_public, False)
        anchors, _ = self.skin_points(placed, details, public['body_format'])
        rotations = global_rotations(placed)[:, :, list(ANCHOR_JOINTS)].detach()
        views = public.get('scene_views', (None,)*len(raw))
        if len(views) != len(raw):
            raise ValueError('One original-video map view is required per example')
        enabled = torch.as_tensor(use_scene, device=raw.device, dtype=torch.bool)
        if enabled.ndim > 1 or (enabled.ndim == 1 and len(enabled) != len(raw)):
            raise ValueError('use_scene must be scalar or a per-example mask')
        enabled = enabled.reshape(-1).expand(len(raw)) & (public['body_format'] > 0)
        features, evidence, near_hit, ground_valid = [], [], [], []
        ground_mask = np.array([j in (7, 8, 10, 11) for j in ANCHOR_JOINTS])
        pending = {}
        if self.query_executor is not None:
            for i, view in enumerate(views):
                if view is not None and enabled[i]:
                    dynamic = tuple(public[k][i].detach().cpu().numpy() for k in
                        ('frame_scene_points', 'frame_scene_owner', 'frame_normals', 'frame_normal_valid'))
                    pending[i] = self.query_executor.submit('features', view,
                        anchors[i].detach().cpu().numpy(), self.spatial_policy,
                        ground_mask=ground_mask, dynamic=dynamic)
        for i, view in enumerate(views):
            if view is None or not enabled[i]:
                features.append(raw.new_zeros(*anchors.shape[1:3], self.refiner.scene_feature_dim))
                empty = torch.zeros(anchors.shape[1:3], device=raw.device, dtype=torch.bool)
                evidence.append(empty);near_hit.append(empty);ground_valid.append(empty)
                continue
            if self.query_executor is None:
                dynamic = tuple(public[k][i].detach().cpu().numpy() for k in
                    ('frame_scene_points', 'frame_scene_owner', 'frame_normals', 'frame_normal_valid'))
                query = query_view(view, anchors[i].detach().cpu().numpy(), self.spatial_policy,
                    ground_mask=ground_mask, dynamic=dynamic)
            else:
                query = pending[i].result()
            features.append(self.refiner.encode(query, anchors[i].detach(), rotations[i]))
            convert = lambda x: torch.as_tensor(x, device=raw.device, dtype=torch.bool)
            near_hit.append(convert(query['near']['valid'].any(-1)))
            ground_valid.append(convert(query['ground']['valid']))
            evidence.append(convert(query['context']['valid'].any(-1) | query['ground']['valid']))
        scene_features, hit = torch.stack(features), torch.stack(evidence)
        omega, extra = self.refiner(motion_features, scene_features, hit)
        changed = apply_rotations(raw, omega, details['rest'])
        final = self.locate(changed, details)
        active = hit.flatten(1).any(-1)
        final = torch.where(active[:, None, None], final, placed)
        return final, dict(details, **extra, rotation_residual=omega, coarse_motion=placed,
            scene_active=active, anchor_near_hit=torch.stack(near_hit), ground_valid=torch.stack(ground_valid))


def build_spatial_model(c, paths, policy, prior_file=None):
    """Strictly load a declared frozen prior; historical 22k runs stay pinned."""
    import yaml
    from scenemi.config import ROOT
    from scenemi.checkpoint import load_trained_model
    from scenemi.data.amass_prepare import sha256
    from scenemi.refinement.training import build_model
    from scenemi.scene_geometry_training import skin_bases

    if prior_file is None:
        original = build_model(c, paths)
        identity = dict(checkpoint=c['continuation']['checkpoint'],
            checkpoint_sha256=c['continuation']['checkpoint_sha256'], step=22000)
    else:
        identity = yaml.safe_load(prior_file.read_text())
        if identity['version'] != 'frozen-scene-prior-v1':
            raise ValueError('Unsupported frozen-prior identity')
        for name in ('checkpoint', 'selection'):
            if sha256(ROOT/identity[name]) != identity[name+'_sha256']:
                raise ValueError(f'Frozen prior {name} changed')
        import json
        selection = json.loads((ROOT/identity['selection']).read_text())
        state = torch.load(ROOT/identity['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        if (state['step'] != identity['step'] or selection['step'] != identity['step']
                or state['contract'].get('stage') != identity['stage']
                or state['contract']['config']['model']['scene_modules_trainable'] is not False):
            raise ValueError('Expected the selected, scene-free prior with its declared stage and step')
        base, _, _ = load_trained_model(ROOT/identity['checkpoint'], paths)
        base.patch_enabled = base.local_enabled = base.feedback_enabled = False
        original = SceneRefinementPipeline(base, skin_bases(c), c['geometry'], c['anchor'], c['refiner'])
    model = SpatialSceneRefinementPipeline(original, c['refiner'], policy)
    model.prior_identity = dict(identity)
    return model


def check(policy_file, output, device='cuda', prior_file=None):
    """Disposable real four-source forward/backward check, with no optimizer."""
    from scenemi.config import ROOT, load_paths
    from scenemi.data.amass_prepare import atomic_json, sha256
    from scenemi.data.video_scene import load_policy
    from scenemi.data.spatial_scene_data import IndexedGeometryEgoData, collate, payload
    from scenemi.execution import configure_runtime
    from scenemi.refinement.training import load_config, fixed_cases, verify_contract
    from scenemi.training import deterministic, tensor_digest
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

    policy = load_policy(policy_file); configure_runtime(policy, device)
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    c = load_config(ROOT/'configs/ego_scene_refinement.yaml');deterministic(c['seed'], device)
    seal = torch.load(ROOT/policy['audit']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)['contract']
    verify_contract(seal)
    model = build_spatial_model(c, paths, policy, prior_file).to(device)
    data = IndexedGeometryEgoData(c, paths, 'validation', spatial_policy=policy,
        video_manifest=ROOT/policy['audit']['video_maps']/'MANIFEST.json')
    cases = fixed_cases(data, c)
    selected = [next(v for v in cases if v['dataset'] == source) for source in c['data']['datasets']]
    pairs = [data.sample_case(case) for case in selected]
    batch = collate([p[0] for p in pairs], device);public = payload(batch)
    before = tensor_digest(model.base.state_dict())
    raw, details = model.coarse(public, steps=c['sampling']['ddim_steps'], seed=c['validation']['seed'])
    baseline = model.locate(raw, details)
    with torch.no_grad():
        initial, extra = model.refine(raw, details, public)
    if not torch.equal(initial, baseline):
        raise AssertionError('New spatial refiner changed the prior at zero initialization')
    # Perturb only disposable residual-head weights to check the full
    # gradient path through both point scales, FK and the final B solve.
    torch.nn.init.normal_(model.refiner.rotation.weight, std=.01)
    final, learned = model.refine(raw, details, public)
    positions = forward_kinematics(final, details['rest'])
    target = forward_kinematics(batch['motion'], batch['target_rest'])
    loss = (positions-target).square().mean()+learned['rotation_residual'].square().mean()
    loss.backward()
    gradients = {name: float(parameter.grad.norm()) for name, parameter in model.refiner.named_parameters()
        if parameter.grad is not None}
    if not gradients.get('point_encoder.0.weight', 0.) > 0 or not all(np.isfinite(v) for v in gradients.values()):
        raise AssertionError('Scene point encoder has no finite nonzero gradient')
    if any(p.grad is not None or p.requires_grad for p in model.base.parameters()) or tensor_digest(model.base.state_dict()) != before:
        raise AssertionError('Spatial refinement altered the frozen motion/shape prior')
    with torch.no_grad():
        fallback, _ = model.refine(raw, details, public, use_scene=False)
    if not torch.equal(fallback, baseline):
        raise AssertionError('Scene-off fallback is not exactly the original B output')
    verify_contract(seal)
    result = dict(state='passed', device=device, real_sources=c['data']['datasets'],
        prior_identity=model.prior_identity,
        source_identities=[p[1] for p in pairs], near_hit_fraction=float(extra['anchor_near_hit'].float().mean()),
        context_or_ground_evidence_fraction=float(extra['anchor_hit'].float().mean()),
        bounded_ground_fraction=float(extra['ground_valid'].float().mean()),
        zero_initialization_exact=True, scene_off_fallback_exact=True, frozen_prior_unchanged=True,
        gradients=gradients, full_maps_on_cpu=True, fixed_neighbours_per_anchor=policy['query']['neighbours_per_scale'],
        optimizer_created=False, formal_training_started=False, new_model_performance_measured=False,
        rich_used=False, test_used=False, policy_sha256=sha256(policy_file),
        dependencies_sha256={str(ROOT/p): sha256(ROOT/p) for p in
            ('scenemi/refinement/spatial_model.py', 'scenemi/data/spatial_scene_data.py',
             'scenemi/geometry/spatial_scene.py', 'scenemi/data/video_scene.py')})
    atomic_json(output, result)
    return result


def main(argv=None):
    import argparse
    from pathlib import Path
    from scenemi.config import ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check',))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ego_scene_spatial_query.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--prior', type=Path, help='Explicit selected frozen-prior identity; default is retained 22k')
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(ROOT/'runs'):
        parser.error('Acceptance records must stay inside runs')
    result = check(args.config, args.output, args.device, args.prior)
    print(result['state'], flush=True)


if __name__ == '__main__':
    main()
