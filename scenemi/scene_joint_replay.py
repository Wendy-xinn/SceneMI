"""Unified EGO/EXO scene conditioning on the completed replay prior.

The shared motion prior is frozen.  EGO and EXO use separate condition paths,
but both query the same causal observed scene and inject scene relations into
the denoiser.  A training update contains 64 EGO and 64 EXO windows.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.grail_augmentation import observed_prefix
from scenemi.data.local_scene import inference_payload as local_payload
from scenemi.data.native_repair import NativeRepairDataset
from scenemi.data.scene_dataset import bounded_frame_points
from scenemi.exo_denoise import channel_reliability, denoising_loss
from scenemi.exo_completion import BoundaryEncoder
from scenemi.joint_model import JointMotionPrior
from scenemi.scene_adapter import (GeometryConditionedPrior, RelationAdapter,
                                    inject_residuals)
from scenemi.scene_model import native_motion_loss
from scenemi.scene_repair_training import StageDataset
from scenemi.geometry.scene_features import relation_features
from scenemi.observations import public_observations
from scenemi.training import ensure_bounded, now, rng_state, tensor_digest
from scenemi.local_model import LocalScenePrior
from scenemi.scene_adapter import resize_temporal
from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def settings(path):
    c = yaml.safe_load(Path(path).read_text())
    if (c['version'] != 'replay-scene-joint-v1'
            or c['training']['batch_size'] != 128
            or c['training']['ego_batch_size'] != 64
            or c['training']['exo_batch_size'] != 64
            or c['data']['modes'] != ['ego', 'exo']
            or c['data']['excluded_datasets'] != ['rich']
            or c['validation']['test_used']):
        raise ValueError('Unexpected unified scene protocol')
    return c


class SceneExoDataset(StageDataset):
    """Native EXO initializer augmented with causal observed scene fields."""
    def sample(self, rng, policy, *, member=None, start=None, yaw=None, condition=None):
        sample, identity = super().sample(rng, policy, member=member, start=start,
                                           yaw=yaw, condition=condition)
        record = self.records[identity['member']]
        meta = self.bundle.metadata(record, self.split)
        first = identity['start']
        span = slice(first, first + 128)
        points = np.asarray(self.bundle.array(meta, 'exo_frame_points_world'))
        owners = np.asarray(self.bundle.array(meta, 'exo_frame_owner'))
        origin = np.asarray(identity['origin'], np.float32)
        anchor = np.asarray(identity['anchor'], np.float32)
        history = observed_prefix(points, owners, first, self.config['data']['history_voxel_m'])
        history = ((history - origin) @ anchor.T).astype(np.float32)
        count = self.config['local']['history_points']
        if len(history) > count:
            history = history[np.linspace(0, len(history)-1, count).astype(int)]
        hp = np.zeros((count, 3), np.float32); hv = np.zeros(count, bool)
        hp[:len(history)], hv[:len(history)] = history, True
        frame, frame_owner = bounded_frame_points(points[span], owners[span],
                                                   self.config['local']['frame_points'])
        frame = ((frame-origin) @ anchor.T).astype(np.float32)
        frame[frame_owner < 0] = 0
        from experiments.offline_camera_retrain_v1.unified_geometry import occupancy_from_points
        sample.update(local_history_points=hp, local_history_valid=hv,
                      local_frame_points=frame,
                      local_frame_owner=frame_owner,
                      occupancy=occupancy_from_points(history),
                      frame_scene_points=frame,
                      frame_scene_owner=frame_owner,
                      body_format=np.int64(3 if identity['native_format'] == 'grail_g1'
                                          else (1 if identity['native_format'] == 'smpl' else 2)),
                      sensor_profile=np.int64(0))
        sample['camera_intrinsics'] = sample['camera_intrinsics'] / self.config['data']['intrinsics_scale_pixels']
        identity.update(scene_history_cutoff=first, scene_current_points='exo_actual_firsthit',
                        local_surface_history_cutoff=first, local_future_history=False)
        return sample, identity


class SceneBoundaryPrior(JointMotionPrior):
    """Parent EXO boundary adapter plus a zero-initialized scene adapter."""
    def __init__(self, base, scene_policy, *, hidden=128, root_limit_m=.5,
                 rotation_limit_rad=1.2, residual_gain=5.):
        super().__init__(base, hidden=hidden, root_limit_m=root_limit_m,
                         rotation_limit_rad=rotation_limit_rad, residual_gain=residual_gain)
        modules = [base.core.unet.mid_block2] + [row[1] for row in base.core.unet.ups]
        widths = [module.time_mlp[1].out_features//2 for module in modules]
        self.scene_adapter = RelationAdapter(widths, scene_policy['hidden'])
        self.scene_policy = dict(scene_policy)

    def _scene_residuals(self, proposal, timestep, payload, shape, formats):
        with torch.no_grad():
            anchors, labels = self.base.skin(proposal, shape, formats)
            rotations = global_rotations(proposal)
            rotations = torch.stack([rotations[i, :, labels[i]] for i in range(len(proposal))])
            velocity = torch.zeros_like(anchors); velocity[:, 1:] = torch.diff(anchors, dim=1)*30
            cp = public_observations(payload)['camera'][..., :3]*2
            body = torch.cat(((anchors-cp[:, :, None])/2, (velocity/.2).clamp(-20,20),
                              rotations[..., :, 0], rotations[..., :, 1]), -1)
            geometry, gate, diagnostic = relation_features(anchors, rotations, payload,
                self.scene_policy['geometry'], None)
        signal = self.alphas[timestep].sqrt()
        residuals = self.scene_adapter(body, geometry, gate, labels, signal)
        frame_gate = gate.amax(-1)*self.alphas[timestep, None].pow(self.scene_policy['noise_gate_power'])
        return residuals, frame_gate, diagnostic

    def forward(self, noisy, timestep, payload, *, return_details=False):
        ego = payload['camera_context'][:, 0] > 0
        if ego.any():
            if not ego.all():
                raise ValueError('Scene EXO batches must be homogeneous')
            return self.base(noisy, timestep, payload, use_scene=True,
                             return_details=return_details)
        initial, rest = payload['initial_motion'], payload['initial_rest']
        reliability = payload['initial_reliability'].clamp(0, 1)
        root_reliability = payload['initial_root_reliability'].clamp(0, 1)
        anchor = self.anchor(payload)
        observations = public_observations(payload)
        conditioned, _, shape, formats = self.base.conditions(payload, True, None)
        shape = dict(shape); shape['rest'] = rest
        empty = (noisy.new_zeros(len(noisy), self.base.core.latent_dim), None, None, None)
        with torch.no_grad():
            reference = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
            proposal = reconcile_motion(reference, rest)
        signal = self.alphas[timestep].sqrt()[:, None, None].expand(-1, 128, 1)
        features = torch.cat((initial, noisy[..., :135], reliability,
            root_reliability[..., None], payload['estimate_available'][..., None],
            payload['joint_visible'], signal), -1)
        hidden = self.condition(features.transpose(1, 2))
        boundary = [layer(hidden) for layer in self.projections]
        scene, frame_gate, diagnostic = self._scene_residuals(proposal, timestep, payload, shape, formats)
        with self.inject(boundary), inject_residuals(self.base.core.unet, scene, frame_gate, late_only=True):
            adapted = self.base._core_pass(noisy, timestep, conditioned, empty, None, False, True)
        difference = (adapted[..., :135]-reference[..., :135])*self.residual_gain
        root = self.root_limit_m*torch.tanh(difference[..., :3]*2/self.root_limit_m)
        root = root/root.norm(dim=-1, keepdim=True).clamp_min(self.root_limit_m)*self.root_limit_m
        root = root*(1-root_reliability[..., None])
        from pytorch3d.transforms import matrix_to_axis_angle
        from scenemi.observations import body_condition
        from experiments.offline_camera_retrain_v1.supervision import rotation_from_6d
        delta = rotation_from_6d(difference[..., 3:135].reshape(len(noisy),128,22,6))
        rotation = self.rotation_limit_rad*torch.tanh(matrix_to_axis_angle(delta)/self.rotation_limit_rad)
        rotation = rotation/rotation.norm(dim=-1, keepdim=True).clamp_min(self.rotation_limit_rad)*self.rotation_limit_rad
        rotation = rotation*(1-reliability[..., None])
        moved = reconcile_motion(anchor, rest, translation_m=root, rotation_axis_angle=rotation)
        fixed = (root_reliability >= 1)&(reliability >= 1).all(-1)
        prediction = torch.where(fixed[..., None], initial, moved)
        details = dict(rest=rest, root_delta_m=root, rotation_delta_rad=rotation,
                       fixed_frames=fixed, reference_x0=reference, anchor_motion=anchor,
                       geometry_diagnostics=diagnostic,
                       reference_gradient='frozen parent with scene residual')
        return (prediction, details) if return_details else prediction


class SceneJointPrior(torch.nn.Module):
    def __init__(self, base, ego_policy, exo):
        super().__init__(); self.base = base; self.ego = GeometryConditionedPrior(base, ego_policy, 'late_only'); self.exo = exo

    @property
    def alphas(self): return self.base.alphas

    def train(self, mode=True):
        super().train(False); self.training = mode; self.base.eval()
        self.ego.adapter.train(mode); self.exo.condition.train(mode); self.exo.projections.train(mode)
        self.exo.scene_adapter.train(mode)
        return self

    def forward(self, noisy, timestep, payload, *, return_details=False):
        ego = payload['camera_context'][:, 0] > 0
        if ego.all(): return self.ego(noisy, timestep, local_payload(payload), return_details=return_details)
        if (~ego).all(): return self.exo(noisy, timestep, payload, return_details=return_details)
        raise ValueError('EGO and EXO must be accumulated as homogeneous microbatches')


def make_model(c, device):
    state = torch.load(ROOT/c['parent']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if state['step'] != c['parent']['step'] or sha256(ROOT/c['parent']['checkpoint']) != c['parent']['sha256']:
        raise ValueError('Pinned parent checkpoint changed')
    ms = state['model']; base_state = {k[5:]:v for k,v in ms.items() if k.startswith('base.')}
    top = {k:v for k,v in ms.items() if not k.startswith('base.')}
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    mount_file = Path(yaml.safe_load((ROOT/c['paths']['scene_paths']).read_text())['pv_mount'])
    mount = json.loads(mount_file.read_text())['camera_wearer']
    bases = {kind:{key:ms['base.native_body.bank.'+kind+'_'+key].clone()
                    for key in ('rest_template','rest_directions','eye_template','eye_directions')}
             for kind in ('smpl','smplx','grail_g1')}
    base = LocalScenePrior(paths['smplh_models']/'neutral/model.npz', mount, local=c['local'],
                           scene_state=base_state, bases=bases)
    base.load_state_dict(base_state, strict=False)
    exo = SceneBoundaryPrior(base, c['adapter'], **c['exo'])
    own = exo.state_dict()
    for key, value in top.items():
        if key not in own or own[key].shape != value.shape:
            raise ValueError('Parent EXO adapter does not match scene EXO initializer: '+key)
        own[key].copy_(value)
    exo.load_state_dict(own, strict=True)
    model = SceneJointPrior(base, c['adapter'], exo).to(device)
    model.train(False)
    return model


def make_data(c, model, split):
    common = model.base.shape.rest_template[:1].detach().cpu().numpy()
    from scenemi.data.checked_scene_dataset import CheckedLocalSceneDataset
    ego = CheckedLocalSceneDataset(ROOT/c['paths']['scene_paths'], common, split=split, config=c,
                                   selection=ROOT/c['paths']['selection'])
    exo_config = dict(c); exo_config['selection'] = c['paths']['selection']
    exo = SceneExoDataset(exo_config, split)
    exo.config = exo_config
    return ego, exo


def optimizer_for(model, c):
    parameters = list(model.ego.adapter.parameters()) + list(model.exo.condition.parameters())
    parameters += list(model.exo.projections.parameters()) + list(model.exo.scene_adapter.parameters())
    return torch.optim.AdamW(parameters, lr=c['optimizer']['lr_ego'], weight_decay=c['optimizer']['weight_decay'])


def update(model, optimizer, ego_data, exo_data, rng, c, device, step, loss_config):
    ego_pairs = [ego_data.sample(rng) for _ in range(c['training']['ego_batch_size'])]
    exo_pairs = [exo_data.sample(rng, c['frontend']) for _ in range(c['training']['exo_batch_size'])]
    ego = collate([x[0] for x in ego_pairs], device); exo = collate([x[0] for x in exo_pairs], device)
    model.train(); optimizer.zero_grad(set_to_none=True)
    t = torch.randint(0, 1000, (len(ego['motion']),), device=device); alpha = model.alphas[t]
    noisy = alpha[:,None,None].sqrt()*ego['motion'] + (1-alpha[:,None,None]).sqrt()*torch.randn_like(ego['motion'])
    prediction, details = model(noisy, t, local_payload(ego), return_details=True)
    ego_terms = native_motion_loss(prediction, ego['motion'], ego, alpha.sqrt(), details=details, config=loss_config)
    joints = forward_kinematics(prediction, details['rest']); baseline = forward_kinematics(details['baseline_motion'], details['rest']).detach()
    ego_terms['preserve_head'] = (joints[:,:,15]-baseline[:,:,15]).square().mean()
    ego_terms['preserve_root'] = (joints[:,:,0]-baseline[:,:,0]).square().mean()
    ego_total = ego_terms['total'] + c['adapter']['head_consistency_weight']*ego_terms['preserve_head'] + c['adapter']['root_consistency_weight']*ego_terms['preserve_root']
    t2 = torch.randint(0, 201, (len(exo['motion']),), device=device); a2 = model.alphas[t2]
    noise = torch.randn_like(exo['motion']); clean = a2[:,None,None].sqrt()*exo['motion'] + (1-a2[:,None,None]).sqrt()*noise
    observed = a2[:,None,None].sqrt()*model.exo.anchor(exo) + (1-a2[:,None,None]).sqrt()*noise
    noisy2 = torch.lerp(clean, observed, channel_reliability(exo))
    prediction2, details2 = model(noisy2, t2, exo, return_details=True)
    exo_terms = denoising_loss(prediction2, exo, details2)
    total = .5*ego_total + .5*exo_terms['total']; total.backward()
    params = [p for p in optimizer.param_groups[0]['params'] if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, c['optimizer']['clip_norm'], error_if_nonfinite=True)
    factor = min(step/c['optimizer']['warmup_updates'],1.)*(.1+.9*(1+math.cos(math.pi*step/c['training']['planned_updates']))/2)
    optimizer.param_groups[0]['lr'] = c['optimizer']['lr_ego']*factor
    optimizer.step()
    return dict(step=step, ego_loss=float(ego_total.detach()), exo_loss=float(exo_terms['total'].detach()),
                total=float(total.detach()), gradient_norm=float(norm), lr=optimizer.param_groups[0]['lr'])


def adapter_state(model):
    return dict(ego=model.ego.adapter_state(),
                exo_condition={k:v.detach().cpu().clone() for k,v in model.exo.condition.state_dict().items()},
                exo_projections={k:v.detach().cpu().clone() for k,v in model.exo.projections.state_dict().items()},
                exo_scene={k:v.detach().cpu().clone() for k,v in model.exo.scene_adapter.state_dict().items()})


def load_adapter_state(model, state):
    model.ego.load_adapter_state(state['ego'])
    model.exo.condition.load_state_dict(state['exo_condition'], strict=True)
    model.exo.projections.load_state_dict(state['exo_projections'], strict=True)
    model.exo.scene_adapter.load_state_dict(state['exo_scene'], strict=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action', choices=('preflight','run'))
    p.add_argument('--config', type=Path, default=ROOT/'configs/scene_joint128_conditioning.yaml')
    p.add_argument('--resume', action='store_true')
    a=p.parse_args(); ensure_bounded(); c=settings(a.config); out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    if a.action == 'preflight':
        model=make_model(c,'cuda'); ego,exo=make_data(c,model,'train')
        loss={'loss': yaml.safe_load((ROOT/c['paths']['loss']).read_text())['loss']}; opt=optimizer_for(model,c); t=time.monotonic()
        rec=update(model,opt,ego,exo,np.random.default_rng(c['seed']),c,'cuda',1,loss)
        atomic_json(out/'preflight_cuda.json',dict(state='passed',batch_size=128,elapsed_seconds=time.monotonic()-t,
            first_update=rec,trainable_parameters=sum(p.numel() for p in opt.param_groups[0]['params']),updated_at_utc=now()))
        return
    check=json.loads((out/'preflight_cuda.json').read_text())
    if check['state']!='passed': raise ValueError('CUDA preflight required')
    model=make_model(c,'cuda'); ego,exo=make_data(c,model,'train'); loss={'loss': yaml.safe_load((ROOT/c['paths']['loss']).read_text())['loss']}; opt=optimizer_for(model,c); rng=np.random.default_rng(c['seed'])
    folder=out/'training'; folder.mkdir(parents=True,exist_ok=True); last=folder/'last.pt'; step=0
    if last.exists():
        if not a.resume: raise ValueError('Existing scene run requires --resume')
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['parent_sha256'] != c['parent']['sha256']: raise ValueError('Resume parent changed')
        load_adapter_state(model,saved['adapter']); opt.load_state_dict(saved['optimizer']); step=int(saved['step'])
        rng.bit_generator.state=saved['rng']
    started=time.monotonic(); log=(folder/'updates.jsonl').open('a')
    try:
        for step in range(step+1,c['training']['planned_updates']+1):
            rec=update(model,opt,ego,exo,rng,c,'cuda',step,loss); log.write(json.dumps(rec)+'\n'); log.flush()
            if step%c['training']['log_every']==0 or step==c['training']['planned_updates']:
                atomic_json(folder/'STATUS.json',dict(state='training',completed_updates=step,planned_updates=c['training']['planned_updates'],batch_size=128,mode_batch={'ego':64,'exo':64},latest=rec,updates_per_second=step/max(time.monotonic()-started,1e-6),updated_at_utc=now()))
            if step%c['training']['checkpoint_every']==0 or step==c['training']['planned_updates']:
                temporary=last.with_suffix('.partial')
                torch.save(dict(step=step,parent_sha256=c['parent']['sha256'],adapter=adapter_state(model),
                                optimizer=opt.state_dict(),rng=rng.bit_generator.state),temporary)
                temporary.replace(last)
        atomic_json(folder/'STATUS.json',dict(state='training_complete',completed_updates=c['training']['planned_updates'],elapsed_seconds=time.monotonic()-started,updated_at_utc=now()))
    finally: log.close()


if __name__ == '__main__': main()
