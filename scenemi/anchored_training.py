"""Manual EGO adaptation with differentiable scheme B and mixed scene inputs.

The retained geometry trainer stays sealed. Both adaptation settings below
share this implementation, prepared data, prediction readout and validation.
"""
import argparse
from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import SOURCE_ORDER, source_cases
from scenemi.data.scene_geometry_data import GeometryEgoData, collate, payload
from scenemi.anchored_scene_model import AnchoredScenePrior, mask_scene_inputs
from scenemi.checkpoint import checkpoint, load_checkpoint
from scenemi.contact_loss import reliable_native_motion_loss
from scenemi.execution import configure_runtime
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.metrics_batch import metric_rows
from scenemi.evaluate import aggregate
from scenemi.prefetch import OrderedPrefetch
from scenemi.monitoring import TrainingReporter
from scenemi.scene_geometry_training import (
    load_config as load_parent_config, skin_bases, contract as geometry_contract,
    frozen_digest, GEOMETRY_LOSSES,
)
from scenemi.training import deterministic, ensure_bounded, now, rng_state, tensor_digest
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

VERSION = 'ego-anchored-geometry-adaptation-v1'
BACKBONE_PREFIXES = ('core.unet.ups.1', 'core.unet.final_conv')


def load_config(path):
    path = Path(path).resolve(); spec = yaml.safe_load(path.read_text())
    if spec['version'] != VERSION:
        raise ValueError('Expected explicit anchored adaptation configuration')
    c = copy.deepcopy(load_parent_config(ROOT/spec['reference_config']))
    for key in ('version', 'output', 'seed', 'reference_config', 'initialization', 'anchor', 'adaptation', 'mix'):
        c[key] = spec[key]
    c['_config_path'] = str(path)
    for key in ('training', 'optimizer', 'validation'):
        c[key].update(spec.get(key, {}))
    c['scene_stage']['backbone'] = 'partial_last_decoder' if c['adaptation']['backbone_prefixes'] else 'frozen'
    if c['adaptation']['backbone_prefixes'] not in ([], list(BACKBONE_PREFIXES)):
        raise ValueError('Only the final decoder level and output head may be unfrozen')
    if (c['training']['effective_batch_size']!=128 or c['data']['source_quota_per_update']!=32
            or 128%c['training']['microbatch_size'] or c['training']['precision']!='fp32'
            or c['mix']['native_scene_fraction']!=.5 or c['validation']['test_used']):
        raise ValueError('Keep batch 128, fixed source quotas, 50% native scene inputs and validation-only evaluation')
    if c['initialization']['step']!=24000 or c['training']['planned_updates']<=24000:
        raise ValueError('This stage starts from the retained 24k checkpoint')
    if (c['anchor']['data_loss'] not in ('l2', 'huber') or c['anchor']['confidence_source']!='uniform_available_frames'
            or not 0<c['optimizer']['backbone_lr']<=c['optimizer']['lr']
            or not (ROOT/c['output']).resolve().is_relative_to(ROOT/'runs')):
        raise ValueError('Unexpected anchoring, learning-rate or output policy')
    if not all(c['adaptation'][key] for key in ('query_after_anchor', 'final_readout_after_sampling', 'geometry_loss_after_anchor')):
        raise ValueError('Training, query placement and final sampling must share scheme B')
    for key in ('checkpoint', 'validation_manifest', 'reference_result'):
        if sha256(ROOT/c['initialization'][key])!=c['initialization'][key+'_sha256']:
            raise ValueError(f'Pinned initialization {key} changed')
    return c


def build_model(c, paths):
    parent_config = load_parent_config(ROOT/c['reference_config'])
    initial = torch.load(ROOT/c['initialization']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    for file, digest in initial['contract']['source_sha256'].items():
        if sha256(ROOT/file)!=digest: raise ValueError(f'Retained trained dependency changed: {file}')
    parent = torch.load(ROOT/parent_config['continuation']['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    weights = parent['model']
    bases = {kind: {key: weights[f'native_body.bank.{kind}_{key}'] for key in
        ('rest_template', 'rest_directions', 'eye_template', 'eye_directions')} for kind in ('smpl', 'smplx', 'grail_g1')}
    mount = dict(rotation=weights['pv_rotation'].tolist(), translation_m=weights['pv_translation'].tolist(),
        position_p90_m=float(weights['pv_uncertainty'][0]), rotation_p90_rad=float(weights['pv_uncertainty'][1]))
    model = AnchoredScenePrior(paths['smplh_models']/'neutral/model.npz', mount, weights=weights, bases=bases,
        geometry=c['geometry'], scene_stage=c['scene_stage'], skin_bases=skin_bases(parent_config),
        anchor_policy=c['anchor'], adaptation=c['adaptation'], latent_dim=c['model']['latent_dim'],
        dim_mults=tuple(c['model']['dim_mults']))
    model.load_state_dict(initial['model'], strict=True)
    return model


def contract(c, paths):
    value = geometry_contract(c, paths)
    for file in (ROOT/c['reference_config'], ROOT/c['initialization']['reference_result'],
                 ROOT/'scenemi/anchored_training.py', ROOT/'scenemi/anchored_scene_model.py',
                 ROOT/'scenemi/geometry/temporal_head_anchor.py'):
        value['source_sha256'][str(file.relative_to(ROOT))] = sha256(file)
    value.update(protocol=VERSION, initialization=c['initialization'], optimizer_migration='fresh two-group optimizer; new explicit adaptation stage',
        additional_updates=c['training']['planned_updates']-c['initialization']['step'],
        additional_windows=(c['training']['planned_updates']-c['initialization']['step'])*128,
        readout='raw clean prediction -> differentiable scheme B -> motion and geometry losses',
        scene_queries='scheme-B-placed clean proposal; static positions, motion query and explicit point-to-body features',
        scene_mix='32 AMASS without scene + 48 native without scene + 48 native with scene per update',
        geometry_supervision='after B; only native samples assigned scene input',
        camera_confidence='uniform on available frames, no GT-derived confidence',
        temporal_scope='128-frame offline window; no full-video stitching or noisy-camera robustness claim')
    value.pop('sha256', None)
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def optimizer_for(model, c):
    scene, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (backbone if any(name.startswith(prefix+'.') for prefix in BACKBONE_PREFIXES) else scene).append(p)
    groups = [dict(params=scene, lr=c['optimizer']['lr'], base_lr=c['optimizer']['lr'], group_name='scene')]
    if backbone:
        groups.append(dict(params=backbone, lr=c['optimizer']['backbone_lr'],
            base_lr=c['optimizer']['backbone_lr'], group_name='backbone'))
    return torch.optim.AdamW(groups, weight_decay=c['optimizer']['weight_decay'])


def scene_keep_mask(pairs, fraction, device):
    keep = torch.zeros(len(pairs), device=device, dtype=torch.bool)
    for source in SOURCE_ORDER:
        if source=='amass': continue
        indices = [i for i, p in enumerate(pairs) if p[1]['dataset']==source]
        selected = torch.randperm(len(indices), device=device)[:round(len(indices)*fraction)]
        if len(indices): keep[torch.tensor(indices, device=device)[selected]] = True
    return keep


def update(model, optimizer, pairs, c, device, step):
    started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
    batch = collate([p[0] for p in pairs], device, pin=c['prepared_data']['pin_memory'])
    keep = scene_keep_mask(pairs, c['mix']['native_scene_fraction'], device)
    sums = defaultdict(float); n = len(pairs); micro = c['training']['microbatch_size']
    loss_config = yaml.safe_load((ROOT/c['reference_loss_config']).read_text())
    local = step-c['initialization']['step']; budget = c['training']['planned_updates']-c['initialization']['step']
    factor = min(1., local/c['optimizer']['warmup_updates'])
    factor *= c['optimizer']['lr_floor_fraction']+(1-c['optimizer']['lr_floor_fraction'])*.5*(1+math.cos(math.pi*local/budget))
    for group in optimizer.param_groups: group['lr'] = group['base_lr']*factor
    ramp = min(1., local/max(1, c['scene_stage']['geometry_warmup_updates']))
    for first in range(0, n, micro):
        part = {k: v[first:first+micro] for k, v in batch.items()}; selected = keep[first:first+micro]
        public = mask_scene_inputs(payload(part), selected)
        times = torch.randint(0, 1000, (len(selected),), device=device); alpha = model.alphas[times]
        noisy = alpha[:, None, None].sqrt()*part['motion']+(1-alpha[:, None, None]).sqrt()*torch.randn_like(part['motion'])
        raw, details = model(noisy, times, public, use_scene=bool(selected.any()), return_details=True)
        final = model.final_motion(raw, details)
        terms = reliable_native_motion_loss(final, part['motion'], part, alpha.sqrt(), details=details,
            config=loss_config, contact_weight=part['contact_weight'])
        # raw_x0/raw_consistency remain the retained explicit auxiliary terms;
        # their raw_motion stays unanchored, preserving supervision of raw root.
        extra, _ = geometry_terms(model, final, details, part, alpha.sqrt()*selected, c['geometry'])
        terms.update(extra)
        terms['total'] = terms['total']+ramp*sum(c['geometry_coefficients'][k]*v for k, v in extra.items())
        if not all(torch.isfinite(v).all() for v in terms.values()): raise FloatingPointError('Nonfinite anchored loss')
        if terms['total'].requires_grad: (terms['total']*(len(selected)/n)).backward()
        for key, value in terms.items(): sums[key] += float(value.detach())*len(selected)/n
    gradients = {g['group_name']: float(torch.stack([p.grad.detach().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt())
        for g in optimizer.param_groups if any(p.grad is not None for p in g['params'])}
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c['optimizer']['clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(step=step, total=sums['total'], terms=dict(sums), gradient_norm=float(norm),
        lr=optimizer.param_groups[0]['lr'], learning_rates={g['group_name']:g['lr'] for g in optimizer.param_groups},
        gradient_groups=gradients, native_with_scene=int(keep.sum()), without_scene=n-int(keep.sum()),
        geometry_ramp=ramp, compute_update_seconds=time.perf_counter()-started, identities=[p[1] for p in pairs])


@torch.inference_mode()
def evaluate(model, data, cases, c, device, out=None, *, steps=None):
    model.eval(); rows = {mode: [] for mode in ('scene', 'no_scene')}; size = c['validation']['eval_batch_size']
    if out is not None:
        identity = dict(model_sha256=tensor_digest(model.state_dict()), cases_sha256=hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()).hexdigest(), seed=c['validation']['seed'],
            ddim_steps=steps or c['validation']['ddim_steps'], anchor=c['anchor'], eval_batch_size=size)
        out.mkdir(parents=True, exist_ok=True)
        identity_file = out/'IDENTITY.json'
        if identity_file.exists() and json.loads(identity_file.read_text())!=identity:
            raise ValueError(f'Existing validation belongs to different weights or cases: {out}')
        atomic_json(identity_file, identity)
        if (out/'RESULT.json').exists():
            return json.loads((out/'RESULT.json').read_text())
    from tqdm.auto import tqdm
    for first in tqdm(range(0, len(cases), size), desc='Validation after B', unit='batch', leave=False):
        chunk = cases[first:first+size]; pairs = [data.sample_case(x) for x in chunk]
        batch = collate([p[0] for p in pairs], device); public = payload(batch)
        truth = forward_kinematics(batch['motion'], batch['target_rest'])
        for mode in rows:
            raw = sample_native30(model, public, steps=steps or c['validation']['ddim_steps'],
                seed=c['validation']['seed'], use_scene=mode=='scene')
            _, details = model(raw, torch.zeros(len(chunk), device=device, dtype=torch.long), public, use_scene=False, return_details=True)
            final = model.final_motion(raw, details)
            _, geometry = geometry_terms(model, final, details, batch, torch.ones(len(chunk), device=device), c['geometry'])
            metrics = metric_rows(forward_kinematics(final, details['rest']), truth, extra=geometry)
            for i, case in enumerate(chunk): rows[mode].append(dict(case=case, identity=pairs[i][1], metrics=metrics[i]))
    result = {mode: aggregate(values) for mode, values in rows.items()}
    result.update(state='complete', test_used=False, readout='scheme B after final diffusion sample')
    if out is not None:
        for mode, values in rows.items(): (out/(mode+'.jsonl')).write_text(''.join(json.dumps(x, allow_nan=False)+'\n' for x in values))
        atomic_json(out/'RESULT.json', result)
    return result


def preflight(c, paths, device):
    deterministic(c['seed'], device); model = build_model(c, paths).to(device)
    data = GeometryEgoData(c, paths, 'train'); validation = GeometryEgoData(c, paths, 'validation')
    pc = copy.deepcopy(c)
    if device=='cpu':
        pc['data']['source_quota_per_update'] = 2
        pc['training'].update(effective_batch_size=8, microbatch_size=4)
        data.c = pc
    rng = np.random.default_rng(c['seed']+41); pairs = data.pairs(rng, forced=True)
    probes = [next(p for p in pairs if p[1]['dataset']==s) for s in SOURCE_ORDER]
    batch = collate([p[0] for p in probes], device); public = payload(batch)
    noise = torch.zeros_like(batch['motion']); times = torch.full((4,), 100, device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        before = model(noise, times, public, use_scene=False)
        empty = mask_scene_inputs(public, torch.zeros(4, device=device, dtype=torch.bool))
        if not torch.equal(before, model(noise, times, empty, use_scene=True)): raise AssertionError('Empty scene fallback differs')
        clean = model(noise, times, public, use_scene=True)
        poisoned = dict(batch, motion=torch.full_like(batch['motion'], float('nan')), target_rest=torch.full_like(batch['target_rest'], float('nan')),
            contact_weight=torch.full_like(batch['contact_weight'], float('nan')))
        if not torch.equal(clean, model(noise, times, payload(poisoned), use_scene=True)): raise AssertionError('Private labels entered inference')
        _, details = model(noise, times, public, use_scene=False, return_details=True)
    probe = (batch['motion']+torch.randn_like(batch['motion'])*.01).detach().requires_grad_()
    final = model.final_motion(probe, details)
    physical, _ = geometry_terms(model, final, details, batch, torch.ones(4, device=device), c['geometry'])
    gradient = torch.autograd.grad(sum(physical.values()), probe)[0]
    if not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient[..., 3:135]): raise AssertionError('Post-B geometry has no finite gradient to pose')
    frozen = frozen_digest(model); optimizer = optimizer_for(model, c); step = c['initialization']['step']+1
    first = update(model, optimizer, pairs, pc, device, step)
    if first['native_with_scene']!=len(pairs)*3//8: raise AssertionError('Scene mix changed')
    if not first['gradient_groups'].get('scene', 0)>0: raise AssertionError('Scene parameters have no gradient')
    if c['adaptation']['backbone_prefixes'] and not first['gradient_groups'].get('backbone', 0)>0: raise AssertionError('Selected backbone has no gradient')
    folder = ROOT/c['output']/'preflight'/device; folder.mkdir(parents=True, exist_ok=True)
    seal = contract(c, paths); temporary = folder/'discarded.pt'
    try:
        checkpoint(model, optimizer, rng, device, step, seal, temporary)
        expected = update(model, optimizer, data.pairs(rng, forced=True), pc, device, step+1)
        digest = tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device)))
        load_checkpoint(model, optimizer, rng, device, seal, temporary)
        resumed = update(model, optimizer, data.pairs(rng, forced=True), pc, device, step+1)
        if digest!=tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device))) or expected['terms']!=resumed['terms'] or expected['identities']!=resumed['identities']:
            raise AssertionError('Next update is not an exact resume')
    finally: temporary.unlink(missing_ok=True)
    if frozen_digest(model)!=frozen: raise AssertionError('Frozen parameters changed')
    model.eval()
    with torch.no_grad():
        after = model(noise, times, public, use_scene=False)
        if not c['adaptation']['backbone_prefixes'] and not torch.equal(before, after): raise AssertionError('Frozen control changed its camera-only prior')
        if not torch.equal(after, model(noise, times, empty, use_scene=True)): raise AssertionError('Empty scene fallback changed after an update')
    cases = source_cases(validation); smoke = [next(x for x in cases if x['dataset']==s) for s in SOURCE_ORDER]
    result = evaluate(model, validation, smoke, c, device, steps=2)
    record = dict(state='passed', device=device, contract=seal, exact_resume=True, frozen_parameters_unchanged=True,
        geometry_gradient_reaches_pose_through_B=True, labels_isolated=True, empty_scene_fallback_exact=True,
        effective_batch_size=pc['training']['effective_batch_size'], first_update=first, validation_smoke=result,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_names=[name for name,p in model.named_parameters() if p.requires_grad],
        positional_encoding_audit=dict(static_patch_centres=True, static_vit_position=True,
            motion_query_contains_pose_and_joint_positions=True, query_proposal_placed_with_B=True,
            explicit_relative_surface_xyz=True, signed_plane_gap=True, unknown_surface_mask=True),
        disposable_updates_only=True, formal_training_started=False, completed_at_utc=now())
    atomic_json(folder/'RESULT.json', record)
    return record


def require_ready(c, paths, device):
    seal = contract(c, paths)
    for kind in ('cpu', device):
        file = ROOT/c['output']/'preflight'/kind/'RESULT.json'
        if not file.exists(): raise ValueError(f'Missing {kind} preflight')
        value = json.loads(file.read_text())
        if value.get('state')!='passed' or value.get('contract')!=seal or not value.get('exact_resume'):
            raise ValueError(f'Stale or failed {kind} acceptance')
    return seal


def train(c, paths, device, resume=False):
    if device!='cuda': raise ValueError('Formal training requires CUDA')
    deterministic(c['seed'], device); model = build_model(c, paths).to(device)
    data = GeometryEgoData(c, paths, 'train'); validation = GeometryEgoData(c, paths, 'validation')
    seal = require_ready(c, paths, device); optimizer = optimizer_for(model, c); rng = np.random.default_rng(c['seed']+41)
    manifest = json.loads((ROOT/c['initialization']['validation_manifest']).read_text())
    full = source_cases(validation); quick = source_cases(validation, quick=True, families_per_dataset=c['validation']['quick_families_per_dataset'])
    if full!=manifest['full'] or quick!=manifest['quick']: raise ValueError('Validation membership changed')
    out = ROOT/c['output']; folder = out/'training'; folder.mkdir(parents=True, exist_ok=True); last = folder/'last.pt'
    if last.exists()!=resume: raise ValueError('Existing run requires --resume; fresh run must not have last.pt')
    step = load_checkpoint(model, optimizer, rng, device, seal, last) if resume else c['initialization']['step']
    if not resume:
        atomic_json(out/'CONTRACT.json', seal); atomic_json(out/'VALIDATION_MANIFEST.json', manifest)
        checkpoint(model, optimizer, rng, device, step, seal, last)
    log_file = folder/'updates.jsonl'
    if resume and log_file.exists(): log_file.write_text(''.join(line+'\n' for line in log_file.read_text().splitlines() if json.loads(line)['step']<=step))
    selection = out/'selection'; selection.mkdir(exist_ok=True)
    best_file = selection/'BEST.json'; best = json.loads(best_file.read_text()) if resume and best_file.exists() else None
    if best and best['step']>step: best = None
    reference = json.loads((ROOT/c['initialization']['reference_result']).read_text())
    method = c['initialization']['reference_method']; base = reference['arms']['base22k'][method]
    reporter = TrainingReporter(out, c, step); reporter.progress.set_description('EGO · scheme B · mixed scenes')
    stopped = False
    def stop(*_):
        nonlocal stopped; stopped = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    def validate(current):
        nonlocal best
        result = evaluate(model, validation, full, c, device, selection/f'full_step_{current:06d}')
        for mode in ('scene', 'no_scene'): reporter.validation(result[mode], current, 'validation_'+mode)
        eligible = all(result['no_scene']['by_dataset'][d]['mpjpe_mm']<=v['mpjpe_mm']+c['validation']['clean_dataset_regression_mm'] for d,v in base['by_dataset'].items())
        eligible &= result['no_scene']['selection_score_mm']<=base['selection_score_mm']+c['validation']['clean_macro_regression_mm']
        eligible &= all(result['no_scene']['by_dataset'][d]['jerk_m_s3']<=v['jerk_m_s3']*c['validation']['jerk_ratio_max'] for d,v in base['by_dataset'].items())
        score = result['scene']['selection_score_mm']
        if eligible and (best is None or score<best['selection_score_mm']-c['validation']['tie_tolerance_mm']):
            checkpoint(model, optimizer, rng, device, current, seal, selection/'best.pt')
            best = dict(step=current, selection_score_mm=score, checkpoint=str(selection/'best.pt'), no_scene_guards_passed=True)
            atomic_json(best_file, best); reporter.best(best)
    boundary = True
    with log_file.open('a', buffering=1) as log:
        try:
            if best is None: validate(step)
            with OrderedPrefetch(lambda r: data.pairs(r), rng, c['prepared_data']['prefetch_batches']) as queue:
                while step<c['training']['planned_updates'] and not stopped:
                    boundary = False; pairs, timing = queue.next()
                    row = update(model, optimizer, pairs, c, device, step+1); step += 1; boundary = True
                    seconds = row['compute_update_seconds']+timing['data_wait_seconds']; row.update(timing=timing, step_seconds=seconds, updated_at_utc=now())
                    log.write(json.dumps(row, allow_nan=False)+'\n'); reporter.update(row, seconds)
                    if reporter.writer is not None:
                        for group in ('learning_rates', 'gradient_groups', 'timing'):
                            for key,value in row[group].items(): reporter.writer.add_scalar(group+'/'+key, value, step)
                        reporter.writer.add_scalar('mix/native_with_scene', row['native_with_scene'], step)
                    if step%c['training']['checkpoint_every']==0 or step==c['training']['planned_updates']:
                        log.flush(); os.fsync(log.fileno()); checkpoint(model, optimizer, rng, device, step, seal, last)
                    if step%c['training']['quick_validation_every']==0:
                        result = evaluate(model, validation, quick, c, device, selection/f'quick_step_{step:06d}')
                        for mode in ('scene', 'no_scene'): reporter.validation(result[mode], step, 'quick_'+mode)
                    if step in c['training']['full_validation_steps'] or step==c['training']['planned_updates']: validate(step)
                    atomic_json(folder/'STATUS.json', dict(state='training', completed_updates=step, planned_updates=c['training']['planned_updates'], latest=row, best=best))
        except BaseException as error:
            atomic_json(folder/'STATUS.json', dict(state='failed', completed_updates=step, failed_update_may_be_partial=not boundary,
                error=f'{type(error).__name__}: {error}', resume='last atomic last.pt only'))
            raise
        finally:
            log.flush(); os.fsync(log.fileno()); reporter.close()
            if boundary: checkpoint(model, optimizer, rng, device, step, seal, last)
            for sig, handler in previous.items(): signal.signal(sig, handler)
    atomic_json(folder/'STATUS.json', dict(state='complete' if step==c['training']['planned_updates'] else 'stopped_at_safe_boundary',
        completed_updates=step, planned_updates=c['training']['planned_updates'], best=best, updated_at_utc=now()))
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args(argv)
    if args.resume and args.action!='train': parser.error('--resume applies only to manual training')
    c = load_config(args.config); ensure_bounded(); configure_runtime(c, args.device)
    os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT/c['prepared_data']['root']/'triton'))
    paths = load_paths(ROOT/'configs/paths.local.yaml'); out = ROOT/c['output']; out.mkdir(parents=True, exist_ok=True)
    with (out/'.anchored_stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='preflight':
            file = out/'preflight'/args.device/'RESULT.json'; file.parent.mkdir(parents=True, exist_ok=True)
            try: result = preflight(c, paths, args.device)
            except BaseException as error:
                atomic_json(file, dict(state='failed', error=f'{type(error).__name__}: {error}', formal_training_started=False, completed_at_utc=now())); raise
            print(json.dumps({k:v for k,v in result.items() if k not in ('contract', 'first_update', 'trainable_names', 'validation_smoke')}, indent=2))
        else:
            print(json.dumps(train(c, paths, args.device, args.resume), indent=2))


if __name__=='__main__':
    main()
