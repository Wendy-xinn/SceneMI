"""Preflight and train the native-format, EGO-only three-corpus scene branch.

Use parallel_resources run to install cgroup/CPU limits and a CUDA allocator
cap. Training requires successful CPU and CUDA real-batch/recovery preflights.
Checkpoints are selected with fixed family-balanced validation, never RICH or
test data. AMASS EGO-only uses its separate full-validation selector.
"""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_config, load_paths
from scenemi.data.amass_dataset import AmassDataset, collate, inference_payload as amass_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.scene_dataset import SceneDataset, inference_payload
from scenemi.evaluation import summarize, verify_training_seal
from scenemi.geometry.native_shape import source_assets
from scenemi.models import build_model
from scenemi.parallel_resources import configure_cuda
from scenemi.scene_model import NativeScenePrior, native_motion_loss
from scenemi.training import deterministic, ensure_bounded, now, rng_state, restore_rng, tensor_digest
from scenemi.metrics import motion_metrics
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

CONFIG = ROOT/'configs/scene_ego128.yaml'
PATHS = ROOT/'configs/scene_paths.local.yaml'


def settings(path):
    config = yaml.safe_load(Path(path).read_text())
    if (config['version'] != 'native-scene-ego128-v2' or
            config['data']['datasets'] != ['egobody', 'trumans', 'grail'] or
            config['data']['excluded_datasets'] != ['rich'] or config['data']['modes'] != ['ego'] or
            config['data']['frames'] != 128 or config['data']['fps'] != 30 or
            config['data']['individual_shape_input'] is not False or config['validation']['test_used'] is not False):
        raise ValueError('Wrong native EGO scene protocol')
    base = load_config(ROOT/config['base_config'])
    config['loss'] = base['loss']
    return config


def make_model(config, paths, scene_paths):
    checkpoint = ROOT/config['prior_checkpoint']
    if sha256(checkpoint) != config['prior_sha256']:
        raise ValueError('The completed mixed AMASS55k warm start changed')
    seal = json.loads((checkpoint.parent/'CONTRACT.json').read_text())
    verify_training_seal(seal, load_config(ROOT/config['base_config']), paths, paths['output']/'index/complete.json')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if state['step'] != 55000 or state['contract'] != seal:
        raise ValueError('Expected the sealed completed 55k prior')
    mount_path = Path(yaml.safe_load(Path(scene_paths).read_text())['pv_mount'])
    mount = json.loads(mount_path.read_text())['camera_wearer']
    if mount['fitting_split'] != 'train':
        raise ValueError('Physical PV population calibration must be TRAIN-only')
    model = NativeScenePrior(paths['smplh_models']/'neutral/model.npz', mount, prior_state=state['model'])
    del state
    return model


def contract(config, scene_paths, paths, cases):
    files = {Path(__file__).resolve(), ROOT/'scenemi/parallel_resources.py'}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and name.endswith('.py'):
            path = Path(name).resolve()
            if (path.is_file() and (path.is_relative_to(ROOT) or
                    path.is_relative_to(ROOT.parent/'diffusion-motion-inbetweening'))
                    and not path.is_relative_to(ROOT/'tests')):
                files.add(path)
    dependencies = {key: sha256(Path(value)) for key, value in yaml.safe_load(Path(scene_paths).read_text()).items()
                    if key != 'verified_cache'}
    value = dict(protocol=config['version'], config=config,
        sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p) for p in sorted(files)},
        inputs=dependencies, assets=source_assets(paths), validation_cases=cases,
        prior_sha256=config['prior_sha256'], known_person_shape_input=False,
        environment={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'einops', 'smplx')})
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def optimizer_for(model, config):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])


def checkpoint_state(model, optimizer, rng, device, step, seal):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(rng, device),
                step=step, contract=seal, initialized_from_amass55k=True, ego_only=True)


def load_state(checkpoint, model, optimizer, rng, device, seal):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['contract'] != seal:
        raise ValueError('Scene checkpoint contract changed')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng, device)
    return state['step']


def update(model, optimizer, data, rng, config, device, step, *, forced_members=None):
    samples, identities = [], []
    for i in range(config['training']['batch_size']):
        sample, identity = data.sample(rng, member=forced_members[i%len(forced_members)] if forced_members else None)
        samples.append(sample)
        identities.append(identity)
    batch = collate(samples, device)
    target = batch['motion']
    t = torch.randint(0, 1000, (len(target),), device=device)
    alpha = model.alphas[t]
    noisy = alpha[:, None, None].sqrt()*target+(1-alpha[:, None, None]).sqrt()*torch.randn_like(target)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, details = model(noisy, t, inference_payload(batch), return_details=True)
    terms = native_motion_loss(prediction, target, batch, alpha.sqrt(), details=details, config=config)
    if not all(torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError('Nonfinite scene loss; no update applied')
    terms['total'].backward()
    gradient = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
        config['optimizer']['clip_norm'], error_if_nonfinite=True)
    shape_gradient = sum(float(p.grad.square().sum()) for p in model.native_body.coefficients.parameters() if p.grad is not None)**.5
    scene_gradient = float(model.patch_condition.weight.grad.norm())
    fraction = step/config['training']['planned_updates']
    lr = config['optimizer']['lr'] * min(1., step/config['optimizer']['warmup_updates']) * (.1+.9*(1+math.cos(math.pi*fraction))/2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return dict(step=step, losses={key:float(value.detach()) for key,value in terms.items()}, gradient_norm=float(gradient),
        native_shape_gradient=shape_gradient, scene_gradient=scene_gradient, lr=lr, identities=identities)


@torch.inference_mode()
def evaluate(model, data, cases, config, device):
    was_training = model.training
    saved_rng = rng_state(np.random.default_rng(0), device)
    model.eval()
    rows = []
    try:
        for case in cases:
            sample, identity = data.sample(np.random.default_rng(config['validation']['seed']),
                                          member=case['member'], start=case['start'], yaw=0.)
            if identity['synthetic_chair'] or identity['dataset']=='rich':
                raise AssertionError('Invalid validation inputs')
            batch = collate([sample], device)
            payload = inference_payload(batch)
            prediction = sample_native30(model, payload, steps=config['validation']['ddim_steps'],
                                          seed=config['validation']['seed'], use_scene=True)
            _, details = model(prediction, torch.zeros(1, device=device, dtype=torch.long), payload, return_details=True)
            joints = forward_kinematics(prediction, details['rest'])[0]
            target = forward_kinematics(batch['motion'], batch['target_rest'])[0]
            rows.append(dict(identity=identity, metrics=motion_metrics(joints, target)))
    finally:
        model.train(was_training)
        restore_rng(saved_rng, np.random.default_rng(0), device)
    groups = {name: summarize([r for r in rows if r['identity']['dataset']==name]) for name in ('egobody', 'trumans', 'grail')
              if any(r['identity']['dataset']==name for r in rows)}
    score = float(np.mean([r['family_macro_mean']['mpjpe_mm'] for r in groups.values()]))
    return dict(selection_score_mm=score, per_dataset=groups, rows=rows)


def forced_members(data):
    result = [next(i for i,r in enumerate(data.records) if r['dataset']=='egobody' and 'camera_wearer' in r['path']),
              next(i for i,r in enumerate(data.records) if r['dataset']=='egobody' and 'interactee' in r['path']),
              next(i for i,r in enumerate(data.records) if r['dataset']=='trumans'),
              next(i for i,r in enumerate(data.records) if r.get('synthetic_chair'))]
    return result


def preflight(model, train, validation, cases, config, device, out, seal, paths):
    folder = out/'preflight'
    folder.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint = folder/f'{device}.json', folder/f'.{device}.pt'
    started = time.monotonic()
    result = dict(state='running', device=device, pid=os.getpid(), started_at_utc=now(), contract_sha256=seal['sha256'])
    atomic_json(result_path, result)
    rng = np.random.default_rng(config['seed']+31)
    optimizer = optimizer_for(model, config)
    try:
        # Real AMASS control: exact warm-start behavior when scene is disabled.
        original = build_model(load_config(), paths['smplh_models']/'neutral/model.npz').to(device).eval()
        state = torch.load(ROOT/config['prior_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        original.load_state_dict(state['model'], strict=True)
        del state
        amass = AmassDataset(paths['output']/'index/complete.json', paths, 'val')
        sample, _ = amass.sample(np.random.default_rng(777), 'ego', member=0, start=0, reflected=False, yaw=0.)
        batch = collate([sample], device)
        payload = amass_payload(batch)
        noisy, times = torch.randn_like(batch['motion']), torch.tensor([500], device=device)
        model.eval()
        with torch.no_grad():
            expected = original(noisy, times, payload)
            actual = model(noisy, times, payload, use_scene=False)
        if not torch.equal(expected, actual):
            raise AssertionError('Native extension changes the scene-free AMASS prior at initialization')
        del original, batch, amass
        gc.collect()
        members = forced_members(train)
        for member in members:
            sample, _ = train.sample(rng, member=member, yaw=0.)
            batch = collate([sample], device)
            truth_fk = forward_kinematics(batch['motion'], batch['target_rest'])
            if not torch.allclose(truth_fk, batch['motion'][..., 135:].reshape(1, 128, 22, 3)*2, atol=3e-4, rtol=0):
                raise AssertionError('Common-reference conversion changes native target FK')
        payload = inference_payload(batch)
        # Forbidden fields may be present in a caller dictionary, never read.
        with torch.no_grad():
            noisy = torch.randn_like(batch['motion'])
            clean = model(noisy, times, payload)
            poison = dict(payload, target_rest=torch.full_like(batch['target_rest'], float('nan')),
                          target_native_betas=torch.full_like(batch['target_native_betas'], float('nan')),
                          target_template=torch.tensor([9999], device=device))
            if not torch.equal(clean, model(noisy, times, poison)):
                raise AssertionError('Target body information entered inference')
            free = model(noisy, times, payload, use_scene=False)
            if not torch.equal(free, clean):
                raise AssertionError('Scene gates do not preserve the warm start')
        first = update(model, optimizer, train, rng, config, device, 1, forced_members=members)
        if first['native_shape_gradient'] <= 0 or first['scene_gradient'] <= 0:
            raise AssertionError('New native shape or scene branch has no gradient')
        save_checkpoint(checkpoint_state(model, optimizer, rng, device, 1, seal), checkpoint)
        expected = update(model, optimizer, train, rng, config, device, 2, forced_members=members)
        digest = tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device)))
        load_state(checkpoint, model, optimizer, rng, device, seal)
        actual = update(model, optimizer, train, rng, config, device, 2, forced_members=members)
        if expected != actual or tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device))) != digest:
            raise AssertionError('Native scene exact-resume check failed')
        checks = evaluate(model, validation, [next(c for c in cases if c['dataset']==name) for name in ('egobody', 'trumans', 'grail')], config, device)
        result.update(state='passed', batch_size=config['training']['batch_size'], exact_resume=True,
            source_modes=['ego'], sources=['egobody', 'trumans', 'grail'], rich_excluded=True,
            prior_scene_off_bitwise_equal=True, zero_scene_gate_equal=True, target_poison_invariant=True,
            real_native_fk_equal=True, initial_update=first, resumed_update=actual, ddim_checks=checks,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            elapsed_seconds=time.monotonic()-started, completed_at_utc=now())
        atomic_json(result_path, result)
        checkpoint.unlink()
        return result
    except BaseException as error:
        result.update(state='failed', error=repr(error), updated_at_utc=now())
        atomic_json(result_path, result)
        raise


def train_model(model, data, validation, cases, config, device, out, seal, resume):
    folder = out/'training'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('cpu', 'cuda'):
        checked = json.loads((out/'preflight'/f'{name}.json').read_text())
        if checked['state'] != 'passed' or checked['contract_sha256'] != seal['sha256']:
            raise ValueError(f'Current native scene protocol needs {name} preflight')
    if not resume and (folder/'last.pt').exists():
        raise ValueError('Existing scene run requires --resume')
    atomic_json(folder/'CONTRACT.json', seal)
    optimizer = optimizer_for(model, config)
    rng = np.random.default_rng(config['seed']+41)
    step = load_state(folder/'last.pt', model, optimizer, rng, device, seal) if resume else 0
    stop = []
    def request_stop(signum, frame):
        stop.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    best_path = out/'selection/BEST.json'
    best_path.parent.mkdir(exist_ok=True)
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    started, start_step = time.monotonic(), step
    counts = Counter()
    def status(state, **extra):
        elapsed = time.monotonic()-started
        speed = (step-start_step)/elapsed if elapsed else 0.
        value = dict(state=state, completed_updates=step, planned_updates=config['training']['planned_updates'],
            updates_per_second=speed, pid=os.getpid(), updated_at_utc=now(), contract_sha256=seal['sha256'],
            datasets=dict(counts), rich_excluded=True, mode='ego', **extra)
        atomic_json(folder/'STATUS.json', value)
        return value
    def score_checkpoint():
        nonlocal best
        evaluation = evaluate(model, validation, cases, config, device)
        evaluation.update(step=step, completed_at_utc=now(), contract_sha256=seal['sha256'])
        atomic_json(out/'selection'/f'step_{step:06d}.json', evaluation)
        if best is None or evaluation['selection_score_mm'] < best['selection_score_mm']:
            temporary = best_path.parent/'.best.pt'
            temporary.unlink(missing_ok=True)
            os.link(folder/'last.pt', temporary)
            temporary.replace(best_path.parent/'best.pt')
            best = dict(step=step, selection_score_mm=evaluation['selection_score_mm'],
                checkpoint_sha256=sha256(best_path.parent/'best.pt'), contract_sha256=seal['sha256'],
                metrics=str(out/'selection'/f'step_{step:06d}.json'))
            atomic_json(best_path, best)
    try:
        if step == 0:
            save_checkpoint(checkpoint_state(model, optimizer, rng, device, step, seal), folder/'last.pt')
            status('initial_validation')
            score_checkpoint()
        status('training')
        with (folder/'updates.jsonl').open('a') as stream:
            while step < config['training']['planned_updates'] and not stop:
                result = update(model, optimizer, data, rng, config, device, step+1)
                step += 1
                counts.update(r['dataset'] for r in result['identities'])
                if step%20==0:
                    stream.write(json.dumps(result, allow_nan=False)+'\n')
                    stream.flush()
                    report = status('training', latest_loss=result['losses']['total'],
                                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
                    if step%100==0:
                        print(json.dumps(report), flush=True)
                if step%config['training']['checkpoint_every']==0 or step==config['training']['planned_updates']:
                    save_checkpoint(checkpoint_state(model, optimizer, rng, device, step, seal), folder/'last.pt')
                if step%config['training']['validation_every']==0 or step==config['training']['planned_updates']:
                    status('validating')
                    score_checkpoint()
        save_checkpoint(checkpoint_state(model, optimizer, rng, device, step, seal), folder/'last.pt')
        return status('complete' if step==config['training']['planned_updates'] else 'stopped', best=best)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train', 'diagnose'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--scene-paths', type=Path, default=PATHS)
    parser.add_argument('--paths', type=Path, default=ROOT/'configs/paths.local.yaml')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--diagnostic-checkpoint', type=Path)
    args = parser.parse_args()
    ensure_bounded()
    config, paths = settings(args.config), load_paths(args.paths)
    deterministic(config['seed'], args.device)
    if args.device=='cuda':
        configure_cuda()
    model = make_model(config, paths, args.scene_paths).to(args.device)
    pelvis = model.shape.rest_template[:1].detach().cpu().numpy()
    training = SceneDataset(args.scene_paths, pelvis, config=config)
    validation = SceneDataset(args.scene_paths, pelvis, split='validation', config=config)
    cases = validation.validation_cases(config['validation']['seed'], config['validation']['families_per_dataset'])
    if args.action=='diagnose':
        if args.diagnostic_checkpoint is None:
            parser.error('diagnose requires --diagnostic-checkpoint')
        print(json.dumps(diagnose_transfer(model, validation, cases, config, args.device, args.diagnostic_checkpoint), indent=2))
        return
    seal = contract(config, args.scene_paths, paths, cases)
    out = ROOT/config['output']
    if args.action=='preflight':
        result = preflight(model, training, validation, cases, config, args.device, out, seal, paths)
        print(json.dumps({key:result[key] for key in ('state', 'device', 'batch_size', 'exact_resume', 'elapsed_seconds', 'peak_cuda_bytes')}, indent=2))
    else:
        print(json.dumps(train_model(model, training, validation, cases, config, args.device, out, seal, args.resume), indent=2))


@torch.inference_mode()
def diagnose_transfer(model, data, cases, config, device, checkpoint):
    """Paired intrinsic-channel transfer ablation on fixed real observations."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval()
    original = model.camera_encoder[0].weight.detach().clone()
    body_original = model.body_encoder[0].weight.detach().clone()
    selected = [next(c for c in cases if c['dataset']==name) for name in ('egobody', 'trumans', 'grail')]
    rows = []
    for condition in ('raw_pixels_random_weights', 'scaled_pixels_random_weights', 'scaled_pixels_zero_weights',
                      'scaled_pixels_all_unused_zero_and_body_tag_copy'):
        model.camera_encoder[0].weight.copy_(original)
        model.body_encoder[0].weight.copy_(body_original)
        if condition in ('scaled_pixels_zero_weights','scaled_pixels_all_unused_zero_and_body_tag_copy'):
            model.camera_encoder[0].weight[:, 13:18].zero_()
        if condition=='scaled_pixels_all_unused_zero_and_body_tag_copy':
            model.camera_encoder[0].weight[:, 34:36].zero_()
            model.body_encoder[0].weight[:, 67].copy_(model.body_encoder[0].weight[:, 66])
        for case in selected:
            sample, identity = data.sample(np.random.default_rng(777), member=case['member'], start=case['start'], yaw=0.)
            batch = collate([sample], device)
            payload = inference_payload(batch)
            if condition=='raw_pixels_random_weights':
                payload['camera_intrinsics'] = payload['camera_intrinsics'] * config['data']['intrinsics_scale_pixels']
            pred = sample_native30(model, payload, steps=20, seed=777, use_scene=True)
            _, details = model(pred, torch.zeros(1,device=device,dtype=torch.long), payload, return_details=True)
            row = dict(condition=condition, identity=identity, metrics=motion_metrics(
                forward_kinematics(pred,details['rest'])[0],forward_kinematics(batch['motion'],batch['target_rest'])[0]))
            rows.append(row)
    result = dict(protocol='intrinsic-transfer-paired-diagnostic-v1', checkpoint_sha256=sha256(checkpoint),
        diagnostic_only=True, windows_per_condition=len(selected), rows=rows,
        means={condition:float(np.mean([r['metrics']['mpjpe_mm'] for r in rows if r['condition']==condition]))
               for condition in sorted({r['condition'] for r in rows})}, completed_at_utc=now())
    out = ROOT/config['output']/'preflight/intrinsic_transfer.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(out,result)
    return result


if __name__ == '__main__':
    main()
