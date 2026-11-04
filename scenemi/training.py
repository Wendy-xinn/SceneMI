"""AMASS training, real-data preflight and fail-closed readiness records."""
import gc
import copy
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from scenemi.config import ROOT
from scenemi.data.amass_dataset import AmassDataset, build_index, collate, inference_payload
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.models import build_model, motion_loss
from scenemi.losses import motion_loss as _loaded_loss_dependency
from experiments.offline_camera_retrain_v1.checkpoint_io import save_checkpoint
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from scenemi.resources import admission, configure_cuda
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_bounded():
    relative = Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2]
    group = Path('/sys/fs/cgroup') / relative.lstrip('/')
    maximum = (group/'memory.max').read_text().strip()
    if maximum == 'max' or int(maximum) > 16*2**30 or len(os.sched_getaffinity(0)) > 4:
        raise RuntimeError('Use the resources run wrapper with <=16GiB memory and <=4 CPU cores')


def deterministic(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def contract(config, paths, index):
    """Actual imported first-party closure plus explicit dynamic entry points."""
    files = {ROOT/'scenemi/__main__.py', ROOT/'scenemi/training.py', ROOT/'scenemi/losses.py'}
    allowed_roots = (ROOT, ROOT.parent/'diffusion-motion-inbetweening')
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and name.endswith('.py') and Path(name).is_absolute():
            path = Path(name).resolve()
            if any(path.is_relative_to(base) for base in allowed_roots):
                files.add(path)
    # Test harnesses do not become runtime dependencies of the model.
    files = {p for p in files if not p.is_relative_to(ROOT/'tests')}
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p)
               for p in sorted(files)}
    value = dict(protocol='amass128-observation-shape-training-v1', config=config,
                 index_sha256=sha256(index), body_model_sha256=sha256(paths['smplh_models']/'neutral/model.npz'),
                 sources=sources, environment={name:importlib.metadata.version(name)
                 for name in ('torch', 'numpy', 'scipy', 'einops')})
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def tensor_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def rng_state(rng, device):
    return dict(data=rng.bit_generator.state, torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None,
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state, rng, device):
    rng.bit_generator.state = state['data']
    torch.set_rng_state(state['torch'].cpu())
    if device == 'cuda':
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def optimizer_for(model, config):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])


def batch_for(data, rng, config, device, *, forced_modes=None, batch_size=None):
    size = config['training']['batch_size'] if batch_size is None else batch_size
    samples, identities = [], []
    for index in range(size):
        mode = forced_modes[index % len(forced_modes)] if forced_modes else (
            'ego' if rng.random() < config['observation']['mode_probabilities']['ego'] else 'exo')
        sample, identity = data.sample(rng, mode,
            temporal_dropout=config['observation']['temporal_dropout'])
        samples.append(sample)
        identities.append(identity)
    return collate(samples, device), identities


def update(model, optimizer, data, rng, config, device, step, *, forced_modes=None, batch_size=None):
    batch, identities = batch_for(data, rng, config, device, forced_modes=forced_modes, batch_size=batch_size)
    target = batch['motion']
    timesteps = torch.randint(0, 1000, (len(target),), device=device)
    alpha = model.alphas[timesteps]
    noise = torch.randn_like(target)
    xt = alpha[:, None, None].sqrt()*target+(1-alpha[:, None, None]).sqrt()*noise
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction, details = model(xt, timesteps, inference_payload(batch), return_details=True)
    terms = motion_loss(prediction, target, batch, alpha.sqrt(), details=details, config=config)
    if not all(torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError('Nonfinite AMASS objective; no optimizer update')
    terms['total'].backward()
    gradient = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                            config['optimizer']['clip_norm'], error_if_nonfinite=True)
    progress = step/config['training']['planned_updates']
    schedule = min(1., step/config['optimizer']['warmup_updates'])*max(
        config['optimizer']['lr_floor_fraction'], .5*(1+math.cos(math.pi*progress)))
    for group in optimizer.param_groups:
        group['lr'] = config['optimizer']['lr']*schedule
    shape_gradient = sum(float(p.grad.detach().square().sum()) for p in model.shape_encoder.parameters() if p.grad is not None)**.5
    optimizer.step()
    record = dict(step=step, identities=identities, terms={k:float(v.detach()) for k,v in terms.items()},
                  grad_norm=float(gradient), shape_grad_norm=shape_gradient,
                  lr=optimizer.param_groups[0]['lr'], timesteps=timesteps.tolist())
    return record


def checkpoint_state(model, optimizer, rng, device, step, seal, **extra):
    return dict(protocol='amass128-observation-shape-checkpoint-v1', model=model.state_dict(),
                optimizer=optimizer.state_dict(), rng=rng_state(rng, device), device=device,
                step=step, contract=seal, **extra)


def load_state(path, model, optimizer, rng, device, seal):
    state = torch.load(path, map_location=device, weights_only=False)
    if state['contract']['sha256'] != seal['sha256'] or state['device'] != device:
        raise ValueError('Checkpoint code/config/data/device identity differs; refuse silent resume')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng, device)
    return state


@torch.no_grad()
def input_isolation_check(model, batch):
    was_training = model.training
    model.eval()
    payload = inference_payload(batch)
    noisy = torch.zeros_like(batch['motion'])
    time_ids = torch.full((len(noisy),), 400, device=noisy.device, dtype=torch.long)
    first, details = model(noisy, time_ids, payload, return_details=True)
    poisoned = dict(payload, motion=torch.full_like(noisy, float('nan')),
                    target_betas=torch.full_like(batch['target_betas'], float('nan')),
                    target_rest=torch.full_like(batch['target_rest'], float('nan')),
                    rest=torch.full_like(batch['target_rest'], float('nan')),
                    mount_translation_m=torch.full((len(noisy), 3), float('nan'), device=noisy.device),
                    occupancy=torch.full((len(noisy), 24, 48, 48), float('nan'), device=noisy.device))
    # In EGO even an apparently confident body track must be ignored.
    ego = payload['camera_context'][:, 0] > 0
    poisoned['trajectory'] = payload['trajectory'].clone()
    poisoned['observation_meta'] = payload['observation_meta'].clone()
    poisoned['trajectory'][ego] = 100.
    poisoned['observation_meta'][ego] = 1.
    second, extra = model(noisy, time_ids, poisoned, return_details=True)
    if not torch.equal(first, second) or not torch.equal(details['predicted_betas'], extra['predicted_betas']):
        raise AssertionError('Body truth or forbidden EGO tracks changed inference')
    if not torch.isfinite(first).all():
        raise AssertionError('Nonfinite body output')
    # Missing camera/joint payloads must never influence shape or denoising.
    masked = {key: value.clone() for key, value in payload.items()}
    masked['camera'][..., :3][payload['camera_available'][..., 0] == 0] = float('nan')
    masked['camera'][..., 3:][payload['camera_available'][..., 1] == 0] = float('nan')
    hidden = payload['observation_meta'][..., 0] == 0
    masked['trajectory'][hidden] = float('nan')
    third, masked_details = model(noisy, time_ids, masked, return_details=True)
    if not torch.equal(first, third) or not torch.equal(details['predicted_betas'], masked_details['predicted_betas']):
        raise AssertionError('Unavailable observation payload changed inference')
    model.train(was_training)
    return dict(label_poison_invariant=True, ego_joint_track_poison_invariant=True,
                masked_observation_poison_invariant=True, observation_only_shape=True, finite_output=True)


@torch.no_grad()
def evaluate(model, data, config, device):
    was_training = model.training
    model.eval()
    try:
        return _evaluate_cases(model, data, config, device)
    finally:
        model.train(was_training)


def _evaluate_cases(model, data, config, device):
    rng = np.random.default_rng(config['validation']['fixed_seeds'][0])
    rows = []
    # Fixed development cases, one eligible source per validation corpus.
    seen, selected = set(), []
    for index, row in enumerate(data.members):
        if row['corpus'] not in seen:
            selected.append(index)
            seen.add(row['corpus'])
    for index in selected:
        for mode in config['validation']['mode']:
            case_seed = int(rng.integers(2**32))
            for condition in config['validation']['conditions']:
                policy = None
                if condition == 'temporal_gap':
                    if config['observation']['temporal_dropout']['probabilities'][mode] == 0:
                        continue
                    policy = copy.deepcopy(config['observation']['temporal_dropout'])
                    policy['probabilities'] = dict(ego=1., exo=1.)
                sample, identity = data.sample(np.random.default_rng(case_seed), mode,
                    member=index, start=0, reflected=False, yaw=0., temporal_dropout=policy)
                batch = collate([sample], device)
                predicted = sample_native30(model, inference_payload(batch), steps=config['validation']['ddim_steps'],
                                            seed=config['validation']['fixed_seeds'][0], use_scene=False)
                # Only observations are used to recover the same static shape.
                _, details = model(predicted, torch.zeros(1, device=device, dtype=torch.long),
                                   inference_payload(batch), return_details=True)
                joints = forward_kinematics(predicted, details['rest'])
                truth = forward_kinematics(batch['motion'], batch['target_rest'])
                error = (joints-truth).norm(dim=-1)
                mask = details['conditioned_observations']['observation_meta'][..., 0] > 0
                frame_valid = ((batch['camera_available'] > 0).any(-1)
                               | (batch['observation_meta'][..., 0] > 0).any(-1))
                visible = float(error[mask].mean()) if mask.any() else None
                hidden = float(error[~mask].mean()) if (~mask).any() else None
                if not torch.isfinite(error).all():
                    raise FloatingPointError('Nonfinite DDIM validation sample')
                rows.append(dict(identity=identity, condition=condition, mpjpe_m=float(error.mean()),
                    visible_mpjpe_m=visible, hidden_mpjpe_m=hidden,
                    missing_frames=int((~frame_valid).sum()),
                    missing_frames_mpjpe_m=float(error[~frame_valid].mean()) if (~frame_valid).any() else None,
                    observed_frames_mpjpe_m=float(error[frame_valid].mean()) if frame_valid.any() else None,
                    root_error_m=float(error[:, :, 0].mean()),
                    shape_beta_mse=float((details['predicted_betas']-batch['target_betas']).square().mean())))
    return rows


def preflight(paths, config, *, index=None, device='cpu', diagnostic=False):
    ensure_bounded()
    folder = paths['output']/'preflight'/('diagnostic_' if diagnostic else 'complete_')
    folder = folder.with_name(folder.name+device)
    folder.mkdir(parents=True, exist_ok=True)
    audit = dict(state='running', device=device, diagnostic_only=diagnostic, started_at_utc=now(),
                 formal_training_started=False)
    audit_path = folder/'RESULT.json'
    if audit_path.exists():
        with (folder/'HISTORY.jsonl').open('a') as history:
            history.write(json.dumps(json.loads(audit_path.read_text()), ensure_ascii=False)+'\n')
    atomic_json(audit_path, audit)
    try:
        if device == 'cuda':
            report = admission('cuda', 12, config=config)
            if not report['admitted']:
                raise RuntimeError('GPU admission refused; no CUDA preflight started')
            audit['gpu_admission'] = report
            audit['gpu_limits'] = configure_cuda(config)
        index = Path(index) if index else build_index(paths, diagnostic=diagnostic)
        data = AmassDataset(index, paths, 'train', diagnostic=diagnostic)
        validation = AmassDataset(index, paths, 'val', diagnostic=diagnostic)
        deterministic(config['seed']+17, device)
        model = build_model(config, paths['smplh_models']/'neutral/model.npz').to(device)
        optimizer = optimizer_for(model, config)
        rng = np.random.default_rng(config['seed']+991)
        seal = contract(config, paths, index)
        size = config['training']['batch_size']
        started = time.monotonic()
        first = update(model, optimizer, data, rng, config, device, 1, forced_modes=('ego', 'exo'), batch_size=size)
        if first['shape_grad_norm'] <= 0:
            raise AssertionError('Shape predictor has no real-data gradient')
        path = folder/'last.pt'
        save_checkpoint(checkpoint_state(model, optimizer, rng, device, 1, seal), path)
        checkpoint_hash, checkpoint_bytes = sha256(path), path.stat().st_size
        expected = update(model, optimizer, data, rng, config, device, 2, forced_modes=('ego', 'exo'), batch_size=size)
        digest = tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device)))
        del model, optimizer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        model = build_model(config, paths['smplh_models']/'neutral/model.npz').to(device)
        optimizer = optimizer_for(model, config)
        load_state(path, model, optimizer, rng, device, seal)
        resumed = update(model, optimizer, data, rng, config, device, 2, forced_modes=('ego', 'exo'), batch_size=size)
        actual_digest = tensor_digest((model.state_dict(), optimizer.state_dict(), rng_state(rng, device)))
        if expected != resumed or digest != actual_digest:
            raise AssertionError('Resumed next update differs from uninterrupted training')
        batch, _ = batch_for(data, rng, config, device, forced_modes=('ego', 'exo'), batch_size=2)
        isolation = input_isolation_check(model, batch)
        forced_gap = copy.deepcopy(config['observation']['temporal_dropout'])
        forced_gap['probabilities'] = {mode: float(p > 0)
            for mode, p in forced_gap['probabilities'].items()}
        gap_samples, gap_identities = [], []
        for mode in ('ego', 'exo'):
            sample, identity = data.sample(rng, mode, temporal_dropout=forced_gap)
            gap_samples.append(sample)
            gap_identities.append(identity)
        gap_batch = collate(gap_samples, device)
        gap_isolation = input_isolation_check(model, gap_batch)
        # Decode a complete DDIM path for one case per mode. Full monitoring
        # uses all four validation corpora during formal training.
        selected = validation.members
        validation.members = selected[:1]
        metrics = evaluate(model, validation, config, device)
        validation.members = selected
        if contract(config, paths, index)['sha256'] != seal['sha256']:
            raise AssertionError('Runtime imported an unsealed first-party dependency')
        audit.update(state='passed', finished_at_utc=now(), contract=seal, batch_size=size,
                     updates_executed=3, steps_for_resume_comparison=[1, 2], resume_bitwise_exact=True,
                     initial_update=first, resumed_update=resumed, isolation=isolation,
                     forced_gap_isolation=gap_isolation, forced_gap_identities=gap_identities,
                     ddim_checks=metrics, elapsed_seconds=time.monotonic()-started,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
                     checkpoint_sha256=checkpoint_hash, checkpoint_bytes=checkpoint_bytes,
                     checkpoint_retention='removed_after_success; diagnostic weights cannot warm-start formal training')
        atomic_json(audit_path, audit)
        path.unlink()
        return audit
    except Exception as error:
        audit.update(state='failed', error=f'{type(error).__name__}: {error}', updated_at_utc=now())
        atomic_json(audit_path, audit)
        raise


def require_preflight(paths, config, index, device):
    path = paths['output']/'preflight'/f'complete_{device}'/'RESULT.json'
    if not path.is_file():
        raise ValueError(f'Missing full-data {device} preflight: {path}')
    result = json.loads(path.read_text())
    expected = contract(config, paths, index)
    if (result.get('state') != 'passed' or result.get('diagnostic_only') or
            result['contract']['sha256'] != expected['sha256'] or
            not result.get('resume_bitwise_exact')):
        raise ValueError('Preflight is incomplete or stale for the current code/config/data')
    if device == 'cuda' and result['batch_size'] != config['training']['batch_size']:
        raise ValueError('GPU preflight did not exercise the formal batch size')
    return expected


def train(paths, config, *, index=None, device='cuda', resume=False):
    ensure_bounded()
    if device != 'cuda':
        raise ValueError('Use CPU preflight for short checks; formal training requires CUDA')
    report = admission('cuda', 12, config=config)
    if not report['admitted']:
        raise RuntimeError('GPU admission refused; no formal training started')
    index = Path(index) if index else paths['output']/'index/complete.json'
    data = AmassDataset(index, paths, 'train')
    validation = AmassDataset(index, paths, 'val')
    seal = require_preflight(paths, config, index, 'cpu')
    require_preflight(paths, config, index, 'cuda')
    gpu_limits = configure_cuda(config)
    folder = paths['output']/'training'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'last.pt'
    if checkpoint.exists() != resume:
        raise ValueError('Existing checkpoint requires --resume; --resume requires a checkpoint')
    if shutil.disk_usage(folder).free < 20*2**30:
        raise RuntimeError('Less than 20 GiB free before training')
    deterministic(config['seed'], device)
    model = build_model(config, paths['smplh_models']/'neutral/model.npz').to(device)
    optimizer = optimizer_for(model, config)
    rng = np.random.default_rng(config['seed']+991)
    start, counts = 0, Counter()
    log_path = folder/'updates.jsonl'
    if resume:
        restored = load_state(checkpoint, model, optimizer, rng, device, seal)
        start = restored['step']
        counts.update(restored['counts'])
        if not log_path.is_file() or log_path.stat().st_size < restored['log_bytes']:
            raise ValueError('Training log is missing committed checkpoint history')
        with log_path.open('r+b') as stream:
            stream.truncate(restored['log_bytes'])
        validation_log = folder/'validation.jsonl'
        if validation_log.is_file():
            committed = [line for line in validation_log.read_text().splitlines(keepends=True)
                         if json.loads(line)['step'] <= start]
            temporary = validation_log.with_suffix('.tmp')
            temporary.write_text(''.join(committed))
            os.replace(temporary, validation_log)
    elif log_path.exists() and log_path.stat().st_size:
        raise ValueError('Refuse to overwrite training history without its checkpoint')
    atomic_json(folder/'CONTRACT.json', seal)
    initialization = dict(type='from_scratch' if not resume else 'exact_resume',
                updated_at_utc=now(), seed=config['seed'], checkpoint=str(checkpoint) if resume else None,
                optimizer_entries=len(optimizer.state), step=start,
                data_sources=['AMASS/train'], scene_enabled=False, ground_truth_shape_input=False)
    if resume:
        with (folder/'RESUMES.jsonl').open('a') as stream:
            stream.write(json.dumps(initialization)+'\n')
    else:
        atomic_json(folder/'INITIALIZATION.json', initialization)
    status = dict(state='training', pid=os.getpid(), started_at_utc=now(), device=device,
                  completed_updates=start, planned_updates=config['training']['planned_updates'],
                  contract_sha256=seal['sha256'], checkpoint=str(checkpoint),
                  gpu_limits=gpu_limits, gpu_admission=report,
                  source_data_complete=True, ground_truth_shape_input=False)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous_term = signal.signal(signal.SIGTERM, stop)
    previous_int = signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    step = start
    try:
        with log_path.open('a', buffering=1) as log:
            def save(step):
                log.flush()
                os.fsync(log.fileno())
                save_checkpoint(checkpoint_state(model, optimizer, rng, device, step, seal,
                                                 counts=dict(counts), log_bytes=log.tell()), checkpoint)
            if not resume:
                save(0)
            atomic_json(folder/'STATUS.json', status)
            for step in range(start+1, config['training']['planned_updates']+1):
                record = update(model, optimizer, data, rng, config, device, step)
                for identity in record['identities']:
                    counts[identity['mode']] += 1
                record['updated_at_utc'] = now()
                log.write(json.dumps(record, ensure_ascii=False)+'\n')
                if step % config['validation']['every_updates'] == 0:
                    metrics = evaluate(model, validation, config, device)
                    with (folder/'validation.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(step=step, seed=config['validation']['fixed_seeds'][0],
                                                     cases=metrics, updated_at_utc=now()), ensure_ascii=False)+'\n')
                final = step == config['training']['planned_updates']
                if stopping or final or step % config['training']['checkpoint_every'] == 0:
                    save(step)
                if stopping or final or step % config['training']['log_every'] == 0:
                    elapsed = time.monotonic()-started
                    speed = (step-start)/elapsed
                    status.update(state='stopped' if stopping else 'complete' if final else 'training',
                                  completed_updates=step, mode_counts=dict(counts),
                                  updated_at_utc=now(), updates_per_second=speed,
                                  eta_hours=(config['training']['planned_updates']-step)/max(speed, 1e-9)/3600,
                                  latest_loss=record['terms']['total'],
                                  peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    atomic_json(folder/'STATUS.json', status)
                    print(json.dumps({k:status[k] for k in ('state', 'completed_updates', 'latest_loss', 'eta_hours')}), flush=True)
                if stopping:
                    break
        return status
    except Exception as error:
        status.update(state='failed', attempted_update=step, updated_at_utc=now(),
                      error=f'{type(error).__name__}: {error}',
                      resume_note='Resume only the last atomic checkpoint; uncommitted log suffix is truncated.')
        atomic_json(folder/'STATUS.json', status)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _preparation_process_alive(pid):
    try:
        command = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        return all(value in command for value in (b'scenemi', b'amass', b'prepare'))
    except FileNotFoundError:
        return False
    except PermissionError:
        return None


def preparation_progress(paths):
    status = json.loads((paths['output']/'preparation/STATUS.json').read_text())
    committed_at = datetime.fromisoformat(status['updated_at_utc'])
    seconds_since_commit = max(0., (datetime.now(timezone.utc)-committed_at).total_seconds())
    process_alive = None
    verifying_resume = False
    if status['state'].startswith('running'):
        process_alive = _preparation_process_alive(status['pid'])
        if process_alive is False:
            # The sealed prepare entry verifies committed bytes before writing
            # its first status. Distinguish that live startup from a dead job.
            launches = sorted((paths['output']/'preparation').glob('RESUME_LAUNCH_*.json'))
            if launches:
                launch = json.loads(launches[-1].read_text())
                resume_pid = launch.get('preparation_pid')
                if resume_pid and _preparation_process_alive(resume_pid):
                    process_alive, verifying_resume = True, True
    summary = json.loads((paths['output']/'raw_inventory.summary.json').read_text())
    excluded = sum(summary['upstream_filename_exclusions'].values())
    remaining = status['total_raw_files']-status['committed_sources']-(excluded-status['counts'].get('filename_excluded', 0))
    # Recent actual processing records exclude the cheap filename skips.
    log = paths['output']/'preparation/prepare.log'
    records = []
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-256*1024))
            for line in stream.read().splitlines():
                try:
                    row = json.loads(line)
                    if 'elapsed' in row and 'source' in row:
                        if records and row['elapsed'] < records[-1]['elapsed']:
                            records = []  # A resumed invocation resets its clock.
                        records.append(row)
                except (ValueError, UnicodeDecodeError):
                    pass
    records = records[-256:]
    eta = None
    if (process_alive is not False and status.get('invocation_completed_sources', 1) > 0 and
            seconds_since_commit <= 15*60 and
            len(records) > 1 and records[-1]['elapsed'] > records[0]['elapsed']):
        seconds = (records[-1]['elapsed']-records[0]['elapsed'])/(len(records)-1)
        eta = max(0, remaining)*seconds
    if verifying_resume:
        eta = None
    state = 'verifying_resume_outputs' if verifying_resume else 'process_missing' if process_alive is False else status['state']
    return dict(state=state, recorded_preparation_state=status['state'], process_alive=process_alive,
                seconds_since_last_commit=seconds_since_commit,
                committed_sources=status['committed_sources'],
                total_raw_files=status['total_raw_files'], failures=status['failures'],
                updated_at_utc=status['updated_at_utc'], estimated_remaining_seconds=eta,
                eta_method='Recent 256 prepared source pairs, excluding cheap filename skips; sequence length varies')


def readiness(paths, config, *, wait=False):
    ensure_bounded()
    report_path = paths['output']/'READINESS.json'
    while True:
        data_status = preparation_progress(paths)
        result = dict(state='waiting_for_full_data', updated_at_utc=now(), data=data_status,
                      formal_training_started=False, automatic_gpu_launch=False)
        if data_status['state'] == 'process_missing':
            result.update(state='data_preparation_process_missing',
                          note='No ETA until preparation resumes; this CPU watcher will keep waiting.')
        if data_status['state'] == 'verifying_resume_outputs':
            result.update(state='verifying_resume_outputs',
                          note='Live resumed preprocessor is verifying committed output hashes before its first status update.')
        if data_status['state'] == 'failed' or data_status['state'].startswith('stopped'):
            result.update(state='data_preparation_'+data_status['state'])
            atomic_json(report_path, result)
            return result
        if data_status['state'] == 'complete':
            break
        atomic_json(report_path, result)
        if not wait:
            return result
        time.sleep(30)
    result.update(state='verifying_full_data')
    atomic_json(report_path, result)
    try:
        index = build_index(paths)
        result.update(state='running_full_data_cpu_preflight', index=str(index))
        atomic_json(report_path, result)
        checked = preflight(paths, config, index=index, device='cpu')
        result.update(state='cpu_verified_gpu_preflight_pending', updated_at_utc=now(),
                      cpu_preflight_contract=checked['contract']['sha256'],
                      resume_bitwise_exact=checked['resume_bitwise_exact'],
                      next='Run an admitted CUDA preflight at the formal batch size, then start fresh training.')
        atomic_json(report_path, result)
        return result
    except Exception as error:
        result.update(state='failed', updated_at_utc=now(), error=f'{type(error).__name__}: {error}')
        atomic_json(report_path, result)
        raise
