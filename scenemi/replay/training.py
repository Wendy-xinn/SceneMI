"""Formal four-corpus EGO/EXO replay: prepare, bounded launch, train and select.

Run preparation on CPU, then launch after admission. Training produces atomic
resumable checkpoints first; complete validation subsequently selects among
the fixed retained candidates. No automatic hyperparameter search is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

from scenemi import native_joint_training as stage
from scenemi.config import ROOT
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.exo_repair_training import verify
from scenemi.geometry.temporal import NativeJointMotionPrior
from scenemi.joint_model import JointMotionPrior
from scenemi.models import AmassMotionPrior
from scenemi.native_joint_data import ego_payload
from scenemi.data.exo_initialization import repair_payload
from scenemi import parallel_resources
from scenemi.parallel_resources import configure_cuda
from scenemi.selection import pin_checkpoint
from scenemi.training import contract, deterministic, ensure_bounded, now, tensor_digest
from experiments.offline_camera_retrain_v1.ops import resources as legacy
from .benchmark import setup as data_setup, PROBE_CHECKPOINT, PROBE_SHA
from .optimization import update
from .runtime import budget
from .stream import SOURCE_WEIGHTS, add_noise, stream_requests
from .validation import prepare as validation_prepare, evaluate, select

CONFIG = ROOT/'configs/scene_joint128_replay.yaml'


class OnlineStream:
    """The screened stream, generated one batch ahead without a huge disk cache."""
    def __init__(self, sampler, seed, batch=128, workers=4):
        self.sampler, self.seed, self.batch = sampler, seed, batch
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = {}

    def submit(self, start):
        if start not in self.pending:
            requests = stream_requests(self.seed, start, self.batch)
            self.pending[start] = [(r, self.pool.submit(self.sampler.sample, r)) for r in requests]

    def get(self, start, device='cpu', next_start=None):
        self.submit(start)
        pairs = [add_noise(f.result(), r, self.seed) for r, f in self.pending.pop(start)]
        if next_start is not None:
            self.submit(next_start)
        values = [collate([p[0] for p in pairs[k::2]], 'cpu') for k in (0, 1)]
        if device == 'cuda':
            values = [{k: v.pin_memory().to(device, non_blocking=True) for k, v in x.items()} for x in values]
        return *values, [p[1] for p in pairs]

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def configuration(path):
    c = yaml.safe_load(path.read_text())
    if (c['version'] != 'native-joint-replay-v1' or c['data']['scene'] or
            c['data']['datasets'] != ['amass', 'egobody', 'trumans', 'grail'] or
            c['data']['excluded_datasets'] != ['rich'] or c['validation']['test_used']):
        raise ValueError('Expected four-corpus, scene-free training without RICH/test')
    t, v = c['training'], c['validation']
    if (t['batch_size'] != 128 or t['sampled_windows'] != t['updates']*128 or
            t['mode_windows_per_update'] != dict(ego=64, exo=64) or
            c['sampling_source_weights'] != list(SOURCE_WEIGHTS)):
        raise ValueError('The screened batch/sample/mode budget changed')
    points = v['sampled_windows']
    if (points != sorted(set(points)) or points[0] != 0 or points[-1] != t['sampled_windows'] or
            any(p % 128 for p in points) or v['stride'] != 128 or not v['include_end_aligned_tail']):
        raise ValueError('Invalid complete-validation milestones')
    if not c['resolution']['formal_training_allowed']:
        raise ValueError('This configuration has not been designated for formal training')
    return c


def initialize(c, out):
    ensure_bounded()
    parent_run = ROOT/c['parent']['run']
    status = json.loads((parent_run/'training/STATUS.json').read_text())
    best = json.loads((parent_run/'selection/BEST.json').read_text())
    if status['state'] != 'complete' or not best['full_validation'] or best['test_used']:
        raise ValueError('Require a completed, full-validation-selected AMASS parent')
    if best['step'] != c['parent']['step'] or best['checkpoint_sha256'] != c['parent']['sha256']:
        raise ValueError('Final AMASS parent selection changed')
    out.mkdir(parents=True, exist_ok=True)
    parent = out/'parent.pt'
    if not parent.exists():
        pin_checkpoint(Path(best['checkpoint']), parent)
    if sha256(parent) != best['checkpoint_sha256']:
        raise ValueError('Pinned final AMASS parent changed')
    state = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)
    verify(state['contract'])
    if state['step'] != best['step'] or state['contract']['config']['variant'] != 'scratch':
        raise ValueError('Wrong AMASS parent format/step')
    atomic_json(out/'PARENT.json', dict(best, final_parent=True, immutable_pin=str(parent)))
    deterministic(c['seed'], 'cpu')
    previous, discarded, sampler, _, reference, paths = data_setup()
    bank = previous.base.native_body.bank
    bases = {kind: {key: getattr(bank, kind+'_'+key).clone() for key in
             ('rest_template', 'rest_directions', 'eye_template', 'eye_directions')}
             for kind in ('smpl', 'smplx', 'grail_g1')}
    mount_path = Path(yaml.safe_load((ROOT/'configs/scene_paths.local.yaml').read_text())['pv_mount'])
    mount = json.loads(mount_path.read_text())['camera_wearer']
    if mount['fitting_split'] != 'train':
        raise ValueError('Camera calibration must be fit on train only')
    prior = {k[5:]: v for k, v in state['model'].items() if k.startswith('base.')}
    deterministic(c['seed'], 'cpu')
    base = stage.NativeMotionPrior(paths['smplh_models']/'neutral/model.npz', mount, prior_state=prior, bases=bases)
    stage.configure_trainable(base, c)
    model = NativeJointMotionPrior(base, **c['model'])
    adapter = {k: v for k, v in state['model'].items() if not k.startswith('base.')}
    missing, unexpected = model.load_state_dict(adapter, strict=False)
    if unexpected or any(not key.startswith('base.') for key in missing):
        raise ValueError('EXO boundary transfer differs')
    if any(not torch.equal(model.state_dict()[key], value) for key, value in adapter.items()):
        raise ValueError('EXO adapter not transferred exactly')
    sampler.seed, sampler.frontend = c['seed'], c['frontend']
    del previous, discarded, bank, bases
    gc.collect()
    data, cases = validation_prepare(model, paths, c, out)
    seal = contract(c, paths, ROOT/'runs/amass128/index/complete.json')
    assets = {str(p): sha256(p) for p in [out/'parent.pt', out/'VALIDATION_CASES.json',
        ROOT/c['selection'], ROOT/c['reference_config'], ROOT/'configs/paths.local.yaml',
        ROOT/'configs/scene_paths.local.yaml', mount_path,
        ROOT/'runs/data_inventory/CURRENT_SELECTION_CHECK.json', ROOT/PROBE_CHECKPOINT, CONFIG]}
    if assets[str(ROOT/PROBE_CHECKPOINT)] != PROBE_SHA:
        raise ValueError('Native basis cache changed')
    for dataset in sampler.native.values():
        bundle = getattr(dataset, 'selection_bundle', getattr(dataset, 'bundle', None))
        for p in bundle.sources.values():
            assets[str(p)] = sha256(p)
    seal.update(protocol='four-corpus-replay-formal-v1', assets=assets,
        trainable_parameter_names=[n for n, p in model.named_parameters() if p.requires_grad],
        scene_input=False, test_used=False, final_parent_sha256=best['checkpoint_sha256'],
        validation_execution='train_first_then_full_validation_all_declared_candidates',
        historical_short_screen_passed=False, historical_failures_not_overridden=True)
    seal.pop('sha256')
    seal['sha256'] = hashlib.sha256(json.dumps(seal, sort_keys=True).encode()).hexdigest()
    protocol = out/'PROTOCOL.json'
    if protocol.exists() and json.loads(protocol.read_text()) != seal:
        raise ValueError('Started replay source/config/data contract changed')
    atomic_json(protocol, seal)
    archive = out/'sources.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as target:
            for p in seal['sources']:
                target.add(ROOT/p, arcname=p.lstrip('/'), recursive=False)
        atomic_json(out/'SOURCE_ARCHIVE.json', dict(sha256=sha256(archive), sources=seal['sources']))
    return model, sampler, data, cases, reference, paths, state, seal


def preflight(model, sampler, reference, paths, parent, c, seal, out, device):
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stream = OnlineStream(sampler, c['seed'], 32 if device == 'cpu' else 128)
    rng = np.random.default_rng(c['seed'])
    optimizer = stage.optimizer_for(model, c)
    folder = out/'preflight'
    folder.mkdir(exist_ok=True)
    snapshot = folder/f'.resume_{device}.pt'
    frozen = lambda: tensor_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    frozen_before = frozen()
    try:
        ego, exo, _ = stream.get(0, device)
        original = JointMotionPrior(AmassMotionPrior(paths['smplh_models']/'neutral/model.npz'), **c['model'])
        original.load_state_dict(parent['model'], strict=True)
        original.to(device).eval()
        model.eval()
        transfer = {}
        with torch.no_grad():
            for mode, batch, payload_fn in [('ego', ego, ego_payload), ('exo', exo, repair_payload)]:
                mask = batch['dataset_id'] == 0
                batch = {k: v[mask] for k, v in batch.items()}
                noisy = batch['motion']
                times = torch.full((len(noisy),), 100, device=device, dtype=torch.long)
                payload = payload_fn(batch)
                a, b = model(noisy, times, payload), original(noisy, times, payload)
                if not torch.equal(a, b):
                    raise AssertionError(f'AMASS {mode} initializer output changed')
                transfer[mode] = dict(max_abs_difference=float((a-b).abs().max()), bitwise_equal=True)
            for batch, payload_fn in [(ego, ego_payload), (exo, repair_payload)]:
                payload = payload_fn(batch)
                times = torch.full((len(batch['motion']),), 100, device=device, dtype=torch.long)
                expected = model(batch['motion'], times, payload)
                poisoned = dict(payload, target_betas=torch.full_like(batch['target_betas'], float('nan')),
                    target_rest=torch.full_like(batch['target_rest'], float('nan')),
                    motion=torch.full_like(batch['motion'], float('nan')),
                    dataset_id=torch.full_like(batch['dataset_id'], 99))
                if not torch.equal(expected, model(batch['motion'], times, poisoned)):
                    raise AssertionError('Private target labels affect inference')
        del original
        first = update(model, optimizer, ego, exo, c, reference, 1, diagnostic=True)
        stage.save(model, optimizer, rng, device, 1, seal, snapshot)
        ego, exo, _ = stream.get(stream.batch, device)
        expected = update(model, optimizer, ego, exo, c, reference, 2)
        digest = tensor_digest((model.state_dict(), optimizer.state_dict()))
        stage.restore(model, optimizer, rng, device, seal, snapshot)
        actual = update(model, optimizer, ego, exo, c, reference, 2)
        if actual != expected or tensor_digest((model.state_dict(), optimizer.state_dict())) != digest:
            raise AssertionError('Formal replay optimizer resume is not exact')
        if frozen() != frozen_before:
            raise AssertionError('Frozen prior parameters changed')
        result = dict(state='passed', device=device, batch_size=stream.batch,
            contract_sha256=seal['sha256'], optimizer_resume_bitwise_exact=True,
            private_labels_ignored=True, original_amass_transfer=transfer,
            frozen_parameters_unchanged=True, full_validation_cases=28741,
            first_update=first, completed_at_utc=now(), formal_training_started=False)
        atomic_json(folder/f'{device}.json', result)
        return result
    finally:
        snapshot.unlink(missing_ok=True)
        stream.close()
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        del optimizer, initial
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def full_selection(model, data, cases, c, seal, out):
    baseline = None
    best, fallback = None, None
    for windows in c['validation']['sampled_windows']:
        step = windows//c['training']['batch_size']
        checkpoint = out/'parent.pt' if not windows else out/'candidates'/f'step_{step:06d}.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        expected = c['parent']['sha256'] if not windows else sha256(checkpoint)
        if not windows:
            if sha256(checkpoint) != expected or state['step'] != c['parent']['step']:
                raise ValueError('Validation parent identity differs')
        elif state['contract'] != seal or state['step'] != step:
            raise ValueError('Validation candidate contract/step differs')
        model.load_state_dict(state['model'], strict=True)
        del state
        atomic_json(out/'training/STATUS.json', dict(state='full_validation', formal_training_started=True,
            completed_updates=c['training']['updates'], planned_updates=c['training']['updates'],
            completed_windows=c['training']['sampled_windows'], evaluating_windows=windows,
            evaluating_step=step, updated_at_utc=now(), pid=os.getpid(), contract_sha256=seal['sha256']))
        metrics = evaluate(model, data, cases, c, out, step, expected, seal['sha256'])
        if baseline is None:
            baseline = metrics
        result = dict(metrics, **select(metrics, baseline, c))
        atomic_json(out/'selection'/f'step_{step:06d}'/'RESULT.json', result)
        if windows:
            record = dict(step=step, sampled_windows=windows, checkpoint=str(checkpoint),
                checkpoint_sha256=expected, selection_score=result['selection_score'],
                selection_eligible=result['selection_eligible'], full_validation=True, test_used=False,
                failed_checks=[k for k, v in result['checks'].items() if not v],
                contract_sha256=seal['sha256'])
            if fallback is None or record['selection_score'] < fallback['selection_score']:
                fallback = record
                atomic_json(out/'selection/BEST_CANDIDATE.json', record)
            if result['selection_eligible'] and (best is None or record['selection_score'] < best['selection_score']):
                best = record
                pin_checkpoint(checkpoint, out/'selection/best.pt')
                atomic_json(out/'selection/BEST.json', dict(record, checkpoint=str(out/'selection/best.pt')))
    return best, fallback


def train(model, sampler, data, cases, reference, c, seal, out):
    folder = out/'training'
    folder.mkdir(exist_ok=True)
    optimizer = stage.optimizer_for(model, c)
    rng = np.random.default_rng(c['seed'])
    last, log = folder/'last.pt', folder/'updates.jsonl'
    completed = 0
    if last.exists():
        saved = stage.restore(model, optimizer, rng, 'cuda', seal, last)
        completed = saved['step']
        with log.open('r+b') as handle:
            handle.truncate(saved['log_bytes'])
        del saved
    else:
        deterministic(c['seed'], 'cuda')
        torch.set_num_threads(1)
        log.touch()
        stage.save(model, optimizer, rng, 'cuda', 0, seal, last)
    stream = OnlineStream(sampler, c['seed'], c['training']['batch_size'])
    stop = {'requested': False}
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.update(requested=True))
    started = time.monotonic()
    try:
        with log.open('a', buffering=1) as handle:
            for step in range(completed+1, c['training']['updates']+1):
                tick = time.monotonic()
                first = (step-1)*128
                ego, exo, identities = stream.get(first, 'cuda', first+128 if step < c['training']['updates'] else None)
                record = update(model, optimizer, ego, exo, c, reference, step, diagnostic=step == 1 or step % 128 == 0)
                record.update(step=step, sampled_windows=step*128, identities=identities,
                    seconds=time.monotonic()-tick, updated_at_utc=now())
                handle.write(json.dumps(record, allow_nan=False)+'\n')
                retained = step*128 in c['validation']['sampled_windows']
                if step % c['training']['checkpoint_every'] == 0 or retained or stop['requested'] or step == c['training']['updates']:
                    handle.flush()
                    os.fsync(handle.fileno())
                    stage.save(model, optimizer, rng, 'cuda', step, seal, last, handle.tell())
                    if retained:
                        candidates = out/'candidates'
                        candidates.mkdir(exist_ok=True)
                        pin_checkpoint(last, candidates/f'step_{step:06d}.pt')
                if step % c['training']['log_every'] == 0 or retained or step == 1 or stop['requested']:
                    status = dict(state='stopped' if stop['requested'] else 'training',
                        formal_training_started=True, completed_updates=step, planned_updates=c['training']['updates'],
                        completed_windows=step*128, planned_windows=c['training']['sampled_windows'],
                        loss_ego=record['loss_ego'], loss_exo=record['loss_exo'],
                        last_update_seconds=record['seconds'], elapsed_seconds_this_invocation=time.monotonic()-started,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                        contract_sha256=seal['sha256'], pid=os.getpid(), updated_at_utc=now())
                    atomic_json(folder/'STATUS.json', status)
                    print(json.dumps(status), flush=True)
                if stop['requested']:
                    return
    finally:
        stream.close()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    best, fallback = full_selection(model, data, cases, c, seal, out)
    verify(seal)
    atomic_json(folder/'STATUS.json', dict(state='complete', formal_training_started=True,
        completed_updates=c['training']['updates'], planned_updates=c['training']['updates'],
        completed_windows=c['training']['sampled_windows'], best=best, best_candidate=fallback,
        all_retention_gates_satisfied=best is not None, last_checkpoint_sha256=sha256(last),
        contract_sha256=seal['sha256'], updated_at_utc=now(), test_used=False))


def launch(args):
    """Same admission/budget policy as replay.runtime; no admission bypass."""
    c = configuration(args.config)
    parallel_resources.MODULES.add('scenemi.replay.training')
    device = args.device
    cap = 0.  # Historical environment field; no allocator ceiling is installed.
    report = parallel_resources.admission(device,c['runtime']['memory_gib'],cap)
    report.update(allocator_gib=None,physical_free_reserve_gib=None,checked_at_utc=now())
    out = ROOT/c['output']
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/f'ADMISSION_{device}.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['admitted']:
        raise SystemExit('Resource admission refused; no job started')
    env = dict(os.environ, SCENEMI_ALLOCATOR_GIB=str(cap),
               SCENEMI_GPU_RESERVE_GIB=str(c['runtime']['gpu_physical_reserve_gib']))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    cores = set(c['runtime']['cpu_affinity'])
    if not cores <= os.sched_getaffinity(0) or len(cores) > 4:
        raise ValueError('Requested bounded CPU affinity unavailable')
    os.sched_setaffinity(0, cores)
    command = [sys.executable, '-m', 'scenemi.replay.training', args.task, '--config', str(args.config), '--device', device]
    return subprocess.call(legacy.scope_command(command, c['runtime']['memory_gib'], len(cores)), env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'prepare', 'train'))
    parser.add_argument('--task', choices=('prepare', 'train'), default='train')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.action == 'launch':
        raise SystemExit(launch(args))
    ensure_bounded()
    c = configuration(args.config)
    out = ROOT/c['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(out/'STATUS.json', dict(state='preparing', action=args.action, device=args.device,
            formal_training_started=(out/'training/last.pt').exists(), updated_at_utc=now(), pid=os.getpid()))
        try:
            if args.device == 'cuda':
                configure_cuda()
            model, sampler, data, cases, reference, paths, parent, seal = initialize(c, out)
            model.to(args.device)
            deterministic(c['seed'], args.device)
            torch.set_num_threads(1)
            preflight_file = out/'preflight'/f'{args.device}.json'
            if not preflight_file.exists() or json.loads(preflight_file.read_text())['contract_sha256'] != seal['sha256']:
                preflight(model, sampler, reference, paths, parent, c, seal, out, args.device)
            del parent
            gc.collect()
            if args.action == 'train':
                if args.device != 'cuda':
                    raise ValueError('Formal training requires CUDA acceptance')
                train(model, sampler, data, cases, reference, c, seal, out)
            else:
                atomic_json(out/'STATUS.json', dict(state='cpu_prepared' if args.device == 'cpu' else 'cuda_prepared',
                    formal_training_started=False, contract_sha256=seal['sha256'],
                    full_validation_cases=28741, updated_at_utc=now()))
        except BaseException as error:
            atomic_json(out/'STATUS.json', dict(state='failed', error=repr(error),
                formal_training_started=(out/'training/last.pt').exists(), updated_at_utc=now()))
            raise


if __name__ == '__main__':
    main()
