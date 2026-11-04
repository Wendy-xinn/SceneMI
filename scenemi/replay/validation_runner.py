"""Resume complete replay selection with larger, noise-equivalent batches.

Training sources/configuration remain sealed. This evaluation-only scheduler
keeps the original cases, per-case inputs, logical DDIM noise and metrics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time

import torch
import yaml

from scenemi import parallel_resources
from scenemi.config import ROOT, load_paths
from scenemi.data.amass_dataset import collate
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.exo_initialization import CONDITIONS, repair_payload
from scenemi.exo_denoise import sample_denoising
from scenemi.exo_repair_training import verify
from scenemi.geometry.temporal import NativeJointMotionPrior
from scenemi.metrics_batch import metric_rows, repair_rows
from scenemi.native_joint_data import ego_payload
from scenemi.selection import pin_checkpoint
from scenemi.training import deterministic, ensure_bounded, now, tensor_digest
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
from . import training, validation
from .data import CORPORA
from .screen import pairs_for
from .sharing import authorization, authorized_admission
from .validation_sampling import reference_noise, sample_ego, sample_exo


def load_state(model, path, *, baseline=False, initializer=None):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if baseline:
        if initializer is None:
            raise ValueError('Baseline requires its complete deterministic native initializer')
        missing = set(initializer)-set(state['model'])
        expected = {k for k in initializer if k.startswith(('base.pv_', 'base.native_body.'))}
        if missing != expected or set(state['model'])-set(initializer):
            raise ValueError('Baseline differs beyond the known native adapter keys')
        model.load_state_dict(initializer, strict=True)
    else:
        model.load_state_dict(state['model'], strict=True)
    return state


def admission(config, out):
    from scenemi.gpu_policy import admission as current
    c=training.configuration(config)
    report=current('cuda',c['runtime']['memory_gib'])
    atomic_json(out/'ADMISSION_validation.json',report)
    if not report['admitted']:raise RuntimeError('Host RAM admission refused')
    parallel_resources.configure_cuda()


def setup(c, out):
    seal = json.loads((out/'PROTOCOL.json').read_text())
    if seal['config'] != c:
        raise ValueError('Formal training configuration changed')
    verify(seal)
    parent = out/'parent.pt'
    if sha256(parent) != c['parent']['sha256']:
        raise ValueError('AMASS parent changed')
    final = torch.load(out/'training/last.pt', map_location='cpu', weights_only=False, mmap=True)
    if final['step'] != c['training']['updates'] or final['contract'] != seal:
        raise ValueError('Require the completed, sealed training checkpoint')
    final_state = final['model']
    bases = {kind: {key: final_state['base.native_body.bank.'+kind+'_'+key].clone()
                   for key in ('rest_template', 'rest_directions', 'eye_template', 'eye_directions')}
             for kind in ('smpl', 'smplx', 'grail_g1')}
    paths = load_paths(ROOT/'configs/paths.local.yaml')
    mount_path = Path(yaml.safe_load((ROOT/'configs/scene_paths.local.yaml').read_text())['pv_mount'])
    mount = json.loads(mount_path.read_text())['camera_wearer']
    if mount['fitting_split'] != 'train':
        raise ValueError('Camera calibration was not fitted on training data')
    old = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)
    prior = {k[5:]: v for k, v in old['model'].items() if k.startswith('base.')}
    deterministic(c['seed'], 'cpu')
    base = training.stage.NativeMotionPrior(paths['smplh_models']/'neutral/model.npz',
                                           mount, prior_state=prior, bases=bases)
    training.stage.configure_trainable(base, c)
    model = NativeJointMotionPrior(base, **c['model'])
    adapter = {k: v for k, v in old['model'].items() if not k.startswith('base.')}
    missing, unexpected = model.load_state_dict(adapter, strict=False)
    if unexpected or any(not k.startswith('base.') for k in missing):
        raise ValueError('EXO boundary transfer differs')
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for key, value in initial.items():
        if key.startswith(('base.native_body.bank.', 'base.pv_')) and not torch.equal(value, final_state[key]):
            raise ValueError('Frozen native initializer differs: '+key)
    for windows in c['validation']['candidate_windows']:
        step = windows//c['training']['batch_size']
        state = torch.load(out/'candidates'/f'step_{step:06d}.pt', map_location='cpu',
                           weights_only=False, mmap=True)
        if state['step'] != step or state['contract'] != seal or set(state['model']) != set(initial):
            raise ValueError('Incomplete candidate checkpoint')
    print(json.dumps(dict(state='weights_verified', native_initializer_sha256=tensor_digest(initial))), flush=True)
    del final, final_state, old, prior, state, bases, adapter
    gc.collect()
    data, cases = validation.prepare(model, paths, c, out)
    return model, initial, data, cases, seal


def predict(model, batch, mode, c, noise=None):
    if mode == 'ego':
        payload = ego_payload(batch)
        steps = c['validation']['ego_ddim_steps']
        motion = (sample_native30(model.base, payload, steps=steps, seed=c['diffusion']['seed'], use_scene=False)
                  if noise is None else sample_ego(model.base, payload, noise, steps=steps))
        _, details = model.base(motion, torch.zeros(len(motion), device=motion.device, dtype=torch.long),
                                payload, return_details=True)
        return forward_kinematics(motion, details['rest'])
    payload = repair_payload(batch)
    options = dict(steps=c['diffusion']['ddim_steps'], start_timestep=c['diffusion']['start_timestep'])
    motion = (sample_denoising(model, payload, seed=c['diffusion']['seed'], **options)
              if noise is None else sample_exo(model, payload, noise, **options))
    return forward_kinematics(motion, batch['initial_rest'])


@torch.inference_mode()
def benchmark_batches(model, data, cases, c, out, sizes):
    rows = []
    for mode in ('ego', 'exo'):
        logical = c['validation']['ego_batch_size' if mode == 'ego' else 'batch_size']
        for condition in (('ego',) if mode == 'ego' else ('tracking', 'occlusion', 'gap')):
            for source in CORPORA:
                selected = [v for v in cases[mode] if v['source'] == source][:sizes[mode]]
                pairs = pairs_for(data[mode], selected, mode, condition, c)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                expected = []
                for offset in range(0, len(pairs), logical):
                    batch = collate([p[0] for p in pairs[offset:offset+logical]], 'cuda')
                    expected.append(predict(model, batch, mode, c))
                expected = torch.cat(expected)
                torch.cuda.synchronize()
                old_seconds = time.monotonic()-started
                batch = collate([p[0] for p in pairs], 'cuda')
                noise = reference_noise(0, len(pairs), len(pairs), logical, seed=c['diffusion']['seed'])
                started = time.monotonic()
                actual = predict(model, batch, mode, c, noise)
                torch.cuda.synchronize()
                new_seconds = time.monotonic()-started
                truth = forward_kinematics(batch['motion'], batch['target_rest'])
                left, right = metric_rows(expected, truth), metric_rows(actual, truth)
                difference = {key: max(abs(a[key]-b[key]) for a, b in zip(left, right))
                              for key in ('mpjpe_mm', 'pa_mpjpe_mm', 'root_error_mm')}
                maximum = float((expected-actual).norm(dim=-1).max()*1000)
                passed = maximum <= .1 and max(difference.values()) <= .05
                row = dict(mode=mode, condition=condition, source=source, batch_size=len(pairs),
                    logical_batch_size=logical, joint_max_difference_mm=maximum,
                    metric_max_difference_mm=difference, original_seconds=old_seconds,
                    packed_seconds=new_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), passed=passed)
                rows.append(row)
                atomic_json(out/'VALIDATION_BATCH_BENCHMARK.json', dict(state='running', rows=rows))
                print(json.dumps(dict(batch_benchmark=row)), flush=True)
                if not passed:
                    raise ValueError('Larger validation batch changes predictions beyond the declared tolerance')
                del pairs, batch, expected, actual, truth, noise
                gc.collect(); torch.cuda.empty_cache()
    result = dict(state='passed', rows=rows, original_noise_preserved=True,
                  joint_tolerance_mm=.1, metric_tolerance_mm=.05, completed_at_utc=now())
    atomic_json(out/'VALIDATION_BATCH_BENCHMARK.json', result)
    return result


def chunks(data, selected, mode, condition, c, size):
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(pairs_for, data, selected[:size], mode, condition, c)
        for offset in range(0, len(selected), size):
            pairs = pending.result()
            if offset+size < len(selected):
                pending = pool.submit(pairs_for, data, selected[offset+size:offset+2*size], mode, condition, c)
            yield offset, pairs


@torch.inference_mode()
def evaluate(model, data, cases, c, out, step, checkpoint_sha, evaluation_sha, sizes):
    folder = out/'selection'/f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(step=step, checkpoint_sha256=checkpoint_sha, evaluation_sha256=evaluation_sha,
                    cases_sha256=sha256(out/'VALIDATION_CASES.json'))
    result_file = folder/'METRICS.json'
    if result_file.exists():
        cached = json.loads(result_file.read_text())
        if any(cached[k] != v for k, v in identity.items()):
            raise ValueError('Cached metrics belong to different weights/protocol')
        return cached
    model.eval()
    rows, completed, started = [], 0, time.monotonic()
    for mode in ('ego', 'exo'):
        logical = c['validation']['ego_batch_size' if mode == 'ego' else 'batch_size']
        for condition in (('ego',) if mode == 'ego' else CONDITIONS):
            for source in CORPORA:
                selected = [v for v in cases[mode] if v['source'] == source]
                shard = folder/f'{mode}_{condition}_{source}.json'
                if shard.exists():
                    cached = json.loads(shard.read_text())
                    if cached['identity'] != identity:
                        raise ValueError('Validation shard identity changed')
                    rows.extend(cached['rows']); completed += len(selected)
                    continue
                items = []
                for offset, pairs in chunks(data[mode], selected, mode, condition, c, sizes[mode]):
                    batch = collate([p[0] for p in pairs], 'cuda')
                    truth = forward_kinematics(batch['motion'], batch['target_rest'])
                    noise = reference_noise(offset, len(pairs), len(selected), logical, seed=c['diffusion']['seed'])
                    repaired = predict(model, batch, mode, c, noise)
                    if mode == 'ego':
                        variants = {'repair': metric_rows(repaired, truth)}
                    else:
                        initial = forward_kinematics(batch['initial_motion'], batch['initial_rest'])
                        visible = (batch['joint_visible'] > 0) & (batch['estimate_available'][..., None] > 0)
                        variants = {name: repair_rows(points, truth, initial, visible, condition)
                                    for name, points in (('input', initial), ('repair', repaired))}
                    for name, values in variants.items():
                        items.extend(dict(mode=mode, condition=condition, source=source, variant=name,
                                          identity=p[1], metrics=m) for p, m in zip(pairs, values))
                    progress = dict(state='evaluating', **identity, mode=mode, condition=condition, source=source,
                        completed_source_windows=offset+len(pairs), source_windows=len(selected),
                        completed_checkpoint_windows=completed+offset+len(pairs), checkpoint_windows=28741,
                        validation_batch_size=sizes, training_complete=True, completed_updates=2000,
                        planned_updates=2000, completed_windows=256000, planned_windows=256000,
                        pid=os.getpid(), updated_at_utc=now(), elapsed_seconds=time.monotonic()-started)
                    atomic_json(folder/'STATUS.json', progress)
                    atomic_json(out/'training/STATUS.json', progress)
                    atomic_json(out/'STATUS.json', progress)
                atomic_json(shard, dict(identity=identity, rows=items))
                rows.extend(items); completed += len(selected)
                print(json.dumps(dict(completed_group=dict(step=step, mode=mode, condition=condition,
                    source=source, completed_checkpoint_windows=completed, elapsed_seconds=time.monotonic()-started))), flush=True)
    if completed != 28741:
        raise ValueError('Incomplete validation window coverage')
    result = dict(validation.summarize(rows), **identity, counts=validation.EXPECTED,
        model_window_evaluations=completed, full_validation=True, test_used=False,
        aggregation='equal_dataset_then_equal_unmirrored_family; per_dataset_metrics_retained',
        completed_at_utc=now(), elapsed_seconds_this_invocation=time.monotonic()-started)
    atomic_json(result_file, result)
    atomic_json(folder/'STATUS.json', dict(state='complete', **identity, updated_at_utc=now()))
    return result


def run(args):
    ensure_bounded()
    c = training.configuration(args.config.resolve())
    out = ROOT/c['output']
    admission(args.config.resolve(), out)
    sizes = dict(ego=args.ego_batch, exo=args.exo_batch)
    model, initializer, data, cases, seal = setup(c, out)
    model.cuda().eval()
    deterministic(c['seed'], 'cuda'); torch.set_num_threads(1)
    sources = {str(ROOT/'scenemi/replay'/name): sha256(ROOT/'scenemi/replay'/name)
               for name in ('validation_runner.py', 'validation_sampling.py', 'validation_sharing.py')}
    protocol = dict(training_contract_sha256=seal['sha256'], validation_batch_size=sizes,
        logical_noise_batch_size=dict(ego=4, exo=8), original_noise_preserved=True,
        cases_sha256=sha256(out/'VALIDATION_CASES.json'), sources=sources,
        joint_parity_tolerance_mm=.1, metric_parity_tolerance_mm=.05, test_used=False)
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    target = out/'VALIDATION_PROTOCOL.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Started validation execution protocol changed')
    atomic_json(target, protocol)
    archive = out/'validation_sources.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as stream:
            for source in sources:
                stream.add(source, arcname=str(Path(source).relative_to(ROOT)))
    benchmark_file = out/'VALIDATION_BATCH_BENCHMARK.json'
    benchmark = json.loads(benchmark_file.read_text()) if benchmark_file.exists() else {}
    if benchmark.get('state') != 'passed' or not benchmark.get('original_noise_preserved'):
        state = load_state(model, out/'training/last.pt')
        del state
        benchmark = benchmark_batches(model, data, cases, c, out, sizes)
    baseline = best = fallback = None
    for windows in c['validation']['sampled_windows']:
        step = windows//c['training']['batch_size']
        checkpoint = out/'parent.pt' if not windows else out/'candidates'/f'step_{step:06d}.pt'
        state = load_state(model, checkpoint, baseline=not windows, initializer=initializer)
        checkpoint_sha = sha256(checkpoint)
        del state
        metrics = evaluate(model, data, cases, c, out, step, checkpoint_sha, protocol['sha256'], sizes)
        if baseline is None:
            baseline = metrics
        selection = dict(metrics, **validation.select(metrics, baseline, c))
        atomic_json(out/'selection'/f'step_{step:06d}'/'RESULT.json', selection)
        record = dict(step=step, sampled_windows=windows, checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha, selection_score=selection['selection_score'],
            selection_eligible=selection['selection_eligible'], full_validation=True, test_used=False,
            failed_checks=[k for k, v in selection['checks'].items() if not v],
            validation_batch_size=sizes, contract_sha256=seal['sha256'], evaluation_sha256=protocol['sha256'])
        if windows and (fallback is None or record['selection_score'] < fallback['selection_score']):
            fallback = record
            atomic_json(out/'selection/BEST_CANDIDATE.json', record)
        if windows and selection['selection_eligible'] and (best is None or record['selection_score'] < best['selection_score']):
            best = record
            pin_checkpoint(checkpoint, out/'selection/best.pt')
            atomic_json(out/'selection/BEST.json', dict(record, checkpoint=str(out/'selection/best.pt')))
    verify(seal)
    result = dict(state='complete', training_complete=True, full_validation=True,
        completed_updates=2000, planned_updates=2000, completed_windows=256000, planned_windows=256000,
        candidates=[0,8192,65536,256000], model_window_evaluations_per_checkpoint=28741,
        validation_batch_size=sizes, best=best, best_candidate=fallback,
        selection_eligible=best is not None, contract_sha256=seal['sha256'],
        evaluation_sha256=protocol['sha256'], test_used=False, completed_at_utc=now())
    for target in ('VALIDATION_RUN.json', 'training/STATUS.json', 'STATUS.json'):
        atomic_json(out/target, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=training.CONFIG)
    parser.add_argument('--ego-batch', type=int, default=16)
    parser.add_argument('--exo-batch', type=int, default=32)
    args = parser.parse_args()
    if args.ego_batch < 4 or args.ego_batch % 4 or args.exo_batch < 8 or args.exo_batch % 8:
        parser.error('Physical batches must be multiples of the original logical batches 4/8')
    out = ROOT/training.configuration(args.config)['output']
    with (out/'validation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as error:
            atomic_json(out/'VALIDATION_RUN.json', dict(state='failed', training_complete=True,
                error=repr(error), pid=os.getpid(), updated_at_utc=now()))
            raise


if __name__ == '__main__':
    main()
