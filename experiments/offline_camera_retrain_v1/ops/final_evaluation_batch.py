"""Per-window seeded batching for the sealed selected-model metric protocol.

The original evaluator and DDIM implementation stay immutable. This extension
uses their input builder, selected model loader, metrics, guards, and durable
row schema. A separate source manifest and a passing VAL-only numerical
benchmark are mandatory before any batched formal result can be committed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from . import final_evaluation as final
from .dependencies import dependency_closure, sha256

PROTOCOL = 'selected-native30-per-window-seeded-batch-v1'
MANIFEST = 'BATCH_RUNNER.json'
POSITION_METRICS = ('w_mpjpe_mm', 'hidden_mpjpe_mm', 'visible_mpjpe_mm',
                    'root_relative_mpjpe_mm')
require, read, write = final.require, final.read, final.write


def stack_known(items):
    """Concatenate singleton payloads, preserving native assets as list items.

    References are computed by the original singleton builder before this
    operation. Native scene/skin objects must not be recursively tensor-stacked.
    """
    import torch
    require(items and all(isinstance(item, dict) for item in items), 'Require singleton input dictionaries')
    forbidden = ('motion', 'joints', 'contact_target', 'contact_valid', 'support_points')
    require(all(not any(key in item for key in forbidden)
                and not any(key.startswith('gt_') for key in item) for item in items), 'GT is not a model input')
    require(all(item.get('mode') in (['ego'], ['exo']) for item in items), 'Require explicit singleton modes')

    def combine(values, path):
        first = values[0]
        if torch.is_tensor(first):
            require(first.ndim > 0 and first.shape[0] == 1, 'Singleton tensor required: ' + path)
            require(all(torch.is_tensor(value) and value.shape == first.shape
                        and value.dtype == first.dtype and value.device == first.device for value in values),
                    'Tensor shape/dtype/device differs: ' + path)
            return torch.cat(values, dim=0)
        if isinstance(first, dict):
            require(all(isinstance(value, dict) and value.keys() == first.keys() for value in values),
                    'Payload keys differ: ' + path)
            return {key: combine([value[key] for value in values], path + '.' + key) for key in first}
        if isinstance(first, (tuple, list)):
            require(all(type(value) is type(first) and len(value) == 1 for value in values),
                    'Singleton metadata list required: ' + path)
            return type(first)(value[0] for value in values)
        require(all(type(value) is type(first) and value == first for value in values),
                'Unequal non-tensor metadata: ' + path)
        return first

    return combine(items, '')


def initial_noise(reference, seeds):
    """Exactly the original [1,T,201] randn call, independently for each seed."""
    import torch
    require(torch.is_tensor(reference) and reference.ndim == 3 and reference.shape[-1] == 201
            and reference.shape[0] > 0 and reference.shape[1] > 0, 'Require [B,T,201] reference')
    require(reference.dtype == torch.float32 and torch.get_default_dtype() == torch.float32,
            'The sealed sampler uses default FP32 initial noise')
    require(len(seeds) == len(reference) and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds),
            'One original uint32 seed is required per window')
    return torch.cat([torch.randn((1, *reference.shape[1:]), device=reference.device,
        generator=torch.Generator(device=reference.device).manual_seed(seed)) for seed in seeds], dim=0)


def sample_motion_batch(model, known, config, *, seeds, steps=20, use_scene=True, state=None):
    """Same deterministic residual-DDIM algebra as native30_refinement.py.

    Only initial noise construction differs: each row retains its singleton
    generator. No GT, metric, confidence gate, or output correction is added.
    """
    import torch
    from experiments.offline_camera_retrain_v1.native30_model import reconcile_motion
    from experiments.offline_camera_retrain_v1.scene_model import cosine_alphas
    require(config.get('diffusion_target') == 'reference_residual', 'Only the sealed residual sampler is supported')
    require(2 <= steps <= 1000, 'Unsupported DDIM steps')
    reference = known['reference_motion']; device = reference.device
    require(len(seeds) == len(reference), 'One independent seed is required per row')
    if state is None:
        state = initial_noise(reference, seeds)
    else:
        require(torch.is_tensor(state) and state.shape == reference.shape
                and state.dtype == reference.dtype and state.device == device
                and bool(torch.isfinite(state).all()), 'Initial noise shape/dtype/device/finite mismatch')
        state = state.clone()
    require(model._sampling_scene is None, 'Nested sampling cache')
    with torch.no_grad():
        alphas = cosine_alphas().to(device)
        schedule = torch.linspace(999, 0, steps, device=device).round().long().unique_consecutive()
        was_training = model.training
        model.eval()
        try:
            if use_scene:
                model._sampling_scene = (known['occupancy'], model.scene_tokens(known))
            for i, step in enumerate(schedule):
                times = torch.full((len(state),), int(step), device=device, dtype=torch.long)
                absolute = model(state, times, known, use_scene=use_scene)
                residual = absolute - reference; alpha = alphas[step]
                epsilon = (state - alpha.sqrt() * residual) / (1-alpha).sqrt().clamp_min(1e-6)
                state = residual if i+1 == len(schedule) else (
                    alphas[schedule[i+1]].sqrt() * residual + (1-alphas[schedule[i+1]]).sqrt() * epsilon)
            return reconcile_motion(reference + state, known['rest'])
        finally:
            model._sampling_scene = None
            model.train(was_training)


def runner_identity(out, plan, batch_size):
    require(type(batch_size) is int and 2 <= batch_size <= 4, 'Supported logical batch size is 2..4')
    require(plan['split'] == 'full-validation' and plan['regimes'] == ['noisy'],
            'This acceleration review admits noisy full VAL only, never TEST')
    require(plan['selection'].get('output_policy', 'raw_model') == 'raw_model'
            and plan['analytical_physics_enabled'] is False and plan['ddim_steps'] == 20
            and (plan['frames'], plan['fps']) == (192, 30), 'Sealed inference parameters differ')
    return dict(protocol=PROTOCOL, parent_plan=str((Path(out) / 'PLAN.json').resolve()),
        parent_plan_sha256=sha256(Path(out) / 'PLAN.json'), batch_size=batch_size,
        runner_source=str(Path(__file__).resolve()), runner_source_sha256=sha256(__file__),
        source_sha256=dependency_closure([Path(__file__)]),
        checkpoint_sha256=plan['selection']['checkpoint_sha256'], model_tensor_sha256=plan['model_tensor_sha256'],
        selection_manifest_sha256=plan['selection_manifest_sha256'],
        frames=192, fps=30, ddim_steps=20, precision='FP32', output_policy='raw_model',
        analytical_physics_enabled=False, reference_policy=plan['reference_policy'],
        initial_noise='cat(original torch.randn([1,192,201], CUDA Generator(original case_seed)) per window)',
        observations_and_reference='Original singleton build_inputs before stacking; original per-row input hashes retained',
        algebra='native30_refinement.sample_refinement residual DDIM; no extra stochastic steps',
        benchmark_mean_abs_metric_tolerance_mm=.1, benchmark_maximum_joint_tolerance_mm=1.,
        optimizer_updates=0, model_selection_allowed=False, TEST_used=False)


def seal_runner(out, plan, batch_size):
    value = runner_identity(out, plan, batch_size)
    path = Path(out) / MANIFEST
    if path.exists():
        require(read(path) == value, 'Immutable batch runner identity/source differs')
    else:
        write(path, value)
    return value


def select_benchmark_rows(plan, rows):
    """First completed identity per corpus/mode, with both noisy scene arms."""
    lookup = {final.row_key(row): row for row in rows}
    result = []
    for dataset in ('rich', 'trumans', 'egobody'):
        for mode in ('ego', 'exo'):
            candidates = [entry for entry in plan['cases'] if (entry['dataset'], entry['mode']) == (dataset, mode)
                and all((entry['case'], 'noisy', scene) in lookup for scene in final.SCENES)]
            require(candidates, 'Need a completed noisy ON/OFF singleton baseline: ' + dataset + '/' + mode)
            entry = candidates[0]
            result.extend(lookup[(entry['case'], 'noisy', scene)] for scene in final.SCENES)
    return result


def _configure(device, memory_fraction):
    import torch
    require(device in ('cpu', 'cuda') and 0 < memory_fraction <= 1, 'Invalid device/memory budget')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _payload(data, builder, entry, plan, device):
    from .validation_review import tensor_fingerprint
    sample, identity, observed, known = final.build_inputs(data, builder, entry, 'noisy', plan['config'], device)
    return dict(entry=entry, sample=sample, identity=identity, observed=observed, known=known,
                input_sha256=tensor_fingerprint(known))


def _predict(model, payloads, plan, scene):
    import torch
    from .scene_comparison import scene_off_guard
    from .validation_review import tensor_fingerprint
    tick = time.monotonic()
    known = stack_known([payload['known'] for payload in payloads])
    seeds = [payload['entry']['case_seed'] for payload in payloads]
    state = initial_noise(known['reference_motion'], seeds)
    noise_hashes = [tensor_fingerprint({'initial_noise': state[i:i+1]}) for i in range(len(state))]
    with (scene_off_guard(model) if scene == 'scene_off' else nullcontext()) as guards:
        prediction = sample_motion_batch(model, known, plan['config'], seeds=seeds, steps=20,
                                         use_scene=scene == 'scene_on', state=state)
    if prediction.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - tick
    require(bool(torch.isfinite(prediction).all()), 'Nonfinite batched prediction')
    for payload in payloads:
        require(tensor_fingerprint(payload['known']) == payload['input_sha256'], 'Mutated singleton inference input')
    return prediction, guards, elapsed, noise_hashes


def _arrays(payload, prediction, checkpoint_sha256, scene):
    import numpy as np
    return dict(prediction=prediction.detach().cpu().numpy(), truth=payload['sample']['motion'],
        rest=payload['sample']['rest'], observations=payload['observed']['trajectory'],
        observation_meta=payload['observed']['observation_meta'],
        reference_motion=payload['known']['reference_motion'][0].detach().cpu().numpy(),
        identity_json=np.array(json.dumps(payload['identity'])), input_sha256=np.array(payload['input_sha256']),
        checkpoint_sha256=np.array(checkpoint_sha256), case_seed=np.array(payload['entry']['case_seed']),
        use_scene=np.array(scene == 'scene_on'))


def atomic_arrays(path, arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def comparison(arrays, baseline):
    """CPU report only; benchmark never changes any prediction or model."""
    import numpy as np
    from .reference_review import metrics
    from .refinement_guard import forward_kinematics_numpy
    for name in ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion'):
        require(np.array_equal(arrays[name], baseline[name]), 'Benchmark input/GT differs: ' + name)
    actual, actual_counts = metrics(arrays); old, old_counts = metrics(baseline)
    require(actual_counts == old_counts, 'Benchmark metric denominator differs')
    differences = {}
    for name, value in actual.items():
        require((value is None) == (old[name] is None), 'Benchmark metric availability differs')
        if value is not None:
            differences[name] = abs(value - old[name])
    delta = forward_kinematics_numpy(arrays['prediction'], arrays['rest']) - forward_kinematics_numpy(
        baseline['prediction'], baseline['rest'])
    joint_difference = np.linalg.norm(delta, axis=-1) * 1000
    require(np.isfinite(joint_difference).all() and all(np.isfinite(v) for v in differences.values()),
            'Nonfinite benchmark comparison')
    return dict(metric_abs_difference=differences, mean_joint_difference_mm=float(joint_difference.mean()),
        maximum_joint_difference_mm=float(joint_difference.max()),
        maximum_motion_channel_difference=float(np.max(np.abs(arrays['prediction'] - baseline['prediction']))))


def benchmark(out, *, batch_size=4, device='cuda', memory_fraction=.7):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    from . import validation_review as common
    out = Path(out).resolve(); plan = final.verify(out)
    selected = select_benchmark_rows(plan, final.load_rows(out, plan))
    manifest = seal_runner(out, plan, batch_size)
    folder = out / f'batch{batch_size}_benchmark'; folder.mkdir(exist_ok=True)
    manifest_sha = sha256(out / MANIFEST)
    snapshot = dict(protocol=PROTOCOL, runner_manifest_sha256=manifest_sha,
        parent_plan_sha256=manifest['parent_plan_sha256'], selected_rows=selected,
        selection='First completed noisy ON/OFF identity for each corpus/mode; metrics not used for selection',
        planned_predictions=12, mean_abs_metric_tolerance_mm=.1, maximum_joint_tolerance_mm=1., TEST_used=False)
    snapshot_path = folder / 'PLAN.json'
    require(not (folder / 'RESULT.json').exists(), 'Benchmark already completed; preserve its record')
    if snapshot_path.exists():
        require(read(snapshot_path) == snapshot, 'Benchmark identity snapshot changed')
    else:
        write(snapshot_path, snapshot)
    write(folder / 'STATUS.json', dict(status='waiting_gpu', checked_at=final.now(), planned_predictions=12))
    entries = {entry['case']: entry for entry in plan['cases']}
    case_ids = list(dict.fromkeys(row['case'] for row in selected))
    records, batches = [], []
    with common.gpu_lock() if device == 'cuda' else nullcontext():
        final.verify(out); _configure(device, memory_fraction)
        model, identity = final.load_selected(plan['selection'], device)
        require(model_digest(model) == plan['model_tensor_sha256'], 'Selected tensor identity differs')
        data, builder = final.make_data(plan, out), common.observation_builder(plan['config'], device=device)
        for offset in range(0, len(case_ids), batch_size):
            payloads = [_payload(data, builder, entries[case], plan, device) for case in case_ids[offset:offset+batch_size]]
            for scene in final.SCENES:
                prediction, guards, elapsed, noises = _predict(model, payloads, plan, scene)
                batches.append(dict(scene=scene, cases=[p['entry']['case'] for p in payloads], sampling_s=elapsed))
                for index, payload in enumerate(payloads):
                    original = next(row for row in selected if final.row_key(row) == (payload['entry']['case'], 'noisy', scene))
                    require(original['input_sha256'] == payload['input_sha256'], 'Original singleton input hash differs')
                    require(sha256(original['artifact']) == original['artifact_sha256'], 'Baseline artifact changed')
                    arrays = _arrays(payload, prediction[index], identity['checkpoint_sha256'], scene)
                    with np.load(original['artifact'], allow_pickle=False) as saved:
                        audit = comparison(arrays, saved)
                    artifact = folder / 'motions' / f"case{original['case']:05d}_noisy_{scene}.npz"
                    atomic_arrays(artifact, arrays)
                    record = dict(case=original['case'], mode=original['mode'], dataset=original['dataset'],
                        scene_variant=scene, seed=original['seed'], input_sha256=payload['input_sha256'],
                        initial_noise_sha256=noises[index], artifact=str(artifact), artifact_sha256=sha256(artifact),
                        baseline_artifact_sha256=original['artifact_sha256'], scene_off_guard_counts=guards,
                        **audit)
                    records.append(record)
                    write(folder / 'PARTIAL.json', dict(completed_predictions=len(records), rows=records, batches=batches))
                del prediction
            del payloads; gc.collect()
        require(model_digest(model) == plan['model_tensor_sha256'], 'Benchmark changed frozen model')
    means = {name: statistics.mean(r['metric_abs_difference'][name] for r in records
        if name in r['metric_abs_difference']) for name in POSITION_METRICS
        if any(name in r['metric_abs_difference'] for r in records)}
    maximum = max(row['maximum_joint_difference_mm'] for row in records)
    passed = len(records) == 12 and all(v <= .1 for v in means.values()) and maximum <= 1.
    singleton_s = sum(row['sampling_s'] for row in selected)
    batch_s = sum(row['sampling_s'] for row in batches)
    result = dict(protocol=PROTOCOL, status='passed' if passed else 'FAILED', checked_at=final.now(), device=device,
        runner_manifest_sha256=manifest_sha, benchmark_plan_sha256=sha256(snapshot_path),
        batch_size=batch_size, completed_predictions=len(records), rows=records, batches=batches,
        mean_abs_position_metric_difference_mm=means, maximum_joint_difference_mm=maximum,
        mean_abs_metric_tolerance_mm=.1, maximum_joint_tolerance_mm=1.,
        singleton_saved_sampling_s=singleton_s, batched_sampling_s=batch_s,
        sampling_only_speedup=singleton_s/batch_s if batch_s else None,
        speed_warning='Saved singleton timings are not a simultaneous controlled timing comparison; input construction and IO excluded',
        GPU_used=device == 'cuda', optimizer_updates=0, TEST_used=False, model_selection_allowed=False)
    write(folder / 'RESULT.json', result)
    write(folder / 'STATUS.json', dict(status=result['status'], completed_predictions=len(records), checked_at=final.now()))
    require(passed, 'Batch numerical benchmark FAILED; formal batched evaluation is blocked')
    return result


def require_benchmark(out, plan, batch_size, device):
    manifest = seal_runner(out, plan, batch_size)
    path = Path(out) / f'batch{batch_size}_benchmark/RESULT.json'
    require(path.exists(), 'Run the numerical batch benchmark before formal evaluation')
    result = read(path)
    require(result['status'] == 'passed' and result['completed_predictions'] == 12
            and result['runner_manifest_sha256'] == sha256(Path(out) / MANIFEST)
            and result['batch_size'] == batch_size and result['device'] == device,
            'No matching passing numerical batch benchmark')
    require(sha256(path.parent / 'PLAN.json') == result['benchmark_plan_sha256'], 'Benchmark plan changed')
    require(result['maximum_joint_difference_mm'] <= manifest['benchmark_maximum_joint_tolerance_mm']
            and set(result['mean_abs_position_metric_difference_mm']) == set(POSITION_METRICS)
            and all(v <= manifest['benchmark_mean_abs_metric_tolerance_mm']
                    for v in result['mean_abs_position_metric_difference_mm'].values()), 'Benchmark guard differs')
    require(len(result['rows']) == 12 and all(sha256(row['artifact']) == row['artifact_sha256']
            for row in result['rows']), 'Benchmark prediction artifacts changed')
    return manifest, path


def evaluate(out, *, batch_size=4, resume=False, max_predictions=0, memory_fraction=.7,
             status_every=10, device='cuda'):
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    from . import validation_review as common
    from .reference_review import metrics
    out = Path(out).resolve(); plan = final.verify(out)
    require(status_every > 0 and max_predictions >= 0, 'Invalid runtime budget')
    require(resume or not (out / 'rows.jsonl').exists(), 'Existing formal results require --resume')
    _, benchmark_path = require_benchmark(out, plan, batch_size, device)
    rows = final.load_rows(out, plan, repair=resume); done = {final.row_key(row) for row in rows}
    manifest_sha = sha256(out / MANIFEST); benchmark_sha = sha256(benchmark_path)
    for row in rows:
        if row.get('execution_protocol') == PROTOCOL:
            require(row['batch_runner_manifest_sha256'] == manifest_sha and row['batch_benchmark_sha256'] == benchmark_sha,
                    'Previously committed batch runtime identity differs')
    final.status(out, 'waiting_gpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                 execution_protocol=PROTOCOL, logical_batch_size=batch_size)
    started, generated = time.monotonic(), 0
    with common.gpu_lock() if device == 'cuda' else nullcontext():
        final.verify(out); _configure(device, memory_fraction)
        # A singleton worker may have committed more rows while this process
        # waited for admission. Reload only after owning the shared GPU lock.
        rows = final.load_rows(out, plan, repair=resume)
        done = {final.row_key(row) for row in rows}
        for row in rows:
            if row.get('execution_protocol') == PROTOCOL:
                require(row['batch_runner_manifest_sha256'] == manifest_sha
                        and row['batch_benchmark_sha256'] == benchmark_sha,
                        'Previously committed batch runtime identity differs')
        model, identity = final.load_selected(plan['selection'], device)
        require(model_digest(model) == plan['model_tensor_sha256'], 'Selected tensor identity differs')
        data, builder = final.make_data(plan, out), common.observation_builder(plan['config'], device=device)
        pending = [entry for entry in plan['cases'] if any((entry['case'], 'noisy', scene) not in done for scene in final.SCENES)]
        for offset in range(0, len(pending), batch_size):
            payloads = [_payload(data, builder, entry, plan, device) for entry in pending[offset:offset+batch_size]]
            for payload in payloads:
                require(all(row['input_sha256'] == payload['input_sha256'] for row in rows
                    if row['case'] == payload['entry']['case'] and row['regime'] == 'noisy'), 'Resumed paired input hash differs')
            for scene in final.SCENES:
                active = [payload for payload in payloads if (payload['entry']['case'], 'noisy', scene) not in done]
                if max_predictions:
                    active = active[:max_predictions-generated]
                if not active:
                    continue
                final.status(out, 'sampling', execution_protocol=PROTOCOL, scene_variant=scene,
                    current_cases=[p['entry']['case'] for p in active], completed_predictions=len(rows),
                    planned_predictions=plan['planned_predictions'], logical_batch_size=len(active))
                prediction, guards, elapsed, noises = _predict(model, active, plan, scene)
                for index, payload in enumerate(active):
                    entry = payload['entry']; arrays = _arrays(payload, prediction[index], identity['checkpoint_sha256'], scene)
                    artifact = out / 'motions' / f"case{entry['case']:05d}_noisy_{scene}.npz"
                    atomic_arrays(artifact, arrays)
                    values, counts = metrics(arrays)
                    row = {key: entry[key] for key in ('case', 'mode', 'dataset', 'recording', 'sequence_id', 'start', 'source_first')}
                    row.update(regime='noisy', scene_variant=scene, seed=entry['case_seed'], input_sha256=payload['input_sha256'],
                        selection_sha256=plan['selection_manifest_sha256'], checkpoint_sha256=identity['checkpoint_sha256'],
                        artifact=str(artifact), artifact_sha256=sha256(artifact), metrics=values, counts=counts,
                        scene_off_guard_counts=guards, sampling_s=elapsed/len(active), checked_at=final.now(), device=device,
                        execution_protocol=PROTOCOL, logical_batch_size=len(active), batch_sampling_s=elapsed,
                        initial_noise_sha256=noises[index], batch_runner_manifest_sha256=manifest_sha,
                        batch_benchmark_sha256=benchmark_sha)
                    with (out / 'rows.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                    rows.append(row); done.add(final.row_key(row)); generated += 1
                    if generated % status_every == 0:
                        final.summarize(out, plan, rows)
                        print(json.dumps(dict(completed=len(rows), planned=plan['planned_predictions'],
                            last_batch_sampling_s=elapsed, logical_batch_size=len(active), elapsed_s=time.monotonic()-started)), flush=True)
                del prediction
                if max_predictions and generated >= max_predictions:
                    require(model_digest(model) == plan['model_tensor_sha256'], 'Frozen tensors changed')
                    final.summarize(out, plan, rows)
                    final.status(out, 'bounded_stop', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                                 execution_protocol=PROTOCOL)
                    return
            del payloads; gc.collect()
        require(model_digest(model) == plan['model_tensor_sha256'], 'Frozen tensors changed')
        final.verify(out)
    final.summarize(out, plan, rows)
    final.status(out, 'sampling_completed_pending_CPU', completed_predictions=len(rows),
                 planned_predictions=plan['planned_predictions'], execution_protocol=PROTOCOL)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('benchmark', 'evaluate'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--memory-fraction', type=float, default=.7)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--status-every', type=int, default=10)
    args = parser.parse_args(argv)
    if args.command == 'benchmark':
        result = benchmark(args.out, batch_size=args.batch_size, device=args.device, memory_fraction=args.memory_fraction)
        print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'batches')}, ensure_ascii=False))
    else:
        evaluate(args.out, batch_size=args.batch_size, device=args.device, memory_fraction=args.memory_fraction,
            resume=args.resume, max_predictions=args.max_predictions, status_every=args.status_every)


if __name__ == '__main__':
    main()
