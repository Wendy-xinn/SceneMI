#!/usr/bin/env python3
"""CPU-only independent checks of every indexed official RICH VAL window.

The runner, its rows, artifacts, source arrays and training files are read only.
Only VERIFY*.json, this script's log and optional watcher PID are written here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[2]
REPO = BASE.parents[1]
RUN = OUT.parents[1] / 'native30_gait_v2_ego55k_oct15'
sys.path.insert(0, str(REPO))


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def compare(expected, actual, label, errors):
    """Compare nested numeric metrics, allowing CPU reduction roundoff only."""
    if isinstance(expected, dict):
        require(set(expected) == set(actual), label + ': metric keys differ')
        for key in expected:
            compare(expected[key], actual[key], label + '/' + key, errors)
    elif isinstance(expected, list):
        require(len(expected) == len(actual), label + ': list size differs')
        for i, value in enumerate(expected):
            compare(value, actual[i], label + '/' + str(i), errors)
    elif isinstance(expected, (bool, int)) or expected is None:
        require(expected == actual, label + ': non-float value differs')
    else:
        require(math.isfinite(float(expected)) and math.isfinite(float(actual)), label + ': nonfinite')
        delta = abs(float(expected) - float(actual))
        require(math.isclose(expected, actual, rel_tol=1e-6, abs_tol=3e-5),
                label + ': CPU metric mismatch ' + repr((expected, actual)))
        errors.append(delta)


def inspect_checkpoint(frozen, effective):
    import torch
    path = OUT / 'checkpoint_eval.pt'
    require(sha(path) == frozen['evaluation_checkpoint_sha256'], 'Frozen compact checkpoint SHA changed')
    require(path.stat().st_size == frozen['evaluation_checkpoint_bytes'], 'Frozen compact size changed')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    require(set(state) == {'model', 'step', 'effective_config', 'planned_steps', 'evaluation_only'},
            'Unexpected frozen state fields')
    require(state['step'] == frozen['checkpoint_step'] and 0 < state['step'] <= 55000,
            'Mixed or invalid frozen checkpoint step')
    require(state['effective_config'] == effective, 'Frozen effective config differs from sealed config')
    require(state['planned_steps'] == 55000 and state['evaluation_only'] is True, 'Frozen role differs')
    model = state['model']
    require(model and all(isinstance(v, torch.Tensor) for v in model.values()), 'Unexpected model state')
    digest = hashlib.sha256()
    n_values = n_bytes = 0
    dtype_counts = Counter()
    schema = []
    for name, value in sorted(model.items()):
        require(value.device.type == 'cpu', 'Checkpoint tensor unexpectedly on GPU')
        require(bool(torch.isfinite(value).all()), 'Nonfinite frozen tensor: ' + name)
        meta = dict(name=name, dtype=str(value.dtype), shape=list(value.shape))
        digest.update(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast('B'))
        n_values += value.numel()
        n_bytes += value.numel() * value.element_size()
        dtype_counts[str(value.dtype)] += 1
        schema.append(meta)
    model_digest = digest.hexdigest()
    require(model_digest == frozen['source_model_tensor_sha256'] == frozen['compact_model_tensor_sha256'],
            'Source/compact/current model tensor digests differ')
    require(frozen['direct_source_tensor_digest_comparison'] is True, 'Freeze omitted source/compact digest check')
    del model, state
    return dict(compact_checkpoint_sha256=frozen['evaluation_checkpoint_sha256'],
        source_full_checkpoint_sha256_recorded_at_freeze=frozen['source_checkpoint_sha256'],
        model_tensor_sha256_sorted_names_dtypes_shapes_and_bytes=model_digest,
        tensor_count=len(schema), tensor_values=n_values, tensor_bytes=n_bytes,
        tensor_dtypes=dict(dtype_counts), all_finite=True, no_optimizer_or_RNG_in_compact=True,
        source_to_compact_evidence='Runner recorded a model tensor digest of the atomically pinned full checkpoint and compared it with a newly loaded compact checkpoint before unlinking the source inode. This verifier independently recomputes the retained compact model digest.',
        source_and_compact_recorded_model_digests_equal=True,
        independently_recomputed_compact_model_digest=True,
        direct_source_tensor_comparison_in_verifier=False,
        boundary='The original full pinned inode was unlinked after publication. The verifier independently validates the retained compact checkpoint against the source-model digest recorded at freeze; it does not reopen the original pinned source or substitute the changing trainer last.pt.')


def inspect_plan(plan):
    """Reconstruct the exact RICH cohort from sealed manifest and legal index."""
    import numpy as np

    entries = plan['plan']
    require(entries and [e['case'] for e in entries] == list(range(len(entries))), 'Plan case numbering differs')
    require(all(e['dataset'] == 'rich' and e['group'] == 'rich' for e in entries), 'Non-RICH plan member')
    require(plan['regimes'] == ['clean', 'noisy'] and plan['mode'] == 'ego' and plan['joints'] == [15],
            'Observation/regime protocol differs')
    require(plan['frames'] == 192 and plan['fps'] == 30 and plan['window_stride'] == 64 and
            plan['ddim_steps'] == 20 and plan['scene'] == 'on', 'Sampling protocol differs')
    require(plan['unique_windows'] == len(entries) and plan['expected_sampling_calls'] == len(entries) * 2,
            'Declared plan counts differ')
    for path, digest in plan['source_sha256'].items():
        require(sha(path) == digest, 'Plan source dependency changed: ' + path)
    manifest = [json.loads(line) for line in Path(plan['source_manifest']).read_text().splitlines() if line.strip()]
    official = {r['path']: r for r in manifest if r.get('dataset') == 'rich' and
                r.get('split') == 'validation' and not r.get('diagnostic_only', False) and
                'ego' in r.get('allowed_modes', [])}
    require(official, 'No official RICH VAL manifest entries')
    index = load(plan['window_index'])
    require(index['fps'] == plan['fps'] and index['frames'] == plan['frames'] and
            index['window_stride'] == plan['window_stride'], 'Persisted grid parameters differ')
    legal = np.load(Path(plan['window_index']).parent / index['files']['validation/ego']['file'], allow_pickle=False)
    expected = [(int(b), int(start)) for b, start in legal if index['bundle_paths'][int(b)] in official]
    require(len(expected) == len(set(expected)), 'Official grid contains repeated RICH pair')
    actual = [(e['source_bundle_ordinal'], e['start']) for e in entries]
    require(len(actual) == len(set(actual)), 'Plan repeats an official RICH window')
    require(set(actual) == set(expected), 'Plan does not exactly cover all official indexed RICH VAL windows')
    per_bundle = Counter(b for b, _ in expected)
    metadata = {}
    scene_windows = Counter()
    for entry in entries:
        ordinal, start = entry['source_bundle_ordinal'], entry['start']
        require(index['bundle_paths'][ordinal] == entry['path'], 'Plan bundle ordinal/path differs')
        source = official[entry['path']]
        if entry['path'] not in metadata:
            meta_path = Path(entry['path']) / 'metadata.json'
            metadata[entry['path']] = (load(meta_path), sha(meta_path))
        meta, metadata_sha = metadata[entry['path']]
        require(metadata_sha == entry['source_metadata_sha256'], 'Source metadata changed')
        require(meta['split'] == 'validation' and not meta['diagnostic_only'] and meta['dataset'] == 'rich',
                'Plan uses a nonofficial/non-RICH source')
        for key in ('dataset', 'group', 'sequence_id', 'source_first', 'frames', 'input_sha256', 'output_sha256'):
            require(source[key] == meta[key], 'Manifest/metadata differs: ' + key)
        require(entry['sequence_id'] == meta['sequence_id'] and entry['recording'] ==
                (meta.get('recording') or meta['sequence_id']), 'Source sequence/recording differs')
        require(entry['mode'] == plan['mode'] and entry['joints'] == plan['joints'], 'Entry observation differs')
        require(entry['eligible_windows'] == per_bundle[ordinal], 'Eligible-window count differs')
        require(meta['source_first'] + start == entry['source_first'] and start >= 0 and
                start + plan['frames'] <= meta['frames'], 'Source frame/window range differs')
        require(meta['output_sha256'] == entry['source_hashes'] and
                meta['input_sha256'] == entry['source_input_hashes'], 'Declared source hashes differ')
        key = json.dumps([entry['group'], 'ego', entry['sequence_id'], entry['source_first']], separators=(',', ':'))
        seed = (plan['case_seed_base'] + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')) % 2**32
        require(seed == entry['case_seed'], 'Identity seed differs')
        scene_windows[meta['scene_id']] += 1
    coverage = []
    for path, meta in sorted(official.items(), key=lambda item: item[1]['sequence_id']):
        covered = np.zeros(meta['frames'], dtype=np.bool_)
        starts = [start for ordinal, start in expected if index['bundle_paths'][ordinal] == path]
        for start in starts:
            covered[start:start + plan['frames']] = True
        coverage.append(dict(sequence_id=meta['sequence_id'], path=path, frames=meta['frames'],
            legal_windows=len(starts), covered_unique_frames=int(covered.sum()),
            uncovered_frames=int((~covered).sum()),
            reason_if_empty=None if starts else 'source_shorter_than_192_frames'))
        require(starts or meta['frames'] < plan['frames'], 'Eligible official sequence silently omitted')
    require(plan['source_coverage'] == coverage, 'Declared full source coverage differs')
    require(plan['covered_unique_frames'] == sum(c['covered_unique_frames'] for c in coverage) and
            plan['window_frame_occurrences'] == len(entries) * plan['frames'], 'Frame coverage totals differ')
    recordings = {e['recording'] for e in entries}
    require(plan['selection']['unique_recordings'] == len(recordings) and
            plan['selection']['source_sequences'] == len(official), 'Sequence count differs')
    return dict(expected_windows=len(expected), expected_predictions=len(expected) * len(plan['regimes']),
        official_source_sequences=len(official), evaluated_sequences=len(metadata), evaluated_recordings=len(recordings),
        full_index_exactly_covered=True, source_coverage=coverage, scene_window_counts=dict(scene_windows),
        covered_unique_frames=plan['covered_unique_frames'], window_frame_occurrences=plan['window_frame_occurrences'],
        overlapping_windows=True, window_stride=plan['window_stride'],
        independent_sample_warning='Stride-64 windows have up to 128 of 192 frames in common. Window count is not an independent recording count; clean/noisy are paired predictions, not extra windows.')


def verify(require_complete=False):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics, rotation_from_6d
    from experiments.offline_camera_retrain_v1.standard_motion_metrics import standard_motion_metrics

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tick = time.monotonic()
    started = now()
    main_before = load(RUN / 'continuous_training_oct16/STATUS.json')
    plan = load(OUT / 'PLAN.json')
    protocol = load(OUT / 'PROTOCOL.json')
    status = load(OUT / 'STATUS.json')
    frozen = load(OUT / 'FROZEN_CHECKPOINT.json')
    effective = load(RUN / 'prepared/effective_config.json')
    continuation = load(RUN / 'continuous_training_oct16/PROTOCOL.json')
    sources = dict(effective['code_sha256'])
    for path, digest in continuation['expected_source_sha256'].items():
        require(path not in sources or sources[path] == digest, 'Conflicting source seals')
        sources[path] = digest
    for path, digest in sources.items():
        require(sha(path) == digest, 'Training source seal changed: ' + path)
    require(sha(RUN / 'config.json') == continuation['original_config_sha256'], 'Training config changed')
    effective_sha = sha(RUN / 'prepared/effective_config.json')
    require(effective_sha == continuation['original_effective_config_sha256'] == protocol['effective_config_sha256'],
            'Effective config hash changed')
    require(sha(OUT / 'PLAN.json') == protocol['plan_sha256'], 'Plan changed')
    require(sha(OUT / 'runner.py') == protocol['runner_sha256'], 'Runner changed')
    require(frozen['evaluation_checkpoint_sha256'] == protocol['checkpoint_sha256'], 'Protocol checkpoint differs')
    require(protocol['checkpoint_step'] == status['checkpoint_step'] == frozen['checkpoint_step'], 'Step differs')
    plan_checks = inspect_plan(plan)
    entries = plan['plan']
    n_windows = plan_checks['expected_windows']
    n_predictions = plan_checks['expected_predictions']
    require(protocol['planned_windows'] == n_windows and protocol['planned_predictions'] == n_predictions,
            'Runner protocol coverage differs')
    require(plan['regimes'] == protocol['regimes'] == ['clean', 'noisy'], 'Regime protocol differs')
    require(plan['frames'] == protocol['frames'] == 192 and plan['fps'] == protocol['fps'] == 30 and
            plan['ddim_steps'] == protocol['ddim_steps'] == 20, 'Sampling protocol differs')
    checkpoint = inspect_checkpoint(frozen, effective)
    rows_path = OUT / 'rows.jsonl'
    payload = rows_path.read_bytes() if rows_path.exists() else b''
    end = payload.rfind(b'\n') + 1
    committed = payload[:end]
    rows = [json.loads(line) for line in committed.splitlines()]
    completed = status['status'] == 'completed'
    if require_complete or completed:
        require(completed and protocol['status'] == 'completed', 'Evaluator has not completed')
        require(len(payload) == end, 'Incomplete final row')
        require(len(rows) == status['completed_predictions'] == n_predictions, 'Final row count differs')
    require(len(rows) <= n_predictions, 'Excess predictions')
    feet_path = BASE / effective['config']['run_control']['foot_metrics_source']
    require(sha(feet_path) == effective['config']['run_control']['foot_metrics_sha256'], 'Foot metrics changed')
    feet = module('independent_cpu_extra_metrics', feet_path)
    seen, metric_checks, deltas = set(), [], []
    for i, row in enumerate(rows):
        case, regime = row['case'], row['regime']
        require(isinstance(case, int) and 0 <= case < n_windows and regime in plan['regimes'], 'Unknown row identity')
        require((case, regime) not in seen, 'Duplicate row')
        seen.add((case, regime))
        entry = entries[case]
        for key in ('dataset', 'recording', 'group', 'sequence_id', 'start', 'source_first', 'case_seed'):
            require(row[key] == entry[key], 'Row differs from plan: ' + key)
        require(row['checkpoint_step'] == frozen['checkpoint_step'], 'Mixed-checkpoint row')
        path = OUT / row['motion_file']
        require(path.resolve().parent == (OUT / 'motions').resolve(), 'Unexpected artifact path')
        require(sha(path) == row['motion_sha256'], 'Motion artifact hash differs: ' + str(path))
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        require(int(arrays['step']) == frozen['checkpoint_step'], 'Artifact checkpoint differs')
        identity = json.loads(arrays['identity_json'].item())
        for key in ('path', 'sequence_id', 'dataset', 'group', 'start', 'mode'):
            require(identity[key] == entry[key], 'Artifact identity differs: ' + key)
        require((identity.get('recording') or identity['sequence_id']) == entry['recording'], 'Artifact recording differs')
        require(identity['source_start_30fps'] == entry['source_first'] and identity['split'] == 'validation' and
                identity['fps'] == 30 and identity['window_stride'] == 64 and identity['yaw_augmentation_rad'] == 0.,
                'Artifact source frame/split/coordinate convention differs')
        require(identity['input_sha256'] == entry['source_input_hashes'] and identity['output_sha256'] == entry['source_hashes'],
                'Artifact source fingerprints differ')
        expected_shapes = dict(prediction=(192, 201), truth=(192, 201), rest=(22, 3),
            observations=(192, 22, 9), observation_meta=(192, 22, 5), reference=(192, 201))
        for key, shape in expected_shapes.items():
            require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(), 'Bad artifact array: ' + key)
        # Partial snapshot: first three committed predictions. Final: every saved prediction.
        if completed or i < 3:
            pred = torch.from_numpy(arrays['prediction'])[None]
            truth = torch.from_numpy(arrays['truth'])[None]
            rest = torch.from_numpy(arrays['rest'])[None]
            conditioned = torch.from_numpy(arrays['observation_meta'][..., 0] > 0)[None]
            batch = dict(motion=truth, rest=rest)
            with torch.no_grad():
                raw = raw_metrics(pred, batch, conditioned)
                extra = feet.extra_metrics(pred, batch)
                g, p = forward_kinematics(truth, rest), forward_kinematics(pred, rest)
                standard = standard_motion_metrics(p[0].numpy(), g[0].numpy(), fps=30.)
                frames, joints = g.shape[1:3]
                benchmark_counts = {key: frames * joints for key in standard}
                benchmark_counts.update(root_position_error_mm=frames, mpjve_mm_s=(frames - 1) * joints,
                                        acceleration_error_mm_s2=(frames - 2) * joints)
                benchmark = {key: dict(value=value, count=benchmark_counts[key]) for key, value in standard.items()}
                gs = g.diff(dim=1)[:, :, (10, 11)].norm(dim=-1) * 30
                ps = p.diff(dim=1)[:, :, (10, 11)].norm(dim=-1) * 30
                low = gs < .2
                root_error = p[:, :, 0] - g[:, :, 0]
                extra.update(gt_low_speed_count=int(low.sum()), gt_foot_velocity_samples=int(low.numel()),
                    gt_foot_speed_on_gt_low_speed_m_s=float(gs[low].mean()) if low.any() else None,
                    pred_speed_over_point2_on_gt_low_speed_count=int(((ps > .2) & low).sum()),
                    root_terminal_horizontal_error_change_m=float((root_error[:, -1] - root_error[:, 0])[:, (0, 2)].norm(dim=-1).mean()))
                gtpath = float(g[:, :, 0][:, :, (0, 2)].diff(dim=1).norm(dim=-1).sum())
                glocal = torch.einsum('btji,btkj->btki', rotation_from_6d(truth[..., 3:9]), g - g[:, :, :1])[:, :, (10, 11)]
                gtamp = (glocal - glocal.mean(1, keepdim=True)).square().sum(-1).mean(1).sqrt()[0]
                activity = dict(gt_root_horizontal_path_m=gtpath, gt_mean_joint_speed_m_s=float(g.diff(dim=1).norm(dim=-1).mean() * 30),
                    gt_local_foot_RMS_left_right_m=gtamp.tolist(), moving_proxy=gtpath >= .5,
                    near_static_proxy=gtpath < .25 and float(gtamp.max()) < .08)
            before = len(deltas)
            compare(row['metrics'], raw, f'{case}/{regime}/metrics', deltas)
            compare(row['extra_metrics'], extra, f'{case}/{regime}/extra_metrics', deltas)
            compare(row['benchmark_metrics'], benchmark, f'{case}/{regime}/benchmark_metrics', deltas)
            compare(row['activity'], activity, f'{case}/{regime}/activity', deltas)
            metric_checks.append(dict(case=case, regime=regime, motion_sha256=row['motion_sha256'],
                                      max_absolute_float_difference=max(deltas[before:], default=0.)))
    expected_pairs = {(e['case'], regime) for e in entries for regime in plan['regimes']}
    if completed:
        require(seen == expected_pairs, 'Final case/regime coverage incomplete')
    main_after = load(RUN / 'continuous_training_oct16/STATUS.json')
    baseline_step = protocol['resources_before']['main_training']['latest_trace_step']
    current_step = main_after['latest_trace_step']
    require(current_step >= main_before['latest_trace_step'] and current_step > baseline_step, 'Main training advancement not observed')
    require(main_after['goal_step'] == 55000 and main_after['status'] == 'training', 'Main trainer no longer running toward 55k')
    require(Path('/proc/' + str(main_after['worker_pid'])).exists(), 'Main worker PID absent')
    require(not torch.cuda.is_initialized(), 'Verifier initialized CUDA')
    result = dict(status='passed_complete' if completed else 'passed_partial', checked_at=now(), started_at=started,
        verification_seconds=time.monotonic()-tick, verifier_pid=os.getpid(), verifier_sha256=sha(__file__),
        CPU_only=True, CPU_threads=1, CUDA_initialized=False, training_or_runner_files_written=False,
        checkpoint_step=frozen['checkpoint_step'], checkpoint=checkpoint,
        sealed_training_sources_verified=len(sources), sealed_original_sources=len(effective['code_sha256']),
        sealed_continuation_sources=len(continuation['expected_source_sha256']),
        effective_config_sha256=effective_sha, plan_sha256=protocol['plan_sha256'], runner_sha256=protocol['runner_sha256'],
        committed_rows_verified=len(rows), rows_committed_prefix_bytes=end,
        rows_committed_prefix_sha256=hashlib.sha256(committed).hexdigest(), uncommitted_trailing_bytes=len(payload)-end,
        artifacts_SHA_identity_shapes_finite_checked=len(rows), expected_predictions=n_predictions,
        expected_windows=n_windows, official_RICH_grid=plan_checks,
        planned_dataset_counts=dict(Counter(e['dataset'] for e in entries)),
        committed_dataset_regime_counts=dict(Counter(r['dataset']+'/'+r['regime'] for r in rows)),
        missing_case_regime_pairs=[list(pair) for pair in sorted(expected_pairs-seen)],
        CPU_metric_recomputations=metric_checks, maximum_absolute_float_difference=max(deltas, default=0.),
        metric_tolerance=dict(relative=1e-6, absolute=3e-5), all_final_predictions_recomputed=completed,
        metric_recomputation_groups=['raw_metrics', 'extra_metrics', 'benchmark_metrics_fps30', 'activity'],
        main_training=dict(before_step=main_before['latest_trace_step'], after_step=current_step,
            evaluation_start_step=baseline_step, latest_committed_step=main_after['latest_committed_step'],
            worker_pid=main_after['worker_pid'], worker_PID_present=True, goal_step=main_after['goal_step'],
            status=main_after['status'], checked_at=main_after['checked_at'], advancement_observed=True),
        source_array_scope='Published source metadata SHA and declared per-array/input hashes checked; large raw arrays were not rehashed or rebuilt. Saved GT is the reference for independent CPU metric recomputation.',
        scope='All persisted official RICH development VAL ego-grid windows; overlapping windows, not an independent TEST or an assumed external-paper benchmark protocol. No model forward, no CUDA, no training update.')
    write(OUT / 'VERIFY.json', result)
    write(OUT / ('VERIFY_COMPLETE.json' if completed else 'VERIFY_PARTIAL.json'), result)
    print(json.dumps(dict(status=result['status'], rows=len(rows), cpu_metric_rows=len(metric_checks),
                         sources=len(sources), max_metric_difference=result['maximum_absolute_float_difference'],
                         main_step=current_step, checked_at=result['checked_at'])), flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Verify the exact full-RICH official grid cohort before checkpoint/runner publication.')
    parser.add_argument('--watch', action='store_true', help='Verify once, then wait for completion without repeated heavy reads.')
    parser.add_argument('--wait-complete', action='store_true', help='Wait using only the standard library, then execute the full CPU verification.')
    args = parser.parse_args()
    try:
        if args.plan_only:
            result = dict(status='passed_plan_only', checked_at=now(), verifier_sha256=sha(__file__),
                          plan_sha256=sha(OUT / 'PLAN.json'), official_RICH_grid=inspect_plan(load(OUT / 'PLAN.json')),
                          model_forward_calls=0, CUDA_used=False, training_or_runner_files_written=False)
            write(OUT / 'VERIFY_PLAN.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return
        complete = False if args.wait_complete else verify(args.require_complete)
        if (args.watch or args.wait_complete) and not complete:
            while True:
                time.sleep(15)
                status = load(OUT / 'STATUS.json')
                if status['status'] == 'completed' and load(OUT / 'PROTOCOL.json')['status'] == 'completed':
                    # Run the final check in a fresh process so torch's thread
                    # initialization is performed once per process.
                    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), '--require-complete'])
                if status['status'] == 'failed':
                    raise RuntimeError('Evaluator failed before complete verification: ' + status.get('error', 'unknown'))
    except Exception as exc:
        failure = dict(status='verification_failed', checked_at=now(), error=repr(exc),
                       verifier_sha256=sha(__file__), training_or_runner_files_written=False)
        write(OUT / 'VERIFY_FAILURE.json', failure)
        print(json.dumps(failure), flush=True)
        raise


if __name__ == '__main__':
    main()
