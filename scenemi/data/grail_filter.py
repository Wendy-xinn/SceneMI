"""Versioned GRAIL EGO support screening; never modifies source observations.

Run this adapter directly while the root CLI and v1 modules remain sealed by
active jobs: python -m scenemi.data.grail_filter --help.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from .grail_chair import observe_seat, precheck, ready_windows
from .grail_contact import ContactBody, intersect_windows, load_source
from .grail_quality import contact_probabilities
from .grail_support import sole_measure, support_events
from .grail_visibility import index_ranges, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/'configs/grail_ego_support_v2.json'
PROTOCOL = 'grail-ego-support-filter-v2'


def load_policy(config):
    policy = json.loads((ROOT/'configs/grail_ego_contact.json').read_text())
    policy.update(json.loads(Path(config).read_text()))
    if (policy['protocol'] != PROTOCOL or policy['mode'] != 'ego' or policy['frames'] != 128
            or policy['fps'] != 30 or policy['source_data_deleted']
            or not policy['canonical_evaluation_unchanged']
            or policy['chair_observation_scope'] != 'before_sitting_causal_recording_history'):
        raise ValueError('Unexpected support selection protocol')
    for key in ('lower_foot_fraction', 'support_normal_y', 'minimum_supported_fraction', 'contact_probability'):
        if not 0 < policy[key] <= 1:
            raise ValueError(f'Invalid fraction: {key}')
    for key in ('support_gap_m', 'ray_origin_lift_m', 'stance_max_speed_mps',
                'chair_support_point_distance_m', 'chair_seat_height_band_m'):
        if not np.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError(f'Invalid distance/speed: {key}')
    if policy['ray_origin_lift_m'] <= policy['support_gap_m']:
        raise ValueError('Ray origin must be above the accepted support interval')
    for key in ('minimum_event_frames', 'chair_full_view_consecutive_frames',
                'chair_ready_frames_per_window', 'chair_min_support_points'):
        if not isinstance(policy[key], int) or not 1 <= policy[key] <= 128:
            raise ValueError(f'Invalid duration/count: {key}')
    return policy


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_baseline(baseline):
    baseline = Path(baseline).resolve()
    state = json.loads((baseline/'STATUS.json').read_text())
    path = baseline/'train/windows.jsonl'
    if (state['state'] != 'complete' or state['protocol'] != 'grail-ego-contact-filter-v1'
            or sha256(path) != state['artifacts_sha256']['train/windows.jsonl']):
        raise ValueError('A complete, unchanged v1 contact selection is required')
    rows = read_rows(path)
    if (len({r['sequence_id'] for r in rows}) != len(rows)
            or any(r['split'] != 'train' or r['mode'] != 'ego' or not r['eligible_windows'] for r in rows)):
        raise ValueError('Baseline must contain unique, retained TRAIN/EGO sources')
    return rows


def metadata(row):
    folder = Path(row['path'])
    if sha256(folder/'metadata.json') != row['metadata_sha256']:
        raise ValueError('Source metadata identity changed')
    meta = dict(json.loads((folder/'metadata.json').read_text()), path=str(folder))
    if (meta['sequence_id'] != row['sequence_id'] or meta['category'] != row['category']
            or meta['split'] != 'train' or meta['fps'] != 30):
        raise ValueError('Source manifest disagrees with metadata')
    return meta


def evaluate(row, body, policy, diagnostics=False):
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    from experiments.offline_camera_retrain_v1.ops.grail_source import load_pickle
    meta = metadata(row)
    n = meta['frames']
    bad = np.zeros(n, bool)
    for a, b in row['bad_frame_ranges']:
        bad[a:b+1] = True
    chair = None
    events, arrays = {}, {}
    eligible = row['eligible_start_ranges']
    needs_body = True
    if row['category'] == 'sitting':
        chair, candidates, seated = precheck(meta, policy)
        needs_body = bool(candidates.any())
    joint_error = None
    if needs_body:
        meta, params, rotation, offset = load_source(row)
        vertices, joints = body.forward(params, rotation, offset, policy['chunk_size'])
        expected = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3)*2
        joint_error = float(np.abs(joints-expected).max())
        if joint_error > 2e-6:
            raise ValueError(f'Joint export mismatch: {joint_error}')
    if row['category'] == 'sitting':
        counts, height = (observe_seat(meta, vertices, body, seated, candidates, policy)
                          if needs_body else (np.zeros(n, int), None))
        good = candidates & (counts >= policy['chair_min_support_points'])
        ready, chair_windows = ready_windows(good, chair['first_stable_sit_frame'], policy)
        chair.update(seat_support_height_m=height, actual_seat_sample_count_max=int(counts.max()),
                     observed_seat_full_outline_ranges=index_ranges(good), history_ready_frame=ready,
                     sample_count_semantics='Actual first-hit static samples out of 512/frame, not pixels')
        allowed = np.zeros(len(chair_windows), bool)
        for a, b in eligible:
            allowed[a:b+1] = True
        eligible = index_ranges(allowed & chair_windows)
        if ready is None:
            events['chair_no_qualified_pre_sit_observation'] = [[0, n-1]]
        arrays = dict(chair_full_outline=candidates, observed_seat_samples=counts,
                      observed_seat_full_outline=good, seated=seated)
    else:
        v = readonly_array(meta, 'scene_object_vertices')
        f = readonly_array(meta, 'scene_object_faces')
        gv = readonly_array(meta, 'scene_ground_vertices')
        gf = readonly_array(meta, 'scene_ground_faces')
        if sha256(meta['source_files']['recon']) != meta['source_hashes']['recon']:
            raise ValueError('Source estimated contact identity changed')
        raw = load_pickle(meta['source_files']['recon'])
        prob = contact_probabilities(raw['human_data']['foot_contact_probs'], n, meta['source_fps'], meta['fps'])
        arrays = sole_measure(vertices, body.feet, np.concatenate([v, gv]),
                              np.concatenate([f, gf+len(v)]), policy)
        support, support_bad = support_events(arrays, prob, row['category'], policy)
        events = support['events']
        bad |= support_bad
        arrays.update(source_estimated_contact_probability=prob, support_bad=support_bad)
    ranges, count = intersect_windows(eligible, bad, policy['frames'])
    result = {k: row[k] for k in ('path', 'metadata_sha256', 'sequence_id', 'category', 'split', 'object_family_id')}
    result.update(protocol=PROTOCOL, mode='ego', frames=n, fps=30, window_frames=128,
                  baseline_windows=row['eligible_windows'], baseline_start_ranges=row['eligible_start_ranges'],
                  eligible_start_ranges=ranges, eligible_windows=count, events=events,
                  baseline_events=row['events'], bad_frame_ranges=index_ranges(bad),
                  exported_joint_max_error_m=joint_error, chair_observation=chair,
                  requires_causal_recording_history=True, physics_validity_proven=False,
                  screen_result='pass_screen' if count else 'withheld_for_review')
    if row['category'] != 'sitting':
        result['support'] = {k: v for k, v in support.items() if k not in ('events', 'bad_frame_ranges')}
    if diagnostics:
        arrays['combined_bad'] = bad
        return result, arrays
    return result


def dependency_seal(config, baseline, selection=None):
    paths = [Path(__file__), Path(config), ROOT/'configs/grail_ego_contact.json']
    paths += [Path(__file__).with_name(name) for name in
              ('grail_chair.py', 'grail_support.py', 'grail_contact.py', 'grail_quality.py', 'grail_visibility.py')]
    ops = ROOT/'experiments/offline_camera_retrain_v1'
    paths += [ops/'ops/grail_source.py', ops/'ops/viewer_geometry.py', ops/'scene_visibility_v2.py',
              ROOT/'body_models/grail_g1_smplx/g1_smplx_tpose.obj',
              ROOT/'body_models/grail_g1_smplx/g1_smplx_param.npz',
              ROOT/'body_models/smplx/SMPLX_NEUTRAL.npz',
              Path(baseline)/'train/windows.jsonl', Path(baseline)/'JOB.json']
    if selection is not None:
        paths.append(Path(selection))
    import smplx
    paths += [Path(smplx.__file__), Path(smplx.__file__).with_name('lbs.py'),
              Path(smplx.__file__).with_name('body_models.py')]
    return {str(p.resolve()): sha256(p) for p in paths}


def resume_ledger(path, rows):
    completed = {}
    by_id = {r['sequence_id']: r for r in rows}
    if not path.exists():
        return completed
    with path.open('rb+') as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if handle.read():
                    raise ValueError('Corrupt ledger before final line')
                handle.truncate(position)
                break
            name = record['sequence_id']
            if (name in completed or name not in by_id
                    or record['metadata_sha256'] != by_id[name]['metadata_sha256']):
                raise ValueError('Ledger source identity changed')
            if not line.endswith(b'\n'):
                handle.write(b'\n')
            completed[name] = record
    return completed


def publish(output, rows, completed):
    (output/'train').mkdir(exist_ok=True)
    (output/'augmentation').mkdir(exist_ok=True)
    summary, reasons = defaultdict(Counter), Counter()
    names = ['train/manifest.jsonl', 'train/windows.jsonl', 'train/excluded.jsonl',
             'augmentation/sitting_candidates.jsonl']
    streams = [(output/(name+'.partial')).open('w') for name in names]
    try:
        for row in rows:
            result = completed[row['sequence_id']]
            retained = bool(result['eligible_windows'])
            for category in (row['category'], 'all'):
                summary[category].update(baseline_sequences=1, retained_sequences=int(retained),
                    baseline_windows=row['eligible_windows'], retained_windows=result['eligible_windows'])
            reasons.update(result['events'].keys())
            streams[1 if retained else 2].write(json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n')
            if retained:
                streams[0].write(json.dumps({k: row[k] for k in ('path', 'metadata_sha256')})+'\n')
            if row['category'] == 'sitting':
                candidate = dict(row, condition_provenance='proposed_synthetic_chair_observation',
                    input_materialized=False, formal_training_integrated=False,
                    real_observation_qualified=retained,
                    use='Training augmentation candidates only; original v1 contact-window restrictions still apply; not a real-visibility or evaluation manifest')
                streams[3].write(json.dumps(candidate, allow_nan=False, separators=(',', ':'))+'\n')
    finally:
        for stream in streams:
            stream.close()
    for name in names:
        (output/(name+'.partial')).replace(output/name)
    return dict(summary), dict(reasons), {name: sha256(output/name) for name in names}


def scan(baseline, output, config=DEFAULT_CONFIG, workers=2, selection=None):
    import torch
    torch.set_num_threads(1)
    if not 1 <= workers <= min(4, len(os.sched_getaffinity(0))):
        raise ValueError('Workers exceed the effective CPU budget')
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = load_policy(config)
    rows = load_baseline(baseline)
    if selection is not None:
        ids = [r['sequence_id'] for r in read_rows(selection)]
        lookup = {r['sequence_id']: r for r in rows}
        if len(set(ids)) != len(ids) or any(name not in lookup for name in ids):
            raise ValueError('Pilot selection must contain unique v1 retained training sources')
        rows = [lookup[name] for name in ids]
    if not rows:
        raise ValueError('Empty source selection')
    seal = dependency_seal(config, baseline, selection)
    job = dict(protocol=PROTOCOL, config=policy, dependencies_sha256=seal,
               baseline=str(baseline), diagnostic_only=selection is not None, sources=len(rows))
    if (output/'JOB.json').exists() and json.loads((output/'JOB.json').read_text()) != job:
        raise ValueError('Sealed job changed; choose a new output root')
    write_json(output/'JOB.json', job)
    ledger_path = output/'RECORDS.jsonl'
    completed = resume_ledger(ledger_path, rows)
    body = ContactBody(policy)
    _, params, rotation, offset = load_source(rows[0])
    write_json(output/'FK_EQUIVALENCE.json', body.validate_full(params, rotation, offset))
    started, initial = time.monotonic(), len(completed)
    state = dict(protocol=PROTOCOL, state='running', pid=os.getpid(), total=len(rows), completed=initial,
                 config=policy, no_gpu=True, workers=workers, failures=0,
                 source_data_deleted=False, canonical_evaluation_unchanged=True,
                 exo_selection_unchanged=True, diagnostic_only=selection is not None,
                 formal_training_integrated=False, updated_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'STATUS.json', state)
    def one(row):
        if selection is not None:
            result, arrays = evaluate(row, body, policy, diagnostics=True)
            np.savez_compressed(output/(row['sequence_id']+'.npz'), **arrays)
            return result
        return evaluate(row, body, policy)
    try:
        pending = iter(row for row in rows if row['sequence_id'] not in completed)
        with ledger_path.open('a') as ledger, ThreadPoolExecutor(max_workers=workers) as pool:
            active = {pool.submit(one, row): row for row in
                      [next(pending, None) for _ in range(workers)] if row is not None}
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    row = active.pop(future)
                    result = future.result()
                    ledger.write(json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n')
                    ledger.flush()
                    completed[row['sequence_id']] = result
                    state.update(completed=len(completed), elapsed_seconds=time.monotonic()-started,
                                 current_sequence=row['sequence_id'], updated_at_utc=datetime.now(timezone.utc).isoformat())
                    if len(completed) % 25 == 0 or len(completed) == len(rows):
                        elapsed = max(time.monotonic()-started, 1e-6)
                        state['eta_seconds'] = ((len(rows)-len(completed))*elapsed/max(len(completed)-initial, 1))
                        write_json(output/'STATUS.json', state)
                        print(json.dumps({k: state[k] for k in ('completed', 'total', 'elapsed_seconds', 'eta_seconds')}), flush=True)
                    row = next(pending, None)
                    if row is not None:
                        active[pool.submit(one, row)] = row
        if len(completed) != len(rows) or any(sha256(p) != digest for p, digest in seal.items()):
            raise ValueError('Incomplete scan or sealed dependency changed')
        summary, reasons, artifacts = publish(output, rows, completed)
        state.update(state='complete', summary=summary, sequences_flagged_by_event=reasons,
                     artifacts_sha256=artifacts, records_sha256=sha256(ledger_path), physics_validity_proven=False)
        write_json(output/'STATUS.json', state)
        return state
    except Exception as error:
        state.update(state='failed', failures=1, error=repr(error), updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'STATUS.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--selection', type=Path, help='Diagnostic source-id JSONL subset; never a formal selection')
    args = parser.parse_args()
    print(json.dumps(scan(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
