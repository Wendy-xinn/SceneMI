"""Fixed GRAIL held-out geometry QC, independent of camera observability.

This creates a versioned quality subset, never rewrites canonical splits or
historical validation lists, and never loads a learned model. Thresholds come
directly from the existing training contact/support configurations.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from .grail_contact import ContactBody, intersect_windows, load_source, measure
from .grail_support import sole_measure, support_events
from .grail_visibility import index_ranges, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'grail-heldout-contact-support-v1'
DEFAULT_CONFIG = ROOT/'configs/grail_split_quality.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
    temporary.replace(path)


def load_policy(config):
    settings = json.loads(Path(config).read_text())
    if (settings['protocol'] != PROTOCOL or settings['splits'] != ['validation', 'test']
            or settings['frames'] != 128 or settings['fps'] != 30
            or not settings['source_splits_unchanged']
            or not settings['historical_evaluation_unchanged'] or settings['training_integration']):
        raise ValueError('Unexpected held-out quality protocol')
    contact = json.loads((ROOT/settings['contact_config']).read_text())
    support = json.loads((ROOT/settings['support_config']).read_text())
    if (contact['protocol'] != 'grail-ego-contact-filter-v1'
            or support['protocol'] != 'grail-ego-support-filter-v2'
            or any(p['frames'] != 128 or p['fps'] != 30 for p in (contact, support))):
        raise ValueError('Unexpected frozen training rules')
    return settings, contact, dict(contact, **support)


def discover(manifest, splits):
    """Verify source metadata and family separation before opening arrays."""
    result, seen, families = [], set(), {}
    source_counts = defaultdict(Counter)
    for row in read_rows(manifest):
        folder = Path(row['path'])
        if not folder.is_absolute():
            folder = (Path(manifest).parent/folder).resolve()
        raw = (folder/'metadata.json').read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != row['metadata_sha256']:
            raise ValueError(f'Source metadata identity changed: {folder}')
        meta = json.loads(raw)
        split, family, name = meta['split'], meta['object_family_id'], meta['sequence_id']
        if (split not in ('train', 'validation', 'test') or name in seen
                or meta['dataset'] != 'grail' or meta['fps'] != 30 or not meta['trainable']):
            raise ValueError(f'Unexpected source identity: {folder}')
        if family in families and families[family] != split:
            raise ValueError(f'Object family crosses source splits: {family}')
        seen.add(name)
        families[family] = split
        source_counts[split].update(sequences=1, frames=meta['frames'])
        if split in splits:
            result.append(dict(path=str(folder), metadata_sha256=row['metadata_sha256'],
                sequence_id=name, category=meta['category'], split=split,
                object_family_id=family, frames=meta['frames']))
    if set(splits) != {r['split'] for r in result}:
        raise ValueError('Requested split is missing')
    return sorted(result, key=lambda r: (r['split'], r['sequence_id'])), dict(
        source_counts=dict(source_counts), source_family_split_intersection_empty=True,
        source_sequence_ids_unique=True, metadata_identities_verified=len(seen))


def usable_frames(ranges, frames, length):
    """Count the union of frame spans, never join disjoint time intervals."""
    spans = []
    previous = -1
    for a, b in ranges:
        if not 0 <= a <= b <= length-frames or a <= previous:
            raise ValueError('Accepted start range outside source sequence')
        previous = b
        end = b+frames  # exclusive
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([a, end])
    return sum(b-a for a, b in spans)


def evaluate(row, body, contact, support):
    from experiments.offline_camera_retrain_v1.ops.viewer_geometry import readonly_array
    meta, params, rotation, offset = load_source(row)
    n = row['frames']
    if any(meta[k] != row[k] for k in ('split', 'sequence_id', 'category', 'frames', 'object_family_id')):
        raise ValueError('Sequence metadata disagrees with sealed selection')
    if n < 2:
        raise ValueError('Recording too short for contact velocity measurement')
    vertices, joints = body.forward(params, rotation, offset, contact['chunk_size'])
    expected = readonly_array(meta, 'motion_world')[:, -66:].reshape(-1, 22, 3)*2
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all() or not np.isfinite(expected).all():
        raise ValueError('Non-finite clean source geometry')
    joint_error = float(np.abs(joints-expected).max())
    if joint_error > 2e-6:
        raise ValueError(f'Joint export mismatch: {joint_error}')
    contact_record, arrays = measure(meta, vertices, joints, body, contact)
    bad = arrays['bad'].copy()
    support_record = None
    if row['category'] != 'sitting':
        v = readonly_array(meta, 'scene_object_vertices')
        f = readonly_array(meta, 'scene_object_faces')
        gv = readonly_array(meta, 'scene_ground_vertices')
        gf = readonly_array(meta, 'scene_ground_faces')
        measured = sole_measure(vertices, body.feet, np.concatenate([v, gv]),
                                np.concatenate([f, gf+len(v)]), support)
        support_record, support_bad = support_events(measured,
            arrays['source_estimated_contact_probability'], row['category'], support)
        bad |= support_bad
    valid = np.asarray(readonly_array(meta, 'observation_valid'))
    if valid.shape != (n,):
        raise ValueError(f'Unexpected source frame-validity shape: {valid.shape}')
    bad |= ~valid.astype(bool)
    ranges, count = intersect_windows([[0, n-128]] if n >= 128 else [], bad, 128)
    result = dict(row, protocol=PROTOCOL, mode='shared_motion_quality', fps=30,
        window_frames=128, eligible_start_ranges=ranges, eligible_windows=count,
        usable_frames=usable_frames(ranges, 128, n), bad_frame_ranges=index_ranges(bad),
        invalid_source_frame_ranges=index_ranges(~valid.astype(bool)),
        contact=contact_record, support=support_record,
        exported_joint_max_error_m=joint_error, physics_validity_proven=False,
        screen_result='pass_screen' if count else 'withheld_for_review',
        visibility_used_for_selection=False, synthetic_chair_added=False)
    counts = np.asarray([s['object_above_ground'] for s in meta['primary_visibility_by_frame']])
    if len(counts) != n:
        raise ValueError('Missing EGO visibility diagnostic frames')
    result['ego_visibility_diagnostic'] = dict(object_visible_frames=int((counts > 0).sum()),
        object_at_least_one_percent_frames=int((counts >= 123).sum()), used_for_selection=False)
    return result


def dependency_seal(config, manifest, settings):
    paths = [Path(config), Path(manifest), Path(__file__), ROOT/settings['contact_config'],
             ROOT/settings['support_config'], ROOT/settings['paths_config']]
    paths += [Path(__file__).with_name(n) for n in
              ('grail_contact.py', 'grail_support.py', 'grail_quality.py', 'grail_visibility.py')]
    ops = ROOT/'experiments/offline_camera_retrain_v1'
    paths += [ops/'ops/grail_source.py', ops/'ops/viewer_geometry.py', ops/'scene_visibility_v2.py',
              ROOT/'body_models/grail_g1_smplx/g1_smplx_tpose.obj',
              ROOT/'body_models/grail_g1_smplx/g1_smplx_param.npz',
              ROOT/'body_models/smplx/SMPLX_NEUTRAL.npz']
    import smplx
    paths += [Path(smplx.__file__), Path(smplx.__file__).with_name('lbs.py'),
              Path(smplx.__file__).with_name('body_models.py')]
    return {str(p.resolve()): sha256(p) for p in paths}


def resume_records(path, rows):
    known = {r['sequence_id']: r for r in rows}
    completed = {}
    if not path.exists():
        return completed
    for result in read_rows(path):
        name = result['sequence_id']
        if name not in known or name in completed or any(result[k] != v for k, v in known[name].items()):
            raise ValueError('Corrupt or incompatible quality ledger')
        expected = usable_frames(result['eligible_start_ranges'], 128, result['frames'])
        if (result['protocol'] != PROTOCOL or result['usable_frames'] != expected
                or result['eligible_windows'] != sum(b-a+1 for a, b in result['eligible_start_ranges'])):
            raise ValueError('Invalid quality ledger frame counts')
        completed[name] = result
    return completed


def publish(output, rows, completed):
    summary, categories, reasons, artifacts = {}, {}, {}, {}
    for split in sorted({r['split'] for r in rows}):
        selected, excluded = [], []
        counts, groups, events = Counter(), defaultdict(Counter), Counter()
        for source in rows:
            if source['split'] != split:
                continue
            r = completed[source['sequence_id']]
            keep = bool(r['eligible_windows'])
            (selected if keep else excluded).append(r)
            contribution = dict(before_sequences=1, before_frames=r['frames'],
                retained_sequences=int(keep), retained_sequence_frames=r['frames']*int(keep),
                usable_frames=r['usable_frames'], flagged_frames=sum(b-a+1 for a, b in r['bad_frame_ranges']))
            counts.update(contribution)
            groups[r['category']].update(contribution)
            for layer in ('contact', 'support'):
                if r[layer]:
                    events.update(r[layer]['events'].keys())
            if r['invalid_source_frame_ranges']:
                events.update(['invalid_source_frame'])
        summary[split], categories[split], reasons[split] = dict(counts), dict(groups), dict(events)
        for name, content in [('selection.jsonl', selected), ('excluded.jsonl', excluded),
                ('manifest.jsonl', [{k: r[k] for k in ('path', 'metadata_sha256')} for r in selected])]:
            relative = split+'/'+name
            write_rows(output/relative, content)
            artifacts[relative] = sha256(output/relative)
    return summary, categories, reasons, artifacts


def scan(output, config=DEFAULT_CONFIG, workers=2, pilot_per_category=0):
    import torch
    import yaml
    torch.set_num_threads(1)
    if not 1 <= workers <= min(4, len(os.sched_getaffinity(0))):
        raise ValueError('Workers exceed effective CPU budget')
    settings, contact, support = load_policy(config)
    paths = yaml.safe_load((ROOT/settings['paths_config']).read_text())
    manifest = Path(paths['grail_manifest'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'scan.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        state = dict(protocol=PROTOCOL, state='checking_source_metadata', pid=os.getpid(),
            updated_at_utc=now(), no_gpu=True, workers=workers, completed=0,
            source_data_modified=False, source_splits_unchanged=True,
            historical_evaluation_unchanged=True, training_integrated=False,
            test_model_inference=False, test_threshold_tuning=False,
            test_source_arrays_opened=False, diagnostic_only=bool(pilot_per_category))
        write_json(output/'STATUS.json', state)
        try:
            rows, metadata_audit = discover(manifest, settings['splits'])
            if pilot_per_category:
                used, subset = Counter(), []
                for row in rows:
                    key = row['split'], row['category']
                    if used[key] < pilot_per_category:
                        subset.append(row)
                        used[key] += 1
                rows = subset
            seal = dependency_seal(config, manifest, settings)
            job = dict(protocol=PROTOCOL, settings=settings, contact_rules=contact, support_rules=support,
                dependencies_sha256=seal, metadata_audit=metadata_audit,
                source_manifest=str(manifest), rows=rows, pilot_per_category=pilot_per_category)
            if (output/'JOB.json').exists() and json.loads((output/'JOB.json').read_text()) != job:
                raise ValueError('Sealed job changed; choose a new output directory')
            write_json(output/'JOB.json', job)
            ledger_path = output/'RECORDS.jsonl'
            completed = resume_records(ledger_path, rows)
            initial = len(completed)
            body = ContactBody(contact)
            _, params, rotation, offset = load_source(rows[0])
            state['test_source_arrays_opened'] = rows[0]['split'] == 'test'
            write_json(output/'FK_EQUIVALENCE.json', body.validate_full(params, rotation, offset))
            state.update(state='running', total=len(rows), completed=initial, metadata_audit=metadata_audit)
            write_json(output/'STATUS.json', state)
            pending = iter(row for row in rows if row['sequence_id'] not in completed)
            with ledger_path.open('a') as ledger, ThreadPoolExecutor(max_workers=workers) as pool:
                active = {}
                def submit(row):
                    state['test_source_arrays_opened'] |= row['split'] == 'test'
                    active[pool.submit(evaluate, row, body, contact, support)] = row
                for _ in range(workers):
                    row = next(pending, None)
                    if row is not None:
                        submit(row)
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        row = active.pop(future)
                        record = future.result()
                        ledger.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                        ledger.flush()
                        completed[row['sequence_id']] = record
                        state.update(completed=len(completed), current_sequence=row['sequence_id'],
                            elapsed_seconds=time.monotonic()-started, updated_at_utc=now())
                        if len(completed) % 25 == 0 or len(completed) == len(rows):
                            state['eta_seconds'] = ((len(rows)-len(completed))*state['elapsed_seconds']
                                                   /max(1, len(completed)-initial))
                            write_json(output/'STATUS.json', state)
                            print(json.dumps({k:state[k] for k in ('completed', 'total', 'elapsed_seconds', 'eta_seconds')}), flush=True)
                        row = next(pending, None)
                        if row is not None:
                            submit(row)
            if len(completed) != len(rows) or any(sha256(p) != digest for p, digest in seal.items()):
                raise ValueError('Incomplete audit or sealed dependency changed')
            for row in rows:
                if sha256(Path(row['path'])/'metadata.json') != row['metadata_sha256']:
                    raise ValueError('Source metadata changed during audit')
            summary, categories, reasons, artifacts = publish(output, rows, completed)
            state.update(state='complete', summary=summary, by_category=categories,
                sequences_flagged_by_event=reasons, artifacts_sha256=artifacts,
                records_sha256=sha256(ledger_path), updated_at_utc=now(),
                physics_validity_proven=False, elapsed_seconds=time.monotonic()-started)
            write_json(output/'STATUS.json', state)
            return state
        except Exception as error:
            state.update(state='failed', error=repr(error), updated_at_utc=now())
            write_json(output/'STATUS.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pilot-per-category', type=int, default=0)
    args = parser.parse_args()
    if args.pilot_per_category < 0:
        parser.error('Pilot sample count cannot be negative')
    result = scan(**vars(args))
    print(json.dumps({k:result[k] for k in ('state', 'summary', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
