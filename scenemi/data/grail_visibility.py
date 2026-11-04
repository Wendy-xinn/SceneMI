"""Screen GRAIL EGO scene observability using sealed first-hit statistics.

This does not rebuild motion, alter cameras, certify contacts, or prune the
canonical evaluation set. Output ranges require causal recording history.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

PROTOCOL = 'grail-ego-causal-visibility-filter-v1'
DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/grail_ego_visibility.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    if (policy['protocol'] != PROTOCOL or policy['frames'] != 128 or policy['fps'] != 30
            or policy['mode'] != 'ego' or policy['memory'] != 'same_recording_causal_static_history'
            or policy['image_size'] != [128, 96] or not policy['training_only_selection']
            or policy['physics_quality_checked']):
        raise ValueError('Unexpected EGO visibility-only protocol')
    if not 1 <= policy['min_ready_frames_per_window'] <= policy['frames']:
        raise ValueError('Invalid usable-frame requirement')
    for profile in policy['profiles'].values():
        if (not 0 < profile['min_object_pixel_fraction'] <= 1
                or not 1 <= profile['min_consecutive_frames'] <= profile['min_observed_frames']):
            raise ValueError('Invalid observability threshold')
    if policy['selected_profile'] not in policy['profiles']:
        raise ValueError('Missing selected profile')
    return policy


def index_ranges(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1]).tolist()


def eligibility(counts, profile, *, frames=128, ready_frames=16, pixels=128*96):
    """Compute causal scene availability and all stride-one window decisions.

    Repeated tiny hits are insufficient. Once enough meaningful views have
    accumulated, already observed STATIC surfaces remain available. This is
    a selection measure; it never supplies unobserved geometry to the model.
    """
    counts = np.asarray(counts, np.int64)
    if counts.ndim != 1 or (counts < 0).any() or (counts > pixels).any():
        raise ValueError('Invalid first-hit object pixel counts')
    threshold = int(np.ceil(profile['min_object_pixel_fraction'] * pixels))
    observed = counts >= threshold
    positions = np.arange(1, len(counts)+1)
    run = positions - np.maximum.accumulate(np.where(~observed, positions, 0))
    ready = ((observed.cumsum() >= profile['min_observed_frames'])
             & (np.maximum.accumulate(run) >= profile['min_consecutive_frames']))
    prefix = np.r_[0, ready.cumsum()]
    window_ready = prefix[frames:]-prefix[:-frames] if len(counts) >= frames else np.array([], int)
    eligible = window_ready >= ready_frames
    current_prefix = np.r_[0, observed.cumsum()]
    current_counts = (current_prefix[frames:]-current_prefix[:-frames]
                      if len(counts) >= frames else np.array([], int))
    return dict(pixel_threshold=threshold, observed_frames=int(observed.sum()),
                longest_observed_run=int(run.max(initial=0)),
                history_ready_frame=int(np.flatnonzero(ready)[0]) if ready.any() else None,
                eligible_start_ranges=index_ranges(eligible), eligible_windows=int(eligible.sum()),
                possible_windows=len(eligible),
                windows_with_no_current_but_known_history=int((eligible & (current_counts == 0)).sum()))


def scan(manifest, output, config=DEFAULT_CONFIG):
    """One bounded pass over completed metadata, verifying every identity."""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = load_policy(config)
    manifest_hash = sha256(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    source_hash = sha256(__file__)
    state = dict(protocol=PROTOCOL, state='scanning', updated_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest=str(manifest), manifest_sha256=manifest_hash, total=len(rows), processed=0,
        mode='ego', no_gpu=True, source_files_modified=False, physics_quality_checked=False,
        config=policy, config_sha256=sha256(config), implementation_sha256=source_hash)
    write_json(output/'STATUS.json', state)
    temporary = output/'visibility_statistics.jsonl.partial'
    seen, families, summary = set(), {}, defaultdict(lambda: defaultdict(Counter))
    peaks = defaultdict(list)
    started = time.monotonic()
    try:
        with temporary.open('w') as stream:
            for index, row in enumerate(rows):
                folder = Path(row['path'])
                if not folder.is_absolute():
                    folder = (manifest.parent/folder).resolve()
                raw = (folder/'metadata.json').read_bytes()
                if set(row) != {'path', 'metadata_sha256'} or hashlib.sha256(raw).hexdigest() != row['metadata_sha256']:
                    raise ValueError(f'Metadata identity mismatch: {folder}')
                meta = json.loads(raw)
                if (meta['sequence_id'] in seen or meta['fps'] != policy['fps'] or not meta['trainable']
                        or meta['dataset'] != 'grail' or meta['split'] not in ('train', 'validation', 'test')):
                    raise ValueError(f'Unexpected source identity: {folder}')
                seen.add(meta['sequence_id'])
                family, split = meta['object_family_id'], meta['split']
                if family in families and families[family] != split:
                    raise ValueError('Object family crosses source splits')
                families[family] = split
                camera = meta['virtual_camera']
                if ([camera['width'], camera['height']] != policy['image_size']
                        or camera['hfov_deg'] != 66.56 or camera['vfov_deg'] != 40.49
                        or camera['pitch_down_deg'] != 0 or camera['near_m'] != .05 or camera['far_m'] != 4):
                    raise ValueError('Mixed EGO camera protocols')
                stats = meta['primary_visibility_by_frame']
                if len(stats) != meta['frames']:
                    raise ValueError('Missing first-hit frames')
                for t, stat in enumerate(stats):
                    if (stat['frame'] != t or stat['rays'] != 128*96
                            or not 0 <= stat['object_above_ground'] <= stat['static'] <= stat['rays']
                            or stat['ground_y0'] + stat['object_above_ground'] != stat['static']):
                        raise ValueError(f'Invalid first-hit counts: {folder}, frame {t}')
                counts = [v['object_above_ground'] for v in stats]
                results = {key: eligibility(counts, profile, frames=policy['frames'],
                                            ready_frames=policy['min_ready_frames_per_window'])
                           for key, profile in policy['profiles'].items()}
                record = dict(path=str(folder), metadata_sha256=row['metadata_sha256'],
                    sequence_id=meta['sequence_id'], split=split, category=meta['category'],
                    object_family_id=family, frames=meta['frames'], object_pixels=counts, profiles=results)
                stream.write(json.dumps(record, separators=(',', ':'))+'\n')
                group = meta['category']
                peaks[split+'/'+group].append(max(counts))
                for key, result in results.items():
                    for section in (split+'/'+group, split+'/all'):
                        summary[key][section].update(sequences=1,
                            eligible_sequences=int(result['eligible_windows'] > 0),
                            possible_windows=result['possible_windows'], eligible_windows=result['eligible_windows'],
                            windows_preserved_by_history=result['windows_with_no_current_but_known_history'])
                if (index+1) % 250 == 0 or index+1 == len(rows):
                    stream.flush()
                    state.update(processed=index+1, elapsed_seconds=time.monotonic()-started,
                                 updated_at_utc=datetime.now(timezone.utc).isoformat(), current_sequence=meta['sequence_id'])
                    write_json(output/'STATUS.json', state)
                    print(json.dumps({k: state[k] for k in ('processed', 'total', 'elapsed_seconds')}), flush=True)
        if sha256(manifest) != manifest_hash or sha256(__file__) != source_hash:
            raise ValueError('Inputs or implementation changed during scan')
        statistics = output/'visibility_statistics.jsonl'
        temporary.replace(statistics)
        report = dict(protocol=PROTOCOL, config=policy, source_manifest_sha256=manifest_hash,
            count=len(rows), statistics_sha256=sha256(statistics),
            summary={k: dict(v) for k, v in summary.items()},
            peak_object_pixel_quantiles={k: np.quantile(v, [0, .1, .5, .9, 1]).tolist() for k, v in peaks.items()},
            quantile_order=['min', 'p10', 'median', 'p90', 'max'])
        write_json(output/'DISTRIBUTION.json', report)
        train = dict(report, summary={k: {g: v for g, v in groups.items() if g.startswith('train/')}
                                     for k, groups in report['summary'].items()},
                     peak_object_pixel_quantiles={k: v for k, v in report['peak_object_pixel_quantiles'].items()
                                                  if k.startswith('train/')})
        write_json(output/'TRAIN_DISTRIBUTION.json', train)
        state.update(state='scan_complete', statistics_sha256=report['statistics_sha256'],
                     completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'STATUS.json', state)
        return report
    except Exception as error:
        state.update(state='failed', error=repr(error), updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'STATUS.json', state)
        raise


def apply(output, config=DEFAULT_CONFIG):
    """Publish EGO training windows and evaluation coverage diagnostics."""
    output = Path(output).resolve()
    policy = load_policy(config)
    state = json.loads((output/'STATUS.json').read_text())
    stats = output/'visibility_statistics.jsonl'
    if state['state'] not in ('scan_complete', 'complete') or sha256(stats) != state['statistics_sha256']:
        raise ValueError('Full verified statistics are required')
    if policy != state['config'] or sha256(state['manifest']) != state['manifest_sha256']:
        raise ValueError('Policy or original manifest changed; use a separate scan output')
    profile = policy['selected_profile']
    (output/'train').mkdir(exist_ok=True)
    counts = Counter()
    paths = [output/'train/manifest.jsonl', output/'train/windows.jsonl',
             output/'train/excluded.jsonl', output/'evaluation_coverage_audit.jsonl']
    streams = [p.with_suffix(p.suffix+'.partial').open('w') for p in paths]
    try:
        with stats.open() as source:
            for line in source:
                row = json.loads(line)
                selected = row['profiles'][profile]
                decision = dict(path=row['path'], metadata_sha256=row['metadata_sha256'],
                    sequence_id=row['sequence_id'], split=row['split'], category=row['category'],
                    object_family_id=row['object_family_id'], mode='ego', frames=policy['frames'], fps=policy['fps'],
                    requires_causal_recording_history=True, physics_quality_checked=False,
                    profile=profile, **selected)
                decision['visibility_pass'] = selected['eligible_windows'] > 0
                decision['reason'] = ('sufficient_causal_object_observation' if decision['visibility_pass'] else
                    'no_non_ground_object_pixels' if not max(row['object_pixels']) else
                    'insufficient_object_observation_or_window_coverage')
                counts[row['split']+'_sequences'] += 1
                counts[row['split']+'_visibility_pass'] += int(decision['visibility_pass'])
                counts[row['split']+'_possible_windows'] += selected['possible_windows']
                counts[row['split']+'_eligible_windows'] += selected['eligible_windows']
                if row['split'] == 'train':
                    stream = streams[1] if decision['visibility_pass'] else streams[2]
                    stream.write(json.dumps(decision, separators=(',', ':'))+'\n')
                    if decision['visibility_pass']:
                        streams[0].write(json.dumps({k: row[k] for k in ('path', 'metadata_sha256')})+'\n')
                else:
                    streams[3].write(json.dumps(decision, separators=(',', ':'))+'\n')
        if sum(counts[s+'_sequences'] for s in ('train', 'validation', 'test')) != state['total']:
            raise ValueError('Incomplete selection')
    finally:
        for stream in streams:
            stream.close()
    for path in paths:
        path.with_suffix(path.suffix+'.partial').replace(path)
    state.update(state='complete', selected_profile=profile, counts=dict(counts),
        artifacts_sha256={str(p.relative_to(output)): sha256(p) for p in paths},
        selected_training_mode='ego', canonical_evaluation_unchanged=True, exo_selection_unchanged=True,
        formal_training_integrated=False, source_data_deleted=False,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'STATUS.json', state)
    return state
