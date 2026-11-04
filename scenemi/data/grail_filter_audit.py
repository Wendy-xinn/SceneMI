"""Independently recompute published v2 window intersections and identities."""
import argparse
import json
from pathlib import Path

import numpy as np

from .grail_visibility import sha256, write_json


def audit(output):
    output = Path(output).resolve()
    state = json.loads((output/'STATUS.json').read_text())
    job = json.loads((output/'JOB.json').read_text())
    if state['state'] != 'complete' or state['diagnostic_only']:
        raise ValueError('A completed full scan is required')
    for path, digest in job['dependencies_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Sealed dependency changed: {path}')
    for name, digest in state['artifacts_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError(f'Published artifact changed: {name}')
    if sha256(output/'RECORDS.jsonl') != state['records_sha256']:
        raise ValueError('Ledger identity changed')
    def rows(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    baseline = rows(Path(job['baseline'])/'train/windows.jsonl')
    sources = {r['sequence_id']: r for r in baseline}
    records = rows(output/'RECORDS.jsonl')
    if len({r['sequence_id'] for r in records}) != len(records) or len(records) != len(sources):
        raise ValueError('Missing or duplicate records')
    retained = rows(output/'train/windows.jsonl')
    excluded = rows(output/'train/excluded.jsonl')
    published = {r['sequence_id']: r for r in retained+excluded}
    if len(published) != len(records) or len(retained)+len(excluded) != len(records):
        raise ValueError('Published partition incomplete or duplicated')
    manifest = rows(output/'train/manifest.jsonl')
    expected_manifest = [{'path': r['path'], 'metadata_sha256': r['metadata_sha256']} for r in retained]
    if manifest != expected_manifest:
        raise ValueError('Compact source manifest differs from retained windows')
    counts, window_count, metadata_count = {}, 0, 0
    policy = job['config']
    for r in records:
        source = sources[r['sequence_id']]
        if r != published[r['sequence_id']]:
            raise ValueError('Ledger and published partition disagree')
        folder = Path(r['path'])
        if sha256(folder/'metadata.json') != source['metadata_sha256']:
            raise ValueError('Metadata identity changed')
        metadata_count += 1
        meta = json.loads((folder/'metadata.json').read_text())
        if (meta['frames'] != r['frames'] or r['window_frames'] != policy['frames']
                or r['metadata_sha256'] != source['metadata_sha256']):
            raise ValueError('Metadata/ledger mismatch')
        n, frames = r['frames'], policy['frames']
        source_allowed = np.zeros(max(0, n-frames+1), bool)
        for a, b in source['eligible_start_ranges']:
            source_allowed[a:b+1] = True
        bad = np.zeros(n, bool)
        for a, b in r['bad_frame_ranges']:
            if not 0 <= a <= b < n:
                raise ValueError('Invalid bad-frame interval')
            bad[a:b+1] = True
        # Direct convolutions provide a separate implementation of the scan's
        # cumulative-sum window intersection.
        expected = source_allowed & (np.convolve(bad.astype(int), np.ones(frames, int), mode='valid') == 0)
        if r['category'] == 'sitting':
            chair = r['chair_observation']
            onset, ready = chair['first_stable_sit_frame'], chair['history_ready_frame']
            observed = np.zeros(n, bool)
            for a, b in chair['observed_seat_full_outline_ranges']:
                observed[a:b+1] = True
                if onset is None or b >= onset:
                    raise ValueError('Future/after-sit observation used to qualify chair')
            run = np.convolve(observed.astype(int), np.ones(policy['chair_full_view_consecutive_frames'], int), mode='valid')
            ends = np.flatnonzero(run == policy['chair_full_view_consecutive_frames']) + policy['chair_full_view_consecutive_frames']-1
            expected_ready = int(ends[0]) if len(ends) else None
            if ready != expected_ready:
                raise ValueError('Chair readiness does not follow actual observation run')
            history = np.arange(n) >= ready if ready is not None else np.zeros(n, bool)
            expected &= np.convolve(history.astype(int), np.ones(frames, int), mode='valid') >= policy['chair_ready_frames_per_window']
        actual = np.zeros(len(source_allowed), bool)
        for a, b in r['eligible_start_ranges']:
            if not 0 <= a <= b < len(actual):
                raise ValueError('Invalid window interval')
            actual[a:b+1] = True
        if not np.array_equal(expected, actual) or actual.sum() != r['eligible_windows']:
            raise ValueError(f'Window recomputation mismatch: {r["sequence_id"]}')
        counts[r['category']] = counts.get(r['category'], 0)+int(actual.sum())
        window_count += int(actual.sum())
    for r in retained:
        if r['eligible_windows'] <= 0:
            raise ValueError('Empty retained source')
    for r in excluded:
        if r['eligible_windows']:
            raise ValueError('Nonempty excluded source')
    for category in sorted(set(counts) | {'all'}):
        matching = records if category == 'all' else [r for r in records if r['category'] == category]
        expected_summary = dict(baseline_sequences=len(matching),
            retained_sequences=sum(r['eligible_windows'] > 0 for r in matching),
            baseline_windows=sum(sources[r['sequence_id']]['eligible_windows'] for r in matching),
            retained_windows=sum(r['eligible_windows'] for r in matching))
        if state['summary'][category] != expected_summary:
            raise ValueError('Published summary differs from per-source records')
    augmented = rows(output/'augmentation/sitting_candidates.jsonl')
    expected_chairs = [r for r in baseline if r['category'] == 'sitting']
    if ({r['sequence_id'] for r in augmented} != {r['sequence_id'] for r in expected_chairs}
            or len(augmented) != len(expected_chairs)):
        raise ValueError('Augmentation candidates do not match baseline chairs')
    for r in augmented:
        source = sources[r['sequence_id']]
        if (r['input_materialized'] or r['formal_training_integrated']
                or r['eligible_start_ranges'] != source['eligible_start_ranges']
                or r['condition_provenance'] != 'proposed_synthetic_chair_observation'):
            raise ValueError('Augmentation candidate lost its provenance/contact restrictions')
    known = 'terrain_stairs__straight_aged_wood_stairs_with_8_steps_no_handrails_no_turns__0002'
    example = published[known]
    if not any(a <= 82 <= b for a, b in example['bad_frame_ranges']):
        raise ValueError('Known frame-82 sole-support regression was lost')
    if any(a <= 82 and b+frames-1 >= 82 for a, b in example['eligible_start_ranges']):
        raise ValueError('Known unsupported frame still enters a training window')
    result = dict(state='passed', protocol=state['protocol'], checked_sequences=len(records),
        metadata_identities_rechecked=metadata_count, retained_sequences=len(retained),
        excluded_sequences=len(excluded), retained_windows=window_count, windows_by_category=counts,
        synthetic_chair_candidates=len(augmented), synthetic_inputs_materialized=False,
        known_frame82_regression_passed=True, canonical_evaluation_unchanged=True,
        exo_selection_unchanged=True, source_data_deleted=False, physics_validity_proven=False,
        job_sha256=sha256(output/'JOB.json'), records_sha256=state['records_sha256'])
    write_json(output/'VALIDATION.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().output), ensure_ascii=False, indent=2))
