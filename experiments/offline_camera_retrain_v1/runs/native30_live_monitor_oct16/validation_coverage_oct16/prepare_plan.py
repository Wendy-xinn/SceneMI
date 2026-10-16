"""Inventory existing VAL indices and prepare plans; never load a model."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
RUN = BASE / 'native30_gait_v2_ego55k_oct15'
SOURCE = BASE.parent


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = read_json(RUN / 'config.json')
    manifest = SOURCE / config['train_manifest']
    descriptor = SOURCE / config['window_index']
    spec = read_json(descriptor)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    entry = spec['files']['validation/ego']
    indexed = descriptor.parent / entry['file']
    assert digest(manifest) == spec['source_manifest_sha256']
    assert digest(indexed) == entry['sha256']
    windows = np.load(indexed, allow_pickle=False)
    assert windows.shape == (entry['windows'], 2)
    by_source, by_recording = {}, defaultdict(list)
    for ordinal in np.unique(windows[:, 0]):
        row = rows[int(ordinal)]
        assert row['split'] == 'validation' and 'ego' in row['allowed_modes']
        starts = windows[windows[:, 0] == ordinal, 1]
        recording = row.get('recording') or row['sequence_id']
        start = int(starts[len(starts) // 2])
        seed_key = json.dumps([row['group'], 'ego', row['sequence_id'], row['source_first'] + start], separators=(',', ':'))
        case = dict(dataset=row['dataset'], recording=recording, group=row['group'],
                    sequence_id=row['sequence_id'], source_bundle_ordinal=int(ordinal),
                    path=row['path'], start=start, source_first=row['source_first'] + start,
                    case_seed=(config['validation']['seed'] + int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], 'little')) % (2 ** 32),
                    eligible_windows=len(starts), mode='ego', joints=[15])
        by_source[int(ordinal)] = case
        by_recording[(row['dataset'], recording)].append(case)
    all_sources = list(by_source.values())
    all_recordings = [sorted(cases, key=lambda r: (r['group'], r['sequence_id']))[0]
                      for _, cases in sorted(by_recording.items())]
    val = [json.loads(line) for line in (RUN / 'training/validation.jsonl').read_text().splitlines()]
    feet = [json.loads(line) for line in (RUN / 'training/foot_monitor.jsonl').read_text().splitlines()]
    seconds = [r['validation_seconds'] for r in feet if r['step'] not in (1, 5000, 10000)]
    seconds_per_pair = float(np.median(seconds) / 6)
    observed = {(c['dataset'], c['sequence_id'], c['start']) for v in val for c in v['cases']
                if c.get('scene_variant', 'scene_on') == 'scene_on'}
    result = dict(prepared_at=datetime.now().astimezone().isoformat(), status='plans_prepared_not_evaluated',
                  model_forward_calls=0, cuda_used=False, training_changed=False, test_used=False,
                  source_manifest_sha256=digest(manifest), index_sha256=digest(indexed),
                  frames=spec['frames'], fps=30, stride=spec['window_stride'],
                  full_ego_windows=len(windows), eligible_bundles=len(by_source), eligible_recordings=len(by_recording),
                  windows_by_dataset=dict(Counter(rows[int(i)]['dataset'] for i in windows[:, 0])),
                  bundles_by_dataset=dict(Counter(c['dataset'] for c in all_sources)),
                  recordings_by_dataset=dict(Counter(c['dataset'] for c in all_recordings)),
                  latest_completed_validation=val[-1]['step'], cumulative_different_model_windows=len(observed),
                  current_monitoring_unique_windows=6, fixed_windows_used_for_trend=3,
                  observed_seconds_per_window_clean_noisy_pair=seconds_per_pair,
                  warm_timing_estimates_seconds=dict(full_pool=len(windows)*seconds_per_pair,
                      one_per_bundle=len(all_sources)*seconds_per_pair, one_per_recording=len(all_recordings)*seconds_per_pair),
                  timing_limitations='Linear estimate from current six-window DDIM20 monitoring; new-window cache preparation and same-GPU contention may increase time. No throughput benchmark launched.',
                  aggregation='Report dataset macro, recording macro within dataset, source/window means and percentiles; clean/noisy separate. Do not treat overlapping windows as independent recordings.',
                  interpretation='Plans cover every eligible recording or bundle, not every legal window. Median starts are selected without reading predictions or selecting favourable outcomes.',
                  checkpoint_policy='Execution must freeze one committed checkpoint and its SHA256 for all cases; no evaluation has started, no checkpoint copy made.',
                  one_per_recording=all_recordings, one_per_bundle=all_sources)
    (OUT / 'EVALUATION_PLAN.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('full_ego_windows','eligible_bundles','eligible_recordings','latest_completed_validation','warm_timing_estimates_seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
