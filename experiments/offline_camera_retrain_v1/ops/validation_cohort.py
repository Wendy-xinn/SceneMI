"""Freeze a larger EGO VAL review: full indexed RICH and recording-balanced peers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROTOCOL = 'native30-expanded-checkpoint-review-ego-v1'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(seed, *identity):
    return hashlib.sha256(json.dumps([seed, *identity], separators=(',', ':')).encode()).hexdigest()


def select_recording_balanced(entries, count, frames, seed):
    """Identity-only selection, one recording per round; no model/GT outcomes."""
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['recording']].append(entry)
    require(0 < count <= len(entries), 'Requested cohort exceeds legal windows')
    order = sorted(groups, key=lambda name: rank(seed, name))
    chosen, per_record, roles = [], defaultdict(list), Counter()
    while len(chosen) < count:
        progress = False
        for name in order:
            available = [entry for entry in groups[name] if (entry['path'], entry['start']) not in
                         {(old['path'], old['start']) for old in chosen}]
            if not available:
                continue
            old = per_record[name]
            # Compare the source recording clock across overlapping role/bundle
            # scopes too, rather than calling different crops independent.
            distance = lambda entry: min((abs(entry['source_first'] - value['source_first']) for value in old), default=frames)
            nonoverlapping = [entry for entry in available if distance(entry) >= frames]
            pool = nonoverlapping or available
            centre = (min(e['source_first'] for e in groups[name]) + max(e['source_first'] for e in groups[name])) / 2.
            def priority(entry):
                temporal = -distance(entry) if old else abs(entry['source_first'] - centre)
                return roles[entry['group']], temporal, rank(seed, entry['path'], entry['start'])
            entry = min(pool, key=priority)
            chosen.append(dict(entry, selection_round=len(old),
                overlaps_selected_same_recording=bool(old and distance(entry) < frames)))
            per_record[name].append(entry)
            roles[entry['group']] += 1
            progress = True
            if len(chosen) == count:
                break
        require(progress, 'Insufficient distinct legal windows')
    return chosen


def recording_representatives_first(entries):
    """One upper-middle source-clock window per recording, then all others."""
    groups = defaultdict(list)
    for row in entries:
        groups[row['recording']].append(row)
    first = [sorted(groups[name], key=lambda row: (row['source_first'], row['path'], row['start']))
             [len(groups[name]) // 2] for name in sorted(groups)]
    selected = {(row['path'], row['start']) for row in first}
    return first + [row for row in entries if (row['path'], row['start']) not in selected]


def common_members(rows, frames, stride):
    """Match Native30Data member order, including EXO-only official sources."""
    import numpy as np
    counters, members, sources = Counter(), {}, {}
    for ordinal, row in enumerate(rows):
        if row['split'] != 'validation':
            continue
        path = Path(row['path'])/'observation_valid.npy'
        digest = sha(path)
        require(digest == row['output_sha256']['observation_valid.npy'], 'Source validity SHA differs')
        valid = np.load(path, allow_pickle=False)
        require(valid.dtype == np.bool_ and valid.shape == (row['frames'],), 'Invalid source clock validity')
        starts = np.arange(0, max(0, len(valid)-frames+1), stride, dtype=np.int64)
        prefix = np.r_[0, np.cumsum(~valid)]
        if np.any(prefix[starts+frames] == prefix[starts]):
            members[ordinal] = counters[row['group']]
            counters[row['group']] += 1
        sources[str(path)] = digest
    return members, sources


def build(config, manifest_path, index_path, *, other_windows=50, seed=2026101807, rich_plan=None):
    import numpy as np
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    index = json.loads(index_path.read_text())
    require(config['frames'] == index['frames'] == 192 and config['fps'] == index['fps'] == 30, 'Expected native192/30Hz')
    require(config['window_stride'] == index['window_stride'] == 64, 'Window grid differs')
    require(index['source_manifest_sha256'] == sha(manifest_path), 'Manifest SHA differs from legal index')
    require(index['bundle_paths'] == [row['path'] for row in rows], 'Manifest source ordering differs')
    info = index['files']['validation/ego']
    array_path = index_path.parent / info['file']
    require(sha(array_path) == info['sha256'], 'Legal-window SHA differs')
    starts = np.load(array_path, allow_pickle=False)
    require(starts.dtype == np.int64 and starts.shape == (info['windows'], 2), 'Invalid legal-window array')
    require(len({(int(i), int(s)) for i, s in starts}) == len(starts), 'Duplicate legal windows')
    members, validity_sources = common_members(rows, 192, 64)
    require(config['validation']['seed'] == 2026101351, 'Historical case-seed protocol differs')
    entries = defaultdict(list)
    for ordinal, start in starts:
        require(0 <= ordinal < len(rows), 'Unknown bundle ordinal')
        meta = rows[int(ordinal)]
        require(meta['split'] == 'validation' and 'ego' in meta['allowed_modes'] and not meta['diagnostic_only'], 'Non-VAL or diagnostic source in VAL index')
        require(0 <= start and start + 192 <= meta['frames'] and start % 64 == 0, 'Illegal indexed window')
        recording = meta.get('recording') or meta['sequence_id']
        source_first = int(meta['source_first'] + start)
        key = json.dumps([meta['group'], 'ego', meta['sequence_id'], source_first], separators=(',', ':'))
        # Same historical full-RICH seed rule; reuse window identities without
        # pretending its 15100-step predictions are 41000-step predictions.
        case_seed = (config['validation']['seed'] + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')) % 2 ** 32
        entries[meta['dataset']].append(dict(dataset=meta['dataset'], recording=recording,
            group=meta['group'], sequence_id=meta['sequence_id'], source_bundle_ordinal=int(ordinal),
            path=meta['path'], member=members[int(ordinal)], start=int(start), source_first=source_first,
            case_seed=case_seed, mode='ego', joints=[15]))
    rich = sorted(entries['rich'], key=lambda row: (row['recording'], row['sequence_id'], row['start']))
    require(len(rich) == 206 and len({row['recording'] for row in rich}) == 27, 'Unexpected full RICH VAL coverage')
    if rich_plan is not None:
        previous = json.loads(rich_plan.read_text())['plan']
        fields = ('path', 'start', 'source_first', 'case_seed')
        require([tuple(row[key] for key in fields) for row in rich] == [tuple(row[key] for key in fields) for row in previous], 'Historical full-RICH cohort or seeds differ')
    selected = [dict(row, panel='full_rich', selection='all_legal_RICH_grid_windows')
                for row in recording_representatives_first(rich)]
    for dataset in ('trumans', 'egobody'):
        selected.extend(dict(row, panel='balanced', selection='recording_round_robin_then_role_balanced_maximum_temporal_spacing')
                        for row in select_recording_balanced(entries[dataset], other_windows, 192, seed))
    coverage = {}
    for dataset in ('rich', 'trumans', 'egobody'):
        chosen = [row for row in selected if row['dataset'] == dataset]
        coverage[dataset] = dict(eligible_windows=len(entries[dataset]), windows=len(chosen),
            eligible_recordings=len({row['recording'] for row in entries[dataset]}),
            independent_recordings=len({row['recording'] for row in chosen}),
            source_bundles=len({row['path'] for row in chosen}), roles=dict(Counter(row['group'] for row in chosen)),
            repeated_recording_windows=len(chosen) - len({row['recording'] for row in chosen}),
            overlapping_supplemental_windows=sum(row.get('overlaps_selected_same_recording', False) for row in chosen))
    for case, entry in enumerate(selected):
        entry['case'] = case
    sources = {str(path): sha(path) for path in (Path(__file__), manifest_path, index_path, array_path)}
    sources.update(validity_sources)
    if rich_plan is not None:
        sources[str(rich_plan)] = sha(rich_plan)
    return dict(protocol=PROTOCOL, prepared_at=datetime.now().astimezone().isoformat(), split='validation',
        mode='ego', frames=192, fps=30, window_stride=64, regimes=['clean', 'noisy'],
        ddim_steps=config['sampling']['ddim_steps'], scene='on', physics='on', seed=seed,
        unique_windows=len(selected), predictions_per_model=2 * len(selected), coverage=coverage,
        source_manifest=str(manifest_path), window_index=str(index_path), source_sha256=sources, entries=selected,
        selection='Identity-only; RICH 27 recording upper-middle representatives first, then remaining179 legal windows; peers one recording per round, role balanced and temporally separated where possible.',
        aggregation='Report valid-joint count weighted window metrics and equal-recording macro; primary RICH equal-recording macro, clean/noisy separately. Peers each equal-recording then equal-dataset. Report paired per-recording deltas; windows within one recording are correlated.',
        outcome_selection=False, TEST_used=False, optimizer_updates=0,
        model_selection='Expanded post-training VAL comparison of retained41k/55k only;41k selection from small monitoring panel was provisional, not best over all historical steps. Previously fixed TEST41k run must keep its original model/protocol and be labelled accordingly.',
        scope='Full persisted legal RICH VAL grid includes overlapping windows; one short RICH source has no legal192-frame window. Other corpora use50windows each but only33TRUMANS/16EgoBody distinct recordings exist; no false50-independent-recording claim. No GT action or prediction value used for selection.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=BASE/'runs/native30_gait_v2_ego55k_oct15/training/best.pt')
    parser.add_argument('--other-windows', type=int, default=50)
    args = parser.parse_args(argv)
    require(not (args.out/'COHORT.json').exists(), 'Existing immutable cohort; use its exact evaluation protocol')
    import torch
    checkpoint_sha = sha(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    require(state['step'] == 41000 and checkpoint_sha == 'f8e2d22b155240c2fda8cb7d75aca451bb4152ede2441d592f3952f52068b4d0', 'Expected exact retained41k parent')
    config = state['effective_config']['config']
    require(config['physical_guidance']['enabled'], 'Registered full system requires physical guidance ON')
    original_config = BASE/'runs/native30_gait_v2_ego55k_oct15/config.json'
    require(json.loads(original_config.read_text()) == config, 'Original config differs from embedded parent config')
    plan = build(config, (BASE/config['train_manifest']).resolve(), (BASE/config['window_index']).resolve(),
        other_windows=args.other_windows, rich_plan=BASE/'runs/native30_live_monitor_oct16/rich_full_validation_oct16/PLAN.json')
    plan['reference_checkpoint'] = dict(path=str(args.checkpoint.resolve()), step=41000, sha256=checkpoint_sha)
    plan['config_sha256'] = sha(original_config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'config.json').write_bytes(original_config.read_bytes())
    (args.out/'COHORT.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(windows=plan['unique_windows'], predictions_per_model=plan['predictions_per_model'], coverage=plan['coverage'], CUDA_initialized=torch.cuda.is_initialized()), ensure_ascii=False))


if __name__ == '__main__':
    main()
