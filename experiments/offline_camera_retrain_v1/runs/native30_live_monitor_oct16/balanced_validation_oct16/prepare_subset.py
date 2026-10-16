#!/usr/bin/env python3
"""Prepare a deterministic 30-window official-VAL plan without model calls.

Only metadata and the persisted legal ego-start index are read. No prediction,
motion label, contact label, training process or GPU is accessed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
SELECTION_SEED = 2026101641
CASE_SEED_BASE = 2026101351  # Same identity-based seed rule as native30_validation.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(label):
    return hashlib.sha256((str(SELECTION_SEED) + '/' + label).encode()).hexdigest()


def build():
    coverage_path = HERE.parent / 'validation_coverage_oct16/EVALUATION_PLAN.json'
    index_path = BASE / 'data/native30_official_windows64_v1/dataset.json'
    config_path = BASE / 'runs/native30_gait_v2_ego55k_oct15/config.json'
    config = json.loads(config_path.read_text())
    manifest_path = BASE / config['train_manifest']
    index = json.loads(index_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    windows_path = index_path.parent / index['files']['validation/ego']['file']
    manifest_sha = sha(manifest_path)
    windows_sha = sha(windows_path)
    assert manifest_sha == index['source_manifest_sha256'] == coverage['source_manifest_sha256']
    assert windows_sha == index['files']['validation/ego']['sha256'] == coverage['index_sha256']
    assert index['frames'] == config['frames'] == 192
    assert index['window_stride'] == config['window_stride'] == 64
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    assert index['bundle_paths'] == [m['path'] for m in manifest]
    starts = np.load(windows_path, allow_pickle=False)
    assert starts.shape == (1806, 2) and starts.dtype == np.int64
    candidates = []
    source_metadata_sha = {}
    for ordinal in np.unique(starts[:, 0]):
        ordinal = int(ordinal)
        meta = manifest[ordinal]
        assert meta['split'] == 'validation' and not meta['diagnostic_only']
        assert 'ego' in meta['allowed_modes'] and meta['fps'] == 30
        metadata_path = Path(meta['path']) / 'metadata.json'
        source = json.loads(metadata_path.read_text())
        for key in ('sequence_id', 'group', 'split', 'frames', 'source_first', 'output_sha256', 'input_sha256'):
            assert meta[key] == source[key], (meta['path'], key)
        source_metadata_sha[meta['path']] = sha(metadata_path)
        legal = starts[starts[:, 0] == ordinal, 1]
        assert len(legal) and (np.diff(legal) > 0).all() and (legal % 64 == 0).all()
        start = int(legal[len(legal) // 2])
        source_first = int(meta['source_first'] + start)
        key = json.dumps([meta['group'], 'ego', meta['sequence_id'], source_first], separators=(',', ':'))
        case_seed = (CASE_SEED_BASE + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')) % 2**32
        candidates.append(dict(dataset=meta['dataset'], recording=meta.get('recording') or meta['sequence_id'],
            group=meta['group'], sequence_id=meta['sequence_id'], source_bundle_ordinal=ordinal,
            path=meta['path'], start=start, source_first=source_first, case_seed=case_seed,
            eligible_windows=int(len(legal)), mode='ego', joints=[15],
            source_hashes=meta['output_sha256'], source_input_hashes=meta['input_sha256'],
            source_metadata_sha256=source_metadata_sha[meta['path']]))
    # Reuse and independently check both previously prepared inventories. Neither
    # inventory contains or selects by prediction quality.
    keys = ('dataset', 'recording', 'group', 'sequence_id', 'path', 'start', 'source_first', 'case_seed', 'eligible_windows')
    def identity(row):
        return tuple(row[k] for k in keys)
    assert {identity(c) for c in candidates} == {identity(c) for c in coverage['one_per_bundle']}
    by_record = defaultdict(list)
    for row in candidates:
        by_record[(row['dataset'], row['recording'])].append(row)
    assert set(by_record) == {(r['dataset'], r['recording']) for r in coverage['one_per_recording']}
    assert all(identity(r) in {identity(c) for c in candidates} for r in coverage['one_per_recording'])
    selected = []
    for dataset in ('trumans', 'rich'):
        records = sorted((r for d, r in by_record if d == dataset), key=lambda r: rank(dataset + '/' + r))[:10]
        assert len(records) == 10
        for record in records:
            selected.append(min(by_record[(dataset, record)], key=lambda row: rank(row['path'])))
    role_records = {}
    for role in ('camera_wearer', 'interactee'):
        role_records[role] = sorted({r['recording'] for r in candidates if r['dataset'] == 'egobody' and r['group'] == role},
            key=lambda record: rank('egobody/' + role + '/' + record))
    # A deterministic bounded search guarantees the requested 5+5 distinct
    # recordings when feasible; records with only one available role are valid.
    from itertools import combinations
    pair = None
    for wearers in combinations(role_records['camera_wearer'], 5):
        interactees = [r for r in role_records['interactee'] if r not in wearers]
        if len(interactees) >= 5:
            pair = dict(camera_wearer=wearers, interactee=interactees[:5])
            break
    if pair is None:
        raise ValueError('Cannot select five camera_wearer and five interactee from ten distinct legal ego recordings')
    for role, records in pair.items():
        for record in records:
            choices = [r for r in by_record[('egobody', record)] if r['group'] == role]
            selected.append(min(choices, key=lambda row: rank(row['path'])))
    selected.sort(key=lambda r: (('trumans', 'egobody', 'rich').index(r['dataset']), rank(r['recording'])))
    for case, row in enumerate(selected):
        row['case'] = case
    assert len(selected) == 30
    assert len({(r['dataset'], r['recording']) for r in selected}) == 30
    assert len({r['sequence_id'] for r in selected}) == 30
    assert Counter(r['dataset'] for r in selected) == dict(trumans=10, egobody=10, rich=10)
    assert Counter(r['group'] for r in selected if r['dataset'] == 'egobody') == dict(camera_wearer=5, interactee=5)
    assert all('member' not in r for r in selected)
    return dict(protocol='native30-balanced-official-VAL-recordings-30-v1',
        status='prepared_not_evaluated', prepared_at=datetime.now(timezone.utc).isoformat(),
        selection=dict(seed=SELECTION_SEED, method='SHA256 ranking of recording identities; distinct records; median legal grid start per selected bundle.',
            source='Previously prepared one_per_recording and one_per_bundle inventories, independently cross-checked against the real legal-start index and source metadata.',
            egobody_roles=dict(camera_wearer=5, interactee=5), different_recordings_per_dataset=10,
            predictions_or_GT_actions_used=False, exclusion='No outcome-based exclusion or selection; previously exposed records may occur.'),
        seed=SELECTION_SEED, case_seed_base=CASE_SEED_BASE,
        frames=192, fps=30, window_stride=64, mode='ego', joints=[15],
        regimes=['clean', 'noisy'], scene='on', ddim_steps=20,
        unique_windows=30, expected_sampling_calls=60,
        source_manifest=str(manifest_path), window_index=str(index_path),
        coverage_plan=str(coverage_path),
        source_sha256={str(p): sha(p) for p in (Path(__file__).resolve(), config_path, manifest_path, index_path, windows_path, coverage_path)},
        checks=dict(official_validation_only=True, legal_index_median_starts=True,
            inventory_reuse_verified=True, source_metadata_matches_manifest=True,
            all_recordings_and_sequences_distinct=True, egobody_five_each_role=True,
            model_forward_calls=0, cuda_used=False, training_changed=False, test_used=False),
        aggregation='Report clean/noisy separately, per-dataset means and equal dataset macro; one record per case. Expanded development VAL, not a pristine TEST or full-window sweep.',
        checkpoint='Execution must independently freeze one committed checkpoint and record its SHA256 before any predictions.',
        plan=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'PLAN.json')
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + '.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(args.output)
    print(json.dumps(dict(output=str(args.output), unique_windows=len(result['plan']),
        datasets=dict(Counter(r['dataset'] for r in result['plan'])),
        egobody_roles=dict(Counter(r['group'] for r in result['plan'] if r['dataset'] == 'egobody'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
