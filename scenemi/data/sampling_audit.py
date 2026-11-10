"""Measure supervised time coverage from immutable training sample logs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def intervals_for_starts(starts, width=128):
    """Return the disjoint union of half-open source-frame intervals."""
    result = []
    for start in sorted(set(int(s) for s in starts)):
        end = start + width
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def covered_frames(starts, source_frames, width=128):
    starts = list(starts)
    if any(s < 0 or s + width > source_frames for s in starts):
        raise ValueError('A logged window crosses its source recording')
    return sum(end-start for start, end in intervals_for_starts(starts, width))


def amass_population(index):
    result = {}
    for row in json.loads(index.read_text())['families']:
        if row['split'] != 'train':
            continue
        lengths = {v['frames_after_relative_pose_drop'] for v in row['variants']}
        if len(lengths) != 1:
            raise ValueError('Reflections have different source clocks')
        n = lengths.pop()
        result[row['source']] = dict(group=row['corpus'], family=row['family'], frames=n,
                                    eligible_frames=n)
    return result


def scene_population(paths, config):
    # Initialization reads source membership and validity masks, never samples
    # target motion or opens the independent test split. Pelvis is unused here.
    from scenemi.data.scene_dataset import SceneDataset
    data = SceneDataset(paths, np.zeros(3), split='train',
                        config=yaml.safe_load(config.read_text()))
    result = {}
    for row in data.records:
        meta = data.metadata(row)
        n = meta['frames']
        result[row['path']] = dict(group=row['dataset'], family=row['family'], frames=n,
            eligible_frames=covered_frames(row['starts'], n))
    return result


def summarize(population, starts, *, run, mode, sequence_writer):
    groups = defaultdict(list)
    for source, meta in population.items():
        chosen = starts.get(source, [])
        covered = covered_frames(chosen, meta['frames'])
        if covered > meta['eligible_frames']:
            raise ValueError('Logged coverage exceeds the accepted population')
        record = dict(run=run, mode=mode, group=meta['group'], family=meta['family'],
            source=source, source_frames=meta['frames'], eligible_frames=meta['eligible_frames'],
            covered_frames=covered, coverage_fraction=covered/meta['eligible_frames'],
            draws=len(chosen))
        sequence_writer.writerow(record)
        groups[meta['group']].append(record)

    def aggregate(rows):
        eligible = sum(r['eligible_frames'] for r in rows)
        covered = sum(r['covered_frames'] for r in rows)
        families = {r['family'] for r in rows}
        seen = {r['family'] for r in rows if r['draws']}
        return dict(sequences=len(rows), sampled_sequences=sum(r['draws']>0 for r in rows),
            families=len(families), sampled_families=len(seen), eligible_frames=eligible,
            supervised_frames_covered=covered, frame_coverage_fraction=covered/eligible,
            sequences_below_half_coverage=sum(r['coverage_fraction']<.5 for r in rows),
            sequence_coverage_quantiles=dict(zip(('min','p10','median','p90','max'),
                np.quantile([r['coverage_fraction'] for r in rows], [0,.1,.5,.9,1]).tolist())),
            draws_quantiles=dict(zip(('min','p10','median','p90','max'),
                np.quantile([r['draws'] for r in rows], [0,.1,.5,.9,1]).tolist())))
    return dict(total=aggregate([r for rows in groups.values() for r in rows]),
                by_group={k:aggregate(rows) for k, rows in sorted(groups.items())})


def audit(output, history, amass_index, scene_paths, scene_config):
    from scenemi.training import ensure_bounded
    ensure_bounded()
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(history.read_text())
    populations = {}
    results = {}
    columns = ('run','mode','group','family','source','source_frames','eligible_frames',
               'covered_frames','coverage_fraction','draws')
    csv_path = output/'sequence_coverage.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for report in expected['reports']:
            name = report['run']
            kind = 'scene' if name.startswith('scene_') else 'amass'
            if kind not in populations:
                populations[kind] = (scene_population(scene_paths, scene_config)
                    if kind=='scene' else amass_population(amass_index))
            population = populations[kind]
            starts = defaultdict(lambda: defaultdict(list))
            digest = hashlib.sha256()
            with Path(report['log_path']).open('rb') as log:
                for line in log:
                    digest.update(line)
                    for item in json.loads(line)['identities']:
                        source = item.get('source', item.get('path'))
                        if source not in population:
                            raise ValueError('Logged source is outside the train population')
                        starts[item['mode']][source].append(int(item['start']))
            if digest.hexdigest() != report['log_sha256']:
                raise ValueError('Training log changed since the initial audit')
            modes = {mode:summarize(population, selected, run=name, mode=mode,
                     sequence_writer=writer) for mode, selected in sorted(starts.items())}
            if len(starts)>1:
                combined = defaultdict(list)
                for selected in starts.values():
                    for source, values in selected.items():
                        combined[source].extend(values)
                modes['all_modes'] = summarize(population, combined, run=name,
                    mode='all_modes', sequence_writer=writer)
            results[name] = dict(log_sha256=digest.hexdigest(), modes=modes)
            print(json.dumps(dict(run=name, total={m:v['total'] for m,v in modes.items()})), flush=True)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        meaning='Fraction of eligible training source frames included in clean motion supervision; reflections merged by source time. This does not measure visible EXO input coverage or the fraction of every possible window sampled.',
        training_modified=False, test_arrays_opened=False,
        input_scope='train membership metadata, train observation-valid masks, completed training logs',
        sources={str(p):sha(p) for p in (Path(__file__),history,amass_index,scene_paths,scene_config)},
        per_sequence_csv=str(csv_path), per_sequence_csv_sha256=sha(csv_path), results=results)
    temporary = output/'RESULT.json.tmp'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output/'RESULT.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--amass-index', type=Path, required=True)
    parser.add_argument('--scene-paths', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path, required=True)
    args = parser.parse_args()
    audit(args.output, args.history, args.amass_index, args.scene_paths, args.scene_config)


if __name__=='__main__':
    main()
