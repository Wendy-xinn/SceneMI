"""Audit the denser ego-visible scene archives before training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.offline_camera_retrain_v1.data import HERE, SOURCE


SPLITS = ('train', 'validation', 'test')
DATASETS = {
    'trumans': ('trumans_sequences', HERE / 'data/visible_trumans_5hz'),
    'egobody': ('egobody_sequences', HERE / 'data/visible_egobody_5hz'),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def manifest_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def describe(values):
    values = np.asarray(values, np.float64)
    return {
        'min': float(values.min()),
        'median': float(np.median(values)),
        'max': float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=HERE / 'visible_maps_5hz_audit.json')
    args = parser.parse_args()
    report = {'status': 'ready', 'sampling_hz': 5, 'datasets': {}}
    hashed = []
    for dataset, (source_name, visible_root) in DATASETS.items():
        dataset_report = {}
        for split in SPLITS:
            source_rows = read_jsonl(SOURCE / source_name / f'{split}.jsonl')
            visible_manifest = visible_root / f'{split}.jsonl'
            rows = read_jsonl(visible_manifest)
            hashed.append(visible_manifest)
            source_ids = {row['sequence_id'] for row in source_rows}
            visible_ids = {row['sequence_id'] for row in rows}
            if len(rows) != len(visible_ids) or visible_ids != source_ids:
                raise ValueError(f'{dataset}/{split}: sequence set mismatch')
            counts = []
            ratios = []
            for row in rows:
                if (row.get('sample_stride') != 4
                        or row.get('candidate_count') != 65536
                        or row.get('query_radius_m') != 6.5
                        or 'project' not in row.get('protocol', '').lower()):
                    raise ValueError(f"{row['sequence_id']}: invalid visibility metadata")
                points = np.load(row['path'], mmap_mode='r')
                if points.ndim != 2 or points.shape[1] != 3 or len(points) != row['points']:
                    raise ValueError(f"{row['sequence_id']}: invalid point array")
                if not np.isfinite(points).all() or len(points) < 32:
                    raise ValueError(f"{row['sequence_id']}: nonfinite or insufficient points")
                counts.append(len(points))
                if dataset == 'trumans':
                    source_row = next(item for item in source_rows
                                      if item['sequence_id'] == row['sequence_id'])
                    old_path = Path(source_row['folder']) / 'visible_static_points_world.npy'
                else:
                    old_path = (HERE / 'data/visible_egobody' / split / row['sequence_id']
                                / 'visible_static_points_world.npy')
                ratios.append(len(points) / len(np.load(old_path, mmap_mode='r')))
            dataset_report[split] = {
                'sequences': len(rows),
                'points': describe(counts),
                'point_count_ratio_vs_1hz': describe(ratios),
            }
        report['datasets'][dataset] = dataset_report
    report['manifest_sha256'] = manifest_hash(hashed)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
