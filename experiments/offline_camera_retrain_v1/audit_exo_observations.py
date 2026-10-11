"""Full frozen-export integrity and visibility-stratification audit."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from experiments.offline_camera_retrain_v1.data import SOURCE
from experiments.offline_camera_retrain_v1.export_egobody_exo import PROTOCOL


def audit(root, output):
    rows = list(map(json.loads, (root / 'manifest.jsonl').read_text().splitlines()))
    grouped = defaultdict(list); records = {}; failures = []
    for split in ('train', 'validation'):
        source = list(map(json.loads, (SOURCE / f'egobody_sequences/{split}.jsonl').read_text().splitlines()))
        assert {r['sequence_id'] for r in source} == {r['sequence_id'] for r in rows if r['split'] == split}
        records[split] = {r['recording'] for r in source}
    assert not records['train'] & records['validation']
    for row in rows:
        folder = Path(row['path']); meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['protocol'] == PROTOCOL and meta['split'] == row['split']
        for name, digest in meta['output_sha256'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        availability = np.load(folder / 'position_available.npy')
        positions = np.load(folder / 'visible_joints_world.npy')
        confidence = np.load(folder / 'position_confidence.npy')
        points = np.load(folder / 'static_visible_points_world.npy')
        assert np.isfinite(positions).all() and np.isfinite(points).all()
        assert np.all(positions[~availability] == 0) and np.all(confidence[~availability] == 0)
        assert np.all((confidence >= 0) & (confidence <= 1))
        assert len(positions) == row['frames']
        selected = availability[:, (15, 20, 21)]
        item = dict(sequence_id=row['sequence_id'], recording=row['recording'], frames=row['frames'],
                    any_head_wrist_frames=int(selected.any(-1).sum()),
                    no_head_wrist_frames=int((~selected.any(-1)).sum()),
                    partial_head_wrist_frames=int((selected.any(-1) & ~selected.all(-1)).sum()),
                    joint_fraction=float(availability.mean()), static_points=len(points))
        grouped[row['split']].append(item)
        if not len(points) or not selected.any(): failures.append(dict(split=row['split'], **item))
    summary = {}
    for split, items in grouped.items():
        summary[split] = dict(segments=len(items), recordings=len(records[split]),
                              person_frames=sum(r['frames'] for r in items),
                              any_head_wrist_frames=sum(r['any_head_wrist_frames'] for r in items),
                              no_head_wrist_frames=sum(r['no_head_wrist_frames'] for r in items),
                              partial_head_wrist_frames=sum(r['partial_head_wrist_frames'] for r in items))
    result = dict(status='passed', protocol=PROTOCOL, summary=summary,
                  missing_scene_or_tracks=failures, missingness='retained as explicit unknown; no full-scan fallback',
                  segments=dict(grouped))
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result
