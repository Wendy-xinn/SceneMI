"""Deterministic family-balanced selection of screened TRAIN chair sources."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .grail_visibility import sha256, write_json


def select(rows, count, seed):
    if type(count) is not int or count <= 0 or count > len(rows):
        raise ValueError('Requested count exceeds screened candidates or is invalid')
    if len({row['sequence_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate source identity')
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: row['sequence_id']):
        if (row['split'] != 'train' or row['category'] != 'sitting'
                or row['condition_provenance'] != 'proposed_synthetic_chair_observation'
                or not row['eligible_start_ranges'] or row['eligible_windows'] <= 0):
            raise ValueError('Use screened TRAIN-only synthetic chair candidates')
        groups[row['object_family_id']].append(row)
    rng = np.random.default_rng(seed)
    families = sorted(groups)
    rng.shuffle(families)
    for family in families:
        rng.shuffle(groups[family])
    chosen = []
    while len(chosen) < count:
        for family in families:
            if groups[family]:
                row = groups[family].pop()
                # Earliest valid window retains the most approach context.
                chosen.append(dict(row, window_start=row['eligible_start_ranges'][0][0]))
                if len(chosen) == count:
                    break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    chosen = select(rows, args.count, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = args.output / 'SELECTION.jsonl'
    content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in chosen)
    if selection.exists() and selection.read_text() != content:
        raise ValueError('Existing selection differs; use another output directory')
    selection.write_text(content)
    counts = Counter(row['object_family_id'] for row in chosen)
    report = dict(protocol='grail-family-balanced-chair-selection-v1', seed=args.seed,
                  candidates=len(rows), selected=len(chosen), object_families=len(counts),
                  eligible_windows=sum(row['eligible_windows'] for row in chosen),
                  sequences_per_family_histogram=dict(Counter(counts.values())),
                  candidate_sha256=sha256(args.candidates), selection_sha256=sha256(selection),
                  implementation_sha256=sha256(__file__), formal_training_integrated=False,
                  physics_validity_proven=False, split='train',
                  condition_provenance='synthetic_chair_completion',
                  selection_rule='Round-robin shuffled object families and sources; earliest contact-qualified window per prototype',
                  evaluation_changed=False, camera_or_motion_modified=False)
    write_json(args.output / 'SELECTION_AUDIT.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
