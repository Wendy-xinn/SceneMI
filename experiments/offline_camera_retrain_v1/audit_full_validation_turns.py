"""Read all cached 55k predictions; anatomical hip-heading audit, no inference.

GT root rotations were not saved in this cache. Use the SAME horizontal hip
axis on GT and generated FK joints, not camera heading or a guessed GT pose.
Net turns omit out-and-back turns. Overlapping windows are not independent.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def hip_turn(joints):
    axis = np.asarray(joints, dtype=np.float64)[:, 1] - joints[:, 2]
    length = np.linalg.norm(axis, axis=-1)
    ratio = np.linalg.norm(axis[:, [0, 2]], axis=-1) / np.maximum(length, 1e-8)
    yaw = np.unwrap(np.arctan2(-axis[:, 2], axis[:, 0]))
    return float(np.rad2deg(yaw[-1] - yaw[0])), float(np.mean(ratio > .3))


def aggregate(rows, threshold):
    eligible = [r for r in rows if r['gt_projection_valid_fraction'] >= .95
                and abs(r['gt_turn_deg']) >= threshold]
    out = {'windows': len(rows), 'sequences': len({r['sequence'] for r in rows}),
           'eligible_windows': len(eligible),
           'eligible_sequences': len({r['sequence'] for r in eligible})}
    for variant in ('scene_on', 'scene_off'):
        predicates = {
            'opposite_any': lambda r: r[variant]['turn_deg'] * r['gt_turn_deg'] < 0,
            'opposite_at_least15deg': lambda r: r[variant]['turn_deg'] * r['gt_turn_deg'] < 0 and abs(r[variant]['turn_deg']) >= 15,
            'under_half_gt': lambda r: abs(r[variant]['turn_deg']) < .5 * abs(r['gt_turn_deg']),
            'near_static_under15deg': lambda r: abs(r[variant]['turn_deg']) < 15,
            'prediction_projection_unreliable': lambda r: r[variant]['projection_valid_fraction'] < .95,
        }
        result = {}
        for label, predicate in predicates.items():
            affected = [r for r in eligible if predicate(r)]
            per_seq = defaultdict(list)
            for r in eligible:
                per_seq[r['sequence']].append(bool(predicate(r)))
            result[label] = {'windows': len(affected),
                             'fraction': len(affected) / len(eligible) if eligible else None,
                             'sequences_with_any': len({r['sequence'] for r in affected}),
                             'sequence_balanced_fraction': float(np.mean([np.mean(v) for v in per_seq.values()])) if per_seq else None}
            reliable = [r for r in eligible if r[variant]['projection_valid_fraction'] >= .95]
            result[label]['reliable_prediction_denominator'] = len(reliable)
            result[label]['reliable_prediction_numerator'] = sum(bool(predicate(r)) for r in reliable)
        out[variant] = result
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Sanity check axis convention, signed left/right and angle wrapping.
    for degrees in (90, -90, 270):
        yaw = np.linspace(0, np.deg2rad(degrees), 128)
        joints = np.zeros((128, 22, 3))
        joints[:, 1, 0] = np.cos(yaw)
        joints[:, 1, 2] = -np.sin(yaw)
        turn, valid = hip_turn(joints)
        assert abs(turn - degrees) < 1e-8 and valid == 1
    source = args.input / 'standard_rows.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    identities = set()
    audited = []
    args.output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        w = row['window']
        identity = (w['group'], w['sequence_index'], w['start_index'], w['length'])
        if identity in identities:
            raise ValueError(f'Duplicate window: {identity}')
        identities.add(identity)
        with np.load(args.input / row['motion_file']) as saved:
            gt_turn, valid = hip_turn(saved['truth'])
            item = {'index': w['index'], 'group': w['group'],
                    'sequence': w['group'] + '/' + w['sequence_id'],
                    'start_index': w['start_index'], 'length': w['length'],
                    'gt_turn_deg': gt_turn, 'gt_projection_valid_fraction': valid}
            for variant in ('scene_on', 'scene_off'):
                assert saved[variant].shape == saved['truth'].shape == (w['length'], 22, 3)
                assert np.isfinite(saved[variant]).all() and np.isfinite(saved['truth']).all()
                turn, pv = hip_turn(saved[variant])
                item[variant] = {'turn_deg': turn, 'projection_valid_fraction': pv}
            audited.append(item)
        if (index + 1) % 3000 == 0:
            print(f'audited {index + 1}/{len(rows)}', flush=True)
    summary = {'protocol': 'horizontal hip-axis net yaw; same FK-joint definition for GT/prediction; y up',
               'scope': 'official55k cached head-camera conditioning, one prediction per window per scene variant; not typed-head trials',
               'limitations': ['hip heading is a geometric proxy, not the previous native root-forward metric',
                               'overlapping windows and sequences are correlated; affected-sequence counts depend on windows per sequence',
                               'net turn excludes out-and-back turns; mismatch to GT is not automatically physically invalid',
                               'projection validity is essential for rolled bodies; report reliable-only denominators too'],
               'rows_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'all': {str(t): aggregate(audited, t) for t in (30, 60, 90)},
               'three_datasets': {g: aggregate([r for r in audited if r['group'] in groups], 30)
                                  for g, groups in [('TRUMANS', ['trumans']),
                                                    ('EgoBody', ['camera_wearer', 'interactee']),
                                                    ('RICH', ['rich'])]},
               'datasets': {g: aggregate([r for r in audited if r['group'] == g], 30)
                            for g in sorted({r['group'] for r in audited})}}
    (args.output / 'turn_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audited))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
