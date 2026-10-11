"""Summarize the freshly recomputed official55k audit, including overlaps.

Run from the repository root after audit_full_validation_turns.py has completed.
Reads only audit results and source provenance; never runs model inference.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(__file__).resolve().parent
EXPERIMENT = RUN.parents[1]
SOURCE = EXPERIMENT / 'runs/native_dynamic_scene20_contact_55k_oct07/full_validation'
PREVIOUS = EXPERIMENT / 'runs/literature_turn_audit_oct10'


def flags(row, variant):
    gt, pred = row['gt_turn_deg'], row[variant]['turn_deg']
    opposite = gt * pred < 0 and abs(pred) >= 15
    under = abs(pred) < .5 * abs(gt)
    return {'opposite_at_least15deg': opposite, 'under_half_gt': under,
            'either': opposite or under, 'both': opposite and under}


def summarize(rows, threshold, variant):
    eligible = [r for r in rows if r['gt_projection_valid_fraction'] >= .95
                and abs(r['gt_turn_deg']) >= threshold]
    all_sequences = {r['sequence'] for r in rows}
    eligible_sequences = {r['sequence'] for r in eligible}
    result = {'all_windows': len(rows), 'all_sequences': len(all_sequences),
              'eligible_windows': len(eligible),
              'eligible_sequences': len(eligible_sequences),
              'prediction_projection_unreliable': sum(
                  r[variant]['projection_valid_fraction'] < .95 for r in eligible)}
    for name in ('opposite_at_least15deg', 'under_half_gt', 'either', 'both'):
        affected = [r for r in eligible if flags(r, variant)[name]]
        per_sequence = defaultdict(list)
        for r in eligible:
            per_sequence[r['sequence']].append(flags(r, variant)[name])
        reliable = [r for r in eligible if r[variant]['projection_valid_fraction'] >= .95]
        result[name] = {
            'windows': len(affected),
            'fraction_eligible': len(affected) / len(eligible) if eligible else None,
            'fraction_all_windows': len(affected) / len(rows) if rows else None,
            'sequences_with_any': len({r['sequence'] for r in affected}),
            'sequence_balanced_fraction_eligible': sum(
                sum(v) / len(v) for v in per_sequence.values()) / len(per_sequence)
                if per_sequence else None,
            'reliable_prediction_denominator': len(reliable),
            'reliable_prediction_numerator': sum(flags(r, variant)[name] for r in reliable),
        }
    assert result['either']['windows'] == (
        result['opposite_at_least15deg']['windows'] + result['under_half_gt']['windows']
        - result['both']['windows'])
    return result


def main():
    rows = [json.loads(s) for s in (RUN / 'turn_rows.jsonl').read_text().splitlines()]
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    assert len(rows) == manifest['eligible_windows'] == len({r['index'] for r in rows})
    assert json.loads((RUN / 'summary.json').read_text()) == json.loads(
        (PREVIOUS / 'summary.json').read_text())
    assert (RUN / 'turn_rows.jsonl').read_bytes() == (PREVIOUS / 'turn_rows.jsonl').read_bytes()
    domains = {'TRUMANS': {'trumans'}, 'EgoBody': {'camera_wearer', 'interactee'},
               'RICH': {'rich'}}
    summary = {
        'date_local': '2026-10-11', 'timezone': 'Asia/Shanghai',
        'checkpoint': manifest['checkpoint'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
        'source': str(SOURCE), 'prediction_protocol': manifest['protocol'],
        'audit_script_sha256': hashlib.sha256(
            (EXPERIMENT / 'audit_full_validation_turns.py').read_bytes()).hexdigest(),
        'source_rows_sha256': hashlib.sha256((SOURCE / 'standard_rows.jsonl').read_bytes()).hexdigest(),
        'turn_rows_sha256': hashlib.sha256((RUN / 'turn_rows.jsonl').read_bytes()).hexdigest(),
        'previous_summary_and_rows_identical': True,
        'length_window_counts': dict(Counter(r['length'] for r in rows)),
        'definition': 'same GT/prediction horizontal hip-axis net yaw; GT projection valid >=95%; '
                      'GT absolute net turn >= threshold; obvious reversal: opposite sign and '
                      'predicted absolute turn >=15deg; underturn: predicted absolute net turn <50% of GT',
        'limitations': [
            'Cached official55k predictions only; one generated realization per window and scene variant.',
            'Overlapping windows and body sequences are correlated; these are not independent turn events.',
            'Net-turn audit omits turns that return to their initial heading and local timing errors.',
            'Hip geometry is a heading proxy; GT root rotations were not archived.',
            'GT mismatch does not automatically imply physically invalid motion.',
            'Noneligible windows are unclassified by this audit, not certified correct.',
        ],
        'thresholds': {},
    }
    for threshold in (30, 60, 90):
        summary['thresholds'][str(threshold)] = {
            variant: {'all': summarize(rows, threshold, variant),
                      'datasets': {name: summarize([r for r in rows if r['group'] in groups],
                                                   threshold, variant)
                                   for name, groups in domains.items()}}
            for variant in ('scene_on', 'scene_off')}
    for variant in ('scene_on', 'scene_off'):
        block = summary['thresholds']['30'][variant]
        for metric in ('all_windows', 'eligible_windows'):
            assert block['all'][metric] == sum(r[metric] for r in block['datasets'].values())
        for metric in ('opposite_at_least15deg', 'under_half_gt', 'either', 'both'):
            assert block['all'][metric]['windows'] == sum(
                r[metric]['windows'] for r in block['datasets'].values())
    (RUN / 'prevalence.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ['index', 'group', 'sequence', 'start_index', 'length', 'gt_turn_deg',
              'prediction_turn_deg', 'opposite_at_least15deg', 'under_half_gt']
    with (RUN / 'affected_windows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            f = flags(r, 'scene_on')
            if r['gt_projection_valid_fraction'] < .95 or abs(r['gt_turn_deg']) < 30 or not f['either']:
                continue
            writer.writerow({**{k: r[k] for k in fields if k in r},
                             'prediction_turn_deg': r['scene_on']['turn_deg'],
                             'opposite_at_least15deg': f['opposite_at_least15deg'],
                             'under_half_gt': f['under_half_gt']})
    print(json.dumps(summary['thresholds']['30']['scene_on'], indent=2))


if __name__ == '__main__':
    main()
