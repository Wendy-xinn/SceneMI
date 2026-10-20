"""CPU-only dev selection and disjoint confirmation for frozen scene policies.

The dev reader discards confirmation lines before JSON decoding their metrics.
One immutable DEV_CHOICE precedes confirmation; confirmation cannot choose a
runner-up. Policy sampling and its existing sealed source are not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

DATASETS = ('rich', 'trumans', 'egobody')
REGIMES = ('clean', 'noisy')
METRICS = ('hidden_mpjpe_mm', 'visible_mpjpe_mm', 'mpjpe_mm',
           'foot_joint_speed_gt_low_m_s', 'mpjve_mm_s', 'acceleration_error_mm_s2')
TOLERANCES = dict(zip(METRICS, (.02, .02, .02, .0001, .05, 2.)))
PROTOCOL = 'frozen-scene-policy-dev-select-confirmation-review-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    state = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            state.update(block)
    return state.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def binding(out):
    out = Path(out)
    selection, rules = read(out / 'SELECTION.json'), read(out / 'ACCEPTANCE.json')
    if file_sha(out / 'SELECTION.json') != rules['selection_manifest_sha256']:
        raise ValueError('Acceptance/selection identity differs')
    if digest({k: v for k, v in selection.items() if k != 'selection_sha256'}) != selection['selection_sha256']:
        raise ValueError('Selection internal hash differs')
    if selection['TEST_used'] or rules['TEST_used'] or selection['optimizer_updates'] or rules['optimizer_updates']:
        raise ValueError('Only zero-update official VAL review is allowed')
    source = Path(selection['source_manifest'])
    if file_sha(source) != selection['source_manifest_sha256']:
        raise ValueError('Original panel manifest changed')
    weights = read(source)['natural_dataset_weights']
    if weights != rules['aggregation']['natural_dataset_weights'] or set(weights) != set(DATASETS) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError('Primary natural-size weights differ from original manifest')
    plan = read(out / 'PLAN.json')
    labels = {v['feedback_policy']: k for k, v in plan['models'].items()}
    required = [rules['baseline_policy'], *rules['candidate_policies']]
    if len(labels) != len(plan['models']) or not all(p in labels for p in required):
        raise ValueError('Missing or duplicate declared policy labels')
    for phase in ('dev', 'confirmation'):
        entries = [e for e in selection['entries'] if e['review_panel'] == phase]
        if Counter(e['dataset'] for e in entries) != Counter(rich=4, trumans=4, egobody=4):
            raise ValueError('Expected four registered windows per dataset and phase')
    return selection, rules, weights, labels


def phase_rows(out, phase, labels):
    """Skip unrequested phase/model lines before deserializing metric values."""
    allowed = {phase, 'ego_retention'} if phase == 'dev' else {phase}
    result = []
    with (Path(out) / 'rows.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                raise ValueError('Incomplete durable ledger tail')
            panel = re.search(r'"review_panel"\s*:\s*"([^"\\]*)"', line)
            model = re.search(r'"model"\s*:\s*"([^"\\]*)"', line)
            if not panel or not model:
                raise ValueError('Missing unambiguous ledger routing envelope')
            if panel.group(1) not in allowed or model.group(1) not in labels:
                continue
            row = json.loads(line)
            if row['review_panel'] != panel.group(1) or row['model'] != model.group(1):
                raise ValueError('Ambiguous ledger routing envelope')
            if row['scene_variant'] != 'scene_on':
                continue
            result.append(row)
    return result


def number(row, metric, field='metrics'):
    value = row[field][metric]
    if isinstance(value, dict):
        value = value['value']
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise ValueError('Nonfinite metric')
    return value


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def percentile(values, fraction=.9):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def groups(rows, label, phase, weights):
    result = {}
    for dataset in DATASETS:
        for regime in REGIMES:
            selected = [r for r in rows if r['model'] == label and r['review_panel'] == phase
                        and r['dataset'] == dataset and r['regime'] == regime]
            if len(selected) != 4:
                raise ValueError('Incomplete panel group')
            result[dataset + '/' + regime] = dict(windows=4,
                metrics={m: dict(mean=mean(number(r, m) for r in selected),
                    p90=percentile(number(r, m) for r in selected),
                    valid_windows=sum(number(r, m) is not None for r in selected)) for m in METRICS},
                reference_hidden_mean_mm=mean(number(r, 'hidden_mpjpe_mm', 'reference_metrics') for r in selected))
    natural, equal = {}, {}
    for regime in REGIMES:
        values = {d: result[d + '/' + regime]['metrics']['hidden_mpjpe_mm']['mean'] for d in DATASETS}
        if any(v is None for v in values.values()):
            raise ValueError('A whole hidden dataset group is empty')
        natural[regime] = sum(weights[d] * values[d] for d in DATASETS)
        equal[regime] = statistics.mean(values.values())
    return dict(groups=result, natural_primary=dict(natural, combined=statistics.mean(natural.values())),
                equal_dataset_secondary=dict(equal, combined=statistics.mean(equal.values())))


def expected_rows(rows, selection, labels, phase):
    entries = [e for e in selection['entries'] if e['review_panel'] == phase]
    if phase == 'dev':
        entries += selection['ego_entries']
    by_case = {e['case']: e for e in entries}
    expected = {(label, case, regime) for label in labels for case in by_case for regime in REGIMES}
    actual = [(r['model'], r['case'], r['regime']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing, duplicate or unexpected phase predictions')
    for row in rows:
        entry = by_case[row['case']]
        for key in ('dataset', 'mode', 'sequence_id', 'recording', 'start', 'source_first', 'review_panel'):
            if row[key] != entry[key]:
                raise ValueError('Selected identity differs: ' + key)
        if row['seed'] != entry['case_seed']:
            raise ValueError('Selected seed differs')
    for case in by_case:
        for regime in REGIMES:
            pair = [r for r in rows if r['case'] == case and r['regime'] == regime]
            if len({r['input_sha256'] for r in pair}) != 1 or len({r['checkpoint_sha256'] for r in pair}) != 1:
                raise ValueError('Paired input or checkpoint differs')
            if any(r['reference_metrics'] != pair[0]['reference_metrics'] for r in pair[1:]):
                raise ValueError('Paired reference metrics differ')


def cpu_verify(rows, baseline):
    import numpy as np
    from .refinement_evaluation import numpy_core
    from . import refinement_guard, refinement_evaluation
    maximum = dict.fromkeys(METRICS, 0.)
    signatures, ego_predictions, ego_checks = {}, {}, []
    for row in rows:
        path = Path(row['artifact'])
        if file_sha(path) != row['artifact_sha256']:
            raise ValueError('Saved artifact changed')
        with np.load(path, allow_pickle=False) as arrays:
            for metric, value in numpy_core(arrays).items():
                old = number(row, metric)
                if (value is None) != (old is None):
                    raise ValueError('CPU metric availability differs')
                if value is not None:
                    maximum[metric] = max(maximum[metric], abs(value - old))
            state = hashlib.sha256()
            for key in ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion'):
                state.update(np.ascontiguousarray(arrays[key]).tobytes())
            key = row['case'], row['regime']
            if key in signatures and signatures[key] != state.hexdigest():
                raise ValueError('Paired saved inputs differ')
            signatures[key] = state.hexdigest()
            if row['mode'] == 'ego':
                ego_predictions[(row['model'],) + key] = arrays['prediction'].copy()
    for (label, case, regime), prediction in ego_predictions.items():
        if label != baseline:
            ego_checks.append(dict(model=label, case=case, regime=regime,
                exact=np.array_equal(prediction, ego_predictions[baseline, case, regime])))
    passed = all(maximum[k] <= TOLERANCES[k] for k in METRICS)
    if not passed:
        raise ValueError('Independent NumPy FK metric check failed: ' + str(maximum))
    return dict(status='independent_CPU_passed', rows=len(rows), maximum_difference=maximum,
        tolerances=TOLERANCES, paired_saved_inputs_exact=True, EGO_retention=ego_checks,
        CUDA_used=False, source_sha256={str(Path(module.__file__).resolve()): file_sha(module.__file__)
            for module in (refinement_guard, refinement_evaluation)})


def assess(rows, weights, baseline, candidate, phase, *, ego_checks=()):
    base, proposed = (groups(rows, label, phase, weights) for label in (baseline, candidate))
    checks = []
    def check(name, passed, **values):
        checks.append(dict(name=name, passed=bool(passed), **values))
    limits = dict(hidden_mpjpe_mm=(5., .05), visible_mpjpe_mm=(1., .02),
        foot_joint_speed_gt_low_m_s=(.01, .05), acceleration_error_mm_s2=(1., .05))
    for key in base['groups']:
        for metric, (absolute, relative) in limits.items():
            b, p = (item['groups'][key]['metrics'][metric]['mean'] for item in (base, proposed))
            cap = b + max(absolute, relative * b) if b is not None else None
            check(key + '/' + metric, (b is None and p is None) or (p is not None and cap is not None and p <= cap),
                  original=b, candidate=p, limit=cap)
    for regime in REGIMES:
        b, p = (v['natural_primary'][regime] for v in (base, proposed))
        check('natural_primary/' + regime + '/nonworsening', p <= b, original=b, candidate=p)
    b, p = (v['natural_primary']['combined'] for v in (base, proposed))
    gain = (b - p) / b if b else 0.
    check('natural_primary/combined/gain3percent', gain >= .03, relative_gain=gain, original=b, candidate=p)
    original = {(r['case'], r['regime']): r for r in rows if r['model'] == baseline and r['review_panel'] == phase}
    proposed_rows = [r for r in rows if r['model'] == candidate and r['review_panel'] == phase]
    paired = []
    for row in proposed_rows:
        old = original[row['case'], row['regime']]
        b, p, ref = number(old, 'hidden_mpjpe_mm'), number(row, 'hidden_mpjpe_mm'), number(row, 'hidden_mpjpe_mm', 'reference_metrics')
        if b is None:
            continue
        paired.append(dict(case=row['case'], dataset=row['dataset'], regime=row['regime'], sequence_id=row['sequence_id'],
            start=row['start'], original_hidden_mm=b, candidate_hidden_mm=p, reference_hidden_mm=ref,
            candidate_minus_original_mm=p - b, original_worse_reference50=b >= ref + 50,
            candidate_worse_reference50=p >= ref + 50))
    new_bad = [r for r in paired if r['candidate_minus_original_mm'] >= 50]
    check('no_new_original_plus50mm', not new_bad, cases=new_bad)
    severe = {}
    for regime in REGIMES:
        part = [r for r in paired if r['regime'] == regime]
        b, p = sum(r['original_worse_reference50'] for r in part), sum(r['candidate_worse_reference50'] for r in part)
        severe[regime] = dict(original=b, candidate=p)
        check('reference_plus50_count/' + regime, p <= b, original=b, candidate=p)
    if phase == 'dev':
        for regime in REGIMES:
            yoga = next(r for r in paired if r['sequence_id'] == 'Pavallion_006_yoga1' and r['start'] == 640 and r['regime'] == regime)
            b, p, ref = (yoga[k] for k in ('original_hidden_mm', 'candidate_hidden_mm', 'reference_hidden_mm'))
            passed = p <= ref + 50 and p <= .5 * b if regime == 'noisy' else p <= b + 5
            check('known_yoga/' + regime, passed, original=b, candidate=p, reference=ref)
        retained = [r for r in ego_checks if r['model'] == candidate]
        check('ego6_exact', len(retained) == 6 and all(r['exact'] for r in retained), results=retained)
    return dict(passed=all(c['passed'] for c in checks), candidate_label=candidate, phase=phase,
        baseline_summary=base, candidate_summary=proposed, checks=checks,
        failed_checks=[r for r in checks if not r['passed']], severe_reference50_counts=severe,
        paired_hidden=sorted(paired, key=lambda r: r['candidate_minus_original_mm'], reverse=True))


def choose(results, preferred):
    eligible = [label for label, result in results.items() if result['passed']]
    if not eligible:
        return None
    score = lambda label: results[label]['candidate_summary']['natural_primary']['combined']
    best = min(eligible, key=score)
    if preferred in eligible and abs(score(preferred) - score(best)) < 1.:
        return preferred
    return best


def freeze_dev_choice(out):
    out = Path(out)
    selection, rules, weights, labels = binding(out)
    destination = out / 'DEV_CHOICE.json'
    identities = dict(selection_sha256=file_sha(out / 'SELECTION.json'), acceptance_sha256=file_sha(out / 'ACCEPTANCE.json'),
                      review_source_sha256=file_sha(__file__))
    if destination.exists():
        result = read(destination)
        if result['identities'] != identities:
            raise ValueError('Immutable dev choice binding changed')
        rows = phase_rows(out, 'dev', [result['baseline_label'], *result['candidates']])
        if digest(rows) != result['selected_rows_sha256']:
            raise ValueError('Frozen dev prediction records changed')
        return result
    baseline = labels[rules['baseline_policy']]
    candidates = [labels[p] for p in rules['candidate_policies']]
    rows = phase_rows(out, 'dev', [baseline, *candidates])
    expected_rows(rows, selection, [baseline, *candidates], 'dev')
    cpu = cpu_verify(rows, baseline)
    results = {label: assess(rows, weights, baseline, label, 'dev', ego_checks=cpu['EGO_retention']) for label in candidates}
    selected = choose(results, labels.get('exo_patch_off'))
    policy = next((p for p, label in labels.items() if label == selected), None)
    result = dict(protocol=PROTOCOL, frozen_at=dt.datetime.now().astimezone().isoformat(), identities=identities,
        status='dev_selected' if selected else 'dev_NO', selected_label=selected, selected_policy=policy,
        baseline_label=baseline, metrics_read_scope=['dev', 'ego_retention'], confirmation_metrics_read=False,
        selected_rows_sha256=digest(rows), rows_used=len(rows), natural_dataset_weights=weights,
        candidates=results, independent_CPU=cpu, TEST_used=False, optimizer_updates=0, automatic_promotion=False)
    write_new(destination, result)
    return result


def confirmation_report(out):
    out = Path(out)
    if not (out / 'DEV_CHOICE.json').exists():
        raise ValueError('Freeze dev choice before reading confirmation')
    selected = freeze_dev_choice(out)
    if selected['selected_label'] is None:
        raise ValueError('Dev selected NO; confirmation cannot rescue or choose another policy')
    selection, _, weights, _ = binding(out)
    labels = [selected['baseline_label'], selected['selected_label']]
    rows = phase_rows(out, 'confirmation', labels)
    expected_rows(rows, selection, labels, 'confirmation')
    cpu = cpu_verify(rows, selected['baseline_label'])
    review = assess(rows, weights, *labels, 'confirmation')
    result = dict(protocol=PROTOCOL, reviewed_at=dt.datetime.now().astimezone().isoformat(),
        status='confirmation_passed' if review['passed'] else 'confirmation_FAILED',
        DEV_CHOICE_sha256=file_sha(out / 'DEV_CHOICE.json'), identities=selected['identities'],
        selected_label=selected['selected_label'], selected_policy=selected['selected_policy'],
        no_runner_up_selection=True, selected_rows_sha256=digest(rows), rows_used=len(rows),
        review=review, independent_CPU=cpu, TEST_used=False, optimizer_updates=0, automatic_promotion=False)
    path = out / 'CONFIRMATION_REVIEW.json'
    if path.exists():
        prior = read(path)
        if prior['selected_rows_sha256'] != result['selected_rows_sha256'] or prior['DEV_CHOICE_sha256'] != result['DEV_CHOICE_sha256']:
            raise ValueError('Cannot replace a completed confirmation')
        return prior
    write_new(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze-dev-choice', 'confirmation'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze_dev_choice(args.out) if args.command == 'freeze-dev-choice' else confirmation_report(args.out)
    print(json.dumps({k: result.get(k) for k in ('status', 'selected_label', 'selected_policy', 'rows_used')}, indent=2))


if __name__ == '__main__':
    main()
