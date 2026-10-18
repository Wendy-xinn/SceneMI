"""Reuse the frozen EGO evaluator for explicitly declared scene-feedback gates.

Only original native30 checkpoints are admitted.  ``near_only`` retains every
scene context/condition but permits learned feedback only for regions covered
by the original 0.08/0.25/0.60 m local-anchor query. ``feedback_off`` disables
the feedback head while retaining learned scene conditioning. No source model,
checkpoint, reference, analytical-guidance setting or historical plan is edited.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from functools import wraps
import json
from pathlib import Path
import statistics
from types import MethodType
from unittest.mock import patch

from . import ego_scene_review as review
from .dependencies import dependency_closure, sha256

PROTOCOL = 'frozen-head-only-ego-scene-feedback-policy-review-v1'
POLICY_PROTOCOL = 'native30-scene-feedback-evidence-gate-v1'
POLICIES = ('original', 'near_only', 'feedback_off')
DEFAULT_OUT = review.BASE / 'runs/native30_ego_scene_feedback_review'
read, write, require = review.read, review.write, review.require


def apply_scene_feedback_policy(model, policy, *, architecture='original'):
    import torch
    require(policy in POLICIES and architecture == 'original'
        and not hasattr(model, 'scene_cross_attention'), 'Only original architecture feedback policies are supported')
    require(not hasattr(model, '_scene_feedback_policy'), 'A feedback policy is already installed')
    if policy == 'near_only':
        original = model.local_geometry

        @wraps(original)
        def local_geometry(self, *args, **kwargs):
            context, bps, valid, geometry = original(*args, **kwargs)
            gated = []
            for item in geometry:
                near = item['valid'].flatten(2).any(-1)
                labels = item['labels']
                regions = torch.stack([near[:, labels == joint].any(-1)
                                       for joint in range(22)], dim=-1)
                gated.append(dict(item, region_valid=regions))
            return context, bps, valid, gated

        model.local_geometry = MethodType(local_geometry, model)
    elif policy == 'feedback_off':
        model.feedback_enabled = False
    metadata = dict(feedback_policy=policy, feedback_policy_protocol=POLICY_PROTOCOL,
        feedback_policy_source_sha256=sha256(__file__),
        feedback_local_radius_m=.6, scene_context_unchanged=True,
        checkpoint_weights_unchanged=True, reference_unchanged=True,
        analytical_guidance_unchanged=True)
    model._scene_feedback_policy = metadata
    return metadata


@contextmanager
def evaluator():
    """Process-scoped extension; leave the sealed evaluator bytes untouched."""
    original_load, original_closure = review.load_model, review.closure

    def load_model(declaration, config, device='cpu'):
        require('feedback_policy' in declaration, 'Missing explicit feedback policy declaration')
        require(declaration['architecture'] == 'original', 'Attention candidates are outside this review')
        model, metadata = original_load(declaration, config, device)
        metadata.update(apply_scene_feedback_policy(model, declaration['feedback_policy'],
                        architecture=declaration['architecture']))
        return model, metadata

    def closure():
        sources = original_closure()
        sources.update(dependency_closure([Path(__file__)]))
        return sources

    with ExitStack() as stack:
        stack.enter_context(patch.object(review, 'PROTOCOL', PROTOCOL))
        stack.enter_context(patch.object(review, 'load_model', load_model))
        stack.enter_context(patch.object(review, 'closure', closure))
        yield


def prepare(out, source=review.DEFAULT_SOURCE, per_dataset=4,
            policies=('near_only', 'feedback_off')):
    require(policies and len(set(policies)) == len(policies)
        and all(policy in POLICIES for policy in policies), 'Invalid policy selection')
    require(not (out / 'PLAN.json').exists(), 'Existing immutable feedback plan')
    original = read(source / 'PLAN.json')['models']['candidate41k']
    declarations = {f'base41k_{policy}': dict(original, architecture='original',
        feedback_policy=policy, arms=['learned_only']) for policy in policies}
    out.mkdir(parents=True, exist_ok=True)
    registry = out / 'MODELS.json'
    write(registry, declarations)
    with evaluator():
        plan = review.prepare(out, source, per_dataset, registry, arms=['learned_only'])
    # This is this new review's own preparation, never a historical plan edit.
    plan.update(feedback_policy_protocol=POLICY_PROTOCOL,
        feedback_policies=list(policies),
        feedback_policy_scope='Context/BPS/valid and checkpoint weights retained; only learned feedback permission changes.',
        near_only_definition='Any valid original local-anchor neighbor within max radius 0.6m; actual query centers can blend reliable tracks.',
        original_reuse='Compare to saved original learned_only EGO baseline by exact paired input fingerprints and saved input arrays; never rerun it here.')
    write(out / 'PLAN.json', plan)
    return plan


def compare(out, baseline):
    """CPU-only matched comparison to separately recorded original predictions."""
    import numpy as np
    with evaluator():
        plan = read(out / 'PLAN.json'); review.verify(plan)
        rows = review.rows_for(out, plan)
    baseline_plan = read(baseline / 'PLAN.json')
    require(baseline_plan['config_sha256'] == plan['config_sha256'], 'Baseline config differs')
    for path, expected in baseline_plan['source_sha256'].items():
        require(sha256(path) == expected, 'Baseline dependency changed: ' + path)
    baseline_rows = review.rows_for(baseline, baseline_plan)
    controls = {}
    for row in baseline_rows:
        declaration = baseline_plan['models'][row['model']]
        if row['arm'] == 'learned_only' and declaration['architecture'] == 'original' \
                and declaration.get('feedback_policy', 'original') == 'original':
            key = row['case'], row['regime']
            require(key not in controls, 'Multiple baseline controls for a window')
            controls[key] = row
    pairs = []
    names = ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion', 'identity_json')
    for row in rows:
        control = controls.get((row['case'], row['regime']))
        if control is None:
            continue
        require(row['checkpoint_sha256'] == control['checkpoint_sha256']
            and row['input_sha256'] == control['input_sha256'] and row['counts'] == control['counts']
            and all(row[name] == control[name] for name in ('dataset','recording','sequence_id','start','source_first','seed','review_panel')),
            'Original/policy pair differs')
        with np.load(row['artifact'], allow_pickle=False) as candidate, np.load(control['artifact'], allow_pickle=False) as original:
            require(all(np.array_equal(candidate[name], original[name]) for name in names), 'Saved pair inputs differ')
        pairs.append(dict(policy=plan['models'][row['model']]['feedback_policy'],
            dataset=row['dataset'], regime=row['regime'], panel=row['review_panel'],
            case=row['case'], original=control['metrics'], candidate=row['metrics']))
    groups = {}
    for policy in plan['feedback_policies']:
        for panel in ('all', 'dev', 'confirmation'):
            for dataset in (*review.DATASETS, 'macro'):
                for regime in plan['regimes']:
                    selected = [r for r in pairs if r['policy'] == policy and r['regime'] == regime
                        and (panel == 'all' or r['panel'] == panel) and (dataset == 'macro' or r['dataset'] == dataset)]
                    if not selected:
                        continue
                    metrics = {}
                    for name in selected[0]['original']:
                        available = [r for r in selected if r['original'][name] is not None and r['candidate'][name] is not None]
                        def mean(which):
                            values = [statistics.mean(r[which][name] for r in available if r['dataset'] == d)
                                for d in review.DATASETS if any(r['dataset'] == d for r in available)]
                            return statistics.mean(values) if values else None
                        a, b = mean('original'), mean('candidate')
                        metrics[name] = dict(original=a, candidate=b, difference=b-a if a is not None else None)
                    groups['/'.join((policy,panel,dataset,regime))] = dict(windows=len(selected),metrics=metrics)
    def verified(folder):
        path = folder / 'CPU_RECHECK.json'
        return path.exists() and read(path)['rows_sha256'] == sha256(folder / 'rows.jsonl')
    result = dict(protocol=PROTOCOL, paired_predictions=len(pairs),
        status='cpu_verified' if verified(out) and verified(baseline) else 'partial_unverified',
        baseline=str(baseline.resolve()), baseline_rows_sha256=sha256(baseline / 'rows.jsonl'),
        candidate_rows_sha256=sha256(out / 'rows.jsonl'), paired_saved_inputs_exact=True,
        paired_recursive_fingerprints_and_denominators_exact=True, groups=groups,
        aggregation='Equal recording within dataset, equal dataset macro on matched windows.', TEST_used=False)
    write(out / 'BASELINE_COMPARISON.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','evaluate','recheck','report','compare'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=review.DEFAULT_SOURCE)
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=['near_only','feedback_off'])
    parser.add_argument('--baseline', type=Path, default=review.DEFAULT_OUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.5)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .8 and args.max_predictions >= 0, 'Invalid GPU bounds')
    if args.command == 'prepare':
        prepare(args.out,args.source,args.per_dataset,args.policies)
    elif args.command == 'compare':
        result = compare(args.out,args.baseline)
        print(json.dumps(dict(status=result['status'],paired_predictions=result['paired_predictions'])))
    else:
        with evaluator():
            if args.command == 'evaluate':
                review.evaluate(args.out,args.resume,args.max_predictions,args.memory_fraction)
            elif args.command == 'recheck':
                review.recheck(args.out)
            else:
                plan=read(args.out/'PLAN.json'); review.verify(plan)
                rows=review.rows_for(args.out,plan)
                path=args.out/'CPU_RECHECK.json'
                checked=path.exists() and read(path)['rows_sha256']==sha256(args.out/'rows.jsonl')
                review.summarize(args.out,plan,rows,checked)


if __name__ == '__main__':
    main()
