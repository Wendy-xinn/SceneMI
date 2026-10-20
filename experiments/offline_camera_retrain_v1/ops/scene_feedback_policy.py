"""Reusable frozen EGO/EXO scene-policy evaluation, without changing weights.

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
import weakref

from . import ego_scene_review as review
from .dependencies import dependency_closure, sha256

PROTOCOL = 'frozen-head-only-ego-scene-feedback-policy-review-v1'
POLICY_PROTOCOL = 'native30-scene-feedback-evidence-gate-v1'
POLICIES = ('original', 'near_only', 'feedback_off', 'cls_off', 'patch_off', 'dynamic_off')
EXO_POLICIES = ('original', 'cls_off', 'patch_off', 'dynamic_off')
EXO_PROTOCOL = 'frozen-EXO-scene-branch-diagnostic-v1'
EXO_DEFAULT_OUT = review.BASE / 'runs/native30_exo_scene_repair_v1/branch_diagnostic'
DEFAULT_OUT = review.BASE / 'runs/native30_ego_scene_feedback_review'
read, write, require = review.read, review.write, review.require


def apply_scene_feedback_policy(model, policy, *, architecture='original'):
    import torch
    require(policy in POLICIES and architecture == 'original'
        and not hasattr(model, 'scene_cross_attention'), 'Only original architecture feedback policies are supported')
    require(not hasattr(model, '_scene_feedback_policy'), 'A feedback policy is already installed')
    require(getattr(model, '_sampling_scene', None) is None, 'Install policy before filling scene cache')
    restore = []
    if policy == 'near_only':
        original = model.local_geometry
        restore.append(lambda: setattr(model, 'local_geometry', original))

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
        original = model.feedback_enabled
        restore.append(lambda: setattr(model, 'feedback_enabled', original))
        model.feedback_enabled = False
    elif policy == 'cls_off':
        original = model.scene_tokens
        restore.append(lambda: setattr(model, 'scene_tokens', original))

        @wraps(original)
        def scene_tokens(self, batch):
            summary, tokens, valid, dynamic = original(batch)
            return torch.zeros_like(summary), tokens, valid, dynamic

        model.scene_tokens = MethodType(scene_tokens, model)
    elif policy == 'patch_off':
        original = model.patch_enabled
        restore.append(lambda: setattr(model, 'patch_enabled', original))
        model.patch_enabled = False
    elif policy == 'dynamic_off':
        handle = model.dynamic_condition.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output))
        restore.append(handle.remove)
    metadata = dict(feedback_policy=policy, feedback_policy_protocol=POLICY_PROTOCOL,
        feedback_policy_source_sha256=sha256(__file__),
        feedback_local_radius_m=.6, scene_context_unchanged=policy not in EXO_POLICIES[1:],
        local_scene_conditions_unchanged=True, reversible=True,
        checkpoint_weights_unchanged=True, reference_unchanged=True,
        analytical_guidance_unchanged=True)
    model._scene_feedback_policy = metadata
    model._scene_feedback_restore = restore
    return metadata


def remove_scene_feedback_policy(model):
    """Undo only process-local methods, flags, and hooks; preserve state_dict."""
    require(getattr(model, '_sampling_scene', None) is None, 'Cannot change policy inside active sampling')
    for restore in reversed(getattr(model, '_scene_feedback_restore', ())):
        restore()
    for name in ('_scene_feedback_restore', '_scene_feedback_policy'):
        if hasattr(model, name):
            delattr(model, name)


@contextmanager
def scene_policy(model, policy):
    metadata = apply_scene_feedback_policy(model, policy)
    try:
        yield metadata
    finally:
        remove_scene_feedback_policy(model)


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


def _exo_summary(out, paired, original_summary):
    """Correct the general evaluator's historical fixed-86 reporting scope."""
    result = original_summary(out)
    out = Path(out); plan = read(out / 'PLAN.json'); rows = paired.rows_for(out)
    expected = plan['planned_predictions_per_model']
    counts = {label: sum(r['model'] == label for r in rows) for label in plan['models']}
    result.update(protocol=EXO_PROTOCOL, planned_predictions=plan['planned_total_predictions'],
        completed_per_policy=counts, partial=any(value != expected for value in counts.values()),
        policies={label: row['feedback_policy'] for label, row in plan['models'].items()},
        scope='Three fixed VAL windows, clean/noisy, original and three isolated global scene branches. Mechanism diagnosis only.',
        automatic_checkpoint_promotion=False)
    write(out / 'SUMMARY.json', result)
    text = ['冻结 EXO 场景分支定位；仅官方 VAL，原10k权重不变，不训练、不读TEST。', '',
        f"{len(plan['entries'])}个固定窗口 × clean/noisy × {len(plan['models'])}种策略，共{plan['planned_total_predictions']}次预测；当前{len(rows)}次。",
        '全部保持use_scene=True、局部查询/BPS/region/feedback；仅逐支关闭CLS、静态patch或dynamic summary。',
        '每数据集仅一个窗口，只用于机制定位，不代表全验证集或最终版。', '',
        '| 策略/输入/数据 | 遮挡mm | 可见mm | 支撑期足速m/s | 加速度mm/s² |',
        '|---|---:|---:|---:|---:|']
    for key, group in result['groups'].items():
        values = group['metrics']
        cells = [values[name]['mean'] for name in ('hidden_mpjpe_mm', 'visible_mpjpe_mm',
                 'foot_joint_speed_gt_low_m_s', 'acceleration_error_mm_s2')]
        text.append('| ' + key + ' | ' + ' | '.join('NA' if v is None else f'{v:.3f}' for v in cells) + ' |')
    text += ['', '指标须经独立CPU复核；不自动推广任何策略或checkpoint。']
    (out / 'REPORT_zh.md').write_text('\n'.join(text) + '\n')
    return result


@contextmanager
def exo_evaluator(out, label=None):
    """Adapt the sealed paired evaluator via reversible loader/report hooks."""
    from . import refinement_evaluation as paired, exo_longrun
    original_load, original_summary = exo_longrun.load_inference_model, paired.summarize
    loaded = []

    def load_model(*args, **kwargs):
        require(label is not None, 'An EXO model policy label is required before loading')
        declaration = read(Path(out) / 'PLAN.json')['models'][label]
        model, metadata = original_load(*args, **kwargs)
        metadata = dict(metadata)
        metadata.update(apply_scene_feedback_policy(model, declaration['feedback_policy']))
        # The original register() verifies arm==label for non-base labels.
        # This is explicitly an inference-policy identity, never a trained arm.
        metadata.update(arm=label, arm_is_inference_policy_label=True)
        loaded.append(weakref.ref(model))
        return model, metadata

    def summarize(path):
        return _exo_summary(path, paired, original_summary)

    with ExitStack() as stack:
        stack.enter_context(patch.object(paired, 'PROTOCOL', EXO_PROTOCOL))
        stack.enter_context(patch.object(paired, 'summarize', summarize))
        stack.enter_context(patch.object(exo_longrun, 'load_inference_model', load_model))
        try:
            yield paired
        finally:
            for reference in loaded:
                model = reference()
                if model is not None:
                    remove_scene_feedback_policy(model)


def prepare_exo(out, source=None, case_ids=(1, 4, 8), policies=EXO_POLICIES):
    from . import refinement_evaluation as paired
    out = Path(out).resolve(); source = Path(source or paired.DEFAULT_SOURCE).resolve()
    case_ids, policies = tuple(case_ids), tuple(policies)
    require(case_ids and len(set(case_ids)) == len(case_ids), 'Unique fixed case IDs required')
    require(policies and len(set(policies)) == len(policies)
        and all(p in EXO_POLICIES for p in policies) and policies[0] == 'original',
        'EXO policies must begin with original and contain only declared global branches')
    manifest = read(source / 'prepared/panels.json')
    entries, _ = paired.select_entries(manifest)
    by_case = {row['case']: row for row in entries}
    require(all(i in by_case for i in case_ids), 'Case ID outside frozen fixed ordering')
    selected = [dict(by_case[i], review_panel='mechanism_diagnostic') for i in case_ids]
    require(all(e['mode'] == 'exo' for e in selected), 'EXO inputs only')
    with exo_evaluator(out) as evaluator_module:
        with patch.object(evaluator_module, 'select_entries', lambda unused: (selected, [])):
            plan = evaluator_module.prepare(out, source, trial=None)
        original = plan['models']['base10k']
        models = {('base10k' if p == 'original' else p): dict(original,
                  feedback_policy=p, architecture='original',
                  identity_kind='same_frozen_weights_with_inference_policy') for p in policies}
        plan.update(models=models, regimes=['clean', 'noisy'], scene_variants=['scene_on'],
            ego_entries=[], case_ids=list(case_ids), feedback_policies=list(policies),
            planned_predictions_per_model=len(selected) * 2,
            planned_total_predictions=len(selected) * 2 * len(models),
            selection='Explicit existing fixed VAL case IDs; known yoga retained for mechanism diagnosis. Not an unbiased effectiveness estimate.',
            policy_protocol=POLICY_PROTOCOL, local_scene_preserved=True,
            checkpoint_weights_unchanged=True, EGO_scope='No EGO predictions or policy installation in this EXO diagnostic.',
            acceptance='Mechanism localization across matched clean/noisy examples only; no automatic policy or checkpoint promotion.')
        plan['source_sha256'].update(dependency_closure([Path(__file__)]))
        write(out / 'PLAN.json', plan)
        # Registration retains strict loading/config/step checks for each label.
        write(out / 'MODELS.json', {})
        evaluator_module.verify(out)
        evaluator_module.status(out, 'prepared', declared_models=list(models),
            planned_predictions=plan['planned_total_predictions'])
    return plan


def evaluate_exo(out, *, device='cuda', deadline='99:99', max_predictions=0, labels=None):
    from . import refinement_evaluation as paired
    out = Path(out); plan = read(out / 'PLAN.json')
    labels = list(labels or plan['models'])
    require(all(label in plan['models'] for label in labels), 'Undeclared policy label')
    before = len(paired.rows_for(out))
    for label in labels:
        completed = len(paired.rows_for(out)) - before
        if max_predictions and completed >= max_predictions:
            break
        with exo_evaluator(out, label) as evaluator_module:
            evaluator_module.evaluate(out, label, device=device, deadline=deadline,
                max_predictions=max_predictions - completed if max_predictions else 0)
    with exo_evaluator(out) as evaluator_module:
        result = evaluator_module.summarize(out)
        evaluator_module.status(out, 'partial' if result['partial'] else 'completed_pending_CPU_recheck',
            completed_predictions=len(paired.rows_for(out)),
            planned_predictions=plan['planned_total_predictions'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','evaluate','recheck','report','compare'))
    parser.add_argument('--backend', choices=('ego', 'exo'), default='ego')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--policies', nargs='+', choices=POLICIES)
    parser.add_argument('--case-ids', type=int, nargs='+', default=[1,4,8])
    parser.add_argument('--model', nargs='+', help='EXO declared labels; default all four policies')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--deadline', default='99:99', help='Local HH:MM; 99:99 disables same-day cutoff')
    parser.add_argument('--baseline', type=Path, default=review.DEFAULT_OUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.5)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .8 and args.max_predictions >= 0, 'Invalid GPU bounds')
    if args.backend == 'exo':
        args.out = args.out or EXO_DEFAULT_OUT
        if args.command == 'prepare':
            result = prepare_exo(args.out, args.source, args.case_ids, args.policies or EXO_POLICIES)
            print(json.dumps(dict(planned_predictions=result['planned_total_predictions'], models=list(result['models']))))
        elif args.command == 'evaluate':
            evaluate_exo(args.out, device=args.device, deadline=args.deadline,
                max_predictions=args.max_predictions, labels=args.model)
        elif args.command == 'compare':
            raise ValueError('EXO policies are already paired in one output; use report/recheck')
        else:
            with exo_evaluator(args.out) as evaluator_module:
                if args.command == 'recheck':
                    evaluator_module.recheck(args.out)
                else:
                    evaluator_module.verify(args.out); evaluator_module.summarize(args.out)
        return
    args.out = args.out or DEFAULT_OUT
    args.source = args.source or review.DEFAULT_SOURCE
    args.policies = args.policies or ['near_only','feedback_off']
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
