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
from contextvars import ContextVar
from functools import wraps
import hashlib
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
POLICIES = ('original', 'near_only', 'feedback_off', 'cls_off', 'patch_off', 'dynamic_off',
            'exo_patch_off', 'exo_static_global_off')
EXO_POLICIES = ('original', 'cls_off', 'patch_off', 'dynamic_off')
EXO_REPAIR_POLICIES = ('original', 'exo_patch_off', 'exo_static_global_off')
EXO_PROTOCOL = 'frozen-EXO-scene-branch-diagnostic-v1'
EXO_DEFAULT_OUT = review.BASE / 'runs/native30_exo_scene_repair_v1/branch_diagnostic'
DEFAULT_OUT = review.BASE / 'runs/native30_ego_scene_feedback_review'
read, write, require = review.read, review.write, review.require


def _explicit_exo_rows(batch, batch_size):
    modes = batch.get('mode') if isinstance(batch, dict) else None
    require(isinstance(modes, (tuple, list)) and len(modes) == batch_size
        and all(mode in ('ego', 'exo') for mode in modes),
        'EXO scene policy requires one explicit ego/exo mode per batch row')
    return tuple(i for i, mode in enumerate(modes) if mode == 'exo')


def _zero_exo_rows(output, selected):
    """EGO-only is the identical tensor; mixed batches retain strides/gradients."""
    import torch
    if not selected:
        return output
    indices = torch.tensor(selected, device=output.device, dtype=torch.long)
    result = output.clone(memory_format=torch.preserve_format)
    result.index_fill_(0, indices, 0.)
    return result


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
    elif policy in ('exo_patch_off', 'exo_static_global_off'):
        active = ContextVar('native30_exo_scene_policy_rows', default=None)
        original_core = model._core_pass
        restore.append(lambda: setattr(model, '_core_pass', original_core))

        @wraps(original_core)
        def core_pass(self, *args, **kwargs):
            noisy = kwargs.get('noisy', args[0] if args else None)
            batch = kwargs.get('batch', args[2] if len(args) > 2 else None)
            require(torch.is_tensor(noisy), 'Scene policy core requires a tensor diffusion state')
            selected = _explicit_exo_rows(batch, len(noisy))
            token = active.set((len(noisy), selected))
            try:
                return original_core(*args, **kwargs)
            finally:
                active.reset(token)

        def patch_output(module, inputs, output):
            context = active.get()
            require(context is not None and len(output) == context[0],
                'Patch policy must run inside an explicit-mode core pass')
            return _zero_exo_rows(output, context[1])

        model._core_pass = MethodType(core_pass, model)
        handle = model.patch_condition.register_forward_hook(patch_output)
        restore.append(handle.remove)
        if policy == 'exo_static_global_off':
            original_scene = model.scene_tokens
            restore.append(lambda: setattr(model, 'scene_tokens', original_scene))

            @wraps(original_scene)
            def exo_scene_tokens(self, batch):
                summary, tokens, valid, dynamic = original_scene(batch)
                selected = _explicit_exo_rows(batch, len(summary))
                return _zero_exo_rows(summary, selected), tokens, valid, dynamic

            model.scene_tokens = MethodType(exo_scene_tokens, model)
    metadata = dict(feedback_policy=policy, feedback_policy_protocol=POLICY_PROTOCOL,
        feedback_policy_source_sha256=sha256(__file__),
        feedback_local_radius_m=.6, scene_context_unchanged=policy not in POLICIES[3:],
        local_scene_conditions_unchanged=True, reversible=True,
        explicit_mode_routing=policy in EXO_REPAIR_POLICIES[1:],
        training_and_inference_same_policy=True,
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
        scope=plan.get('selection_scope', 'Three fixed VAL windows, clean/noisy, original and three isolated global scene branches. Mechanism diagnosis only.'),
        automatic_checkpoint_promotion=False)
    write(out / 'SUMMARY.json', result)
    text = ['冻结 EXO 场景分支定位；仅官方 VAL，原10k权重不变，不训练、不读TEST。', '',
        f"{len(plan['entries'])}个EXO窗口、{len(plan['ego_entries'])}个EGO锚点 × clean/noisy × {len(plan['models'])}种策略，计划上限{plan['planned_total_predictions']}次预测；当前{len(rows)}次。",
        '全部保持use_scene=True、局部查询/BPS/region/feedback；策略和适用模式由PLAN逐项声明。',
        plan.get('selection_scope', '每数据集仅一个窗口，只用于机制定位，不代表全验证集或最终版。'), '',
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
def exo_evaluator(out, label=None, phase='all'):
    """Adapt the sealed paired evaluator via reversible loader/report hooks."""
    from . import refinement_evaluation as paired, exo_longrun
    original_load, original_summary, original_verify = exo_longrun.load_inference_model, paired.summarize, paired.verify
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

    def verify(path):
        plan = original_verify(path)
        require(phase in ('all', 'dev', 'confirmation'), 'Unknown evaluation phase')
        if phase == 'all':
            return plan
        require(plan.get('selection_file'), 'Phase filtering requires a declared selection')
        return dict(plan, entries=[e for e in plan['entries'] if e['review_panel'] == phase],
            ego_entries=plan['ego_entries'] if phase == 'dev' else [])

    with ExitStack() as stack:
        stack.enter_context(patch.object(paired, 'PROTOCOL', EXO_PROTOCOL))
        stack.enter_context(patch.object(paired, 'summarize', summarize))
        stack.enter_context(patch.object(paired, 'verify', verify))
        stack.enter_context(patch.object(exo_longrun, 'load_inference_model', load_model))
        try:
            yield paired
        finally:
            for reference in loaded:
                model = reference()
                if model is not None:
                    remove_scene_feedback_policy(model)


def _read_exo_selection(path, source):
    path = Path(path).resolve(); selection = read(path)
    require(selection['protocol'] == 'exo-patch-suppression-identity-only-VAL-selection-v1'
        and selection['split'] == 'validation' and not selection['TEST_used']
        and selection['optimizer_updates'] == 0 and selection['frames'] == 192
        and selection['fps'] == 30 and selection['regimes'] == ['clean', 'noisy'],
        'Only frozen identity-only official VAL selections are admitted')
    payload = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                            allow_nan=False).encode()).hexdigest()
    require(digest == selection['selection_sha256'], 'Selection internal digest differs')
    manifest_path = Path(selection['source_manifest']).resolve()
    require(manifest_path == (Path(source) / 'prepared/panels.json').resolve()
        and sha256(manifest_path) == selection['source_manifest_sha256'], 'Selection source manifest changed')
    manifest = read(manifest_path)
    rows = selection['entries'] + selection['ego_entries']
    require(len({e['case'] for e in rows}) == len(rows), 'Duplicate selected case IDs')
    for entry in rows:
        original = manifest
        for key in entry['source_manifest_pointer'].strip('/').split('/'):
            original = original[int(key)] if isinstance(original, list) else original[key]
        row_hash = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode()).hexdigest()
        require(row_hash == entry['source_row_sha256'], 'Selected source row changed')
        require(all(entry.get(k) == v for k, v in original.items() if k != 'case'),
            'Selection altered source observation/seed/identity fields')
    require(all(e['mode'] == 'exo' and e['review_panel'] in ('dev', 'confirmation')
        for e in selection['entries']), 'Invalid EXO phase')
    require(all(e['mode'] == 'ego' for e in selection['ego_entries'])
        and len(selection['ego_entries']) == 3, 'Three EGO retention anchors required')
    dev = {(e['dataset'], e['recording']) for e in selection['entries'] if e['review_panel'] == 'dev'}
    confirmation = {(e['dataset'], e['recording']) for e in selection['entries'] if e['review_panel'] == 'confirmation'}
    require(dev and confirmation and not dev & confirmation, 'Dev/confirmation recordings overlap')
    return selection


def prepare_exo(out, source=None, case_ids=(1, 4, 8), policies=None, *, selection=None, acceptance=None):
    from . import refinement_evaluation as paired
    out = Path(out).resolve(); source = Path(source or paired.DEFAULT_SOURCE).resolve()
    policies = tuple(policies or (EXO_REPAIR_POLICIES if selection else EXO_POLICIES))
    case_ids = tuple(case_ids)
    require(case_ids and len(set(case_ids)) == len(case_ids), 'Unique fixed case IDs required')
    require(policies and len(set(policies)) == len(policies)
        and all(p in EXO_POLICIES + EXO_REPAIR_POLICIES[1:] for p in policies) and policies[0] == 'original',
        'EXO policies must begin with original and contain only declared global branches')
    selection_path = Path(selection).resolve() if selection else None
    if selection_path:
        choice = _read_exo_selection(selection_path, source)
        selected, ego = choice['entries'], choice['ego_entries']
        acceptance = Path(acceptance or selection_path.with_name('ACCEPTANCE.json')).resolve()
        require(acceptance.is_file(), 'Frozen acceptance declaration required before preparing repair validation')
        acceptance_rules = read(acceptance)
        require(acceptance_rules['selection_manifest_sha256'] == sha256(selection_path)
            and not acceptance_rules['TEST_used'] and acceptance_rules['optimizer_updates'] == 0,
            'Acceptance declaration must bind this frozen VAL selection')
    else:
        manifest = read(source / 'prepared/panels.json')
        entries, _ = paired.select_entries(manifest)
        by_case = {row['case']: row for row in entries}
        require(all(i in by_case for i in case_ids), 'Case ID outside frozen fixed ordering')
        selected = [dict(by_case[i], review_panel='mechanism_diagnostic') for i in case_ids]
        ego = []
    require(all(e['mode'] == 'exo' for e in selected), 'EXO inputs only')
    with exo_evaluator(out) as evaluator_module:
        with patch.object(evaluator_module, 'select_entries', lambda unused: (selected, ego)):
            plan = evaluator_module.prepare(out, source, trial=None)
        original = plan['models']['base10k']
        baseline_label = 'original' if selection_path else 'base10k'
        models = {(baseline_label if p == 'original' else p): dict(original,
                  feedback_policy=p, architecture='original',
                  identity_kind='same_frozen_weights_with_inference_policy') for p in policies}
        plan.update(models=models, regimes=['clean', 'noisy'], scene_variants=['scene_on'],
            ego_entries=ego, case_ids=[e['case'] for e in selected], feedback_policies=list(policies),
            planned_predictions_per_model=(len(selected) + len(ego)) * 2,
            planned_total_predictions=(len(selected) + len(ego)) * 2 * len(models),
            selection='Explicit existing fixed VAL case IDs; known yoga retained for mechanism diagnosis. Not an unbiased effectiveness estimate.',
            policy_protocol=POLICY_PROTOCOL, local_scene_preserved=True,
            checkpoint_weights_unchanged=True, EGO_scope='No EGO predictions or policy installation in this EXO diagnostic.',
            acceptance='Mechanism localization across matched clean/noisy examples only; no automatic policy or checkpoint promotion.')
        if selection_path:
            plan.update(selection_file=str(selection_path), selection_file_sha256=sha256(selection_path),
                acceptance_file=str(acceptance), acceptance_file_sha256=sha256(acceptance),
                selection_scope='Fixed12 dev + independent recording-disjoint rotating12 confirmation + three EGO retention anchors. Dev chooses a policy before reading candidate confirmation.',
                baseline_label=baseline_label, historical_comparison='Disabled for this selection: rotating case numbers overlap fixed historical indices. Baseline is freshly paired original10k.',
                EGO_scope='Every candidate must match all three original EGO anchors exactly under clean/noisy.',
                allowed_prediction_upper_bound=(len(selected) + len(ego)) * 2 * len(models))
            plan['source_files_sha256'].update({str(selection_path): sha256(selection_path),
                                              str(acceptance): sha256(acceptance)})
        plan['source_sha256'].update(dependency_closure([Path(__file__)]))
        write(out / 'PLAN.json', plan)
        # Registration retains strict loading/config/step checks for each label.
        write(out / 'MODELS.json', {})
        evaluator_module.verify(out)
        evaluator_module.status(out, 'prepared', declared_models=list(models),
            planned_predictions=plan['planned_total_predictions'])
    return plan


def evaluate_exo(out, *, device='cuda', deadline='99:99', max_predictions=0, labels=None, phase='all'):
    from . import refinement_evaluation as paired
    out = Path(out); plan = read(out / 'PLAN.json')
    labels = list(labels or plan['models'])
    require(all(label in plan['models'] for label in labels), 'Undeclared policy label')
    before = len(paired.rows_for(out))
    for label in labels:
        completed = len(paired.rows_for(out)) - before
        if max_predictions and completed >= max_predictions:
            break
        with exo_evaluator(out, label, phase=phase) as evaluator_module:
            evaluator_module.evaluate(out, label, device=device, deadline=deadline,
                max_predictions=max_predictions - completed if max_predictions else 0)
    with exo_evaluator(out) as evaluator_module:
        result = evaluator_module.summarize(out)
        selected_entries = plan['entries'] if phase == 'all' else [e for e in plan['entries'] if e['review_panel'] == phase]
        ego = plan['ego_entries'] if phase != 'confirmation' else []
        expected_per_model = (len(selected_entries) + len(ego)) * len(plan['regimes'])
        selected_cases = {e['case'] for e in selected_entries + ego}
        phase_counts = {label: sum(r['model'] == label and r['case'] in selected_cases
                       for r in paired.rows_for(out)) for label in labels}
        phase_complete = all(value == expected_per_model for value in phase_counts.values())
        result.update(last_phase=phase, phase_complete=phase_complete, phase_models=labels,
            phase_expected_predictions=expected_per_model * len(labels), phase_completed_per_model=phase_counts)
        write(out / 'SUMMARY.json', result)
        evaluator_module.status(out, 'phase_completed_pending_CPU_recheck' if phase_complete else 'partial',
            completed_predictions=len(paired.rows_for(out)),
            planned_predictions=plan['planned_total_predictions'], phase=phase,
            phase_expected_predictions=expected_per_model * len(labels), phase_completed_per_model=phase_counts)
    return result


def recheck_exo(out):
    """Independent FK plus mandatory, non-vacuous EGO exactness for repairs."""
    import numpy as np
    out = Path(out)
    with exo_evaluator(out) as paired:
        paired.recheck(out)
        plan = paired.verify(out); rows = paired.rows_for(out)
    if not plan.get('selection_file'):
        return read(out / 'CPU_RECHECK.json')
    report = read(out / 'CPU_RECHECK.json'); exact = []
    try:
        baseline = plan['baseline_label']
        expected = {(e['case'], regime) for e in plan['ego_entries'] for regime in plan['regimes']}
        original = {(r['case'], r['regime']): r for r in rows
                    if r['mode'] == 'ego' and r['model'] == baseline}
        require(set(original) == expected, 'All original EGO anchors must be present before repair acceptance')
        active_labels = {r['model'] for r in rows} - {baseline}
        require(active_labels, 'EGO retention cannot pass without a candidate')
        for label in sorted(active_labels):
            candidate = {(r['case'], r['regime']): r for r in rows
                         if r['mode'] == 'ego' and r['model'] == label}
            require(set(candidate) == expected, 'Missing candidate EGO anchors: ' + label)
            for key in sorted(expected):
                require(candidate[key]['input_sha256'] == original[key]['input_sha256'], 'EGO inputs differ')
                with np.load(candidate[key]['artifact'], allow_pickle=False) as a, \
                        np.load(original[key]['artifact'], allow_pickle=False) as b:
                    equal = np.array_equal(a['prediction'], b['prediction'])
                exact.append(dict(model=label, case=key[0], regime=key[1], exact=bool(equal)))
        require(exact and all(r['exact'] for r in exact), 'An EXO-only policy changed an EGO output')
        report.update(EGO_retention=exact, EGO_retention_expected=len(expected) * len(active_labels),
            EGO_retention_nonvacuous=True)
    except Exception as error:
        report.update(status='independent_CPU_FAILED', EGO_retention=exact,
            EGO_retention_nonvacuous=False, EGO_retention_error=str(error))
        write(out / 'CPU_RECHECK.json', report)
        raise
    write(out / 'CPU_RECHECK.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','evaluate','recheck','report','compare'))
    parser.add_argument('--backend', choices=('ego', 'exo'), default='ego')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--policies', nargs='+', choices=POLICIES)
    parser.add_argument('--case-ids', type=int, nargs='+', default=[1,4,8])
    parser.add_argument('--selection', type=Path, help='Frozen fixed/dev and rotating/confirmation VAL declaration')
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--phase', choices=('all','dev','confirmation'), default='all')
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
            result = prepare_exo(args.out, args.source, args.case_ids, args.policies,
                selection=args.selection, acceptance=args.acceptance)
            print(json.dumps(dict(planned_predictions=result['planned_total_predictions'], models=list(result['models']))))
        elif args.command == 'evaluate':
            evaluate_exo(args.out, device=args.device, deadline=args.deadline,
                max_predictions=args.max_predictions, labels=args.model, phase=args.phase)
        elif args.command == 'compare':
            raise ValueError('EXO policies are already paired in one output; use report/recheck')
        else:
            if args.command == 'recheck':
                recheck_exo(args.out)
            else:
                with exo_evaluator(args.out) as evaluator_module:
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
