"""Fixed recording-level VAL cohorts and original/fusion scene evaluation.

This extends the sealed EGO evaluator in process; historical sources and plans
remain immutable. Observation noise has its own fixed seed. A second DDIM seed
uses a separate immutable output directory and never changes the observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from unittest.mock import patch

from . import ego_scene_review as review
from .dependencies import dependency_closure, sha256

PROTOCOL = 'fixed-recording-ego-scene-fusion-evaluation-v1'
COHORT_PROTOCOL = 'fixed-recording-ego-scene-validation-cohort-v1'
ARCHITECTURES = ('original', 'scene_fusion_v2')
PANELS = ('dev', 'confirmation', 'all')
DEFAULT_COHORT = review.BASE / 'runs/native30_ego_scene_fusion_cohort/COHORT.json'
DEFAULT_OUT = review.BASE / 'runs/native30_ego_scene_fusion_evaluation'
DEFAULT_ACCEPTANCE = review.BASE / 'runs/native30_ego_scene_fusion_v2/ACCEPTANCE.json'
FORBIDDEN_ROOTS = ('native30_ego_scene_attention_trial', 'native30_ego_scene_attention_review')
read, write, require = review.read, review.write, review.require


def recording_key(row):
    """Collapse EgoBody roles to their common recording, within each corpus."""
    return row['dataset'], row.get('recording') or row['sequence_id']


def diffusion_seed(entry, index):
    require(index in (0, 1), 'Only the two prespecified DDIM seeds are admitted')
    identity = [COHORT_PROTOCOL, 'DDIM', index, entry['dataset'], entry['recording'],
                entry['group'], entry['sequence_id'], entry['source_first']]
    return int.from_bytes(hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).digest()[:4], 'little')


def select_cohort(source_entries, *, trumans=8, egobody=8, dev_per_dataset=4):
    require(trumans >= 8 and egobody >= 8 and dev_per_dataset >= 1, 'Insufficient recording coverage')
    selected = []
    for dataset in review.DATASETS:
        seen, chosen = set(), []
        limit = None if dataset == 'rich' else dict(trumans=trumans, egobody=egobody)[dataset]
        for entry in source_entries:
            if entry['dataset'] != dataset or recording_key(entry) in seen:
                continue
            seen.add(recording_key(entry))
            chosen.append(dict(entry, observation_seed=entry['case_seed'],
                diffusion_seeds=[diffusion_seed(entry, i) for i in (0, 1)],
                review_panel='dev' if len(chosen) < dev_per_dataset else 'confirmation'))
            if limit is not None and len(chosen) == limit:
                break
        require(chosen and (limit is None or len(chosen) == limit)
                and len(chosen) > dev_per_dataset, 'Insufficient dev/confirmation recordings: ' + dataset)
        selected.extend(chosen)
    require(len({recording_key(e) for e in selected}) == len(selected), 'Duplicate cohort recording')
    return selected


def split_proof(manifest_rows, selected, eligible_rich):
    train = {recording_key(r) for r in manifest_rows if r['split'] == 'train'}
    validation = {recording_key(r) for r in manifest_rows if r['split'] == 'validation'}
    require(not train & validation, 'TRAIN/VAL share recordings, including across EgoBody roles')
    require(all(recording_key(e) in validation and recording_key(e) not in train for e in selected), 'Cohort is not exclusively VAL')
    by_identity = {(r['group'], r['sequence_id'], str(Path(r['path']).resolve())): r for r in manifest_rows}
    for e in selected:
        row = by_identity.get((e['group'], e['sequence_id'], str(Path(e['path']).resolve())))
        require(row is not None and row['split'] == 'validation', 'Selected source is not VAL')
    actual_rich = {e['recording'] for e in selected if e['dataset'] == 'rich'}
    require(actual_rich == set(eligible_rich), 'Cohort must cover every eligible L192 RICH recording')
    all_rich = {r for d, r in validation if d == 'rich'}
    excluded = []
    for recording in sorted(all_rich - actual_rich):
        matches = [r for r in manifest_rows if r['split'] == 'validation'
            and recording_key(r) == ('rich', recording)]
        excluded.append(dict(recording=recording, maximum_source_frames=max(r['frames'] for r in matches),
            reason='shorter_than_L192' if max(r['frames'] for r in matches) < 192 else 'no_legal_head_observed_L192_window'))
    return dict(TRAIN_VAL_recording_overlap=[], recording_identity='dataset + recording; EgoBody roles collapsed',
        train_recordings=len(train), validation_recordings=len(validation), selected_recordings=len(selected),
        rich_validation_recordings=len(all_rich), rich_eligible_L192_recordings=len(actual_rich),
        rich_exclusions=excluded, TEST_inputs_loaded=False)


def prepare_cohort(path=DEFAULT_COHORT, source=review.DEFAULT_SOURCE, acceptance=DEFAULT_ACCEPTANCE):
    review.common.cpu_setup()
    require(not path.exists(), 'Existing cohort is immutable; choose a new destination')
    source = source.resolve(); original = read(source / 'PLAN.json'); source_cohort = source / 'COHORT.json'
    require(sha256(source_cohort) == original['cohort_sha256'], 'Sealed source identities changed')
    entries = select_cohort(read(source_cohort)['entries'])
    config = original['config']; manifest = review.BASE / config['train_manifest']
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    data = review.common.validation_data(config); review.validate_entries(data, entries)
    eligible_rich = {m.get('recording') or m['sequence_id'] for i, (m, _) in enumerate(data.members['rich'])
        if len(data.starts_for('rich', i, 'ego'))}
    proof = split_proof(manifest_rows, entries, eligible_rich)
    require(read(acceptance)['protocol'] == 'ego-scene-fusion-v2-preregistered-acceptance-v1', 'Unknown acceptance protocol')
    value = dict(protocol=COHORT_PROTOCOL, prepared_at=review.common.now(), entries=entries,
        source=str(source), source_cohort_sha256=sha256(source_cohort), source_plan_sha256=sha256(source / 'PLAN.json'),
        source_manifest=str(manifest.resolve()), source_manifest_sha256=sha256(manifest),
        selection_source=str(Path(__file__).resolve()), selection_source_sha256=sha256(__file__),
        preregistered_acceptance=dict(path=str(acceptance.resolve()), sha256=sha256(acceptance)),
        counts=dict(Counter(e['dataset'] for e in entries)),
        panel_counts=dict(Counter(e['review_panel'] for e in entries)), split_proof=proof,
        frames=192, fps=30, regimes=['clean', 'noisy'], diffusion_seed_indices=[0, 1],
        observation_seed_policy='Original identity-derived case_seed; fixed for both DDIM seeds.',
        diffusion_seed_policy='SHA256 of fixed protocol/index/source identity; both prespecified, never best-of-two.',
        selection='First legal window per recording in the sealed identity-only cohort; all eligible RICH, first8 TRUMANS/EgoBody; never ranked by errors.',
        gate_policy='Only the first4 recordings/dataset (12 dev) are used at interim stage checks; remaining31 confirmation reserved for final stage.',
        historical_scope='Experiment-specific confirmation holdout; previous base-model reports may contain these VAL identities.',
        metric_scope='Joint/foot GT errors and low-speed proxies are not full-scene penetration rates.',
        optimizer_updates=0, TEST_used=False)
    write(path, value)
    return value


def verify_cohort(path):
    cohort = read(path); require(cohort['protocol'] == COHORT_PROTOCOL, 'Unknown cohort protocol')
    for file, digest in [(Path(cohort['source']) / 'COHORT.json', cohort['source_cohort_sha256']),
            (Path(cohort['source']) / 'PLAN.json', cohort['source_plan_sha256']),
            (Path(cohort['source_manifest']), cohort['source_manifest_sha256']),
            (Path(cohort['selection_source']), cohort['selection_source_sha256']),
            (Path(cohort['preregistered_acceptance']['path']), cohort['preregistered_acceptance']['sha256'])]:
        require(sha256(file) == digest, 'Cohort dependency changed: ' + str(file))
    expected = select_cohort(read(Path(cohort['source']) / 'COHORT.json')['entries'])
    require(cohort['entries'] == expected, 'Cohort identities/seeds/panels differ')
    return cohort


def entries_for(cohort, panel, seed_index):
    require(panel in PANELS and seed_index in (0, 1), 'Unsupported panel/seed')
    return [dict(e, case_seed=e['diffusion_seeds'][seed_index], diffusion_seed_index=seed_index)
        for e in cohort['entries'] if panel == 'all' or e['review_panel'] == panel]


def check_declaration(declaration):
    require(declaration['architecture'] in ARCHITECTURES, 'Only original and scene_fusion_v2 architectures are admitted')
    require(not any(root in Path(declaration['checkpoint']).parts for root in FORBIDDEN_ROOTS),
        'Retired attention500 checkpoints are forbidden dependencies')


def fusion_options(saved):
    spec, declared = saved.get('fusion_spec'), saved.get('fusion_options')
    allowed = ('token_dim', 'heads', 'fusion_dim', 'include_global', 'include_local')
    require(isinstance(spec, dict) and isinstance(declared, dict)
        and {'token_dim', 'heads', 'fusion_dim'} <= declared.keys()
        and not declared.keys() - set(allowed), 'Fusion checkpoint needs explicit supported fusion_options')
    require(all(key in spec and spec[key] == value for key, value in declared.items()),
        'fusion_options disagree with saved fusion_spec')
    return {key: spec[key] for key in allowed if key in spec}


def load_model(declaration, config, device='cpu'):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config, model_digest
    check_declaration(declaration)
    require(sha256(declaration['checkpoint']) == declaration['checkpoint_sha256'], 'Checkpoint bytes changed')
    saved = torch.load(declaration['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    require(saved['step'] == declaration['expected_step'] and saved.get('architecture', 'original') == declaration['architecture'], 'Checkpoint step/architecture differs')
    require(review.digest_json(checkpoint_config(saved)) == review.digest_json(config), 'Checkpoint source config differs')
    model = make_model(config); fusion_spec = None
    if declaration['architecture'] == 'scene_fusion_v2':
        from .scene_fusion import attach_scene_fusion, scene_fusion_metadata
        spec = saved.get('fusion_spec')
        require(isinstance(spec, dict) and saved.get('parent_step') == 41000, 'Fusion checkpoint needs complete metadata and parent41000')
        options = fusion_options(saved)
        attach_scene_fusion(model, **options); fusion_spec = scene_fusion_metadata(model)
        require(all(key in spec and spec[key] == value for key, value in fusion_spec.items()), 'Fusion metadata differs from runtime implementation')
    model.load_state_dict(saved['model'], strict=True)
    require(all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in model.state_dict().values()), 'Nonfinite checkpoint tensors')
    digest = model_digest(model)
    require('model_tensor_sha256' not in declaration or declaration['model_tensor_sha256'] == digest, 'Model tensors differ')
    metadata = dict(model_tensor_sha256=digest, fusion_spec=fusion_spec,
        fusion_options=saved.get('fusion_options'), attention_spec=None,
        parent_step=saved.get('parent_step'), trained_guidance_enabled=saved.get('guidance_enabled'),
        stage_plan_sha256=saved.get('stage_plan_sha256'))
    del saved; return model.to(device).eval(), metadata


@contextmanager
def evaluator():
    original_observation = review.common.source_observation
    original_verify = review.verify

    def source_observation(data, builder, entry, regime):
        return original_observation(data, builder, dict(entry, case_seed=entry['observation_seed']), regime)

    def closure():
        sources = review.common.closure()
        sources.update(dependency_closure([Path(__file__), Path(review.__file__),
            Path(__file__).with_name('scene_fusion.py'), Path(__file__).with_name('ego_scene_trial.py')]))
        return sources

    def verify(plan):
        original_verify(plan)
        cohort = verify_cohort(Path(plan['evaluation_cohort']))
        require(sha256(plan['evaluation_cohort']) == plan['evaluation_cohort_sha256']
            and plan['entries'] == entries_for(cohort, plan['evaluation_panel'], plan['diffusion_seed_index']), 'Evaluation cohort/panel/seed changed')
        for declaration in plan['models'].values(): check_declaration(declaration)

    with ExitStack() as stack:
        for name, value in dict(PROTOCOL=PROTOCOL, ARCHITECTURES=ARCHITECTURES,
                load_model=load_model, closure=closure, verify=verify).items():
            stack.enter_context(patch.object(review, name, value))
        stack.enter_context(patch.object(review.common, 'source_observation', source_observation))
        yield


def pin_models(out, models):
    """Pin published atomic-last inodes; source paths are provenance, not live dependencies."""
    require(isinstance(models, dict) and models, 'Models registry must be nonempty')
    directory = out / 'checkpoints'; directory.mkdir(parents=True, exist_ok=True)
    pinned = {}
    for label, declaration in models.items():
        require(re.fullmatch(r'[A-Za-z0-9_-]+', label) is not None, 'Unsafe model label')
        check_declaration(declaration)
        source = Path(declaration['checkpoint']).resolve(); target = directory / (label + '.pt')
        if not target.exists(): os.link(source, target)
        require(sha256(target) == declaration['checkpoint_sha256'], 'Pinned checkpoint differs; choose a new output')
        pinned[label] = dict(declaration, source_checkpoint=str(source),
            source_checkpoint_sha256=declaration['checkpoint_sha256'], checkpoint=str(target.resolve()))
    return pinned


def prepare(out, cohort_path=DEFAULT_COHORT, *, models_path=None, panel='dev', seed_index=0, arms=('learned_only',)):
    require(not (out / 'PLAN.json').exists(), 'Existing evaluation plan is immutable')
    cohort = verify_cohort(cohort_path); selected = entries_for(cohort, panel, seed_index)
    models = read(models_path) if models_path is not None else dict(base41k=dict(
        read(Path(cohort['source']) / 'PLAN.json')['models']['candidate41k'], architecture='original', arms=list(arms)))
    registry = out / 'MODELS_PINNED.json'; write(registry, pin_models(out, models))
    with evaluator(), patch.object(review, 'select_entries', lambda source_entries, per_dataset: selected):
        plan = review.prepare(out, Path(cohort['source']), 4, registry, arms)
    plan['source_sha256'][str(cohort_path.resolve())] = sha256(cohort_path)
    plan.update(evaluation_cohort=str(cohort_path.resolve()), evaluation_cohort_sha256=sha256(cohort_path),
        evaluation_panel=panel, diffusion_seed_index=seed_index, observation_seed_fixed_across_DDIM_seeds=True,
        preregistered_acceptance=cohort['preregistered_acceptance'],
        selection=cohort['selection'], panels=cohort['gate_policy'],
        split_proof=cohort['split_proof'], seed_aggregation='Final seed0 and seed1 use separate immutable outputs; average both, never best-of-two.',
        retired_attention_checkpoints_forbidden=True,
        input_models_registry=None if models_path is None else dict(path=str(models_path.resolve()),
            sha256_at_prepare=sha256(models_path), role='Provenance only; mutable input registry is not a sealed dependency.'),
        checkpoint_pinning='os.link pins the published inode; only pinned registry/checkpoints are active dependencies. Training may atomically replace last.pt.')
    write(out / 'PLAN.json', plan)
    return plan


def verified_seed_outputs(inputs):
    require(len(inputs) == 2, 'Need both prespecified seed outputs')
    sources = []
    for folder in inputs:
        proof = read(folder / 'CPU_RECHECK.json')
        require(proof['status'] == 'independent_cpu_passed' and proof['rows_sha256'] == sha256(folder / 'rows.jsonl'), 'Seed output is not CPU verified')
        with evaluator():
            plan = read(folder / 'PLAN.json'); review.verify(plan); rows = review.rows_for(folder, plan)
        require(len(rows) == plan['planned_predictions'], 'Seed output is incomplete')
        sources.append((plan, rows, folder))
    a, b = sources
    require({a[0]['diffusion_seed_index'], b[0]['diffusion_seed_index']} == {0, 1}, 'Must include seed0 and seed1')
    require(all(a[0][key] == b[0][key] for key in ('evaluation_cohort_sha256', 'evaluation_panel', 'config_sha256')), 'Seed protocols differ')
    def model_identity(plan):
        return {label: {key: value for key, value in declaration.items()
            if key not in ('checkpoint', 'source_checkpoint')} for label, declaration in plan['models'].items()}
    require(model_identity(a[0]) == model_identity(b[0]), 'Seed models differ')
    return sources


def combine(out, inputs):
    """Average both prespecified DDIM seeds per identity before aggregation."""
    require(not (out / 'SEED_COMPARISON.json').exists(), 'Existing seed summary is immutable')
    sources = verified_seed_outputs(inputs); a, b = sources
    index = {review.row_key(row): row for row in b[1]}; merged = []
    for row in a[1]:
        other = index[review.row_key(row)]
        require(row['input_sha256'] == other['input_sha256'] and row['counts'] == other['counts'], 'Observation inputs differ between DDIM seeds')
        require(all((value is None) == (other['metrics'][key] is None) for key, value in row['metrics'].items()), 'Seed metric missingness differs')
        merged.append(dict(row, stages=row['stages'] + other['stages'],
            metrics={key: None if value is None else (value + other['metrics'][key]) / 2 for key, value in row['metrics'].items()}))
    with evaluator(): review.summarize(out, a[0], merged, verified=True)
    write(out / 'SEED_COMPARISON.json', dict(protocol=PROTOCOL, status='two_prespecified_seeds_averaged',
        source_outputs=[dict(path=str(folder.resolve()), rows_sha256=sha256(folder / 'rows.jsonl'), seed_index=plan['diffusion_seed_index']) for plan, rows, folder in sources],
        paired_predictions=len(merged), observation_fingerprints_exact=True,
        policy='Mean of seed0 and seed1 for each recording/regime/model/arm; no seed selected.', TEST_used=False))


def paired_bootstrap(first, second):
    """Fixed-seed percentile bootstrap of recording-paired candidate minus comparator."""
    import numpy as np
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    require(a.ndim == 1 and a.shape == b.shape and len(a) >= 2
        and np.isfinite(a).all() and np.isfinite(b).all() and b.mean() > 0, 'Need finite recording-paired positive-error comparator inputs')
    delta = a - b
    indices = np.random.default_rng(2026101807).integers(0, len(delta), size=(10000, len(delta)))
    lower, upper = np.quantile(delta[indices].mean(-1), [.025, .975])
    return dict(recordings=len(delta), candidate_mean=float(a.mean()), comparator_mean=float(b.mean()),
        mean_delta_mm=float(delta.mean()), confidence_interval_95pct_mm=[float(lower), float(upper)],
        relative_improvement_fraction=float((b.mean() - a.mean()) / b.mean()) if b.mean() > 0 else None,
        bootstrap_draws=10000, bootstrap_seed=2026101807,
        method='Recording-paired percentile bootstrap; clean/noisy and both DDIM seeds averaged within recording first.')


def decision(out, inputs, acceptance, *, base_label='base41k', control_label='control', fusion_label='fusion'):
    """Apply the registered final criteria; never promote weights automatically."""
    review.common.cpu_setup()
    require(not (out / 'DECISION.json').exists(), 'Existing decision is immutable')
    sources = verified_seed_outputs(inputs); plan = sources[0][0]; rules = read(acceptance)
    require(rules['protocol'] == 'ego-scene-fusion-v2-preregistered-acceptance-v1', 'Unknown acceptance protocol')
    require(sha256(acceptance) == plan['preregistered_acceptance']['sha256'], 'Acceptance thresholds differ from the preregistered cohort')
    require(plan['evaluation_panel'] == 'all' and dict(Counter(e['dataset'] for e in plan['entries']))
        == dict(rich=27, trumans=8, egobody=8), 'A final decision requires all43 recordings, not an interim dev panel')
    labels = [base_label, control_label, fusion_label]
    require(len(set(labels)) == 3 and all(label in plan['models'] for label in labels), 'Need base41k, matched control and fusion labels')
    base, control, fusion = [plan['models'][label] for label in labels]
    require(base['architecture'] == control['architecture'] == 'original' and base['expected_step'] == 41000
        and fusion['architecture'] == 'scene_fusion_v2' and control['expected_step'] == fusion['expected_step'] == 3000
        and control['stage_plan_sha256'] == fusion['stage_plan_sha256'], 'Final checkpoint budget/architecture pairing differs')
    indices = [{review.row_key(row): row for row in rows} for _, rows, _ in sources]
    averaged = {}
    for entry in plan['entries']:
        for regime in plan['regimes']:
            frozen_input = None
            for label in labels:
                pair = [index[(entry['case'], regime, label, 'learned_only')] for index in indices]
                require(pair[0]['input_sha256'] == pair[1]['input_sha256'] and pair[0]['counts'] == pair[1]['counts'], 'DDIM seeds changed observations/denominators')
                signature = pair[0]['input_sha256'], pair[0]['counts']
                require(frozen_input is None or frozen_input == signature, 'Model comparisons changed input observations/denominators')
                frozen_input = signature
                require(all(row['calls']['guide_motion'] == 0 for row in pair), 'Final comparison requires physics OFF for all models')
                for row in pair:
                    require(all(value is None or math.isfinite(value) for value in row['metrics'].values()), 'Nonfinite evaluation metric')
                require(all((value is None) == (pair[1]['metrics'][key] is None) for key, value in pair[0]['metrics'].items()), 'DDIM seed metric coverage differs')
                averaged[entry['case'], regime, label] = {key: None if value is None else
                    (value + pair[1]['metrics'][key]) / 2 for key, value in pair[0]['metrics'].items()}
    def mean(metric, label, dataset, regime):
        datasets = review.DATASETS if dataset == 'macro' else [dataset]
        means = []
        for name in datasets:
            values = [averaged[e['case'], regime, label][metric] for e in plan['entries'] if e['dataset'] == name]
            values = [value for value in values if value is not None]
            require(values, 'No evaluation opportunities for metric: ' + metric)
            means.append(statistics.mean(values))
        return statistics.mean(means)
    comparisons = {}; spec = rules['fusion_acceptance']
    for comparator in (control_label, base_label):
        rich = [entry for entry in plan['entries'] if entry['dataset'] == 'rich']
        def per_recording(label):
            return [statistics.mean(averaged[e['case'], regime, label]['w_mpjpe_mm']
                for regime in plan['regimes']) for e in rich]
        primary = paired_bootstrap(per_recording(fusion_label), per_recording(comparator))
        checks = []
        def check(name, candidate, reference, tolerance, *, strict=False):
            difference = candidate - reference
            checks.append(dict(name=name, candidate=candidate, comparator=reference, delta=difference,
                maximum_allowed_regression=tolerance, passed=difference < tolerance if strict else difference <= tolerance))
        check('RICH_mean_World_relative_improvement', primary['candidate_mean'], primary['comparator_mean'],
            -spec['rich_world_relative_improvement_minimum_fraction'] * primary['comparator_mean'])
        check('RICH_bootstrap_upper_delta_mm', primary['confidence_interval_95pct_mm'][1], 0.,
            spec['rich_world_paired_bootstrap_95pct_upper_delta_below_mm'], strict=True)
        monitors = []
        for regime in plan['regimes']:
            world_a, world_b = [mean('w_mpjpe_mm', label, 'rich', regime) for label in (fusion_label, comparator)]
            if spec['rich_world_improves_each_regime']: check('rich/' + regime + '/World', world_a, world_b, 0., strict=True)
            check('rich/' + regime + '/PA', mean('pa_mpjpe_mm', fusion_label, 'rich', regime),
                mean('pa_mpjpe_mm', comparator, 'rich', regime), spec['rich_pa_no_regression_above_mm'])
            for dataset in ('trumans', 'egobody'):
                a, b = [mean('w_mpjpe_mm', label, dataset, regime) for label in (fusion_label, comparator)]
                tolerance = spec['other_dataset_world_maximum_regression']
                check(dataset + '/' + regime + '/World', a, b, max(tolerance['absolute_mm'], tolerance['relative_fraction'] * b))
            for dataset in ('rich', 'macro'):
                a, b = [mean('foot_joint_horizontal_speed_gt_low_m_s', label, dataset, regime) for label in (fusion_label, comparator)]
                tolerance = spec['foot_speed_maximum_regression']
                check(dataset + '/' + regime + '/foot_horizontal_m_s', a, b, max(tolerance['absolute_m_s'], tolerance['relative_fraction'] * b))
                check(dataset + '/' + regime + '/foot_y_abs_mm', mean('foot_joint_y_abs_error_mm', fusion_label, dataset, regime),
                    mean('foot_joint_y_abs_error_mm', comparator, dataset, regime), spec['foot_y_absolute_error_maximum_regression_mm'])
                a, b = [mean('acceleration_error_mm_s2', label, dataset, regime) for label in (fusion_label, comparator)]
                check(dataset + '/' + regime + '/acceleration_mm_s2', a, b, spec['acceleration_error_maximum_relative_regression_fraction'] * b)
                a, b = [mean('observed_head_position_error_mm', label, dataset, regime) for label in (fusion_label, comparator)]
                if 'observed_head_position_error_maximum_regression_mm' in spec:
                    check(dataset + '/' + regime + '/observed_head_mm', a, b,
                        spec['observed_head_position_error_maximum_regression_mm'])
                monitors.append(dict(name=dataset + '/' + regime + '/observed_head_mm', candidate=a, comparator=b, delta=a-b,
                    binding='observed_head_position_error_maximum_regression_mm' in spec))
        comparisons[comparator] = dict(primary=primary, guards=checks, monitors=monitors,
            passed=all(item['passed'] for item in checks), failed_checks=[item['name'] for item in checks if not item['passed']])
    result = dict(protocol=PROTOCOL, status='pass' if all(v['passed'] for v in comparisons.values()) else 'reject',
        checked_at=review.common.now(), acceptance=str(acceptance.resolve()), acceptance_sha256=sha256(acceptance),
        checkpoint_labels=dict(base41k=base_label, control=control_label, fusion=fusion_label),
        seed_outputs=[dict(path=str(folder.resolve()), plan_sha256=sha256(folder/'PLAN.json'), rows_sha256=sha256(folder/'rows.jsonl'),
            CPU_recheck_sha256=sha256(folder/'CPU_RECHECK.json')) for _, _, folder in sources],
        comparisons=comparisons, foot_y_and_acceleration_guard_scope='RICH and equal-dataset macro, separately each clean/noisy regime.',
        interpretation='Pass requires all registered checks against both comparators. A pass supports this fixed VAL cohort only; no automatic promotion or TEST claim.',
        automatic_checkpoint_promotion=False, TEST_used=False)
    write(out / 'DECISION.json', result)
    lines = ['# 场景融合最终采纳判定', '', '判定：**' + result['status'] + '**。基于43个独立录制窗口、clean/noisy及两个预注册DDIM种子；先在录制内均值，再进行配对bootstrap。', '',
        '| 对照 | RICH World差值 mm | 95% CI mm | 相对改善 | 全部门槛 |', '|---|---:|---|---:|---|']
    for comparator, value in comparisons.items():
        primary = value['primary']; interval = primary['confidence_interval_95pct_mm']
        lines.append(f"| {comparator} | {primary['mean_delta_mm']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] | {primary['relative_improvement_fraction']:.2%} | {'通过' if value['passed'] else '未通过'} |")
    for comparator, value in comparisons.items():
        if value['failed_checks']: lines.extend(['', comparator + ' 未通过：' + '；'.join(value['failed_checks']) + '。'])
    lines += ['', '负差值表示融合模型误差更低；需同时对原41k和同预算control满足1%改善、置信上界及各项守卫。完整每项阈值、观测头误差监测和证据SHA见JSON。此结果不能替代完整窗口验证或官方test指标，也不自动替换保留模型。']
    (out / 'DECISION_zh.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cohort', 'prepare', 'evaluate', 'recheck', 'report', 'combine', 'decision'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    parser.add_argument('--source', type=Path, default=review.DEFAULT_SOURCE)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--panel', choices=PANELS, default='dev')
    parser.add_argument('--seed-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--arms', nargs='+', choices=review.ARMS, default=['learned_only'])
    parser.add_argument('--inputs', type=Path, nargs='+')
    parser.add_argument('--acceptance', type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument('--base-label', default='base41k')
    parser.add_argument('--control-label', default='control')
    parser.add_argument('--fusion-label', default='fusion')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.5)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .8 and args.max_predictions >= 0, 'Invalid bounded GPU options')
    if args.command == 'cohort': prepare_cohort(args.cohort, args.source, args.acceptance)
    elif args.command == 'prepare': prepare(args.out, args.cohort, models_path=args.models, panel=args.panel, seed_index=args.seed_index, arms=args.arms)
    elif args.command == 'combine': combine(args.out, args.inputs or [])
    elif args.command == 'decision': decision(args.out, args.inputs or [], args.acceptance,
        base_label=args.base_label, control_label=args.control_label, fusion_label=args.fusion_label)
    else:
        with evaluator():
            if args.command == 'evaluate': review.evaluate(args.out, args.resume, args.max_predictions, args.memory_fraction)
            elif args.command == 'recheck': review.recheck(args.out)
            else:
                plan = read(args.out / 'PLAN.json'); review.verify(plan); rows = review.rows_for(args.out, plan)
                proof = args.out / 'CPU_RECHECK.json'
                verified = proof.exists() and read(proof)['rows_sha256'] == sha256(args.out / 'rows.jsonl')
                review.summarize(args.out, plan, rows, verified)


if __name__ == '__main__': main()
