"""Paired, corrected-reference EXO validation of explicitly registered models.

One identity-selected window per recording is reused from the existing EXO
cohort. The input reference, original 41k, and retained 41k+3000 control are
reported together. Analytical physical guidance is always disabled. This
module never trains, selects a checkpoint, or opens TEST data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from . import adapter_evaluation as evaluation
from . import reference_review as reference
from . import reference_temporal
from . import validation_review as common
from .dependencies import dependency_closure, sha256
from .scene_comparison import append_row, digest_json

BASE = common.BASE
SOURCE = BASE / 'runs/native30_exo_expanded_review_v1'
SELECTED = BASE / 'runs/native30_ego_scene_fusion_v2/SELECTED_MODEL.json'
DEFAULT_OUT = BASE / 'runs/native30_exo_control_review'
PROTOCOL = 'corrected-reference-frozen-exo-control-review-v1'
REFERENCE = 'input_reference'
DATASETS = ('rich', 'trumans', 'egobody')
REGIMES = ('clean', 'noisy')
read, write, require = common.read, common.write, common.require


def status(out, state, **values):
    write(out / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=common.now(), pid=os.getpid(), optimizer_updates=0,
        TEST_used=False, **values))


def select_entries(entries, panel='all64'):
    require(panel in ('all64', 'preflight13'), 'Unknown fixed panel')
    require(len({(r['dataset'], r['recording']) for r in entries}) == len(entries),
        'Cohort repeats a recording')
    if panel == 'all64':
        chosen = list(entries)
    else:
        chosen = []
        for dataset in DATASETS:
            pool = [r for r in entries if r['dataset'] == dataset and r['panel'] == 'new55']
            require(len(pool) >= 4, 'Insufficient preflight identities')
            chosen.extend(pool[:4])
        regression = [r for r in entries
            if r['sequence_id'] == 'recording_20220215_S21_S22_01_interactee_1901-4901_master_full'
            and r['start'] == 1984]
        require(len(regression) == 1, 'Missing unique known jitter regression')
        chosen.extend(regression)
    return [dict(r, review_panel='primary' if r['panel'] == 'new55' else 'regression') for r in chosen]


def checkpoint_metadata(saved, declaration, config):
    """Stage update clocks are explicit; 3000 is not a 3000-from-scratch model."""
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config
    require(saved['step'] == declaration['expected_step'], 'Checkpoint stage step differs')
    require(saved.get('architecture', 'original') == declaration.get('architecture', 'original'),
        'Checkpoint architecture differs')
    require(digest_json(checkpoint_config(saved)) == digest_json(config), 'Checkpoint source config differs')
    if declaration.get('expected_parent_step') is not None:
        require(saved.get('parent_step') == declaration['expected_parent_step'], 'Checkpoint parent step differs')
    if 'expected_guidance_enabled' in declaration:
        require(saved.get('guidance_enabled') is declaration['expected_guidance_enabled'],
            'Checkpoint training guidance differs')
    return dict(stage_step=saved['step'], parent_step=saved.get('parent_step'),
        architecture=saved.get('architecture', 'original'),
        trained_guidance_enabled=saved.get('guidance_enabled'))


def load_model(declaration, config, device='cpu'):
    """Strict registered loader; optional new adapters supply a sealed loader."""
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    require(sha256(declaration['checkpoint']) == declaration['checkpoint_sha256'], 'Checkpoint bytes changed')
    if declaration.get('kind', 'original') == 'original':
        saved = torch.load(declaration['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        metadata = checkpoint_metadata(saved, declaration, config)
        model = make_model(config)
        model.load_state_dict(saved['model'], strict=True)
        del saved
    else:
        require(declaration.get('kind') == 'registered_adapter', 'Unsupported checkpoint kind')
        # The loader module/function and its complete dependency closure are
        # frozen by prepare before any output is generated.
        module = importlib.import_module(declaration['loader_module'])
        model, metadata = getattr(module, declaration['loader_function'])(
            declaration['checkpoint'], device='cpu',
            expected_sha256=declaration['checkpoint_sha256'])
        require(metadata['stage_step'] == declaration['expected_step'], 'Adapter stage step differs')
        require(metadata['frozen_base_sha256'] == declaration['frozen_base_tensor_sha256'],
            'Adapter frozen base differs')
    require(all(not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in model.state_dict().values()), 'Nonfinite checkpoint tensors')
    tensor_sha = model_digest(model)
    if declaration.get('model_tensor_sha256'):
        require(tensor_sha == declaration['model_tensor_sha256'], 'Model tensor identity differs')
    require(hasattr(model, 'guidance_enabled'), 'Model lacks explicit physics switch')
    model.guidance_enabled = False
    metadata = dict(metadata, model_tensor_sha256=tensor_sha,
        strict_full_state_loaded=True, analytical_guidance_enabled=False)
    return model.to(device).eval(), metadata


def declarations():
    selected = read(SELECTED)
    require(selected['architecture'] == 'original' and selected['expected_step'] == 3000
        and selected['parent_step'] == 41000, 'Selected control identity changed')
    base = read(SOURCE / 'MODELS.json')['base41k']
    return dict(base41k=dict(kind='original', architecture='original',
        checkpoint=base['checkpoint'], checkpoint_sha256=base['checkpoint_sha256'],
        expected_step=41000, model_tensor_sha256=base['model_tensor_sha256']),
        control3000=dict(kind='original', architecture='original',
            checkpoint=selected['checkpoint'], checkpoint_sha256=selected['checkpoint_sha256'],
            expected_step=3000, expected_parent_step=41000, expected_guidance_enabled=False,
            model_tensor_sha256=selected['model_tensor_sha256']))


def closure(models):
    sources = common.closure()
    sources.update(dependency_closure([Path(__file__), Path(reference_temporal.__file__)]))
    for declaration in models.values():
        if declaration.get('kind') == 'registered_adapter':
            module = importlib.import_module(declaration['loader_module'])
            sources.update(dependency_closure([Path(module.__file__)]))
    return sources


def prepare(out, panel='all64', models_path=None):
    common.cpu_setup()
    out = Path(out).resolve()
    require(not (out / 'PLAN.json').exists(), 'Existing immutable review plan')
    selected = read(SELECTED)
    config_path = Path(selected['source_config']['path'])
    require(sha256(config_path) == selected['source_config']['file_sha256'], 'Control source config changed')
    config = read(config_path)
    cohort = read(SOURCE / 'COHORT.json')
    data = common.validation_data(config)
    coverage = evaluation.validate_entries(data, cohort['entries'], config)
    entries = select_entries(cohort['entries'], panel)
    models = declarations() if models_path is None else read(models_path)
    require(models and REFERENCE not in models and len(models) == len(set(models)), 'Invalid model declarations')
    registered = {}
    for label, declaration in models.items():
        require(label.replace('_', '').replace('-', '').isalnum(), 'Unsafe model label')
        declaration = dict(declaration, checkpoint=str(Path(declaration['checkpoint']).resolve()))
        model, metadata = load_model(declaration, config)
        registered[label] = dict(declaration, model_tensor_sha256=metadata['model_tensor_sha256'],
            registered_metadata=metadata)
        del model
        gc.collect()
    sources = closure(models)
    sources.update(cohort['source_sha256'])
    sources.update(common.metadata_identity(data, config))
    source_paths = [SELECTED, config_path, SOURCE / 'COHORT.json', SOURCE / 'MODELS.json']
    if models_path is not None:
        source_paths.append(Path(models_path).resolve())
    for path in source_paths:
        sources[str(path.resolve())] = sha256(path)
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(protocol=PROTOCOL, prepared_at=common.now(), config=config,
        config_sha256=digest_json(config), source_sha256=sources, entries=entries,
        models=registered, model_order=list(registered), panel=panel, source_coverage=coverage,
        regimes=list(REGIMES), reference_policy=reference_temporal.POLICY,
        input_reference_label=REFERENCE, scene=True, analytical_physics=False,
        frames=192, fps=30, ddim_steps=20, planned_model_predictions=len(entries) * 2 * len(models),
        planned_reference_rows=len(entries) * 2, planned_rows=len(entries) * 2 * (len(models) + 1),
        selection='Existing immutable EXO identity-only cohort; one window per recording; no score ranking.',
        reporting='new55 primary, old9 regression, all64 separately. Equal recordings within dataset; macro equal dataset.',
        disclosure='All are VAL. Historical use of this cohort prevents claiming a fresh held-out TEST or architecture-selection holdout.',
        inference='Same corrected reference, camera, visible scene, position/orientation masks, noise and DDIM seed for every model; original observation anchors retained.',
        missingness='Visible/missing means position evidence available/unavailable; missing is not exclusively physical occlusion.',
        optimizer_updates=0, TEST_used=False, automatic_checkpoint_promotion=False)
    write(out / 'PLAN.json', plan)
    status(out, 'prepared', completed_rows=0, planned_rows=plan['planned_rows'])
    return plan


def verify(plan):
    require(plan['protocol'] == PROTOCOL and plan['reference_policy'] == reference_temporal.POLICY,
        'Reference review protocol differs')
    require(plan['analytical_physics'] is False and plan['scene'] is True, 'Scene/physics protocol differs')
    require(digest_json(plan['config']) == plan['config_sha256'], 'Sealed config differs')
    for path, expected in plan['source_sha256'].items():
        require(sha256(path) == expected, 'Sealed dependency changed: ' + path)
    for declaration in plan['models'].values():
        require(sha256(declaration['checkpoint']) == declaration['checkpoint_sha256'], 'Sealed checkpoint changed')


@contextmanager
def physics_off(model):
    from experiments.offline_camera_retrain_v1 import native30_physical_guidance as physics
    previous = model.guidance_enabled
    model.guidance_enabled = False
    def forbidden(*args, **kwargs):
        raise RuntimeError('Analytical physics unexpectedly called in EXO control review')
    with patch.object(physics, 'guide_motion', forbidden):
        try:
            yield
        finally:
            model.guidance_enabled = previous


def row_key(row):
    return row['case'], row['regime'], row['model']


def rows_for(out, plan):
    path = out / 'rows.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    require(len({row_key(row) for row in rows}) == len(rows), 'Duplicate result identity')
    entries = {r['case']: r for r in plan['entries']}
    paired_inputs = {}
    paired_counts = {}
    for row in rows:
        require(row['case'] in entries and row['model'] in [REFERENCE, *plan['model_order']]
            and row['regime'] in plan['regimes'], 'Unregistered result')
        entry = entries[row['case']]
        require(all(row[k] == entry[k] for k in ('dataset', 'recording', 'sequence_id', 'start', 'source_first')),
            'Result source identity differs')
        require(row['seed'] == entry['case_seed'] and row['policy'] == plan['reference_policy'], 'Result seed/policy differs')
        require(row['analytical_physics'] is False and row['use_scene'] is True, 'Result physics/scene differs')
        if row['model'] != REFERENCE:
            declaration = plan['models'][row['model']]
            require(row['checkpoint_sha256'] == declaration['checkpoint_sha256']
                and row['checkpoint_step'] == declaration['expected_step'], 'Result checkpoint differs')
        require(sha256(row['artifact']) == row['artifact_sha256'], 'Prediction artifact changed')
        pair = row['case'], row['regime']
        require(pair not in paired_inputs or row['input_sha256'] == paired_inputs[pair],
            'Paired recursive inputs differ')
        require(pair not in paired_counts or row['counts'] == paired_counts[pair],
            'Paired metric denominators differ')
        paired_inputs[pair] = row['input_sha256']
        paired_counts[pair] = row['counts']
    return rows


def summarize(out, plan, rows, verified=False):
    groups = {}
    for panel in ('all', 'primary', 'regression'):
        for dataset in (*DATASETS, 'macro'):
            for regime in plan['regimes']:
                for label in [REFERENCE, *plan['model_order']]:
                    selected = [r for r in rows if r['regime'] == regime and r['model'] == label
                        and (panel == 'all' or r['review_panel'] == panel)
                        and (dataset == 'macro' or r['dataset'] == dataset)]
                    if not selected:
                        continue
                    values = {}
                    for metric in selected[0]['metrics']:
                        means = []
                        for name in DATASETS:
                            valid = [r['metrics'][metric] for r in selected
                                if r['dataset'] == name and r['metrics'][metric] is not None]
                            if valid:
                                means.append(statistics.mean(valid))
                        values[metric] = statistics.mean(means) if means else None
                    groups['/'.join((panel, dataset, regime, label))] = dict(windows=len(selected),
                        recordings=len({(r['dataset'], r['recording']) for r in selected}), metrics=values,
                        counts=dict(Counter({key: sum(r['counts'].get(key, 0) for r in selected)
                            for key in selected[0]['counts']})))
    write(out / 'SUMMARY.json', dict(protocol=PROTOCOL, checked_at=common.now(),
        status='cpu_verified' if verified else 'partial_unverified', completed_rows=len(rows),
        planned_rows=plan['planned_rows'], groups=groups, aggregation=plan['reporting'],
        reference_policy=plan['reference_policy'], analytical_physics=False,
        missingness=plan['missingness'], disclosure=plan['disclosure'], TEST_used=False))


def evaluate(out, resume=False, memory_fraction=.85):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    out = Path(out).resolve()
    plan = read(out / 'PLAN.json')
    verify(plan)
    require(0 < memory_fraction <= 1, 'Invalid GPU memory fraction')
    require(resume or not (out / 'rows.jsonl').exists(), 'Existing results require --resume')
    rows = rows_for(out, plan)
    done = {row_key(row) for row in rows}
    status(out, 'waiting_for_gpu', completed_rows=len(rows), planned_rows=plan['planned_rows'])
    with common.gpu_lock():
        verify(plan)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        data = common.validation_data(plan['config'])
        builder = common.observation_builder(plan['config'], device='cuda')
        models = {label: load_model(declaration, plan['config'], device='cuda')[0]
            for label, declaration in plan['models'].items()}
        folder = out / 'motions'
        folder.mkdir(exist_ok=True)
        started = time.monotonic()
        for entry in plan['entries']:
            for regime in plan['regimes']:
                labels = [REFERENCE, *plan['model_order']]
                if all((entry['case'], regime, label) in done for label in labels):
                    continue
                sample, observed, scene, asset = evaluation.source_observation(data, builder, entry, regime)
                with reference_temporal.installed():
                    _, known = evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cuda')
                input_sha = common.tensor_fingerprint(known)
                previous = [r for r in rows if (r['case'], r['regime']) == (entry['case'], regime)]
                require(all(r['input_sha256'] == input_sha for r in previous), 'Resumed recursive inputs differ')
                for label in labels:
                    current = entry['case'], regime, label
                    if current in done:
                        continue
                    status(out, 'sampling', completed_rows=len(rows), planned_rows=plan['planned_rows'],
                        case=entry['case'], dataset=entry['dataset'], regime=regime, model=label)
                    before = time.monotonic()
                    if label == REFERENCE:
                        prediction = known['reference_motion']
                    else:
                        with torch.no_grad(), physics_off(models[label]):
                            prediction = sample_motion(models[label], known, plan['config'],
                                steps=plan['ddim_steps'], seed=entry['case_seed'], use_scene=True)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - before
                    require(bool(torch.isfinite(prediction).all()), 'Nonfinite model output')
                    arrays = dict(prediction=prediction[0].detach().cpu().numpy(), truth=sample['motion'],
                        rest=sample['rest'], observations=observed['trajectory'],
                        observation_meta=observed['observation_meta'],
                        reference_motion=known['reference_motion'][0].detach().cpu().numpy(),
                        reference_policy=np.array(plan['reference_policy']))
                    artifact = folder / f"case{entry['case']:04d}_{regime}_{label}.npz"
                    temporary = artifact.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(artifact)
                    values, counts = reference.metrics(arrays)
                    declaration = plan['models'].get(label, {})
                    row = dict(case=entry['case'], dataset=entry['dataset'], recording=entry['recording'],
                        sequence_id=entry['sequence_id'], start=entry['start'], source_first=entry['source_first'],
                        panel=entry['panel'], review_panel=entry['review_panel'], regime=regime, model=label,
                        policy=plan['reference_policy'], seed=entry['case_seed'],
                        checkpoint_sha256=declaration.get('checkpoint_sha256'),
                        checkpoint_step=declaration.get('expected_step'),
                        parent_step=declaration.get('expected_parent_step'), input_sha256=input_sha,
                        artifact=str(artifact), artifact_sha256=sha256(artifact), metrics=values, counts=counts,
                        analytical_physics=False, use_scene=True, sampling_s=elapsed, checked_at=common.now())
                    append_row(out, row)
                    rows.append(row)
                    done.add(current)
                    summarize(out, plan, rows)
                    write(out / 'PROGRESS.json', dict(completed_rows=len(rows), planned_rows=plan['planned_rows'],
                        elapsed_s=time.monotonic() - started, last_sampling_s=elapsed, checked_at=common.now()))
        for label, model in models.items():
            require(model_digest(model) == plan['models'][label]['model_tensor_sha256'], 'Inference changed model tensors')
        verify(plan)
    status(out, 'sampling_completed_pending_cpu', completed_rows=len(rows), planned_rows=plan['planned_rows'])


def recheck(out):
    import numpy as np
    import torch
    common.cpu_setup()
    out = Path(out).resolve()
    plan = read(out / 'PLAN.json')
    verify(plan)
    rows = rows_for(out, plan)
    require(len(rows) == plan['planned_rows'], 'Incomplete frozen evaluation')
    data = common.validation_data(plan['config'])
    builder = common.observation_builder(plan['config'])
    indexed = {row_key(row): row for row in rows}
    maximum = independent_maximum = 0.
    for entry in plan['entries']:
        for regime in plan['regimes']:
            sample, observed, scene, asset = evaluation.source_observation(data, builder, entry, regime)
            with reference_temporal.installed():
                _, known = evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cpu')
            fingerprints = []
            denominators = []
            for label in [REFERENCE, *plan['model_order']]:
                row = indexed[entry['case'], regime, label]
                with np.load(row['artifact'], allow_pickle=False) as saved:
                    arrays = {key: saved[key].copy() for key in saved.files}
                for name, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                        ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                    require(np.array_equal(arrays[name], actual), 'Source array changed: ' + name)
                require(np.allclose(arrays['reference_motion'], known['reference_motion'][0].numpy(), atol=2e-5, rtol=2e-5),
                    'Saved corrected reference differs from CPU reconstruction')
                if label == REFERENCE:
                    require(np.array_equal(arrays['prediction'], arrays['reference_motion']), 'Input reference prediction differs')
                actual, counts = reference.metrics(arrays)
                require(counts == row['counts'], 'Metric denominator differs')
                for name, value in actual.items():
                    require((value is None) == (row['metrics'][name] is None), 'Metric missingness changed')
                    if value is not None:
                        maximum = max(maximum, abs(value - row['metrics'][name]))
                for name, value in common.independent_core(arrays).items():
                    require((value is None) == (actual[name] is None), 'Independent metric missingness differs')
                    if value is not None:
                        independent_maximum = max(independent_maximum, abs(value - actual[name]))
                fingerprints.append(row['input_sha256'])
                denominators.append(counts)
            require(len(set(fingerprints)) == 1 and all(c == denominators[0] for c in denominators),
                'Models used different recursive inputs or metric masks')
    require(maximum < 1e-6 and independent_maximum < 1e-3 and not torch.cuda.is_initialized(),
        'Independent CPU metric verification failed')
    verify(plan)
    write(out / 'CPU_RECHECK.json', dict(status='independent_cpu_passed', checked_at=common.now(),
        rows=len(rows), maximum_recompute_difference=maximum,
        maximum_independent_formula_difference=independent_maximum, CUDA_initialized=False,
        source_GT_rest_observations_and_missing_masks_exact=True,
        recursive_model_input_fingerprints_and_denominators_paired=True,
        reference_CPU_reconstruction_tolerance=2e-5, analytical_physics=False,
        rows_sha256=sha256(out / 'rows.jsonl'), TEST_used=False))
    summarize(out, plan, rows, verified=True)
    status(out, 'completed', completed_rows=len(rows), planned_rows=plan['planned_rows'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate', 'recheck'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--panel', choices=('all64', 'preflight13'), default='all64')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=.85)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args.out, args.panel, args.models)
    elif args.command == 'evaluate':
        evaluate(args.out, args.resume, args.memory_fraction)
    else:
        recheck(args.out)


if __name__ == '__main__':
    main()
