"""Manifest-driven EGO scene-path review for frozen original/attention models.

The same head-only observations, reference, known shape, seeds and DDIM20 are
used in every arm. ``none`` disables scene input and guidance; ``learned_only``
keeps all learned scene paths and disables analytical guidance; ``full`` keeps
both. Checkpoint configuration and weights are never modified on disk.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import gc
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from unittest.mock import patch

from . import reference_review as reference
from . import validation_review as common
from .dependencies import dependency_closure, sha256
from .scene_comparison import append_row, digest_json, scene_off_guard

BASE = common.BASE
DEFAULT_SOURCE = BASE / 'runs/native30_checkpoint_review_v1'
DEFAULT_OUT = BASE / 'runs/native30_ego_scene_review'
PROTOCOL = 'frozen-head-only-ego-scene-path-review-v1'
DATASETS = ('rich', 'trumans', 'egobody')
ARMS = ('none', 'learned_only', 'full')
ARCHITECTURES = ('original', 'scene_cross_attention')
read, write, require = common.read, common.write, common.require


def status(out, state, **values):
    write(out / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=common.now(), pid=os.getpid(), optimizer_updates=0, TEST_used=False, **values))


def closure():
    sources = common.closure()
    # Attention is selected dynamically at load; the training CLI is not an eval dependency.
    sources.update(dependency_closure([Path(__file__), Path(__file__).with_name('scene_attention.py')]))
    return sources


def select_entries(source_entries, per_dataset):
    entries = []
    for dataset in DATASETS:
        chosen, seen = [], set()
        for entry in source_entries:
            if entry['dataset'] != dataset or entry['recording'] in seen:
                continue
            chosen.append(dict(entry, review_panel='dev' if len(chosen) < (per_dataset + 1) // 2 else 'confirmation'))
            seen.add(entry['recording'])
            if len(chosen) == per_dataset:
                break
        require(len(chosen) == per_dataset, 'Insufficient distinct EGO recordings: ' + dataset)
        entries.extend(chosen)
    return entries


def validate_entries(data, entries):
    require(data.split == 'validation', 'Only official validation inputs allowed')
    require(len({(e['dataset'], e['recording']) for e in entries}) == len(entries), 'Repeated recording')
    for entry in entries:
        meta, _ = data.members[entry['group']][entry['member']]
        require(entry['mode'] == 'ego' and entry['joints'] == [15], 'EGO must be head-only')
        require(meta['split'] == 'validation' and meta['sequence_id'] == entry['sequence_id']
            and Path(meta['path']).resolve() == Path(entry['path']).resolve(), 'EGO source identity differs')
        require(entry['start'] in data.starts_for(entry['group'], entry['member'], 'ego')
            and entry['source_first'] == meta['source_first'] + entry['start'], 'Illegal EGO source clock')
    dev = {(e['dataset'], e['recording']) for e in entries if e['review_panel'] == 'dev'}
    confirm = {(e['dataset'], e['recording']) for e in entries if e['review_panel'] == 'confirmation'}
    require(not dev & confirm, 'Dev and confirmation share recordings')


def load_model(declaration, config, device='cpu'):
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model
    from experiments.offline_camera_retrain_v1.train_native30 import checkpoint_config, model_digest
    require(sha256(declaration['checkpoint']) == declaration['checkpoint_sha256'], 'Checkpoint bytes changed')
    saved = torch.load(declaration['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    require(saved['step'] == declaration['expected_step'], 'Checkpoint step differs')
    require(digest_json(checkpoint_config(saved)) == digest_json(config), 'Checkpoint source config differs')
    architecture = saved.get('architecture', 'original')
    require(architecture == declaration['architecture'] and architecture in ARCHITECTURES, 'Checkpoint architecture differs')
    model = make_model(config)
    if architecture == 'scene_cross_attention':
        from .scene_attention import attach_scene_attention, scene_attention_metadata
        spec = saved.get('attention_spec') or declaration.get('attention_spec') or {}
        options = {k: spec[k] for k in ('token_dim', 'heads', 'include_global', 'include_local') if k in spec}
        attach_scene_attention(model, **options)
        attention_spec = scene_attention_metadata(model)
    else:
        attention_spec = None
    model.load_state_dict(saved['model'], strict=True)
    require(all(not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in model.state_dict().values()), 'Nonfinite checkpoint parameters')
    tensor_sha = model_digest(model)
    if 'model_tensor_sha256' in declaration:
        require(tensor_sha == declaration['model_tensor_sha256'], 'Model tensors differ')
    metadata = dict(model_tensor_sha256=tensor_sha, attention_spec=attention_spec,
        parent_step=saved.get('parent_step'), trained_guidance_enabled=saved.get('guidance_enabled'),
        stage_plan_sha256=saved.get('stage_plan_sha256'))
    del saved
    return model.to(device).eval(), metadata


def prepare(out, source=DEFAULT_SOURCE, per_dataset=4, models_path=None, arms=ARMS):
    common.cpu_setup()
    require(per_dataset >= 2 and not (out / 'PLAN.json').exists(), 'Need >=2 recordings/dataset and new output')
    source = source.resolve()
    original = read(source / 'PLAN.json')
    cohort = read(source / 'COHORT.json')
    require(sha256(source / 'COHORT.json') == original['cohort_sha256'], 'Original EGO cohort changed')
    config = original['config']
    require(config['frames'] == 192 and config['fps'] == 30 and config['sampling']['ddim_steps'] == 20
        and config['ego_head_only_probability'] == 1 and config['physical_guidance']['enabled'], 'Unexpected source protocol')
    entries = select_entries(cohort['entries'], per_dataset)
    data = common.validation_data(config)
    validate_entries(data, entries)
    if models_path is None:
        models = dict(base41k=dict(original['models']['candidate41k'], architecture='original', arms=list(arms)))
    else:
        models = read(models_path)
        require(isinstance(models, dict) and models, 'Models JSON must map names to declarations')
    registered = {}
    for label, specification in models.items():
        require(re.fullmatch(r'[A-Za-z0-9_-]+', label) is not None, 'Unsafe model label')
        declaration = dict(specification, checkpoint=str(Path(specification['checkpoint']).resolve()))
        declaration.setdefault('arms', list(arms))
        require(declaration['architecture'] in ARCHITECTURES and declaration['arms']
            and len(set(declaration['arms'])) == len(declaration['arms'])
            and all(arm in ARMS for arm in declaration['arms']), 'Unsupported model/arm')
        model, metadata = load_model(declaration, config)
        if models_path is not None and 'arms' not in specification and metadata['trained_guidance_enabled'] is False:
            declaration['arms'] = ['learned_only']
        registered[label] = dict(declaration, **metadata)
        del model
        gc.collect()
    sources = closure()
    for key in ('source_sha256', 'metadata_sha256', 'cohort_source_sha256'):
        sources.update(original.get(key, {}))
    sources.update(common.metadata_identity(data, config))
    for path in (source / 'PLAN.json', source / 'COHORT.json'):
        sources[str(path)] = sha256(path)
    if models_path is not None:
        sources[str(models_path.resolve())] = sha256(models_path)
    plan = dict(protocol=PROTOCOL, prepared_at=common.now(), source=str(source), config=config,
        config_sha256=digest_json(config), source_sha256=sources, entries=entries, models=registered,
        regimes=['clean', 'noisy'], frames=192, fps=30, ddim_steps=20,
        planned_predictions=len(entries) * 2 * sum(len(d['arms']) for d in registered.values()),
        selection='First N unique recordings per dataset in the existing sealed identity-only EGO cohort, never prediction-ranked.',
        panels='First ceil(N/2) identities per dataset are dev, remainder confirmation; recordings disjoint. Both are VAL, not TEST.',
        arms=dict(none='use_scene=False; all learned scene paths and physical guidance off; observation/IK anchors retained',
            learned_only='use_scene=True; all learned scene conditioning and feedback retained; physical guidance off',
            full='use_scene=True; original learned scene paths and original physical guidance on'),
        reference_policy='original sealed observation-derived EGO reference; no EXO policy or yaw override',
        diagnostic_scope='Per-DDIM-step compact coverage, feedback, contact and physical diagnostics; no new GT inference inputs.',
        candidate_registration='Completed candidates use a separate immutable --out with the same source cohort/config and explicit --models JSON.',
        optimizer_updates=0, TEST_used=False, automatic_checkpoint_promotion=False,
        reuse_policy='No old prediction reuse: stages require original sampler diagnostics for every arm.')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'PLAN.json', plan)
    status(out, 'prepared', completed_predictions=0, planned_predictions=plan['planned_predictions'])
    return plan


def verify(plan):
    require(plan['protocol'] == PROTOCOL and digest_json(plan['config']) == plan['config_sha256'], 'Plan config/protocol differs')
    for path, expected in plan['source_sha256'].items():
        require(sha256(path) == expected, 'Sealed dependency changed: ' + path)
    for declaration in plan['models'].values():
        require(sha256(declaration['checkpoint']) == declaration['checkpoint_sha256'], 'Sealed checkpoint changed')


def scalar_summary(value):
    value = value.detach().float()
    return dict(mean=float(value.mean()), maximum=float(value.max()))


def stage_diagnostic(timestep, details):
    import torch
    geometry = details.get('geometry', [])
    result = dict(timestep=int(timestep[0]), geometry_queried=bool(geometry), near_anchor_fraction=None,
        near_region_fraction=None, relation_region_fraction=None, far_only_region_fraction=None)
    if geometry:
        near_anchor, near_region, relation_region, far_only = [], [], [], []
        for item in geometry:
            labels = item['labels']
            near = item['valid'].flatten(2).any(-1)
            regions = torch.stack([near[:, labels == joint].any(-1) for joint in range(22)], -1)
            relation = item.get('relation_valid', torch.zeros_like(regions))
            near_anchor.append(near.float().mean())
            near_region.append(regions.float().mean())
            relation_region.append(relation.float().mean())
            far_only.append((relation & ~regions).float().mean())
        for name, values in (('near_anchor_fraction', near_anchor), ('near_region_fraction', near_region),
                ('relation_region_fraction', relation_region), ('far_only_region_fraction', far_only)):
            result[name] = float(torch.stack(values).mean())
    for name in ('root_delta_m', 'rotation_delta_rad', 'physical_root_delta_m'):
        if name in details:
            norms = scalar_summary(details[name].norm(dim=-1))
            result[name + '_mean'] = norms['mean']
            result[name + '_max'] = norms['maximum']
    for name in ('contact_logits', 'correspondence_logits'):
        if name in details:
            result[name.replace('_logits', '_probability_mean')] = float(details[name].detach().sigmoid().mean())
    result['physical_accepted_steps'] = float(details.get('physical_accepted_steps', timestep.new_zeros(1)).sum())
    result['physical_energy_before'] = float(details.get('physical_energy_before', timestep.new_zeros(1)).sum())
    result['physical_energy_after'] = float(details.get('physical_energy_after', timestep.new_zeros(1)).sum())
    return result


@contextmanager
def instrumented_arm(model, arm):
    from experiments.offline_camera_retrain_v1 import native30_physical_guidance
    stages, calls = [], dict(guide_motion=0)
    original_forward = model.forward
    original_guide = native30_physical_guidance.guide_motion

    def observe_forward(*args, **kwargs):
        requested = kwargs.pop('return_details', False)
        output, details = original_forward(*args, **kwargs, return_details=True)
        stages.append(stage_diagnostic(args[1], details))
        return (output, details) if requested else output

    def observe_guide(*args, **kwargs):
        calls['guide_motion'] += 1
        require(arm == 'full', 'Analytical guidance called in disabled arm')
        return original_guide(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, 'guidance_enabled', arm == 'full'))
        stack.enter_context(patch.object(model, 'forward', observe_forward))
        stack.enter_context(patch.object(native30_physical_guidance, 'guide_motion', observe_guide))
        if arm == 'none':
            guards = stack.enter_context(scene_off_guard(model))
        else:
            guards = {}
        yield stages, calls, guards
    require(arm == 'full' or calls['guide_motion'] == 0, 'Disabled guidance path ran')


def row_key(row):
    return row['case'], row['regime'], row['model'], row['arm']


def rows_for(out, plan):
    path = out / 'rows.jsonl'
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        tail = lines.pop()
        (out / ('TRUNCATED_ROW_TAIL_' + digest_json(list(tail)) + '.bin')).write_bytes(tail)
        with path.open('r+b') as stream:
            stream.truncate(sum(map(len, lines))); stream.flush(); os.fsync(stream.fileno())
    rows = [json.loads(line) for line in lines]
    require(len({row_key(r) for r in rows}) == len(rows), 'Duplicate committed EGO result')
    entries = {e['case']: e for e in plan['entries']}
    paired = {}
    for row in rows:
        require(row['case'] in entries and row['regime'] in plan['regimes'] and row['model'] in plan['models'], 'Unknown EGO result')
        entry, declaration = entries[row['case']], plan['models'][row['model']]
        require(row['arm'] in declaration['arms'] and row['checkpoint_sha256'] == declaration['checkpoint_sha256']
            and row['checkpoint_step'] == declaration['expected_step']
            and row['architecture'] == declaration['architecture'], 'Wrong model/arm provenance')
        require(row['seed'] == entry['case_seed'] and all(row[k] == entry[k]
            for k in ('dataset', 'recording', 'sequence_id', 'source_first', 'start', 'review_panel')), 'Wrong source identity')
        artifact = out / 'motions' / f"case{row['case']:04d}_{row['regime']}_{row['model']}_{row['arm']}.npz"
        require(Path(row['artifact']).resolve() == artifact.resolve() and sha256(artifact) == row['artifact_sha256'], 'Changed EGO artifact')
        pair = row['case'], row['regime']
        require(pair not in paired or paired[pair] == (row['input_sha256'], row['counts']), 'Different paired inputs/denominators')
        paired[pair] = row['input_sha256'], row['counts']
        require(len(row['stages']) == plan['ddim_steps'], 'Missing DDIM stage diagnostics')
        require(row['arm'] == 'full' or row['calls']['guide_motion'] == 0, 'Disabled guidance ran')
        require(row['arm'] != 'none' or len(row['scene_off_guards']) == 3
            and all(value == 0 for value in row['scene_off_guards'].values()), 'Scene-off guard proof differs')
    return rows


def summarize(out, plan, rows, verified=False):
    groups = {}
    for panel in ('all', 'dev', 'confirmation'):
        for dataset in (*DATASETS, 'macro'):
            for regime in plan['regimes']:
                for label, declaration in plan['models'].items():
                    for arm in declaration['arms']:
                        selected = [r for r in rows if r['model'] == label and r['arm'] == arm and r['regime'] == regime
                            and (panel == 'all' or r['review_panel'] == panel) and (dataset == 'macro' or r['dataset'] == dataset)]
                        if not selected:
                            continue
                        values = {}
                        for metric in selected[0]['metrics']:
                            means = [statistics.mean(r['metrics'][metric] for r in selected
                                if r['dataset'] == d and r['metrics'][metric] is not None)
                                for d in DATASETS if any(r['dataset'] == d and r['metrics'][metric] is not None for r in selected)]
                            values[metric] = statistics.mean(means) if means else None
                        diagnostics = {}
                        for name in selected[0]['stages'][0]:
                            if name in ('timestep', 'geometry_queried'):
                                continue
                            numbers = [step[name] for r in selected for step in r['stages'] if step.get(name) is not None]
                            diagnostics[name] = (max(numbers) if name.endswith('_max') else statistics.mean(numbers)) if numbers else None
                        groups['/'.join((panel, dataset, regime, label, arm))] = dict(windows=len(selected), metrics=values, stage_diagnostics=diagnostics)
    write(out / 'SUMMARY.json', dict(protocol=PROTOCOL, checked_at=common.now(),
        status='cpu_verified' if verified else 'partial_unverified', completed_predictions=len(rows),
        planned_predictions=plan['planned_predictions'], groups=groups,
        aggregation='One recording per window; equal recording per dataset, equal dataset macro. Dev/confirmation separately reported.',
        optimizer_updates=0, TEST_used=False))


def evaluate(out, resume=False, max_predictions=0, memory_fraction=.5):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    plan = read(out / 'PLAN.json'); verify(plan)
    require(resume or not (out / 'rows.jsonl').exists(), 'Existing EGO results require --resume')
    rows = rows_for(out, plan); done = {row_key(r) for r in rows}
    generated = 0; started = time.monotonic()
    status(out, 'waiting_for_gpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
    with common.gpu_lock():
        verify(plan)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        data = common.validation_data(plan['config']); validate_entries(data, plan['entries'])
        builder = common.observation_builder(plan['config'], device='cuda')
        models = {label: load_model(d, plan['config'], 'cuda')[0] for label, d in plan['models'].items()}
        (out / 'motions').mkdir(exist_ok=True)
        for entry in plan['entries']:
            for regime in plan['regimes']:
                if all((entry['case'], regime, label, arm) in done for label, d in plan['models'].items() for arm in d['arms']):
                    continue
                sample, observed, scene, asset = common.source_observation(data, builder, entry, regime)
                _, identity = data.sample(entry['group'], 'ego', member=entry['member'], start=entry['start'], yaw=0)
                batch, known = common.collated_inputs(plan['config'], observed, scene, asset, 'cuda')
                input_sha = common.tensor_fingerprint(known)
                previous = [r for r in rows if (r['case'], r['regime']) == (entry['case'], regime)]
                require(all(r['input_sha256'] == input_sha for r in previous), 'Resumed EGO inputs differ')
                for label, declaration in plan['models'].items():
                    for arm in declaration['arms']:
                        current = entry['case'], regime, label, arm
                        if current in done:
                            continue
                        status(out, 'sampling', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                            dataset=entry['dataset'], case=entry['case'], regime=regime, model=label, arm=arm)
                        before = time.monotonic()
                        with instrumented_arm(models[label], arm) as (stages, calls, guards), torch.no_grad():
                            prediction = sample_motion(models[label], known, plan['config'], steps=plan['ddim_steps'],
                                seed=entry['case_seed'], use_scene=arm != 'none')
                        torch.cuda.synchronize(); sampling_s = time.monotonic() - before
                        require(bool(torch.isfinite(prediction).all()) and len(stages) == plan['ddim_steps'], 'Invalid prediction/stages')
                        require(common.tensor_fingerprint(known) == input_sha, 'Sampler changed frozen EGO inputs')
                        arrays = dict(prediction=prediction[0].detach().cpu().numpy(), truth=sample['motion'], rest=sample['rest'],
                            observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                            reference_motion=known['reference_motion'][0].detach().cpu().numpy(),
                            identity_json=np.array(json.dumps(identity)), checkpoint_step=np.array(declaration['expected_step']),
                            checkpoint_sha256=np.array(declaration['checkpoint_sha256']), input_sha256=np.array(input_sha),
                            model=np.array(label), arm=np.array(arm), stages_json=np.array(json.dumps(stages, allow_nan=False)))
                        artifact = out / 'motions' / f"case{entry['case']:04d}_{regime}_{label}_{arm}.npz"
                        temporary = artifact.with_suffix('.tmp')
                        with temporary.open('wb') as stream:
                            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
                        temporary.replace(artifact)
                        values, counts = reference.metrics(arrays)
                        row = dict(case=entry['case'], dataset=entry['dataset'], recording=entry['recording'], sequence_id=entry['sequence_id'],
                            start=entry['start'], source_first=entry['source_first'], review_panel=entry['review_panel'], regime=regime,
                            model=label, architecture=declaration['architecture'], arm=arm, seed=entry['case_seed'],
                            checkpoint_sha256=declaration['checkpoint_sha256'], checkpoint_step=declaration['expected_step'],
                            input_sha256=input_sha, artifact=str(artifact.resolve()), artifact_sha256=sha256(artifact),
                            metrics=values, counts=counts, stages=stages, calls=calls, scene_off_guards=guards,
                            sampling_s=sampling_s, checked_at=common.now())
                        append_row(out, row); rows.append(row); done.add(current); generated += 1
                        summarize(out, plan, rows)
                        write(out / 'PROGRESS.json', dict(completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                            elapsed_s=time.monotonic() - started, last_sampling_s=sampling_s, checked_at=common.now()))
                        if max_predictions and generated >= max_predictions:
                            status(out, 'bounded_sampling_stop', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
                            return
        for label, model in models.items():
            require(model_digest(model) == plan['models'][label]['model_tensor_sha256'], 'Model changed during evaluation')
        verify(plan)
    status(out, 'sampling_completed_pending_cpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def recheck(out):
    import numpy as np
    import torch
    common.cpu_setup(); plan = read(out / 'PLAN.json'); verify(plan)
    rows = rows_for(out, plan)
    require(len(rows) == plan['planned_predictions'], 'Incomplete EGO scene review')
    data = common.validation_data(plan['config']); validate_entries(data, plan['entries'])
    builder = common.observation_builder(plan['config']); maximum = 0.
    for entry in plan['entries']:
        for regime in plan['regimes']:
            sample, observed, scene, asset = common.source_observation(data, builder, entry, regime)
            batch, known = common.collated_inputs(plan['config'], observed, scene, asset, 'cpu')
            stable = None
            for row in [r for r in rows if (r['case'], r['regime']) == (entry['case'], regime)]:
                with np.load(row['artifact'], allow_pickle=False) as saved:
                    arrays = {k: saved[k].copy() for k in saved.files}
                require(np.isfinite(arrays['prediction']).all(), 'Nonfinite saved prediction')
                for name, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                        ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                    require(np.array_equal(arrays[name], actual), 'Source EGO arrays differ: ' + name)
                require(np.allclose(arrays['reference_motion'], known['reference_motion'][0].numpy(), atol=2e-5, rtol=2e-5), 'CPU reference differs')
                for name, value in [('checkpoint_sha256', row['checkpoint_sha256']), ('model', row['model']),
                        ('arm', row['arm']), ('input_sha256', row['input_sha256'])]:
                    require(str(arrays[name]) == value, 'Saved identity differs: ' + name)
                require(int(arrays['checkpoint_step']) == row['checkpoint_step']
                    and json.loads(str(arrays['stages_json'])) == row['stages'], 'Saved step/diagnostics differ')
                paired = {name: arrays[name] for name in ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion', 'identity_json')}
                require(stable is None or all(np.array_equal(value, stable[name]) for name, value in paired.items()), 'Paired saved inputs differ')
                stable = paired
                actual, counts = reference.metrics(arrays)
                require(counts == row['counts'], 'EGO metric denominators differ')
                for name, value in actual.items():
                    require((value is None) == (row['metrics'][name] is None), 'Metric missingness differs')
                    if value is not None:
                        maximum = max(maximum, abs(value - row['metrics'][name]))
                for name, value in common.independent_core(arrays).items():
                    require((value is None) == (actual[name] is None), 'Independent metric missingness differs')
                    if value is not None:
                        maximum = max(maximum, abs(value - actual[name]))
                for step in row['stages']:
                    require(all(isinstance(value, bool) or value is None or math.isfinite(value) for value in step.values()), 'Nonfinite stage diagnostic')
    require(maximum < 1e-3 and not torch.cuda.is_initialized(), 'Independent CPU review failed')
    verify(plan)
    write(out / 'CPU_RECHECK.json', dict(status='independent_cpu_passed', checked_at=common.now(), predictions=len(rows),
        maximum_metric_difference=maximum, CUDA_initialized=False, source_GT_rest_observations_exact=True,
        paired_saved_inputs_exact=True, paired_recursive_input_fingerprints_exact=True,
        reference_CPU_tolerance=2e-5, stage_diagnostics_persistence_verified=True,
        stage_diagnostics_note='Internal sampler diagnostics are archived and checked for integrity/finite values, not independently regenerated on CPU.',
        rows_sha256=sha256(out / 'rows.jsonl')))
    summarize(out, plan, rows, verified=True)
    status(out, 'completed', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate', 'recheck', 'report'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.5)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .8 and args.max_predictions >= 0, 'Invalid bounded GPU options')
    if args.command == 'prepare':
        prepare(args.out, args.source, args.per_dataset, args.models, args.arms)
    elif args.command == 'evaluate':
        evaluate(args.out, args.resume, args.max_predictions, args.memory_fraction)
    elif args.command == 'recheck':
        recheck(args.out)
    else:
        plan = read(args.out / 'PLAN.json'); verify(plan)
        rows = rows_for(args.out, plan)
        verified = (args.out / 'CPU_RECHECK.json').exists() and read(args.out / 'CPU_RECHECK.json')['rows_sha256'] == sha256(args.out / 'rows.jsonl')
        summarize(args.out, plan, rows, verified)


if __name__ == '__main__':
    main()
