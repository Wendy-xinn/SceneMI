"""Frozen scene-on/off comparison with reusable, CPU-verified on controls.

Only the sampler's use_scene switch changes. References, observations, anchors,
model weights, DDIM settings and seeds remain identical within each pair. This
is an inference ablation, not scene-free retraining or a TEST benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from . import adapter_evaluation as evaluation
from . import reference_review as reference
from . import reference_temporal
from . import validation_review as common
from .dependencies import dependency_closure, sha256

BASE = common.BASE
DEFAULT_SOURCE = BASE / 'runs/native30_exo_reference_review'
DEFAULT_OUT = BASE / 'runs/native30_exo_scene_comparison'
MODEL = 'balancedv2adapter1000'
PROTOCOL = 'frozen-exo-scene-on-off-paired-review-v1'
POLICIES = reference.POLICIES
DATASETS = ('trumans', 'egobody', 'rich')
read, write, require = common.read, common.write, common.require


def status(out, state, **values):
    write(out / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=common.now(), pid=os.getpid(), optimizer_updates=0,
        TEST_used=False, **values))


def closure():
    sources = reference.closure()
    sources.update(dependency_closure([Path(__file__)]))
    return sources


def policy_context(policy):
    require(policy in POLICIES, 'Unknown reference policy')
    return nullcontext() if policy == 'original' else reference_temporal.installed()


@contextmanager
def scene_off_guard(model):
    """Fail before any disabled scene path can affect an off prediction."""
    from unittest.mock import patch
    from experiments.offline_camera_retrain_v1 import native30_physical_guidance
    counts = dict(scene_tokens=0, local_geometry=0, guide_motion=0)

    def sentinel(name):
        def forbidden(*args, **kwargs):
            counts[name] += 1
            raise RuntimeError('Scene-off unexpectedly called ' + name)
        return forbidden

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, 'scene_tokens', sentinel('scene_tokens')))
        stack.enter_context(patch.object(model, 'local_geometry', sentinel('local_geometry')))
        stack.enter_context(patch.object(native30_physical_guidance, 'guide_motion', sentinel('guide_motion')))
        yield counts
    require(not any(counts.values()), 'Scene-off guard recorded a scene/physics call')


def key(row):
    return row['case'], row['regime'], row['policy'], row['use_scene']


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def append_row(out, row):
    with (out / 'rows.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(out, source=DEFAULT_SOURCE, per_dataset=4):
    common.cpu_setup()
    require(not (out / 'PLAN.json').exists(), 'Existing immutable scene comparison plan')
    require(per_dataset > 0, 'Positive per-dataset count required')
    source = source.resolve()
    source_plan = read(source / 'PLAN.json')
    reference.verify(source_plan)
    source_rows = reference.rows_for(source, source_plan)
    checked = read(source / 'CPU_RECHECK.json')
    require(checked['status'] == 'independent_cpu_passed'
        and checked['rows_sha256'] == sha256(source / 'rows.jsonl'), 'Source controls lack CPU verification')
    require(len(source_rows) == source_plan['planned_predictions'], 'Incomplete source controls')
    entries = []
    for dataset in DATASETS:
        candidates = [e for e in source_plan['entries']
            if e['review_panel'] == 'primary' and e['dataset'] == dataset]
        require(len(candidates) >= per_dataset, 'Insufficient primary controls: ' + dataset)
        entries.extend(candidates[:per_dataset])
    require(len({(e['dataset'], e['recording']) for e in entries}) == len(entries),
        'Scene comparison needs distinct recordings')
    cases = {e['case'] for e in entries}
    controls = [r for r in source_rows if r['case'] in cases and r['model'] == MODEL]
    require(len(controls) == len(entries) * len(POLICIES) * len(source_plan['regimes']),
        'Missing scene-on control')
    declaration = source_plan['models'][MODEL]
    evaluation.verify_registered(declaration)
    sources = closure()
    sources.update(source_plan['source_sha256'])
    for path in (source / 'PLAN.json', source / 'rows.jsonl', source / 'CPU_RECHECK.json'):
        sources[str(path)] = sha256(path)
    plan = dict(protocol=PROTOCOL, prepared_at=common.now(), source=str(source),
        source_protocol=source_plan['protocol'], config=source_plan['config'],
        config_sha256=digest_json(source_plan['config']), entries=entries,
        model=MODEL, declaration=declaration, policies=list(POLICIES),
        regimes=source_plan['regimes'], scene_conditions=[True, False],
        source_sha256=sources, source_scene_on_rows=controls,
        planned_pairs=len(controls), reused_predictions=len(controls),
        planned_new_predictions=len(controls), planned_predictions=2 * len(controls),
        selection='First N previously sealed primary identities per dataset, one recording per window; diagnostic case excluded.',
        frames=source_plan['frames'], fps=source_plan['fps'], ddim_steps=source_plan['ddim_steps'],
        optimizer_updates=0, TEST_used=False, automatic_checkpoint_promotion=False,
        scene_off_guard=['model.scene_tokens', 'model.local_geometry', 'native30_physical_guidance.guide_motion'],
        scene_off_semantics='sample_motion(use_scene=False): network scene conditioning and physical/scene feedback disabled; observations, observation-derived reference, known body/rest and observation anchors preserved.',
        comparison_type='inference-time removal of learned scene input plus physical scene feedback; not scene-free training',
        noncausal_offline_window=True)
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'PLAN.json', plan)
    for row in controls:
        append_row(out, dict(row, use_scene=True, reused=True,
            source_rows=str(source / 'rows.jsonl'), source_row_sha256=digest_json(row)))
    rows = rows_for(out, plan)
    summarize(out, plan, rows)
    status(out, 'prepared', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
        reused_predictions=len(controls), new_predictions=0)
    return plan


def verify(plan):
    require(plan['protocol'] == PROTOCOL and plan['policies'] == list(POLICIES), 'Scene comparison protocol differs')
    require(plan['model'] == MODEL and plan['scene_conditions'] == [True, False], 'Scene comparison conditions differ')
    require(digest_json(plan['config']) == plan['config_sha256'], 'Scene comparison config changed')
    for path, expected in plan['source_sha256'].items():
        require(sha256(path) == expected, 'Sealed dependency changed: ' + path)
    evaluation.verify_registered(plan['declaration'])


def rows_for(out, plan):
    path = out / 'rows.jsonl'
    if not path.exists():
        return []
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        tail = lines.pop()
        saved = out / ('TRUNCATED_ROW_TAIL_' + hashlib.sha256(tail).hexdigest() + '.bin')
        saved.write_bytes(tail)
        with path.open('r+b') as stream:
            stream.truncate(sum(map(len, lines)))
            stream.flush()
            os.fsync(stream.fileno())
    rows = [json.loads(line) for line in lines]
    require(len({key(row) for row in rows}) == len(rows), 'Duplicate scene comparison result')
    entries = {e['case']: e for e in plan['entries']}
    controls = {(r['case'], r['regime'], r['policy']): r for r in plan['source_scene_on_rows']}
    for row in rows:
        require(row['case'] in entries and row['regime'] in plan['regimes']
            and row['policy'] in POLICIES and type(row['use_scene']) is bool, 'Unexpected scene result')
        entry = entries[row['case']]
        require(row['model'] == plan['model']
            and row['checkpoint_sha256'] == plan['declaration']['checkpoint_sha256']
            and row['checkpoint_step'] == plan['declaration']['expected_step'], 'Scene model identity differs')
        require(row['seed'] == entry['case_seed']
            and all(row[name] == entry[name] for name in ('dataset', 'recording', 'sequence_id', 'start', 'source_first')),
            'Scene source identity differs')
        require(sha256(row['artifact']) == row['artifact_sha256'], 'Scene prediction changed')
        control = controls[row['case'], row['regime'], row['policy']]
        require(row['input_sha256'] == control['input_sha256'], 'Scene pair recursive inputs differ')
        require(row['counts'] == control['counts'], 'Scene pair metric denominators differ')
        if row['use_scene']:
            require(row['reused'] is True and row['source_row_sha256'] == digest_json(control)
                and all(row[name] == control[name] for name in control), 'Reused scene-on control differs')
        else:
            artifact = out / 'motions' / f"case{row['case']:04d}_{row['regime']}_{row['policy']}_scene_off_{MODEL}.npz"
            require(Path(row['artifact']).resolve() == artifact.resolve() and row['reused'] is False,
                'Scene-off artifact path differs')
            require(row['scene_off_guard_counts'] == dict(scene_tokens=0, local_geometry=0, guide_motion=0),
                'Scene-off guard proof differs')
    return rows


def summarize(out, plan, rows, verified=False):
    groups, paired = {}, {}
    indexed = {key(r): r for r in rows}
    for policy in POLICIES:
        for regime in plan['regimes']:
            for dataset in (*DATASETS, 'macro'):
                for use_scene in (True, False):
                    selected = [r for r in rows if r['policy'] == policy and r['regime'] == regime
                        and r['use_scene'] == use_scene and (dataset == 'macro' or r['dataset'] == dataset)]
                    if not selected:
                        continue
                    values = {}
                    for metric in selected[0]['metrics']:
                        averages = [statistics.mean(r['metrics'][metric] for r in selected
                            if r['dataset'] == d and r['metrics'][metric] is not None)
                            for d in DATASETS if any(r['dataset'] == d and r['metrics'][metric] is not None for r in selected)]
                        values[metric] = statistics.mean(averages) if averages else None
                    label = '/'.join((policy, regime, dataset, 'scene_on' if use_scene else 'scene_off'))
                    groups[label] = dict(windows=len(selected), metrics=values)
                pairs = [(r, indexed[key(r)[:-1] + (False,)]) for r in rows
                    if r['use_scene'] and key(r)[:-1] + (False,) in indexed
                    and r['policy'] == policy and r['regime'] == regime
                    and (dataset == 'macro' or r['dataset'] == dataset)]
                if not pairs:
                    continue
                differences = {}
                for metric in pairs[0][0]['metrics']:
                    valid = [(a, b) for a, b in pairs if a['metrics'][metric] is not None and b['metrics'][metric] is not None]
                    means = [statistics.mean(a['metrics'][metric] - b['metrics'][metric]
                        for a, b in valid if a['dataset'] == d) for d in DATASETS
                        if any(a['dataset'] == d for a, b in valid)]
                    differences[metric] = dict(scene_on_minus_off=statistics.mean(means) if means else None,
                        paired_windows=len(valid), scene_on_lower=sum(a['metrics'][metric] < b['metrics'][metric] - 1e-6 for a, b in valid),
                        scene_off_lower=sum(b['metrics'][metric] < a['metrics'][metric] - 1e-6 for a, b in valid))
                paired['/'.join((policy, regime, dataset))] = dict(windows=len(pairs), metrics=differences)
    write(out / 'SUMMARY.json', dict(protocol=PROTOCOL, checked_at=common.now(),
        status='cpu_verified' if verified else 'partial_unverified', completed_predictions=len(rows),
        planned_predictions=plan['planned_predictions'], groups=groups, paired_differences=paired,
        aggregation='Equal recording within dataset, equal dataset macro; same-window scene-on/off pairs only.',
        metric_note='Unavailable observed joints include dropout and projection/ray support failures as well as occlusion; angles are diagnostics.',
        comparison_type=plan['comparison_type'], optimizer_updates=0, TEST_used=False))


def evaluate(out, resume=False, max_predictions=0, memory_fraction=.5):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    plan = read(out / 'PLAN.json')
    verify(plan)
    # Prepared output intentionally already contains reused controls.
    rows = rows_for(out, plan)
    require(resume or all(r['reused'] for r in rows), 'Existing scene-off results require --resume')
    done = {key(row) for row in rows}
    controls = {(r['case'], r['regime'], r['policy']): r for r in plan['source_scene_on_rows']}
    status(out, 'waiting_for_gpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
    generated = 0
    started = time.monotonic()
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
        model, metadata = evaluation.load_registered(plan['declaration'], device='cuda')
        require(model_digest(model) == plan['declaration']['model_tensor_sha256'], 'Loaded scene model differs')
        folder = out / 'motions'
        folder.mkdir(exist_ok=True)
        for entry in plan['entries']:
            for regime in plan['regimes']:
                if all((entry['case'], regime, policy, False) in done for policy in POLICIES):
                    continue
                sample, observed, scene, asset = evaluation.source_observation(data, builder, entry, regime)
                _, identity = data.sample(entry['group'], 'exo', member=entry['member'], start=entry['start'], yaw=0)
                for policy in POLICIES:
                    current = entry['case'], regime, policy, False
                    if current in done:
                        continue
                    with policy_context(policy):
                        batch, known = evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cuda')
                        input_sha = common.tensor_fingerprint(known)
                        control = controls[entry['case'], regime, policy]
                        require(input_sha == control['input_sha256'], 'Scene-off input differs from existing scene-on control')
                        status(out, 'sampling', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                            case=entry['case'], dataset=entry['dataset'], regime=regime, policy=policy, use_scene=False)
                        before = time.monotonic()
                        with scene_off_guard(model) as guard_counts, torch.no_grad():
                            prediction = sample_motion(model, known, plan['config'], steps=plan['ddim_steps'],
                                seed=entry['case_seed'], use_scene=False)
                        torch.cuda.synchronize()
                        sampling_s = time.monotonic() - before
                        require(bool(torch.isfinite(prediction).all()), 'Nonfinite scene-off prediction')
                        require(common.tensor_fingerprint(known) == input_sha, 'Scene-off sampler mutated source inputs')
                        arrays = dict(prediction=prediction[0].detach().cpu().numpy(), truth=sample['motion'],
                            rest=sample['rest'], observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                            reference_motion=known['reference_motion'][0].detach().cpu().numpy(),
                            identity_json=np.array(json.dumps(identity)), checkpoint_step=np.array(plan['declaration']['expected_step']),
                            checkpoint_sha256=np.array(plan['declaration']['checkpoint_sha256']), reference_policy=np.array(policy),
                            use_scene=np.array(False))
                    artifact = folder / f"case{entry['case']:04d}_{regime}_{policy}_scene_off_{MODEL}.npz"
                    temporary = artifact.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(artifact)
                    values, counts = reference.metrics(arrays)
                    require(counts == control['counts'], 'Scene-off denominators differ from scene-on control')
                    row = dict(case=entry['case'], dataset=entry['dataset'], sequence_id=entry['sequence_id'],
                        recording=entry['recording'], review_panel='primary', start=entry['start'], source_first=entry['source_first'],
                        regime=regime, policy=policy, model=plan['model'], use_scene=False, reused=False,
                        scene_off_guard_counts=guard_counts,
                        seed=entry['case_seed'], checkpoint_sha256=plan['declaration']['checkpoint_sha256'],
                        checkpoint_step=plan['declaration']['expected_step'], input_sha256=input_sha,
                        artifact=str(artifact.resolve()), artifact_sha256=sha256(artifact), metrics=values, counts=counts,
                        sampling_s=sampling_s, checked_at=common.now())
                    append_row(out, row)
                    rows.append(row)
                    done.add(current)
                    generated += 1
                    summarize(out, plan, rows)
                    write(out / 'PROGRESS.json', dict(completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                        new_predictions=len(rows) - plan['reused_predictions'], elapsed_s=time.monotonic() - started,
                        last_sampling_s=sampling_s, checked_at=common.now()))
                    if max_predictions and generated >= max_predictions:
                        status(out, 'bounded_sampling_stop', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
                        return
        require(model_digest(model) == plan['declaration']['model_tensor_sha256'], 'Model changed during scene comparison')
        verify(plan)
    status(out, 'sampling_completed_pending_cpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def recheck(out):
    import numpy as np
    import torch
    common.cpu_setup()
    plan = read(out / 'PLAN.json')
    verify(plan)
    rows = rows_for(out, plan)
    require(len(rows) == plan['planned_predictions'], 'Incomplete scene comparison')
    data = common.validation_data(plan['config'])
    builder = common.observation_builder(plan['config'])
    indexed = {key(r): r for r in rows}
    maximum = 0.
    for entry in plan['entries']:
        for regime in plan['regimes']:
            sample, observed, scene, asset = evaluation.source_observation(data, builder, entry, regime)
            for policy in POLICIES:
                with policy_context(policy):
                    batch, known = evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cpu')
                paired_arrays = []
                for use_scene in (True, False):
                    row = indexed[entry['case'], regime, policy, use_scene]
                    with np.load(row['artifact'], allow_pickle=False) as saved:
                        arrays = {k: saved[k].copy() for k in saved.files}
                    for name, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                            ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                        require(np.array_equal(arrays[name], actual), 'CPU scene source differs: ' + name)
                    require(np.allclose(arrays['reference_motion'], known['reference_motion'][0].numpy(), atol=2e-5, rtol=2e-5),
                        'CPU scene reference reconstruction differs')
                    require(int(arrays['checkpoint_step']) == plan['declaration']['expected_step']
                        and str(arrays['checkpoint_sha256']) == plan['declaration']['checkpoint_sha256']
                        and str(arrays['reference_policy']) == policy, 'Saved scene checkpoint/policy differs')
                    if not use_scene:
                        require(not bool(arrays['use_scene']), 'Saved scene-off flag differs')
                    actual, counts = reference.metrics(arrays)
                    require(counts == row['counts'], 'CPU scene denominators differ')
                    for name, value in actual.items():
                        require((value is None) == (row['metrics'][name] is None), 'CPU scene metric missingness differs')
                        if value is not None:
                            maximum = max(maximum, abs(value - row['metrics'][name]))
                    for name, value in common.independent_core(arrays).items():
                        require((value is None) == (actual[name] is None), 'Independent scene missingness differs')
                        if value is not None:
                            maximum = max(maximum, abs(value - actual[name]))
                    paired_arrays.append(arrays)
                for name in ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion', 'identity_json'):
                    require(np.array_equal(paired_arrays[0][name], paired_arrays[1][name]), 'Scene-on/off saved input differs: ' + name)
    require(maximum < 1e-3 and not torch.cuda.is_initialized(), 'Independent CPU scene verification failed')
    verify(plan)
    write(out / 'CPU_RECHECK.json', dict(status='independent_cpu_passed', checked_at=common.now(),
        predictions=len(rows), pairs=plan['planned_pairs'], maximum_metric_difference=maximum, CUDA_initialized=False,
        source_GT_rest_observations_exact=True, scene_on_off_recursive_input_fingerprints_exact=True,
        scene_on_off_saved_references_exact=True, scene_off_guard_all_counts_zero=True,
        reference_CPU_reconstruction_tolerance=2e-5,
        rows_sha256=sha256(out / 'rows.jsonl')))
    summarize(out, plan, rows, verified=True)
    status(out, 'completed', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate', 'recheck'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.5)
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .8 and args.max_predictions >= 0, 'Invalid bounded GPU options')
    if args.command == 'prepare':
        prepare(args.out, args.source, args.per_dataset)
    elif args.command == 'evaluate':
        evaluate(args.out, args.resume, args.max_predictions, args.memory_fraction)
    else:
        recheck(args.out)


if __name__ == '__main__':
    main()
