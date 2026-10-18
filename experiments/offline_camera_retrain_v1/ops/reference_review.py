"""Frozen paired EXO evaluation of explicitly versioned reference policies.

The model checkpoints, observations, scenes, seeds and DDIM settings are held
fixed. Only the observation-derived reference initializer changes. No training
or TEST access occurs. Every prediction is durable and independently checked
on CPU before the completed report is published.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import time

from . import adapter_evaluation as evaluation
from . import reference_temporal
from . import validation_review as common
from .dependencies import dependency_closure, sha256


BASE = common.BASE
SOURCE = BASE / 'runs/native30_exo_expanded_review_v1'
DEFAULT_OUT = BASE / 'runs/native30_exo_reference_review'
MODELS = ('base41k', 'balancedv2adapter1000')
POLICIES = ('original', reference_temporal.POLICY)
PROTOCOL = 'frozen-exo-reference-policy-paired-review-v1'
read, write, require = common.read, common.write, common.require


def status(out, state, **values):
    write(out / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=common.now(), pid=os.getpid(), optimizer_updates=0,
        TEST_used=False, **values))


def closure():
    sources = evaluation.closure()
    sources.update(dependency_closure([Path(__file__), Path(reference_temporal.__file__)]))
    return sources


def prepare(out, per_dataset=4):
    common.cpu_setup()
    require(not (out / 'PLAN.json').exists(), 'Existing immutable review plan')
    require(per_dataset > 0, 'Positive per-dataset count required')
    cohort = read(SOURCE / 'COHORT.json')
    original = read(SOURCE / 'PLAN.json')
    config = original['config']
    data = common.validation_data(config)
    evaluation.validate_entries(data, cohort['entries'], config)
    entries = []
    # This prefix follows the previously sealed identity/hash ordering. Scores
    # and predictions play no role in this selection.
    pools = {d: [r for r in cohort['entries'] if r['dataset'] == d and r['panel'] == 'new55']
             for d in ('trumans', 'egobody', 'rich')}
    require(all(len(v) >= per_dataset for v in pools.values()), 'Insufficient new recordings')
    for index in range(per_dataset):
        for dataset in pools:
            entries.append(dict(pools[dataset][index], review_panel='primary'))
    regression = next(r for r in cohort['entries']
        if r['sequence_id'] == 'recording_20220215_S21_S22_01_interactee_1901-4901_master_full'
        and r['start'] == 1984)
    entries.insert(0, dict(regression, review_panel='diagnostic_regression'))
    registry = read(SOURCE / 'MODELS.json')
    models = {key: registry[key] for key in MODELS}
    for declaration in models.values():
        evaluation.verify_registered(declaration)
    sources = closure()
    # Bind the legal source index/manifests as well as scientific source code.
    sources.update(cohort['source_sha256'])
    for path in (SOURCE / 'COHORT.json', SOURCE / 'PLAN.json', SOURCE / 'MODELS.json'):
        sources[str(path.resolve())] = sha256(path)
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(protocol=PROTOCOL, prepared_at=common.now(), config=config,
        entries=entries, models=models, policies=list(POLICIES), regimes=['clean', 'noisy'],
        source_sha256=sources, planned_predictions=len(entries) * 8,
        selection='First N new55 identities per dataset in the pre-existing identity-only cohort; one window per recording. Known jitter case reported separately.',
        primary_windows=3 * per_dataset, diagnostic_windows=1,
        frames=192, fps=30, ddim_steps=20, optimizer_updates=0, TEST_used=False,
        reference_change_only=True, observations_scenes_seeds_physics_unchanged=True,
        noncausal_offline_window=True, automatic_checkpoint_promotion=False)
    write(out / 'PLAN.json', plan)
    status(out, 'prepared', completed_predictions=0, planned_predictions=plan['planned_predictions'])
    return plan


def verify(plan):
    require(plan['protocol'] == PROTOCOL and plan['policies'] == list(POLICIES), 'Review protocol differs')
    for path, expected in plan['source_sha256'].items():
        require(sha256(path) == expected, 'Sealed dependency changed: ' + path)
    for declaration in plan['models'].values():
        evaluation.verify_registered(declaration)


def temporal_metrics(arrays):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations
    rotations = []
    with torch.no_grad():
        for key in ('prediction', 'truth'):
            tensor = torch.from_numpy(np.array(arrays[key], dtype=np.float32, copy=True))[None]
            rotations.append(global_rotations(tensor)[0].double().numpy())
    pred, truth = rotations

    def angles(a, b):
        rel = np.swapaxes(a, -1, -2) @ b
        return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))

    pstep, gstep = angles(pred[:-1], pred[1:]), angles(truth[:-1], truth[1:])
    result = dict(arm_global_orientation_error_deg=float(angles(pred, truth)[:, 16:22].mean()),
        arm_step_angle_error_deg=float(np.abs(pstep[:, 16:22] - gstep[:, 16:22]).mean()))
    for label, joint in (('left_wrist', 20), ('right_wrist', 21)):
        result[label + '_step_p95_deg'] = float(np.percentile(pstep[:, joint], 95))
        result[label + '_gt_step_p95_deg'] = float(np.percentile(gstep[:, joint], 95))
        result[label + '_step_over30_count'] = int((pstep[:, joint] > 30).sum())
    return result


def metrics(arrays):
    values, counts = common.metrics_for(arrays)
    values.update(temporal_metrics(arrays))
    return values, counts


def key(row):
    return row['case'], row['regime'], row['policy'], row['model']


def rows_for(out, plan):
    path = out / 'rows.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    require(len({key(row) for row in rows}) == len(rows), 'Duplicate result identity')
    entries = {e['case']: e for e in plan['entries']}
    for row in rows:
        require(row['case'] in entries and row['model'] in MODELS and row['policy'] in POLICIES
                and row['regime'] in plan['regimes'], 'Unregistered result')
        require(row['checkpoint_sha256'] == plan['models'][row['model']]['checkpoint_sha256'], 'Wrong checkpoint')
        require(sha256(row['artifact']) == row['artifact_sha256'], 'Prediction artifact changed')
        require(row['seed'] == entries[row['case']]['case_seed'], 'Wrong sampling seed')
    return rows


def summarize(out, plan, rows, verified=False):
    groups = {}
    for panel in ('primary', 'diagnostic_regression'):
        for dataset in ('trumans', 'egobody', 'rich', 'macro'):
            for regime in plan['regimes']:
                for policy in POLICIES:
                    for model in MODELS:
                        selected = [r for r in rows if (r['review_panel'], r['regime'], r['policy'], r['model'])
                                    == (panel, regime, policy, model)
                                    and (dataset == 'macro' or r['dataset'] == dataset)]
                        if not selected:
                            continue
                        values = {}
                        for metric in selected[0]['metrics']:
                            by_dataset = []
                            for d in ('trumans', 'egobody', 'rich'):
                                valid = [r['metrics'][metric] for r in selected
                                         if r['dataset'] == d and r['metrics'][metric] is not None]
                                if valid:
                                    by_dataset.append(statistics.mean(valid))
                            values[metric] = statistics.mean(by_dataset) if by_dataset else None
                        groups['/'.join((panel, dataset, regime, policy, model))] = dict(
                            windows=len(selected), metrics=values)
    write(out / 'SUMMARY.json', dict(protocol=PROTOCOL, checked_at=common.now(),
        status='cpu_verified' if verified else 'partial_unverified', completed_predictions=len(rows),
        planned_predictions=plan['planned_predictions'], groups=groups,
        aggregation='Equal recording within dataset, equal dataset macro; diagnostic case excluded from primary.',
        metric_note='hidden/visible mean unavailable/available observed 3D positions, not exclusively physical occlusion. Angle-step metrics are diagnostics, not a benchmark protocol.',
        optimizer_updates=0, TEST_used=False))


def evaluate(out, resume=False):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    plan = read(out / 'PLAN.json')
    verify(plan)
    require(resume or not (out / 'rows.jsonl').exists(), 'Existing results require --resume')
    rows = rows_for(out, plan)
    done = {key(row) for row in rows}
    status(out, 'waiting_for_gpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
    with common.gpu_lock():
        verify(plan)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.5)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        data = common.validation_data(plan['config'])
        builder = common.observation_builder(plan['config'], device='cuda')
        models = {}
        for label, declaration in plan['models'].items():
            model, metadata = evaluation.load_registered(declaration, device='cuda')
            require(model_digest(model) == declaration['model_tensor_sha256'], 'Loaded model identity differs')
            models[label] = model
        folder = out / 'motions'
        folder.mkdir(exist_ok=True)
        started = time.monotonic()
        for entry in plan['entries']:
            for regime in plan['regimes']:
                if all((entry['case'], regime, policy, label) in done for policy in POLICIES for label in MODELS):
                    continue
                sample, observed, scene, asset = evaluation.source_observation(data, builder, entry, regime)
                _, identity = data.sample(entry['group'], 'exo', member=entry['member'], start=entry['start'], yaw=0)
                for policy in POLICIES:
                    context = nullcontext() if policy == 'original' else reference_temporal.installed()
                    with context:
                        batch, known = evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cuda')
                        input_sha = common.tensor_fingerprint(known)
                        previous = [r for r in rows if (r['case'], r['regime'], r['policy']) == (entry['case'], regime, policy)]
                        require(all(r['input_sha256'] == input_sha for r in previous), 'Resumed input differs')
                        for label in MODELS:
                            current = entry['case'], regime, policy, label
                            if current in done:
                                continue
                            status(out, 'sampling', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                                   case=entry['case'], dataset=entry['dataset'], regime=regime, policy=policy, model=label)
                            before = time.monotonic()
                            with torch.no_grad():
                                prediction = sample_motion(models[label], known, plan['config'], steps=20,
                                    seed=entry['case_seed'], use_scene=True)
                            torch.cuda.synchronize()
                            sampling_s = time.monotonic() - before
                            require(bool(torch.isfinite(prediction).all()), 'Nonfinite model output')
                            arrays = dict(prediction=prediction[0].detach().cpu().numpy(), truth=sample['motion'],
                                rest=sample['rest'], observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                                reference_motion=known['reference_motion'][0].detach().cpu().numpy(),
                                identity_json=np.array(json.dumps(identity)), checkpoint_step=np.array(plan['models'][label]['expected_step']),
                                checkpoint_sha256=np.array(plan['models'][label]['checkpoint_sha256']), reference_policy=np.array(policy))
                            artifact = folder / f"case{entry['case']:04d}_{regime}_{policy}_{label}.npz"
                            temporary = artifact.with_suffix('.tmp')
                            with temporary.open('wb') as stream:
                                np.savez_compressed(stream, **arrays)
                                stream.flush(); os.fsync(stream.fileno())
                            temporary.replace(artifact)
                            values, counts = metrics(arrays)
                            row = dict(case=entry['case'], dataset=entry['dataset'], sequence_id=entry['sequence_id'],
                                recording=entry['recording'], review_panel=entry['review_panel'], start=entry['start'],
                                source_first=entry['source_first'], regime=regime, policy=policy, model=label,
                                seed=entry['case_seed'], checkpoint_sha256=plan['models'][label]['checkpoint_sha256'],
                                checkpoint_step=plan['models'][label]['expected_step'], input_sha256=input_sha,
                                artifact=str(artifact.resolve()), artifact_sha256=sha256(artifact),
                                metrics=values, counts=counts, sampling_s=sampling_s, checked_at=common.now())
                            with (out / 'rows.jsonl').open('a') as stream:
                                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                            rows.append(row); done.add(current)
                            summarize(out, plan, rows)
                            write(out / 'PROGRESS.json', dict(completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
                                elapsed_s=time.monotonic() - started, last_sampling_s=sampling_s, checked_at=common.now()))
        for label, model in models.items():
            require(model_digest(model) == plan['models'][label]['model_tensor_sha256'], 'Model changed during inference')
        verify(plan)
    status(out, 'sampling_completed_pending_cpu', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def recheck(out):
    import numpy as np
    import torch
    common.cpu_setup()
    plan = read(out / 'PLAN.json'); verify(plan)
    rows = rows_for(out, plan)
    require(len(rows) == plan['planned_predictions'], 'Incomplete frozen evaluation')
    data = common.validation_data(plan['config'])
    builder = common.observation_builder(plan['config'])
    indexed = {key(r): r for r in rows}
    maximum = 0.
    for entry in plan['entries']:
        for regime in plan['regimes']:
            sample, observed, scene, asset = evaluation.source_observation(data, builder, entry, regime)
            for policy in POLICIES:
                fingerprints = []
                context = nullcontext() if policy == 'original' else reference_temporal.installed()
                with context:
                    batch, known = evaluation.collated_inputs(plan['config'], observed, scene, asset, 'cpu')
                for label in MODELS:
                    row = indexed[entry['case'], regime, policy, label]
                    with np.load(row['artifact'], allow_pickle=False) as saved:
                        arrays = {k: saved[k].copy() for k in saved.files}
                    for name, actual in [('truth', sample['motion']), ('rest', sample['rest']),
                            ('observations', observed['trajectory']), ('observation_meta', observed['observation_meta'])]:
                        require(np.array_equal(arrays[name], actual), 'Source array changed: ' + name)
                    require(np.allclose(arrays['reference_motion'], known['reference_motion'][0].numpy(), atol=2e-5, rtol=2e-5),
                            'Saved reference differs from CPU reconstruction')
                    actual, counts = metrics(arrays)
                    require(counts == row['counts'], 'Metric denominator changed')
                    for name, value in actual.items():
                        require((value is None) == (row['metrics'][name] is None), 'Metric missingness changed')
                        if value is not None:
                            maximum = max(maximum, abs(value - row['metrics'][name]))
                    for name, value in common.independent_core(arrays).items():
                        require((value is None) == (actual[name] is None), 'Independent metric missingness differs')
                        if value is not None:
                            maximum = max(maximum, abs(value - actual[name]))
                    fingerprints.append(row['input_sha256'])
                require(len(set(fingerprints)) == 1, 'Models used different recursive inputs')
    require(maximum < 1e-3 and not torch.cuda.is_initialized(), 'Independent CPU verification failed')
    verify(plan)
    write(out / 'CPU_RECHECK.json', dict(status='independent_cpu_passed', checked_at=common.now(),
        predictions=len(rows), maximum_metric_difference=maximum, CUDA_initialized=False,
        source_GT_rest_observations_exact=True, recursive_model_input_fingerprints_paired=True,
        reference_CPU_reconstruction_tolerance=2e-5, rows_sha256=sha256(out / 'rows.jsonl')))
    summarize(out, plan, rows, verified=True)
    status(out, 'completed', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate', 'recheck'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.per_dataset)
    elif args.command == 'evaluate':
        evaluate(args.out, args.resume)
    else:
        recheck(args.out)


if __name__ == '__main__':
    main()
