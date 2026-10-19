"""Read-only query traces for an existing, identity-matched EGO evaluation.

Run with ``python -m experiments.offline_camera_retrain_v1.ops.scene_query_review``.
Queries are observed in place, never repeated or replaced. Ground truth enters
only the post-sampling diagnostic reduction. Historical evaluations are immutable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from . import ego_scene_evaluation as evaluation
from . import ego_scene_review as review
from .dependencies import dependency_closure, sha256
from .scene_query_metrics import summarize_query_step, aggregate_query_steps

BASE = review.BASE
DEFAULT_SOURCE = BASE / 'runs/native30_ego_scene_fusion_v2'
DEFAULT_OUT = BASE / 'runs/native30_ego_scene_query_review'
PROTOCOL = 'retained-ego-readonly-query-review-v1'
read, write, require = review.read, review.write, review.require


def prepare(out, source):
    require(not (out / 'PLAN.json').exists(), 'Query plan already exists')
    selected = read(source / 'SELECTED_MODEL.json')
    old = read(source / 'final_seed0/PLAN.json')
    cohort = evaluation.verify_cohort(evaluation.DEFAULT_COHORT)
    entries = evaluation.entries_for(cohort, 'dev', 0)
    declaration = old['models']['control']
    require(declaration['checkpoint_sha256'] == selected['checkpoint_sha256'], 'Selected model differs')
    rows = [json.loads(line) for line in (source / 'final_seed0/rows.jsonl').read_text().splitlines()]
    paired = [r for r in rows if r['model'] == 'control' and r['arm'] == 'learned_only'
              and r['case'] in {e['case'] for e in entries}]
    require(len(paired) == 2 * len(entries) == 24, 'Expected twelve fixed dev recordings, two regimes')
    files = [source / 'SELECTED_MODEL.json', source / 'final_seed0/PLAN.json',
             source / 'final_seed0/rows.jsonl', evaluation.DEFAULT_COHORT,
             Path(selected['source_config']['path'])]
    sources = review.common.closure()
    sources.update(dependency_closure([Path(__file__), BASE / 'native30_scene_relations.py',
                                      BASE / 'native30_refinement.py', BASE / 'prepare_native30.py',
                                      Path(__file__).with_name('test_scene_query_metrics.py')]))
    files.extend(Path(r['artifact']) for r in paired)
    plan = dict(protocol=PROTOCOL, prepared_at=review.common.now(), config=old['config'],
        entries=entries, declaration=declaration, baselines=paired, regimes=['clean', 'noisy'],
        planned_predictions=24, ddim_steps=20, source_sha256=sources,
        historical_source_sha256=old['source_sha256'],
        dependencies={str(p.resolve()): sha256(p) for p in files},
        selected_checkpoint_sha256=selected['checkpoint_sha256'],
        query_slots=dict(near='radius .6 m / first returned neighbor',
                         relation='first nearest neighbor; wide upward slots excluded'),
        selection='Unchanged first four recordings per dataset from previous identity-only dev panel; no error ranking.',
        instrumentation='Actual query centres and returned points captured without an extra query; original outputs retained.',
        GT_policy='No GT in tracing context; proposal FK and optional GT errors reduced only after sample_motion returns.',
        interpretation='Descriptive noise/temporal diagnostics, not an intervention or proof of causal query failure.',
        optimizer_updates=0, TEST_used=False, model_or_config_modified=False)
    write(out / 'PLAN.json', plan)
    return plan


def verify(plan):
    require(plan['protocol'] == PROTOCOL, 'Unknown query protocol')
    for file, digest in {**plan['source_sha256'], **plan['historical_source_sha256'], **plan['dependencies']}.items():
        require(sha256(file) == digest, 'Query dependency changed: ' + file)
    require(sha256(plan['declaration']['checkpoint']) == plan['selected_checkpoint_sha256'], 'Model changed')


@contextmanager
def trace_queries(model, known):
    """Observe actual calls; CPU copies cannot become model inputs."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1 import native30_scene_relations as relations
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    records, current = [], {}
    relation_depth = 0
    original_relation, original_forward = relations.relation_query, model.forward

    def cpu(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.array(value, copy=True)

    def observe_relation(scene, centres, **kwargs):
        nonlocal relation_depth
        outer = relation_depth == 0
        relation_depth += 1
        try:
            result = original_relation(scene, centres, **kwargs)
        finally:
            relation_depth -= 1
        if outer:
            require('relation' not in current, 'Expected batch size one relation query')
            current['relation'] = dict(centres=cpu(centres), points=cpu(result['points'][:, :, 0]),
                valid=cpu(result['valid'][:, :, 0]), owners=cpu(result['owners'][:, :, 0]),
                reliable=cpu(result['normal_reliable'][:, :, 0]))
        return result

    def query_wrapper(original):
        def observe(centres, rotations, *args, **kwargs):
            result = original(centres, rotations, *args, **kwargs)
            require('near' not in current, 'Expected batch size one near query')
            current['near'] = dict(centres=cpu(centres), points=cpu(result['points'][:, :, 2, 0]),
                valid=cpu(result['valid'][:, :, 2, 0]), owners=cpu(result['owners'][:, :, 2, 0]),
                reliable=cpu(result['normal_reliable'][:, :, 2, 0]))
            return result
        return observe

    def observe_forward(*args, **kwargs):
        require(not current, 'Unconsumed query trace')
        requested = kwargs.pop('return_details', False)
        output, details = original_forward(*args, **kwargs, return_details=True)
        require(set(current) == {'near', 'relation'}, 'Missing actual query')
        timestep = int(args[1][0])
        records.append(dict(timestep=timestep, alpha_bar=float(model.alphas[timestep]),
            proposal=cpu(forward_kinematics(details['proposal'], known['rest'])[0]),
            context=cpu(details['region_context'][0]),
            **current))
        current.clear()
        return (output, details) if requested else output

    with ExitStack() as stack:
        require(len(known['surface_windows']) == 1, 'Query review requires batch size one')
        for scene in known['surface_windows']:
            name = 'query_motion' if hasattr(scene, 'query_motion') else 'query'
            stack.enter_context(patch.object(scene, name, query_wrapper(getattr(scene, name))))
        stack.enter_context(patch.object(relations, 'relation_query', observe_relation))
        stack.enter_context(patch.object(model, 'forward', observe_forward))
        yield records


def reduce_trace(arrays, window_id):
    rows = []
    for index, timestep in enumerate(arrays['timesteps']):
        for kind in ('near', 'relation'):
            kwargs = {name: arrays[kind + '_' + source][index] for name, source in (
                ('query_centres_m', 'centres'), ('surface_points_m', 'points'),
                ('surface_valid', 'valid'), ('surface_owners', 'owners'),
                ('surface_normal_reliable', 'reliable'))}
            rows.append(summarize_query_step(timestep=int(timestep), alpha_bar=float(arrays['alphas'][index]),
                proposal_joints_m=arrays['proposal'][index], gt_joints_m=arrays['truth_joints'],
                feature_values=arrays['context'][index] if kind == 'relation' else None,
                window_id=window_id, query_kind=kind, **kwargs))
    return rows


def load_rows(out):
    path = out / 'rows.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(out, limit=0):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    plan = read(out / 'PLAN.json'); verify(plan)
    rows = load_rows(out); done = {(r['case'], r['regime']) for r in rows}
    require(len(done) == len(rows), 'Duplicate trace result')
    for row in rows:
        require(sha256(row['artifact']) == row['artifact_sha256'], 'Existing trace changed')
    generated = 0
    with review.common.gpu_lock():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.5)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        data = review.common.validation_data(plan['config']); review.validate_entries(data, plan['entries'])
        builder = review.common.observation_builder(plan['config'], device='cuda')
        model, _ = evaluation.load_model(plan['declaration'], plan['config'], 'cuda')
        (out / 'traces').mkdir(exist_ok=True)
        for entry in plan['entries']:
            for regime in plan['regimes']:
                if (entry['case'], regime) in done:
                    continue
                write(out / 'STATUS.json', dict(status='tracing', case=entry['case'], dataset=entry['dataset'],
                    regime=regime, completed=len(rows), planned=24, checked_at=review.common.now(), pid=os.getpid()))
                sample, observed, scene, asset = review.common.source_observation(data, builder,
                    dict(entry, case_seed=entry['observation_seed']), regime)
                _, known = review.common.collated_inputs(plan['config'], observed, scene, asset, 'cuda')
                baseline = next(r for r in plan['baselines'] if (r['case'], r['regime']) == (entry['case'], regime))
                require(baseline['seed'] == entry['case_seed'] and baseline['checkpoint_sha256']
                        == plan['selected_checkpoint_sha256'], 'Baseline seed/checkpoint differs')
                fingerprint = review.common.tensor_fingerprint(known)
                require(fingerprint == baseline['input_sha256'], 'Baseline inputs differ')
                started = time.monotonic()
                with review.instrumented_arm(model, 'learned_only') as (stages, calls, guards), \
                        trace_queries(model, known) as traces, torch.no_grad():
                    prediction = sample_motion(model, known, plan['config'], steps=20,
                        seed=entry['case_seed'], use_scene=True)
                predicted = prediction[0].detach().cpu().numpy()
                with np.load(baseline['artifact'], allow_pickle=False) as previous:
                    delta = float(np.abs(predicted - previous['prediction']).max())
                    require(np.array_equal(predicted, previous['prediction']), f'Trace changed output: {delta}')
                    require(np.array_equal(sample['motion'], previous['truth']), 'Baseline truth differs')
                    require(np.array_equal(sample['rest'], previous['rest']), 'Baseline rest differs')
                    require(np.array_equal(observed['trajectory'], previous['observations'])
                            and np.array_equal(observed['observation_meta'], previous['observation_meta']),
                            'Baseline observations differ')
                    require(np.array_equal(known['reference_motion'][0].detach().cpu().numpy(),
                                           previous['reference_motion']), 'Baseline reference differs')
                require(len(traces) == 20 and calls['guide_motion'] == 0, 'Unexpected query/physics calls')
                require(review.common.tensor_fingerprint(known) == fingerprint, 'Trace mutated model inputs')
                # GT is first used here, after all denoising/model calls completed.
                truth = torch.from_numpy(np.array(sample['motion'], copy=True))[None]
                rest = torch.from_numpy(np.array(sample['rest'], copy=True))[None]
                arrays = dict(timesteps=np.array([r['timestep'] for r in traces]),
                    alphas=np.array([r['alpha_bar'] for r in traces]),
                    proposal=np.stack([r['proposal'] for r in traces]),
                    context=np.stack([r['context'] for r in traces]),
                    truth_joints=forward_kinematics(truth, rest)[0].numpy())
                for kind in ('near', 'relation'):
                    for name in ('centres', 'points', 'valid', 'owners', 'reliable'):
                        arrays[kind + '_' + name] = np.stack([r[kind][name] for r in traces])
                identity = f"{entry['dataset']}/{entry['recording']}/{regime}"
                diagnostics = reduce_trace(arrays, identity)
                artifact = out / 'traces' / f"case{entry['case']:04d}_{regime}.npz"
                with artifact.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
                artifact.with_suffix('.tmp').replace(artifact)
                row = dict(case=entry['case'], dataset=entry['dataset'], recording=entry['recording'], regime=regime,
                    seed=entry['case_seed'], input_sha256=fingerprint, prediction_exactly_matches_baseline=True,
                    prediction_max_difference=delta, baseline_artifact_sha256=baseline['artifact_sha256'],
                    artifact=str(artifact.resolve()), artifact_sha256=sha256(artifact), query_steps=diagnostics,
                    physics_calls=calls['guide_motion'], elapsed_s=time.monotonic()-started)
                review.append_row(out, row); rows.append(row); done.add((entry['case'], regime)); generated += 1
                print(json.dumps(dict(completed=len(rows), case=entry['case'], regime=regime, elapsed_s=row['elapsed_s'])), flush=True)
                if limit and generated >= limit:
                    return
        require(model_digest(model) == plan['declaration']['model_tensor_sha256'], 'Model state changed')
    verify(plan)
    write(out / 'STATUS.json', dict(status='traced_pending_cpu_recheck', completed=len(rows), planned=24,
                                   checked_at=review.common.now(), optimizer_updates=0))


def recheck(out):
    import numpy as np
    import torch
    review.common.cpu_setup()
    plan = read(out / 'PLAN.json'); verify(plan); rows = load_rows(out)
    require(len(rows) == 24 and len({(r['case'], r['regime']) for r in rows}) == 24, 'Incomplete trace')
    require({(r['case'], r['regime']) for r in rows} ==
            {(e['case'], regime) for e in plan['entries'] for regime in plan['regimes']}, 'Unexpected trace cases')
    for row in rows:
        require(sha256(row['artifact']) == row['artifact_sha256'], 'Trace artifact changed')
        with np.load(row['artifact'], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        identity = f"{row['dataset']}/{row['recording']}/{row['regime']}"
        require(reduce_trace(arrays, identity) == row['query_steps'], 'CPU metric reduction differs')
        require(row['prediction_exactly_matches_baseline'] and row['prediction_max_difference'] == 0
                and row['physics_calls'] == 0, 'Trace intervention detected')
    aggregated = aggregate_query_steps([step for row in rows for step in row['query_steps']])
    write(out / 'QUERY_SUMMARY.json', aggregated)
    write(out / 'CPU_RECHECK.json', dict(status='passed', checked_at=review.common.now(), predictions=24,
        steps=480, query_records=960, prediction_baseline_exact=True, CUDA_initialized=torch.cuda.is_initialized(),
        source_and_model_seals_verified=True, gt_used_only_post_sampling=True,
        scope='Recomputes query statistics from traces, not an independent geometry backend audit.'))
    write(out / 'STATUS.json', dict(status='completed_cpu_verified', completed=24, planned=24,
        checked_at=review.common.now(), optimizer_updates=0, TEST_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'recheck'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.out, args.source)
    elif args.action == 'run':
        run(args.out, args.limit)
    else:
        recheck(args.out)


if __name__ == '__main__':
    main()
