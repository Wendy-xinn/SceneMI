"""Paired fixed-weight EGO scene ablations using the existing validation sampler.

The trained sources remain immutable. Run this evaluator through ``python -m
scenemi resources run``. Donors come only from the pinned validation manifest;
all scene channels move together in the donor's camera-start local frame.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

CONDITIONS = ('matched', 'disabled', 'empty', 'mismatched')
SCENE_SOURCES = ('egobody', 'trumans', 'grail')
SCENE_KEYS = ('occupancy', 'frame_scene_points', 'frame_scene_owner')


def scene_identity(case, metadata):
    """Reject same-room swaps even when their recordings differ."""
    key = metadata.get('object_family_id') if case['dataset'] == 'grail' else metadata.get('scene_id')
    if not key:
        raise ValueError(f'Missing physical scene identity for {case}')
    return f'{case["dataset"]}:{key}'


def donor_plan(cases, scenes, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for case in cases:
        if case['dataset'] != 'amass':
            groups[case['dataset']][scenes[case['case_id']]].append(case)
    plan = {}
    for case in cases:
        cid, dataset = case['case_id'], case['dataset']
        if dataset == 'amass':
            plan[cid] = None
            continue
        eligible = {}
        for scene, pool in groups[dataset].items():
            if scene != scenes[cid]:
                candidates = [x for x in pool if x['family'] != case['family']]
                if candidates:
                    eligible[scene] = candidates
        if not eligible:
            raise ValueError(f'No different-scene, different-family validation donor for {case}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, cid]))
        scene = sorted(eligible)[int(rng.integers(len(eligible)))]
        pool = eligible[scene]
        plan[cid] = pool[int(rng.integers(len(pool)))]
    return plan


def tensor_identity(sample, keys):
    """Digest all camera/body inputs and private truth before/after scene removal."""
    digest = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(sample[key])
        digest.update(str((key, value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def recover_prefix(out, cases, batch_size):
    """Use only full batches present in every arm, even after a host reboot."""
    parsed, lines_by_condition = {}, {}
    status = json.loads((out / 'STATUS.json').read_text())
    for condition in CONDITIONS:
        lines = (out / f'{condition}.jsonl').read_text().splitlines(keepends=True)
        values = []
        for index, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError('Corrupt metrics before the final interrupted row')
                break
            if value['case'] != cases[index]:
                raise ValueError('Saved validation order differs')
            values.append(value)
        parsed[condition], lines_by_condition[condition] = values, lines
    count = min(status['completed_windows'], *(len(v) for v in parsed.values()))
    if count != len(cases):
        count = count // batch_size * batch_size
    return count, {key: value[:count] for key, value in parsed.items()}, {
        key: ''.join(lines[count:]) for key, lines in lines_by_condition.items()}


def condition_payload(public, condition, donor=None):
    """Change scene channels only; an absent scene has invalid owners, not zeros."""
    import torch
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown scene condition {condition}')
    value = dict(public)
    if condition == 'disabled':
        for key in SCENE_KEYS:
            value.pop(key)
    elif condition == 'empty':
        for key in SCENE_KEYS:
            value[key] = torch.full_like(public[key], -1) if key == 'frame_scene_owner' else torch.zeros_like(public[key])
    elif condition == 'mismatched':
        if donor is None:
            raise ValueError('Mismatched input requires a declared donor batch')
        for key in SCENE_KEYS:
            if donor[key].shape != public[key].shape or donor[key].dtype != public[key].dtype:
                raise ValueError(f'Donor scene schema differs: {key}')
            value[key] = donor[key]
    return value


def summarize(rows, bootstrap_samples, seed):
    """Preserve original equal-dataset/equal-family metrics and paired deltas."""
    from scenemi.evaluate import aggregate
    expected = [row['case'] for row in rows['matched']]
    for values in rows.values():
        if [row['case'] for row in values] != expected:
            raise ValueError('Unpaired validation windows')
    results = {condition: aggregate(values) for condition, values in rows.items()}
    for result in results.values():
        result['scene_dataset_macro_mean'] = {
            key: float(np.mean([result['by_dataset'][d][key] for d in SCENE_SOURCES]))
            for key in result['dataset_macro_mean']}
    base = results['matched']
    deltas = {}
    for condition in rows:
        if condition == 'matched':
            continue
        delta = {scope: {key: results[condition][scope][key] - base[scope][key]
                         for key in base[scope]}
                 for scope in ('dataset_macro_mean', 'scene_dataset_macro_mean')}
        delta['by_dataset'] = {d: {k: results[condition]['by_dataset'][d][k] - base['by_dataset'][d][k]
                                   for k in base['by_dataset'][d]}
                               for d in base['by_dataset']}
        # Resample paired family means, not correlated, overlapping windows.
        grouped = defaultdict(lambda: defaultdict(list))
        if len(rows['matched']) != len(rows[condition]):
            raise ValueError('Unpaired validation window counts')
        for a, b in zip(rows['matched'], rows[condition]):
            if a['case'] != b['case']:
                raise ValueError('Unpaired validation windows')
            grouped[a['identity']['dataset']][a['identity']['family']].append(
                b['metrics']['mpjpe_mm'] - a['metrics']['mpjpe_mm'])
        rng = np.random.default_rng(seed)
        draws = {}
        for dataset, families in grouped.items():
            means = np.array([np.mean(v) for _, v in sorted(families.items())])
            draws[dataset] = means[rng.integers(len(means), size=(bootstrap_samples, len(means)))].mean(1)
        delta['mpjpe_family_bootstrap_95ci_mm'] = {
            'by_dataset': {d: np.quantile(v, [.025, .975]).tolist() for d, v in draws.items()},
            'scene_dataset_macro_mean': np.quantile(np.mean([draws[d] for d in SCENE_SOURCES], axis=0), [.025, .975]).tolist(),
            'dataset_macro_mean': np.quantile(np.mean(list(draws.values()), axis=0), [.025, .975]).tolist(),
            'replicates': bootstrap_samples,
            'unit': 'paired family within dataset; descriptive validation uncertainty, no checkpoint-selection correction'}
        changes = defaultdict(lambda: defaultdict(list))
        for row in rows[condition]:
            changes[row['identity']['dataset']][row['identity']['family']].append(row['prediction_change_from_matched_mm'])
        delta['prediction_change_from_matched_mm'] = {
            dataset: float(np.mean([np.mean(v) for v in families.values()]))
            for dataset, families in changes.items()}
        deltas[condition] = delta
    return dict(conditions=results, delta_from_matched= deltas,
                delta_convention='condition minus matched; positive error means matched is better')


def run(config_path, *, device='cuda', output=None, preflight=False, resume=False):
    import torch
    import yaml
    from tqdm.auto import tqdm
    from scenemi.config import ROOT, load_paths
    from scenemi.checkpoint import load_trained_model
    from scenemi.data.amass_prepare import atomic_json, sha256
    from scenemi.data.ego import ALL_KEYS, source_cases
    from scenemi.data.scene_ego import SceneEgoData, collate, payload
    from scenemi.execution import configure_runtime
    from scenemi.metrics import motion_metrics
    from scenemi.training import deterministic, now
    from experiments.offline_camera_retrain_v1.native30_model import sample_native30
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

    spec = yaml.safe_load(config_path.read_text())
    if (spec['version'] != 'ego-scene-input-ablation-v1' or tuple(spec['conditions']) != CONDITIONS
            or spec['donor_policy'] != 'same_dataset_different_scene_and_family'
            or spec['scene_coordinates'] != 'donor_camera_start_local_yup_meters'
            or spec['amass_policy'] != 'unchanged_empty_scene_no_donor'):
        raise ValueError('Unsupported fixed-weight scene ablation protocol')
    for key in ('checkpoint', 'validation_manifest'):
        if sha256(ROOT / spec[key]) != spec[key + '_sha256']:
            raise ValueError(f'Pinned {key} differs')
    model, c, step = load_trained_model(ROOT / spec['checkpoint'])
    if c['version'] != 'ego-scene-adaptation-v1' or step != spec['checkpoint_step']:
        raise ValueError('Expected the declared scene checkpoint')
    configure_runtime(c, device)
    deterministic(c['seed'], device)
    data = SceneEgoData(c, load_paths(ROOT / 'configs/paths.local.yaml'), 'validation')
    manifest = json.loads((ROOT / spec['validation_manifest']).read_text())
    cases = manifest['full']
    if manifest['test_used'] or source_cases(data) != cases:
        raise ValueError('Fixed validation membership/order differs')
    scenes = {case['case_id']: scene_identity(case, data.native.selection_bundle.metadata(
                  data.native.records[case['member']], 'validation'))
              for case in cases if case['dataset'] != 'amass'}
    plan = donor_plan(cases, scenes, spec['donor_seed'])
    out = (ROOT / spec['output']) if output is None else output
    if resume and preflight:
        raise ValueError('Resume is for full validation only')
    out.mkdir(parents=True, exist_ok=resume)
    if preflight:
        # One real recipient from each source; donors still use the full plan.
        seen, selected = set(), []
        for case in cases:
            if case['dataset'] not in seen:
                seen.add(case['dataset'])
                selected.append(case)
        cases = selected
    source_roots = (ROOT, ROOT.parent / 'diffusion-motion-inbetweening')
    sources = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and name.endswith('.py') and Path(name).is_absolute():
            path = Path(name).resolve()
            if any(path.is_relative_to(root) for root in source_roots):
                sources[str(path)] = sha256(path)
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    protocol = dict(spec, config_sha256=sha256(config_path), checkpoint_step=step,
        model_config=c, validation=dict(c['validation']), preflight=preflight, windows=len(cases), device=device,
        physical_scene_counts={dataset: len({scene for cid, scene in scenes.items()
            if scene.startswith(dataset + ':')}) for dataset in SCENE_SOURCES},
        test_used=False, inference_only=True, causal=False, created_at_utc=now(), source_sha256=sources,
        scene_map='whole-recording union of observed first-hit surfaces; no complete scene mesh',
        matched_reference_sha256=sha256(ROOT / spec['reference_metrics']),
        recipient_and_donor_cases=[dict(case=case, recipient_scene=scenes.get(case['case_id']),
            donor=plan[case['case_id']], donor_scene=scenes.get(plan[case['case_id']]['case_id'])
            if plan[case['case_id']] is not None else None) for case in cases])
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    size = c['validation']['eval_batch_size']
    resume_offset, rows = 0, {condition: [] for condition in CONDITIONS}
    if resume:
        previous = json.loads((out / 'PROTOCOL.json').read_text())
        if any(previous[key] != spec[key] for key in spec) or previous['device'] != device:
            raise ValueError('Resume configuration or device differs')
        if previous['recipient_and_donor_cases'] != protocol['recipient_and_donor_cases']:
            raise ValueError('Resume donor plan differs')
        for filename, digest in previous['source_sha256'].items():
            path = Path(filename)
            if path == Path(__file__).resolve() and sha256(path) != digest:
                path = out / 'recovery/evaluator_before_resume.py'
            if sha256(path) != digest:
                raise ValueError(f'Resume dependency changed: {filename}')
        resume_offset, rows, tails = recover_prefix(out, cases, size)
        recovery = dict(state='prepared', parent_protocol_sha256=previous['sha256'],
            resumed_at_utc=now(), resume_offset=resume_offset,
            remaining_windows=len(cases)-resume_offset, source_sha256=sources,
            uncommitted_rows=tails, original_status=json.loads((out / 'STATUS.json').read_text()),
            unchanged_model_data_metrics_rng=True,
            runtime_change='Recover common complete batches; fsync metrics and persist running audit')
        recovery_file = out / 'recovery/RESUME.json'
        if recovery_file.exists():
            recovery_file = out / 'recovery' / f'RESUME_{resume_offset:06d}.json'
        atomic_json(recovery_file, recovery)
        for condition in CONDITIONS:
            path = out / f'{condition}.jsonl'
            temporary = path.with_suffix('.jsonl.recovering')
            with temporary.open('w') as stream:
                stream.write(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows[condition]))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        protocol = previous
    else:
        atomic_json(out / 'PROTOCOL.json', protocol)
    atomic_json(out / 'STATUS.json', dict(state='caching_scene_inputs', completed_windows=resume_offset, windows=len(cases)))
    replay_start = (resume_offset-1)//size*size if resume_offset else 0
    remaining = cases[replay_start:]
    required = {case['case_id']: case for case in remaining if case['dataset'] != 'amass'}
    required.update({plan[case['case_id']]['case_id']: plan[case['case_id']]
                     for case in remaining if plan[case['case_id']] is not None})
    cache, identities, input_hashes = {}, {}, {}
    for case in tqdm(sorted(required.values(), key=lambda x: x['case_id']), desc='Scene inputs', unit='window'):
        sample, identity = data.sample_case(case)
        cid = case['case_id']
        cache[cid] = {key: sample[key].copy() for key in SCENE_KEYS}
        identities[cid] = identity
        input_hashes[cid] = tensor_identity(sample, ALL_KEYS)
    cache_bytes = sum(value.nbytes for scene in cache.values() for value in scene.values())
    print(f'Cached {len(cache)} observed scene windows ({cache_bytes / 2**30:.2f} GiB); no prediction arrays saved', flush=True)
    model = model.to(device).eval()
    audit = dict(camera_and_truth_equal=True, scene_channels_changed_windows=0,
                 mismatched_scene_windows=0, amass_unavailable_windows=0, cache_bytes=cache_bytes,
                 predicted_rest_max_change_m=0., empty_vs_disabled_prediction_max_abs_m=0.,
                 amass_intervention_prediction_max_abs_m=0.)
    audit.update(measured_from_window=replay_start,
        historical_in_memory_audit_lost=bool(resume),
        camera_truth_check_scope='All committed rows passed the fail-closed input identity check; runtime maxima cover this process')
    replay_errors = defaultdict(float)
    streams = {condition: (out / f'{condition}.jsonl').open('a' if resume else 'w') for condition in CONDITIONS}
    try:
        with torch.inference_mode():
            for offset in tqdm(range(replay_start, len(cases), size), desc='Paired ablation', unit='batch'):
                chunk = cases[offset:offset+size]
                samples, chunk_identities, donors = [], [], []
                for case in chunk:
                    sample, identity = data.sample_camera_case(case)
                    cid = case['case_id']
                    if case['dataset'] == 'amass':
                        scene = {key: sample[key] for key in SCENE_KEYS}
                        donor_scene = scene
                        audit['amass_unavailable_windows'] += 1
                    else:
                        if tensor_identity(sample, ALL_KEYS) != input_hashes[cid]:
                            raise ValueError(f'Scene preparation changed recipient camera/body/truth: {case}')
                        scene = cache[cid]
                        donor_scene = cache[plan[cid]['case_id']]
                        identity = identities[cid]
                        audit['mismatched_scene_windows'] += 1
                        audit['scene_channels_changed_windows'] += int(any(
                            not np.array_equal(scene[key], donor_scene[key]) for key in SCENE_KEYS))
                    sample.update(scene)
                    samples.append(sample)
                    chunk_identities.append(identity)
                    donors.append(donor_scene)
                batch = collate(samples, device)
                public = payload(batch)
                donor_batch = {key: torch.as_tensor(np.stack([sample[key] for sample in donors]), device=device)
                               for key in SCENE_KEYS}
                truth = forward_kinematics(batch['motion'], batch['target_rest'])
                predictions = {}
                matched_rest = None
                for condition in CONDITIONS:
                    condition_inputs = condition_payload(public, condition, donor_batch)
                    enabled = condition != 'disabled'
                    prediction = sample_native30(model, condition_inputs, steps=c['validation']['ddim_steps'],
                        seed=c['validation']['seed'], use_scene=enabled)
                    _, details = model(prediction, torch.zeros(len(chunk), device=device, dtype=torch.long),
                        condition_inputs, use_scene=enabled, return_details=True)
                    predicted = forward_kinematics(prediction, details['rest'])
                    predictions[condition] = predicted
                    if condition == 'matched':
                        matched_rest = details['rest']
                    else:
                        audit['predicted_rest_max_change_m'] = max(audit['predicted_rest_max_change_m'],
                            float((details['rest'] - matched_rest).abs().max()))
                    if condition == 'empty':
                        audit['empty_vs_disabled_prediction_max_abs_m'] = max(
                            audit['empty_vs_disabled_prediction_max_abs_m'],
                            float((predicted - predictions['disabled']).abs().max()))
                    for index, case in enumerate(chunk):
                        row = dict(case=case, identity=chunk_identities[index],
                            metrics=motion_metrics(predicted[index], truth[index]),
                            prediction_change_from_matched_mm=float(
                                (predicted[index] - predictions['matched'][index]).norm(dim=-1).mean() * 1000))
                        if case['dataset'] == 'amass' and condition in ('empty', 'mismatched'):
                            audit['amass_intervention_prediction_max_abs_m'] = max(
                                audit['amass_intervention_prediction_max_abs_m'],
                                float((predicted[index] - predictions['matched'][index]).abs().max()))
                        if condition == 'mismatched':
                            row.update(donor=plan[case['case_id']], recipient_scene=scenes.get(case['case_id']),
                                donor_scene=scenes.get(plan[case['case_id']]['case_id'])
                                if plan[case['case_id']] is not None else None)
                        if offset < resume_offset:
                            old = rows[condition][offset+index]
                            for key, value in row['metrics'].items():
                                error = abs(value-old['metrics'][key])
                                replay_errors[key] = max(replay_errors[key], error)
                                if error > .001:
                                    raise ValueError(f'Resumed replay differs: {condition}, {case}, {key}, {error}')
                        else:
                            rows[condition].append(row)
                            streams[condition].write(json.dumps(row, allow_nan=False) + '\n')
                    streams[condition].flush()
                    os.fsync(streams[condition].fileno())
                if offset < resume_offset:
                    atomic_json(out / 'recovery/REPLAY_VERIFICATION.json', dict(state='passed',
                        start=offset, windows=len(chunk), conditions=list(CONDITIONS),
                        maximum_metric_difference=dict(replay_errors), tolerance=.001,
                        checkpoint_sha256=spec['checkpoint_sha256']))
                atomic_json(out / 'RUNTIME_AUDIT.json', audit)
                atomic_json(out / 'STATUS.json', dict(state='evaluating',
                    completed_windows=max(resume_offset, min(offset+size, len(cases))), windows=len(cases), updated_at_utc=now()))
        result = summarize(rows, spec['bootstrap_samples'], spec['donor_seed'])
        reference = json.loads((ROOT / spec['reference_metrics']).read_text())
        reference_delta = {}
        for condition, original in [('matched', reference), ('disabled', reference['no_scene_fallback'])]:
            reference_delta[condition] = {key: result['conditions'][condition]['dataset_macro_mean'][key]
                                         - original['dataset_macro_mean'][key]
                                         for key in original['dataset_macro_mean']}
        result.update(state='complete', checkpoint_step=step, checkpoint_sha256=spec['checkpoint_sha256'],
            protocol_sha256=protocol['sha256'], completed_at_utc=now(), test_used=False, inference_only=True,
            preflight=preflight, audit=audit, reference_macro_delta=reference_delta,
            recovery=dict(resumed=resume, completed_prefix_windows=resume_offset,
                replay_verification='recovery/REPLAY_VERIFICATION.json' if resume_offset else None),
            reference_comparable=not preflight and all(abs(value) < .001
                for values in reference_delta.values() for value in values.values()),
            interpretation='AMASS has no scene; use the three-scene-dataset macro for geometric input effects. '
                'Empty retains the trained branch; disabled bypasses it. '
                'Donor geometry is canonicalized to its own starting camera, with recipient camera and truth unchanged.')
        atomic_json(out / 'RESULT.json', result)
        atomic_json(out / 'STATUS.json', dict(state='complete', windows=len(cases), completed_windows=len(cases),
            protocol_sha256=protocol['sha256'], completed_at_utc=now()))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for stream in streams.values():
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Recover complete paired batches and replay the preceding batch')
    args = parser.parse_args(argv)
    run(args.config, device=args.device, output=args.output, preflight=args.preflight, resume=args.resume)


if __name__ == '__main__':
    main()
