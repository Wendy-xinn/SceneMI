"""Fixed-weight scene interventions for the prepared observed-geometry model.

All physical metrics use the recipient's true observed map, including when
inference receives a different map. No optimizer or training launch is used.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import source_cases
from scenemi.data.scene_ego import SCENE_KEYS
from scenemi.data.scene_geometry_data import GeometryEgoData, GEOMETRY_KEYS, collate, payload
from scenemi.execution import configure_runtime
from scenemi.geometry.alignment import place_ego_motion
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.metrics import motion_metrics
from scenemi.scene_geometry_training import build_model, load_config
from scenemi.scene_input_evaluation import (
    CONDITIONS, donor_plan, recover_prefix, scene_identity, summarize,
)
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

SCENE_INPUTS = SCENE_KEYS + GEOMETRY_KEYS


def condition_payload(public, condition, donor=None):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown scene condition: {condition}')
    value = dict(public)
    for key in SCENE_INPUTS:
        if condition == 'disabled':
            value.pop(key)
        elif condition == 'empty':
            value[key] = torch.full_like(public[key], -1) if key == 'frame_scene_owner' else torch.zeros_like(public[key])
        elif condition == 'mismatched':
            if donor is None or donor[key].shape != public[key].shape or donor[key].dtype != public[key].dtype:
                raise ValueError(f'Missing or incompatible donor scene channel: {key}')
            value[key] = donor[key]
    for key in public.keys() - set(SCENE_INPUTS):
        if value[key] is not public[key]:
            raise AssertionError(f'Intervention changed a camera/body input: {key}')
    return value


def metrics(model, motion, details, batch, policy):
    joints = forward_kinematics(motion, details['rest'])
    truth = forward_kinematics(batch['motion'], batch['target_rest'])
    _, geometry = geometry_terms(model, motion, details, batch, torch.ones(len(motion), device=motion.device), policy)
    placed = place_ego_motion(motion, details['rest'], details['head_camera_translation'], details['observations']['camera'], rotation=False)
    placed_joints = forward_kinematics(placed, details['rest'])
    _, placed_geometry = geometry_terms(model, placed, details, batch, torch.ones(len(motion), device=motion.device), policy)
    rows = []
    for i in range(len(motion)):
        row = motion_metrics(joints[i], truth[i])
        row.update({k: float(v[i]) for k, v in geometry.items()})
        other = motion_metrics(placed_joints[i], truth[i])
        other.update({k: float(v[i]) for k, v in placed_geometry.items()})
        row.update({'camera_translation_' + k: v for k, v in other.items()})
        rows.append(row)
    return joints, rows


def run(config_path, device, *, preflight=False, resume=False):
    spec = yaml.safe_load(config_path.read_text())
    if spec['version'] != 'ego-geometry-input-ablation-v1' or tuple(spec['conditions']) != CONDITIONS:
        raise ValueError('Unexpected geometry input-intervention protocol')
    for key in ('checkpoint', 'validation_manifest', 'reference_metrics', 'disabled_reference_metrics'):
        if sha256(ROOT / spec[key]) != spec[key + '_sha256']:
            raise ValueError(f'Pinned {key} changed')
    state = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    c = state['contract']['config']
    if c != load_config(ROOT / spec['training_config']) or state['step'] != spec['checkpoint_step']:
        raise ValueError('Selected checkpoint/configuration changed')
    for file, digest in state['contract']['source_sha256'].items():
        if sha256(ROOT / file) != digest:
            raise ValueError(f'Trained source changed: {file}')
    configure_runtime(c, device); deterministic(c['seed'], device)
    os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / c['prepared_data']['root'] / 'triton'))
    paths = load_paths(ROOT / 'configs/paths.local.yaml')
    model = build_model(c, paths)
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False).eval().to(device)
    data = GeometryEgoData(c, paths, 'validation')
    manifest = json.loads((ROOT / spec['validation_manifest']).read_text())
    cases = manifest['full']
    if manifest['test_used'] or source_cases(data) != cases or len({x['case_id'] for x in cases}) != len(cases):
        raise ValueError('Fixed validation membership/order changed')
    scenes = {case['case_id']: scene_identity(case, data.native.selection_bundle.metadata(
        data.native.records[case['member']], 'validation')) for case in cases if case['dataset'] != 'amass'}
    plan = donor_plan(cases, scenes, spec['donor_seed'])
    if preflight:
        cases = [next(case for case in cases if case['dataset'] == source) for source in c['data']['datasets']]
    out = ROOT / (spec['preflight_output'] if preflight else spec['output'])
    out.mkdir(parents=True, exist_ok=resume)
    source_roots = (ROOT, ROOT.parent / 'diffusion-motion-inbetweening')
    sources = {str(Path(m.__file__).resolve()): sha256(m.__file__) for m in list(sys.modules.values())
        if isinstance(getattr(m, '__file__', None), str) and m.__file__.endswith('.py')
        and Path(m.__file__).is_absolute() and any(Path(m.__file__).resolve().is_relative_to(p) for p in source_roots)}
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    protocol = dict(spec=spec, spec_sha256=sha256(config_path), source_sha256=sources, device=device,
        preflight=preflight, cases=cases, donor_plan=plan, scenes=scenes,
        scene_inputs=SCENE_INPUTS, physical_metrics_map='original recipient map in every condition',
        donor_policy='same dataset, different physical scene and source family; donor camera-start coordinates',
        amass_policy='no scene; after first-batch equivalence check, reuse unchanged prediction in all conditions',
        inference_only=True, test_used=False, formal_training_started=False)
    # JSON normalizes integer donor-plan keys and tuples for exact resume checks.
    protocol = json.loads(json.dumps(protocol, sort_keys=True))
    start = 0; rows = {key: [] for key in CONDITIONS}; size = c['validation']['eval_batch_size']
    if resume:
        if json.loads((out / 'PROTOCOL.json').read_text()) != protocol:
            raise ValueError('Evaluation resume protocol changed')
        start, rows, _ = recover_prefix(out, cases, size)
        for key in CONDITIONS:
            (out / f'{key}.jsonl').write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in rows[key]))
    else:
        atomic_json(out / 'PROTOCOL.json', protocol)
    audit_file = out / 'RUNTIME_AUDIT.json'
    audit = json.loads(audit_file.read_text()) if resume else dict(empty_disabled_max_abs_m=0., amass_condition_max_abs_m=0.,
        amass_equivalence_checked=False, reused_amass_windows=0, camera_body_inputs_unchanged=True,
        all_geometry_channels_intervened=True, recipient_map_used_for_all_physics=True)
    streams = {key: (out / f'{key}.jsonl').open('a') for key in CONDITIONS}
    from tqdm.auto import tqdm
    try:
        with torch.inference_mode():
            for offset in tqdm(range(start, len(cases), size), desc='Geometry scene interventions', unit='batch'):
                chunk = cases[offset:offset + size]
                pairs = [data.sample_case(case) for case in chunk]
                batch = collate([x[0] for x in pairs], device)
                public = payload(batch)
                donors = [data.sample_case(plan[case['case_id']])[0] if plan[case['case_id']] is not None else pairs[i][0]
                    for i, case in enumerate(chunk)]
                donor_batch = {key: torch.as_tensor(np.stack([x[key] for x in donors]), device=device) for key in SCENE_INPUTS}
                if any(k in public for k in ('motion', 'target_rest', 'target_native_betas', 'contact_weight')):
                    raise AssertionError('Private labels in inference payload')
                predictions = {}; joints = {}; details = None
                only_amass = all(case['dataset'] == 'amass' for case in chunk)
                reuse = only_amass and audit['amass_equivalence_checked']
                for condition in CONDITIONS:
                    inputs = condition_payload(public, condition, donor_batch)
                    if reuse and condition != 'matched':
                        prediction = predictions['matched']
                    else:
                        prediction = sample_native30(model, inputs, steps=c['validation']['ddim_steps'],
                            seed=c['validation']['seed'], use_scene=condition != 'disabled')
                    predictions[condition] = prediction
                    if details is None:
                        _, details = model(prediction, torch.zeros(len(chunk), dtype=torch.long, device=device),
                            public, use_scene=False, return_details=True)
                    if reuse and condition != 'matched':
                        predicted, scores = joints['matched'], scores_by_condition['matched']
                    else:
                        predicted, scores = metrics(model, prediction, details, batch, c['geometry'])
                    if condition == 'matched': scores_by_condition = {}
                    joints[condition] = predicted; scores_by_condition[condition] = scores
                    if condition == 'empty':
                        audit['empty_disabled_max_abs_m'] = max(audit['empty_disabled_max_abs_m'],
                            float((predicted - joints['disabled']).abs().max()))
                    for i, case in enumerate(chunk):
                        if case['dataset'] == 'amass':
                            error = float((predicted[i] - joints['matched'][i]).abs().max())
                            audit['amass_condition_max_abs_m'] = max(audit['amass_condition_max_abs_m'], error)
                            if error != 0: raise AssertionError('Scene intervention changed AMASS')
                        row = dict(case=case, identity=pairs[i][1], metrics=scores[i],
                            prediction_change_from_matched_mm=float((predicted[i] - joints['matched'][i]).norm(dim=-1).mean() * 1000))
                        if condition == 'mismatched':
                            donor = plan[case['case_id']]
                            row.update(donor=donor, recipient_scene=scenes.get(case['case_id']),
                                donor_scene=scenes.get(donor['case_id']) if donor else None)
                        rows[condition].append(row)
                        streams[condition].write(json.dumps(row, allow_nan=False) + '\n')
                    streams[condition].flush(); os.fsync(streams[condition].fileno())
                if only_amass: audit['amass_equivalence_checked'] = True
                if reuse: audit['reused_amass_windows'] += len(chunk)
                atomic_json(audit_file, audit)
                atomic_json(out / 'STATUS.json', dict(state='evaluating', completed_windows=min(offset + size, len(cases)),
                    windows=len(cases), updated_at_utc=now(), inference_only=True))
        result = summarize(rows, spec['bootstrap_samples'], spec['donor_seed'])
        differences = {}
        if not preflight:
            for condition, key in (('matched', 'reference_metrics'), ('disabled', 'disabled_reference_metrics')):
                reference = json.loads((ROOT / spec[key]).read_text())['dataset_macro_mean']
                actual = result['conditions'][condition]['dataset_macro_mean']
                differences[condition] = {k: actual[k] - v for k, v in reference.items()}
                if any(abs(v) > .001 for v in differences[condition].values()):
                    raise AssertionError(f'Fixed validation reference differs: {condition}: {differences[condition]}')
        if audit['empty_disabled_max_abs_m'] != 0: raise AssertionError('Empty-map fallback differs from disabled scene')
        result.update(state='passed' if preflight else 'complete', checkpoint_step=state['step'],
            checkpoint_sha256=spec['checkpoint_sha256'], windows=len(cases), preflight=preflight,
            reference_macro_delta=differences, audit=audit, inference_only=True, test_used=False,
            physical_metrics_map='original recipient map in every condition', completed_at_utc=now())
        atomic_json(out / 'RESULT.json', result)
        atomic_json(out / 'STATUS.json', dict(state=result['state'], completed_windows=len(cases), windows=len(cases), completed_at_utc=now()))
        print(json.dumps({key: value['dataset_macro_mean']['mpjpe_mm'] for key, value in result['conditions'].items()}), flush=True)
    except BaseException as error:
        atomic_json(out / 'FAILURE.json', dict(error=f'{type(error).__name__}: {error}', completed_at_utc=now()))
        raise
    finally:
        for stream in streams.values(): stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    ensure_bounded()
    spec = yaml.safe_load(args.config.read_text())
    output = ROOT / (spec['preflight_output'] if args.preflight else spec['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.config, args.device, preflight=args.preflight, resume=args.resume)


if __name__ == '__main__':
    main()
