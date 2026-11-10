"""Fixed-weight, validation-only comparison of temporal head anchoring.

Run under ``python -m scenemi resources run``. Only root translations change.
All settings are global; no per-case or per-dataset choice uses ground truth.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from scenemi.config import ROOT, load_paths
from scenemi.data.amass_prepare import atomic_json, sha256
from scenemi.data.ego import source_cases
from scenemi.data.scene_geometry_data import GeometryEgoData, collate, payload
from scenemi.evaluate import aggregate
from scenemi.execution import configure_runtime
from scenemi.geometry.head_anchor import camera_residual, correction_operator, translate_motion, translation_grid_metrics
from scenemi.geometry.scene_objectives import geometry_terms
from scenemi.scene_geometry_training import build_model, load_config
from scenemi.training import deterministic, ensure_bounded, now
from experiments.offline_camera_retrain_v1.native30_model import sample_native30
from experiments.offline_camera_retrain_v1.supervision import forward_kinematics

ARMS = ('base22k', 'scene24k')


def methods(spec):
    result = [dict(name=k, kind=k) for k in ('raw', 'hard', 'constant')]
    a = spec['scheme_a']
    for kind in a['kinds']:
        for window in a['windows']:
            for strength in a['strengths']:
                result.append(dict(name=f'a_{kind}_{window}_x{strength:g}', kind=kind,
                    window=window, strength=strength, polyorder=a['savgol_polyorder']))
    for acc, jerk in spec['scheme_b']['timescales_s']:
        result.append(dict(name=f'b_acc{acc:g}_jerk{jerk:g}', kind='quadratic',
            data_weight=spec['scheme_b']['data_weight'], acceleration_weight=acc**4, jerk_weight=jerk**6,
            acceleration_timescale_s=acc, jerk_timescale_s=jerk))
    if len({m['name'] for m in result}) != len(result):
        raise ValueError('Duplicate anchoring settings')
    return result


def aggregate_grid(rows, settings):
    return {arm: {m['name']: aggregate([dict(identity=x['identity'], metrics=x['arms'][arm][m['name']])
        for x in rows]) for m in settings} for arm in ARMS}


def select_policy(summary, settings, policy):
    """One base-selected policy is applied unchanged to both models."""
    base = summary['base22k']; raw = base['raw']; eligible = []
    for m in settings:
        name = m['name']; scores = base[name]
        if name in ('raw', 'hard'):
            continue
        macro_ok = scores['dataset_macro_mean']['jerk_m_s3'] <= raw['dataset_macro_mean']['jerk_m_s3'] * policy['macro_jerk_ratio_max']
        dataset_ok = all(v['jerk_m_s3'] <= raw['by_dataset'][d]['jerk_m_s3'] * policy['per_dataset_jerk_ratio_max']
            for d, v in scores['by_dataset'].items())
        if macro_ok and dataset_ok:
            eligible.append(name)
    score = lambda name: base[name]['dataset_macro_mean']['mpjpe_mm']
    selected = min(eligible, key=score) if eligible else 'raw'
    def category(prefix):
        candidates = [m['name'] for m in settings if m['name'].startswith(prefix)]
        feasible = [name for name in candidates if name in eligible]
        return dict(method=min(feasible or candidates, key=score), satisfies_jerk_guard=bool(feasible))
    return dict(selected=selected, best_a=category('a_'), best_b=category('b_'), eligible_methods=eligible,
        basis=policy, same_policy_for_both_models=True, independent_test=False)


def run(config_path, device, *, preflight=False, resume=False):
    spec = yaml.safe_load(config_path.read_text())
    if (spec['version'] != 'ego-head-anchor-evaluation-v1' or spec['frames'] != 128 or spec['fps'] != 30
            or spec['test_used'] or spec['input_condition'] != 'unchanged_clean_camera'):
        raise ValueError('Unexpected anchoring diagnostic protocol')
    for key in ('checkpoint', 'validation_manifest'):
        if sha256(ROOT / spec[key]) != spec[key + '_sha256']:
            raise ValueError(f'Pinned {key} changed')
    for arm, file in spec['references'].items():
        if sha256(ROOT / file) != spec['reference_sha256'][arm]:
            raise ValueError(f'Pinned reference changed: {arm}')
    state = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    c = state['contract']['config']
    if state['step'] != spec['checkpoint_step'] or c != load_config(ROOT / spec['training_config']):
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
    if manifest['test_used'] or source_cases(data) != cases:
        raise ValueError('Fixed validation membership/order changed')
    if preflight:
        cases = [next(x for x in cases if x['dataset'] == d) for d in c['data']['datasets']]
    settings = methods(spec)
    operators = torch.as_tensor(np.stack([correction_operator(m, spec['frames'], spec['fps']) for m in settings]), device=device)
    out = ROOT / spec['preflight_output' if preflight else 'output']
    out.mkdir(parents=True, exist_ok=resume)
    sources = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and name.endswith('.py') and Path(name).is_absolute():
            file = Path(name).resolve()
            if file.is_relative_to(ROOT) or file.is_relative_to(ROOT.parent / 'diffusion-motion-inbetweening'):
                sources[str(file)] = sha256(file)
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    protocol = json.loads(json.dumps(dict(spec=spec, spec_sha256=sha256(config_path), source_sha256=sources,
        settings=settings, cases=cases, device=device, preflight=preflight,
        temporal_extent='independent 128-frame window, offline/bidirectional; not full source recording',
        unchanged='model weights, camera observations, predicted shape, root rotation, local rotations',
        anchoring_inputs='predicted body, predicted mount and observed camera only; no GT',
        parameter_selection='global validation choice on base22k; same correction applied to scene24k',
        test_used=False, train_updates=0), sort_keys=True))
    if resume:
        if json.loads((out / 'PROTOCOL.json').read_text()) != protocol:
            raise ValueError('Evaluation resume protocol changed')
    else:
        atomic_json(out / 'PROTOCOL.json', protocol)
        atomic_json(out / 'STATUS.json', dict(state='grid', completed_windows=0, windows=len(cases)))
    cache = out / 'temporary_predictions'; cache.mkdir(exist_ok=True)
    grid_file = out / 'metrics.jsonl'; size = c['validation']['eval_batch_size']; rows = []
    if resume and grid_file.exists():
        for line in grid_file.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: break
        saved = json.loads((out / 'STATUS.json').read_text())
        count = min(saved['completed_windows'], len(rows))
        if count != len(cases): count = count // size * size
        rows = rows[:count]
        if [r['case'] for r in rows] != cases[:count]: raise ValueError('Saved case order changed')
        grid_file.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    audit = dict(rotations_unchanged=True, original_camera_unchanged=True, target_used_for_anchoring=False,
        max_pa_change_mm=0., max_root_relative_change_mm=0., original_model_reference_checked=False)
    with grid_file.open('a') as stream, torch.inference_mode():
        for first in tqdm(range(len(rows), len(cases), size), desc='Head anchoring grid', unit='batch'):
            chunk = cases[first:first + size]; pairs = [data.sample_case(x) for x in chunk]
            batch = collate([p[0] for p in pairs], device); public = payload(batch)
            if not torch.all(batch['camera_available'] == 1): raise ValueError('Clean complete camera stream required')
            base = sample_native30(model, public, steps=c['validation']['ddim_steps'], seed=c['validation']['seed'], use_scene=False)
            if all(x['dataset'] == 'amass' for x in chunk):
                scene = base
            else:
                scene = sample_native30(model, public, steps=c['validation']['ddim_steps'], seed=c['validation']['seed'], use_scene=True)
            _, details = model(base, torch.zeros(len(chunk), dtype=torch.long, device=device), public,
                use_scene=False, return_details=True)
            saved = {k: details[k].cpu() for k in ('rest', 'predicted_native_betas', 'predicted_template', 'head_camera_translation')}
            saved['camera'] = details['observations']['camera'].cpu()
            saved['predictions'] = dict(base22k=base.cpu(), scene24k=scene.cpu())
            saved['cases'] = chunk
            file = cache / f'{first:06d}.pt'; temporary = file.with_suffix('.tmp')
            torch.save(saved, temporary); temporary.replace(file)
            target = forward_kinematics(batch['motion'], batch['target_rest'])
            records = [dict(case=case, identity=pairs[i][1], arms={}) for i, case in enumerate(chunk)]
            for arm, motion in (('base22k', base), ('scene24k', scene)):
                points = forward_kinematics(motion, details['rest'])
                residual = camera_residual(motion, details['rest'], details['head_camera_translation'], details['observations']['camera'])
                values = translation_grid_metrics(points, target, residual, operators, fps=spec['fps'])
                names = list(values); matrix = torch.stack([values[k] for k in names], -1).permute(1, 0, 2).cpu().tolist()
                for i, scores in enumerate(matrix):
                    records[i]['arms'][arm] = {m['name']: dict(zip(names, v)) for m, v in zip(settings, scores)}
                for key, audit_key in (('pa_mpjpe_mm', 'max_pa_change_mm'), ('root_relative_mpjpe_mm', 'max_root_relative_change_mm')):
                    audit[audit_key] = max(audit[audit_key], float((values[key] - values[key][0:1]).abs().max()))
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + '\n'); rows.append(record)
            stream.flush(); os.fsync(stream.fileno())
            atomic_json(out / 'STATUS.json', dict(state='grid', completed_windows=len(rows), windows=len(cases), updated_at_utc=now()))
    summary = aggregate_grid(rows, settings)
    differences = {}
    if not preflight:
        for arm, file in spec['references'].items():
            reference = json.loads((ROOT / file).read_text())['dataset_macro_mean']
            differences[arm] = {k: v - reference[k] for k, v in summary[arm]['raw']['dataset_macro_mean'].items() if k in reference}
            if max(map(abs, differences[arm].values())) > .001:
                raise AssertionError(f'Original model metrics not reproduced: {arm}: {differences[arm]}')
        audit['original_model_reference_checked'] = True
    selection = select_policy(summary, settings, spec['selection'])
    atomic_json(out / 'GRID_RESULT.json', dict(arms=summary, selection=selection, reference_macro_delta=differences))
    selected = list(dict.fromkeys(['raw', 'hard', 'constant', selection['selected'], selection['best_a']['method'], selection['best_b']['method']]))
    indices = [next(i for i, m in enumerate(settings) if m['name'] == name) for name in selected]
    physical = []
    with (out / 'geometry.jsonl').open('w') as stream, torch.inference_mode():
        for first in tqdm(range(0, len(cases), size), desc='Anchoring physical checks', unit='batch'):
            chunk = cases[first:first + size]
            saved = torch.load(cache / f'{first:06d}.pt', map_location=device, weights_only=False)
            if saved['cases'] != chunk: raise ValueError('Temporary prediction identities changed')
            pairs = [data.sample_case(x) for x in chunk]; batch = collate([x[0] for x in pairs], device)
            details = {k: saved[k] for k in ('rest', 'predicted_native_betas', 'predicted_template', 'head_camera_translation')}
            records = [dict(case=case, identity=pairs[i][1], arms={arm: {} for arm in ARMS}) for i, case in enumerate(chunk)]
            for arm in ARMS:
                motion = saved['predictions'][arm]
                residual = camera_residual(motion, saved['rest'], saved['head_camera_translation'], saved['camera'])
                corrections = torch.einsum('stq,bqc->sbtc', operators[indices], residual)
                for name, correction in zip(selected, corrections):
                    shifted = translate_motion(motion, details['rest'], correction)
                    _, scores = geometry_terms(model, shifted, details, batch, torch.ones(len(chunk), device=device), c['geometry'])
                    names = list(scores); matrix = torch.stack([scores[k] for k in names], -1).cpu().tolist()
                    for i, v in enumerate(matrix): records[i]['arms'][arm][name] = dict(zip(names, v))
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + '\n'); physical.append(record)
            stream.flush()
            atomic_json(out / 'STATUS.json', dict(state='geometry', completed_windows=len(cases), geometry_completed_windows=len(physical), windows=len(cases), updated_at_utc=now()))
    physics = {arm: {name: aggregate([dict(identity=x['identity'], metrics=dict(
        mpjpe_mm=rows[i]['arms'][arm][name]['mpjpe_mm'], **x['arms'][arm][name])) for i, x in enumerate(physical)])
        for name in selected} for arm in ARMS}
    if not preflight:
        for arm, file in spec['references'].items():
            reference = json.loads((ROOT / file).read_text())['dataset_macro_mean']
            observed = physics[arm]['raw']['dataset_macro_mean']
            delta = {k: v - reference[k] for k, v in observed.items()}
            if max(map(abs, delta.values())) > .001:
                raise AssertionError(f'Original physical metrics not reproduced: {arm}: {delta}')
        audit['original_physical_reference_checked'] = True
    result = dict(state='passed' if preflight else 'complete', windows=len(cases), settings=settings, arms=summary,
        selection=selection, geometry=physics, reference_macro_delta=differences, audit=audit,
        preflight=preflight, checkpoint_step=state['step'], test_used=False, train_updates=0,
        temporal_extent=protocol['temporal_extent'], completed_at_utc=now())
    atomic_json(out / 'RESULT.json', result)
    inventory = [dict(path=str(file.relative_to(out)), bytes=file.stat().st_size, sha256=sha256(file)) for file in sorted(cache.glob('*.pt'))]
    atomic_json(out / 'CACHE_CLEANUP.json', dict(files=inventory, reason='Temporary predictions consumed by completed grid and physical checks; retain all per-case metrics and exact reproduction protocol'))
    for file in inventory: (out / file['path']).unlink()
    cache.rmdir()
    atomic_json(out / 'STATUS.json', dict(state=result['state'], completed_windows=len(cases), windows=len(cases), completed_at_utc=now()))
    print(json.dumps(dict(state=result['state'], windows=len(cases), selection=selection)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv); ensure_bounded()
    spec = yaml.safe_load(args.config.read_text())
    out = ROOT / spec['preflight_output' if args.preflight else 'output']; out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.config, args.device, preflight=args.preflight, resume=args.resume)
        except BaseException as error:
            if out.exists(): atomic_json(out / 'FAILURE.json', dict(error=f'{type(error).__name__}: {error}', at_utc=now()))
            raise


if __name__ == '__main__':
    main()
