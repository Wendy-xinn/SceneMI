"""Resumable, paired official-VAL review of frozen EXO refinement candidates.

Selection is identity-only. A scene switch, model, and input regime are crossed
on identical observations, reference, and DDIM seeds. No optimization or TEST.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import copy
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from . import adapt, reference_temporal
from .dependencies import sha256

PROTOCOL = 'frozen-reference-refinement-paired-VAL-v1'
BASE = adapt.BASE
DEFAULT_SOURCE = BASE / 'runs/native30_exo_natural10k_v1'
DEFAULT_OUT = BASE / 'runs/native30_exo_refinement150_v1/paired_validation'
METRICS = ('hidden_mpjpe_mm', 'visible_mpjpe_mm', 'mpjpe_mm',
           'foot_joint_speed_gt_low_m_s', 'mpjve_mm_s', 'acceleration_error_mm_s2')
read, write = adapt.read, adapt.write


def status(out, state, **values):
    write(Path(out) / 'STATUS.json', dict(protocol=PROTOCOL, status=state,
        checked_at=adapt.now(), pid=os.getpid(), optimizer_updates=0,
        TEST_used=False, automatic_checkpoint_promotion=False, **values))


def select_entries(manifest):
    """All registered RICH, first four distinct recordings per other corpus."""
    selected = []
    for dataset in ('rich', 'trumans', 'egobody'):
        used = set()
        for index, row in enumerate(manifest['fixed_exo']):
            if row['dataset'] != dataset:
                continue
            if dataset != 'rich' and row['recording'] in used:
                continue
            selected.append(dict(row, historical_case=index, case=len(selected),
                                 review_panel='primary'))
            used.add(row['recording'])
            if dataset != 'rich' and len(used) == 4:
                break
    counts = Counter(r['dataset'] for r in selected)
    if counts != dict(rich=12, trumans=4, egobody=4):
        raise ValueError('Expected registered RICH12 + TRUMANS4 + EgoBody4')
    rich = [r for r in selected if r['dataset'] == 'rich']
    selected = rich[:4] + [r for r in selected if r['dataset'] != 'rich'] + rich[4:]
    selected = [dict(r, case=i, review_panel='primary' if i < 12 else 'rich_extension')
                for i, r in enumerate(selected)]
    ego = []
    for dataset in ('rich', 'trumans', 'egobody'):
        row = next(r for r in manifest['ego_retention'] if r['dataset'] == dataset)
        ego.append(dict(row, case=100 + len(ego), review_panel='ego_retention'))
    return selected, ego


def prepare(out, source=DEFAULT_SOURCE, trial=None):
    out, source = Path(out).resolve(), Path(source).resolve()
    if (out / 'PLAN.json').exists():
        raise ValueError('Existing paired plan is immutable')
    manifest = read(source / 'prepared/panels.json')
    entries, ego = select_entries(manifest)
    best = read(source / 'training/best.json')
    if best['stage_step'] != 10000:
        raise ValueError('The authorized parent must be EXO10k')
    checkpoint = Path(best['source_checkpoint'])
    if sha256(checkpoint) != best['checkpoint_sha256']:
        raise ValueError('Parent milestone changed')
    config = read(source / 'config.json')
    effective = read(source / 'prepared/effective_config.json')
    sources = dict(effective['code_sha256'])
    for filename in ('refinement_evaluation.py', 'refinement_guard.py', 'scene_comparison.py',
                     'reference_review.py', 'validation_review.py', 'adapter_evaluation.py'):
        file = Path(__file__).with_name(filename)
        sources[str(file.resolve())] = sha256(file)
    fixed = source / 'fixed_validation/step_010000'
    files = [source / 'config.json', source / 'prepared/panels.json',
             fixed / 'validation.jsonl', fixed / 'validation_latest.npz']
    declarations = dict(base10k=dict(path=str(checkpoint.resolve()),
        loader='exo_longrun', expected_step=10000, expected_sha256=best['checkpoint_sha256']))
    if trial is not None:
        trial = Path(trial).resolve()
        for arm in ('control', 'nondegradation'):
            declarations[arm] = dict(path=str(trial / arm / 'last.pt'), loader='refinement_trial',
                                     expected_step=150, expected_sha256=None)
    plan = dict(protocol=PROTOCOL, created_at=adapt.now(), config=config,
        source_run=str(source), entries=entries, ego_entries=ego, models=declarations,
        source_sha256=sources, source_files_sha256={str(f): sha256(f) for f in files},
        historical_arrays=str(fixed / 'validation_latest.npz'),
        historical_rows=str(fixed / 'validation.jsonl'), regimes=['clean', 'noisy'],
        scene_variants=['scene_on', 'scene_off'], reference_policy=reference_temporal.POLICY,
        analytical_physics_enabled=False, frames=192, fps=30, ddim_steps=20,
        planned_predictions_per_model=86, TEST_used=False, optimizer_updates=0,
        selection='All12 RICH fixed identities, first4 distinct TRUMANS/EgoBody recordings in sealed fixed ordering; no error/GT selection. One predeclared EGO anchor per corpus.',
        aggregation='Equal window within dataset, report each regime and scene separately; macro three datasets equally weighted. Window P90 is not joint-sample P90.',
        acceptance='Compare candidate vs base10k and paired control; hidden mean/P90 and reference-worse>=50mm, visible, stance-foot speed and acceleration must be reviewed; no automatic promotion')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'PLAN.json', plan)
    write(out / 'MODELS.json', dict(base10k=declarations['base10k']))
    status(out, 'prepared', declared_models=list(declarations), planned_predictions_per_model=86)
    return plan


def verify(out):
    plan = read(Path(out) / 'PLAN.json')
    if plan['protocol'] != PROTOCOL or plan['TEST_used'] or plan['optimizer_updates']:
        raise ValueError('Only zero-update official VAL allowed')
    for path, digest in {**plan['source_sha256'], **plan['source_files_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Sealed paired source changed: ' + path)
    return plan


def register(out, label):
    out = Path(out)
    plan = verify(out)
    declaration = plan['models'][label]
    path = Path(declaration['path'])
    if not path.is_file():
        raise ValueError('Declared candidate has not been saved')
    if declaration['loader'] == 'refinement_trial':
        from .refinement_trial import load_model
    else:
        from .exo_longrun import load_inference_model as load_model
    model, identity = load_model(path, device='cpu', expected_sha256=declaration['expected_sha256'])
    if identity['stage_step'] != declaration['expected_step'] or identity['config'] != plan['config']:
        raise ValueError('Candidate step/config mismatch')
    if label != 'base10k' and identity.get('arm') != label:
        raise ValueError('Candidate arm mismatch')
    del model
    gc.collect()
    models = read(out / 'MODELS.json')
    registered = dict(declaration, expected_sha256=identity['checkpoint_sha256'], registered_at=adapt.now())
    if label in models and models[label]['expected_sha256'] != registered['expected_sha256']:
        raise ValueError('Cannot replace an already registered candidate')
    models[label] = registered
    write(out / 'MODELS.json', models)
    return registered


def rows_for(out):
    path = Path(out) / 'rows.jsonl'
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b'\n'):
        raise ValueError('Incomplete durable JSONL tail; inspect before resuming')
    rows = [json.loads(line) for line in content.splitlines()]
    keys = [(r['model'], r['case'], r['regime'], r['scene_variant']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate paired prediction')
    return rows


def append_row(out, row):
    with (Path(out) / 'rows.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def numpy_core(arrays):
    """Independent NumPy/SciPy FK: positions, velocity, acceleration and feet."""
    import numpy as np
    from .refinement_guard import forward_kinematics_numpy
    p, g = (forward_kinematics_numpy(arrays[k], arrays['rest']) for k in ('prediction', 'truth'))
    error = np.linalg.norm(p - g, axis=-1) * 1000
    visible = arrays['observation_meta'][..., 0] > 0
    vp, vg = np.diff(p, axis=0) * 30, np.diff(g, axis=0) * 30
    low = np.linalg.norm(vg[:, (10, 11)], axis=-1) < .2
    return dict(hidden_mpjpe_mm=float(error[~visible].mean()) if (~visible).any() else None,
        visible_mpjpe_mm=float(error[visible].mean()) if visible.any() else None,
        mpjpe_mm=float(error.mean()), mpjve_mm_s=float(np.linalg.norm(vp - vg, axis=-1).mean() * 1000),
        acceleration_error_mm_s2=float(np.linalg.norm(np.diff(vp, axis=0) - np.diff(vg, axis=0), axis=-1).mean() * 30000),
        foot_joint_speed_gt_low_m_s=float(np.linalg.norm(vp[:, (10, 11)], axis=-1)[low].mean()) if low.any() else None)


def source_inputs(data, builder, entry, regime, config, device):
    import numpy as np
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from .validation_review import inference_inputs, assert_input_device
    sample, identity = data.sample(entry['group'], entry['mode'], member=entry['member'],
                                   start=entry['start'], yaw=0)
    if identity['source_start_30fps'] != entry['source_first'] or identity['split'] != 'validation':
        raise ValueError('Only exact registered official VAL identity allowed')
    sensors = {k: v for k, v in sample.items() if k not in ('motion', 'joints', 'contact_target', 'contact_valid') and not k.startswith('gt_')}
    selected = builder if regime == 'clean' else Native30ObservationBuilder(builder.skin_cache,
                builder.mounts, dict(builder.noise_config, clean_probability=0.))
    observed, scene, asset = selected.build(data, sensors, identity, entry['mode'], entry['joints'],
        noise_rng=None if regime == 'clean' else np.random.default_rng(entry['case_seed']))
    batch = collate_observations([(observed, scene, asset, entry['mode'])], device=device)
    known = inference_inputs(config, batch)
    assert_input_device(known, device)
    return sample, identity, observed, batch, known


def historical_rows(plan):
    all_rows = []
    for line in Path(plan['historical_rows']).read_text().splitlines():
        all_rows.extend(json.loads(line)['cases'])
    return {(r['case'], r['regime']): r for r in all_rows if not r.get('scene_variant')}


def evaluate(out, label, *, device='cuda', deadline='21:20', max_predictions=0):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
    from . import validation_review as common
    from .reference_review import metrics
    from .scene_comparison import scene_off_guard
    out = Path(out)
    plan = verify(out)
    declarations = read(out / 'MODELS.json')
    if label not in declarations:
        register(out, label)
        declarations = read(out / 'MODELS.json')
    declaration = declarations[label]
    if sha256(declaration['path']) != declaration['expected_sha256']:
        raise ValueError('Registered model bytes changed')
    rows = rows_for(out)
    done = {(r['model'], r['case'], r['regime'], r['scene_variant']) for r in rows}
    status(out, 'waiting_for_gpu' if device == 'cuda' else 'loading_CPU', model=label,
           completed_predictions=len(rows))
    lock = common.gpu_lock() if device == 'cuda' else nullcontext()
    with lock:
        verify(out)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        if device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(.7)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if declaration['loader'] == 'refinement_trial':
            from .refinement_trial import load_model
        else:
            from .exo_longrun import load_inference_model as load_model
        model, metadata = load_model(declaration['path'], device=device,
            expected_sha256=declaration['expected_sha256'])
        if metadata['config'] != plan['config'] or model.guidance_enabled:
            raise ValueError('Model config or analytical physics differs')
        config = copy.deepcopy(plan['config'])
        data = common.validation_data(config)
        builder = common.observation_builder(config, device=device)
        prior = historical_rows(plan)
        artifacts = out / 'motions' / label
        artifacts.mkdir(parents=True, exist_ok=True)
        newly_done = 0
        for entry in plan['entries'] + plan['ego_entries']:
            variants = ['scene_on'] if entry['mode'] == 'ego' else plan['scene_variants']
            for regime in plan['regimes']:
                if all((label, entry['case'], regime, v) in done for v in variants):
                    continue
                with reference_temporal.installed():
                    sample, identity, observed, batch, known = source_inputs(data, builder, entry, regime, config, device)
                input_sha = common.tensor_fingerprint(known)
                other = [r for r in rows if r['case'] == entry['case'] and r['regime'] == regime]
                if any(r['input_sha256'] != input_sha for r in other):
                    raise ValueError('Paired/resumed observation-derived input differs')
                for variant in variants:
                    key = label, entry['case'], regime, variant
                    if key in done:
                        continue
                    if dt.datetime.now().strftime('%H:%M') >= deadline:
                        status(out, 'deadline_stop', model=label, completed_predictions=len(rows), deadline=deadline)
                        summarize(out)
                        return
                    status(out, 'sampling', model=label, case=entry['case'], dataset=entry['dataset'],
                           regime=regime, scene_variant=variant, completed_predictions=len(rows))
                    start = time.monotonic()
                    guard = scene_off_guard(model) if variant == 'scene_off' else nullcontext()
                    with guard as counts, torch.no_grad():
                        prediction = sample_motion(model, known, config, steps=20,
                            seed=entry['case_seed'], use_scene=variant == 'scene_on')
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    if not bool(torch.isfinite(prediction).all()):
                        raise FloatingPointError('Nonfinite frozen prediction')
                    arrays = dict(prediction=prediction[0].cpu().numpy(), truth=sample['motion'], rest=sample['rest'],
                        observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                        reference_motion=known['reference_motion'][0].cpu().numpy(),
                        identity_json=np.array(json.dumps(identity)), case_seed=np.array(entry['case_seed']),
                        checkpoint_sha256=np.array(metadata['checkpoint_sha256']), input_sha256=np.array(input_sha),
                        use_scene=np.array(variant == 'scene_on'))
                    filename = artifacts / f"case{entry['case']:03d}_{regime}_{variant}.npz"
                    temporary = filename.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(filename)
                    values, denominators = metrics(arrays)
                    reference_values, _ = metrics(dict(arrays, prediction=arrays['reference_motion']))
                    historical = None
                    if label == 'base10k' and variant == 'scene_on' and entry['mode'] == 'exo':
                        metric_batch = dict(batch, motion=torch.from_numpy(sample['motion'])[None].to(device))
                        native = raw_metrics(prediction, metric_batch, batch['observation_meta'][..., 0] > 0)
                        before = prior[(entry['historical_case'], regime)]
                        ref_native = raw_metrics(known['reference_motion'], metric_batch,
                                                batch['observation_meta'][..., 0] > 0)
                        historical = dict(raw_metrics_max_difference=max(abs(native[k]['value'] - before['metrics'][k]['value'])
                            for k in native if native[k]['value'] is not None and before['metrics'][k]['value'] is not None),
                            reference_metric_max_difference=max(abs(ref_native[k]['value'] - before['reference_metrics'][k]['value'])
                            for k in native if ref_native[k]['value'] is not None and before['reference_metrics'][k]['value'] is not None))
                    row = dict(model=label, case=entry['case'], dataset=entry['dataset'], mode=entry['mode'],
                        regime=regime, scene_variant=variant, review_panel=entry['review_panel'],
                        sequence_id=entry['sequence_id'], recording=entry['recording'], start=entry['start'],
                        source_first=entry['source_first'], seed=entry['case_seed'], input_sha256=input_sha,
                        checkpoint_sha256=metadata['checkpoint_sha256'], checkpoint_step=metadata['stage_step'],
                        artifact=str(filename.resolve()), artifact_sha256=sha256(filename),
                        sampling_s=time.monotonic() - start, metrics=values, reference_metrics=reference_values,
                        counts=denominators, scene_off_guard_counts=counts, historical_crosscheck=historical)
                    append_row(out, row)
                    rows.append(row)
                    done.add(key)
                    newly_done += 1
                    if newly_done % 10 == 0:
                        print(json.dumps(dict(model=label, completed=newly_done, total_rows=len(rows),
                            last_sampling_s=row['sampling_s']), ensure_ascii=False), flush=True)
                    if max_predictions and newly_done >= max_predictions:
                        summarize(out)
                        status(out, 'bounded_stop', model=label, completed_predictions=len(rows))
                        return
                del known, batch, sample, observed
                gc.collect()
        verify(out)
        if sha256(declaration['path']) != declaration['expected_sha256']:
            raise ValueError('Weights changed during frozen review')
    summarize(out)
    status(out, 'model_completed', model=label, completed_predictions=len(rows))


def summarize(out):
    import numpy as np
    out = Path(out)
    plan = read(out / 'PLAN.json')
    rows = rows_for(out)
    groups = {}
    for model in sorted({r['model'] for r in rows}):
        for variant in plan['scene_variants']:
            for regime in plan['regimes']:
                for dataset in ('rich', 'trumans', 'egobody'):
                    chosen = [r for r in rows if r['model'] == model and r['scene_variant'] == variant
                              and r['regime'] == regime and r['dataset'] == dataset and r['mode'] == 'exo']
                    if not chosen:
                        continue
                    values = {}
                    for metric in METRICS:
                        valid = [r['metrics'][metric] for r in chosen if r['metrics'].get(metric) is not None]
                        refs = [r['reference_metrics'][metric] for r in chosen if r['reference_metrics'].get(metric) is not None]
                        values[metric] = dict(mean=float(np.mean(valid)) if valid else None,
                            window_p90=float(np.percentile(valid, 90)) if valid else None,
                            reference_mean=float(np.mean(refs)) if refs else None)
                    valid = [r for r in chosen if r['metrics'].get('hidden_mpjpe_mm') is not None]
                    groups['/'.join((model, variant, regime, dataset))] = dict(windows=len(chosen), metrics=values,
                        hidden_worse_than_reference_50mm=sum(r['metrics']['hidden_mpjpe_mm'] >=
                            r['reference_metrics']['hidden_mpjpe_mm'] + 50 for r in valid))
    summary = dict(protocol=PROTOCOL, checked_at=adapt.now(), rows=len(rows), groups=groups,
        rows_sha256=sha256(out / 'rows.jsonl') if rows else None, TEST_used=False,
        partial=any(sum(r['model'] == m for r in rows) != 86 for m in plan['models']),
        automatic_checkpoint_promotion=False, window_p90_definition='90th percentile of per-window mean errors')
    write(out / 'SUMMARY.json', summary)
    text = ['冻结 EXO 配对验证；仅官方 VAL，不训练、不读取 TEST。', '',
        '每模型20个 EXO 窗口 × clean/noisy × 有/无场景，另3个 EGO 保持锚点 × clean/noisy。',
        '以下为每窗口等权均值；P90是窗口均值的P90，不能作为逐关节P90。未完成组保留实际窗口数。', '',
        '| 模型/场景/输入/数据 | 窗口 | 遮挡均值 mm | 遮挡 P90 mm | 可见 mm | 支撑期足速 m/s | 加速度 mm/s² | 比参考差≥50mm窗 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.3f}'
    for key, group in groups.items():
        m = group['metrics']
        text.append('| ' + ' | '.join([key, str(group['windows']), fmt(m['hidden_mpjpe_mm']['mean']),
            fmt(m['hidden_mpjpe_mm']['window_p90']), fmt(m['visible_mpjpe_mm']['mean']),
            fmt(m['foot_joint_speed_gt_low_m_s']['mean']), fmt(m['acceleration_error_mm_s2']['mean']),
            str(group['hidden_worse_than_reference_50mm'])]) + ' |')
    text += ['', '场景OFF同时关闭学习场景编码、局部查询与反馈；本版解析物理引导始终关闭。',
             '原模型重跑与历史指标的差异逐行保存；CPU_RECHECK通过后才确认指标与输入一致性。']
    (out / 'REPORT_zh.md').write_text('\n'.join(text) + '\n')
    return summary


def recheck(out):
    import numpy as np
    import torch
    out = Path(out)
    verify(out)
    if torch.cuda.is_initialized():
        raise ValueError('Independent CPU review cannot initialize CUDA')
    rows = rows_for(out)
    pairs = {}
    maximum = dict.fromkeys(METRICS, 0.)
    ego = []
    for row in rows:
        file = Path(row['artifact'])
        if sha256(file) != row['artifact_sha256']:
            raise ValueError('Saved prediction changed')
        with np.load(file, allow_pickle=False) as arrays:
            actual = numpy_core(arrays)
            for metric, value in actual.items():
                expected = row['metrics'][metric]
                if (value is None) != (expected is None):
                    raise ValueError('Independent metric availability differs')
                if value is not None:
                    maximum[metric] = max(maximum[metric], abs(value - expected))
            key = row['case'], row['regime']
            digest = hashlib.sha256()
            for field in ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion'):
                digest.update(np.ascontiguousarray(arrays[field]).tobytes())
            fingerprint = digest.hexdigest()
            if key in pairs and pairs[key] != fingerprint:
                raise ValueError('Paired input arrays changed across models/scenes')
            pairs[key] = fingerprint
            if row['mode'] == 'ego' and row['model'] != 'base10k':
                base = next((r for r in rows if r['model'] == 'base10k' and r['case'] == row['case']
                              and r['regime'] == row['regime']), None)
                if base:
                    with np.load(base['artifact'], allow_pickle=False) as original:
                        exact = np.array_equal(arrays['prediction'], original['prediction'])
                    ego.append(dict(model=row['model'], case=row['case'], regime=row['regime'], exact=exact))
    limits = dict(hidden_mpjpe_mm=.02, visible_mpjpe_mm=.02, mpjpe_mm=.02,
        mpjve_mm_s=.05, acceleration_error_mm_s2=2., foot_joint_speed_gt_low_m_s=.0001)
    passed = all(maximum[k] <= limits[k] for k in maximum) and all(item['exact'] for item in ego)
    report = dict(status='independent_CPU_passed' if passed else 'independent_CPU_FAILED',
        checked_at=adapt.now(), rows=len(rows), independent_numpy_FK_max_difference=maximum,
        tolerances=limits, paired_arrays_exact=True, EGO_retention=ego,
        rows_sha256=sha256(out / 'rows.jsonl'), CUDA_initialized=False, TEST_used=False)
    write(out / 'CPU_RECHECK.json', report)
    summarize(out)
    if not passed:
        raise ValueError('Independent metric/EGO retention review failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'register', 'evaluate', 'recheck', 'report'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--trial', type=Path, default=BASE / 'runs/native30_exo_refinement150_v1')
    parser.add_argument('--model', choices=('base10k', 'control', 'nondegradation'), default='base10k')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--deadline', default='21:20')
    parser.add_argument('--max-predictions', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.source, args.trial)
    elif args.command == 'register':
        register(args.out, args.model)
    elif args.command == 'evaluate':
        evaluate(args.out, args.model, device=args.device, deadline=args.deadline,
                 max_predictions=args.max_predictions)
    elif args.command == 'recheck':
        recheck(args.out)
    else:
        summarize(args.out)


if __name__ == '__main__':
    main()
