"""One immutable selected model: full official VAL and independent RICH TEST.

``entry`` indexes identities only and can run before model selection. ``freeze``
requires an explicitly frozen, VAL-selected model manifest. Only ``evaluate``
samples motions; TEST is never admitted for checkpoint ranking or tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from .dependencies import dependency_closure, sha256

BASE = Path(__file__).resolve().parents[1]
PROTOCOL = 'selected-native30-final-metrics-v1'
KINDS = ('longrun', 'refinement')
SCENES = ('scene_on', 'scene_off')
DEFAULT_TEST_PLAN = BASE / 'runs/rich_test_ego41k/PLAN.json'
DEFAULT_CONFIG = BASE / 'runs/native30_exo_natural10k_v1/config.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def status(out, state, **values):
    write(Path(out) / 'STATUS.json', dict(protocol=PROTOCOL, status=state, checked_at=now(),
        pid=os.getpid(), optimizer_updates=0, automatic_model_selection=False, **values))


def validate_selection(value):
    """Validate an explicit decision, never infer a winner from TEST metrics."""
    from .reference_temporal import POLICY
    require(value.get('frozen') is True and value.get('selected_on') == 'validation'
            and value.get('TEST_used_for_selection') is False,
            'Require frozen=true, selected_on=validation and TEST_used_for_selection=false')
    require(value.get('kind') in KINDS, 'Selected model kind must be longrun/refinement')
    for key in ('checkpoint_sha256', 'config_sha256'):
        require(isinstance(value.get(key), str) and len(value[key]) == 64
                and all(c in '0123456789abcdef' for c in value[key]), 'Explicit SHA256 required: ' + key)
    for key in ('checkpoint', 'config'):
        require(isinstance(value.get(key), str) and Path(value[key]).is_absolute(),
                'Selected file paths must be absolute: ' + key)
    require(value.get('reference_policy') == POLICY and value.get('analytical_physics_enabled') is False,
            'Require registered temporal EXO reference policy and analytical physics OFF')
    require(type(value.get('expected_step')) is int and value['expected_step'] > 0, 'Require explicit checkpoint step')
    require(value.get('output_policy', 'raw_model') == 'raw_model',
            'This evaluator accepts raw selected model output, not an undeclared output projection')
    return value


def _seed(config, row, mode, start):
    key = json.dumps([row['group'], mode, row['sequence_id'], int(row['source_first']) + int(start)],
                     separators=(',', ':'))
    return (config['validation']['seed'] + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')) % 2**32


def entry(out, selection, *, split, config_path=DEFAULT_CONFIG, test_plan=DEFAULT_TEST_PLAN):
    """Prepare legal source identities; no checkpoint, prediction or metric is read."""
    import numpy as np
    out, config_path = Path(out).resolve(), Path(config_path).resolve()
    require(split in ('rich-test', 'full-validation'), 'Unknown evaluation split')
    require(not (out / 'ENTRY.json').exists() and not (out / 'PLAN.json').exists(), 'Existing entry is immutable')
    config = read(config_path)
    require((config['frames'], config['fps'], config['window_stride']) == (192, 30, 64), 'Require native192/30Hz/S64')
    sources, cases = {str(config_path): sha256(config_path)}, []
    if split == 'rich-test':
        test_plan = Path(test_plan).resolve()
        source = read(test_plan)  # Input identities only, not its results or old model.
        require(source['status'] == 'prepared' and source['unique_windows'] == 335,
                'Require all-subject prepared RICH TEST index')
        require((source['frames'], source['fps'], source['stride']) == (192, 30, 64), 'TEST source clock differs')
        manifest = Path(source['independent_test_manifest']).resolve()
        require(sha256(manifest) == source['independent_manifest_sha256'], 'TEST input manifest changed')
        index = test_plan.parent / 'test_ego_windows.npy'
        require(sha256(index) == source['window_index_sha256'], 'TEST EGO window index changed')
        array = np.load(index, allow_pickle=False)
        require(array.shape == (335, 2) and array.dtype == np.int64, 'Invalid TEST EGO index')
        for old in source['cases']:
            row = dict(old, group='rich', dataset='rich', mode='ego', joints=[15])
            require(np.array_equal(array[row['case']], [row['member'], row['start']]), 'TEST identity/index differs')
            meta = read(Path(row['path']) / 'metadata.json')
            require(meta['split'] == 'test' and meta.get('allowed_modes') == ['ego'],
                    'Do not invent EXO TEST observations')
            require(sha256(Path(row['path']) / 'metadata.json') == row['metadata_sha256'], 'TEST metadata changed')
            cases.append(row)
        sources.update({str(test_plan): sha256(test_plan), str(index): sha256(index)})
        window_index = None
    else:
        from .validation_cohort import common_members
        manifest, window_index = ((BASE / config[key]).resolve() for key in ('train_manifest', 'window_index'))
        source = read(window_index)
        require((source['frames'], source['fps'], source['window_stride']) == (192, 30, 64),
                'VAL legal index clock differs')
        require(source['source_manifest_sha256'] == sha256(manifest), 'VAL input manifest differs from index')
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        require(source['bundle_paths'] == [r['path'] for r in rows], 'VAL manifest order differs')
        members, validity = common_members(rows, 192, 64)
        sources.update(validity)
        for mode in ('ego', 'exo'):
            info = source['files']['validation/' + mode]
            index = window_index.parent / info['file']
            require(sha256(index) == info['sha256'], 'VAL legal index changed')
            array = np.load(index, allow_pickle=False)
            require(array.dtype == np.int64 and array.shape == (info['windows'], 2), 'Invalid legal VAL index')
            for ordinal, first in array:
                ordinal, first = int(ordinal), int(first)
                meta = rows[ordinal]
                require(meta['split'] == 'validation' and not meta['diagnostic_only']
                        and mode in meta['allowed_modes'], 'Nonofficial VAL observer or source')
                path = Path(meta['path'])
                cases.append(dict(case=len(cases), path=str(path), mode=mode, dataset=meta['dataset'],
                    group=meta['group'], joints=[15] if mode == 'ego' else None,
                    member=members[ordinal], recording=meta.get('recording') or meta['sequence_id'],
                    sequence_id=meta['sequence_id'], start=first, source_first=int(meta['source_first']) + first,
                    case_seed=_seed(config, meta, mode, first), metadata_sha256=sha256(path / 'metadata.json')))
            sources[str(index)] = sha256(index)
        sources[str(window_index)] = sha256(window_index)
    sources[str(manifest)] = sha256(manifest)
    require(len({(r['mode'], r['path'], r['start']) for r in cases}) == len(cases), 'Duplicate input windows')
    coverage = Counter((r['dataset'], r['mode']) for r in cases)
    result = dict(protocol=PROTOCOL, state='identities_only_not_model_frozen', created_at=now(), split=split,
        selection_manifest=str(Path(selection).resolve()), frames=192, fps=30, stride=64, ddim_steps=20,
        manifest=str(manifest), window_index=str(window_index) if window_index else None,
        cases=cases, coverage={'/'.join(k): v for k, v in sorted(coverage.items())},
        regimes=['clean', 'noisy'], scene_variants=list(SCENES), unique_mode_windows=len(cases),
        planned_predictions=len(cases) * 4, input_source_sha256=sources,
        prediction_outputs_read=False, model_loaded=False, optimizer_updates=0,
        disclosure='Native192/30Hz/S64,22 common joints,known body profile,synthetic/noisy sensor observations,observed static recording union; not a matched AMASS-only paper protocol.')
    write(out / 'ENTRY.json', result)
    status(out, 'identities_ready_selection_pending', split=split, planned_predictions=result['planned_predictions'])
    return result


def load_selected(selection, device='cpu'):
    validate_selection(selection)
    require(sha256(selection['checkpoint']) == selection['checkpoint_sha256'], 'Selected checkpoint bytes changed')
    require(sha256(selection['config']) == selection['config_sha256'], 'Selected config bytes changed')
    if selection['kind'] == 'longrun':
        from .exo_longrun import load_inference_model as loader
    else:
        from .refinement_trial import load_model as loader
    model, identity = loader(selection['checkpoint'], device=device,
                             expected_sha256=selection['checkpoint_sha256'])
    require(identity['stage_step'] == selection['expected_step'] and identity['config'] == read(selection['config']),
            'Selected checkpoint step or embedded config differs')
    require(identity['guidance_enabled'] is False and model.guidance_enabled is False
            and identity['config']['physical_guidance']['enabled'] is False,
            'Analytical physics must remain OFF')
    return model, identity


def freeze(out):
    import torch
    from .reference_temporal import POLICY
    out = Path(out).resolve()
    require(not (out / 'PLAN.json').exists(), 'Existing frozen metric plan is immutable')
    original = read(out / 'ENTRY.json')
    selection_path = Path(original['selection_manifest'])
    selection = validate_selection(read(selection_path))
    for path, expected in original['input_source_sha256'].items():
        require(sha256(path) == expected, 'Evaluation input changed: ' + path)
    model, identity = load_selected(selection)
    require(not torch.cuda.is_initialized(), 'CPU freezing must not initialize CUDA')
    config = identity['config']
    require((config['frames'], config['fps'], config['window_stride'], config['sampling']['ddim_steps'])
            == (192, 30, 64, 20), 'Selected inference clock differs')
    if original['split'] == 'full-validation':
        require((BASE / config['train_manifest']).resolve() == Path(original['manifest'])
                and (BASE / config['window_index']).resolve() == Path(original['window_index']),
                'Selected data manifest/index differs')
    sources = dependency_closure([Path(__file__)])
    from vit_pytorch import vit
    external = BASE.parents[2] / 'diffusion-motion-inbetweening/sample/virtual_head_camera.py'
    sources.update({str(p.resolve()): sha256(p) for p in (Path(vit.__file__), external)})
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    result = dict(original, state='frozen', frozen_at=now(), selection=selection,
        selection_manifest_sha256=sha256(selection_path), config=config, reference_policy=POLICY,
        source_sha256=sources, model_tensor_sha256=model_digest(model),
        model_selection='Completed on VAL before this freeze; no TEST metric may change model or coefficients',
        analytical_physics_enabled=False, output_policy='raw_model', optimizer_updates=0)
    del model
    gc.collect()
    write(out / 'PLAN.json', result)
    status(out, 'frozen_ready_to_evaluate', split=result['split'], planned_predictions=result['planned_predictions'])
    return result


def verify(out):
    plan = read(Path(out) / 'PLAN.json')
    require(plan['protocol'] == PROTOCOL and plan['state'] == 'frozen', 'Metric plan is not frozen')
    require(sha256(plan['selection_manifest']) == plan['selection_manifest_sha256'], 'Final selection changed')
    validate_selection(plan['selection'])
    require(read(plan['selection_manifest']) == plan['selection'], 'Selected declaration differs')
    for path, expected in {**plan['input_source_sha256'], **plan['source_sha256']}.items():
        require(sha256(path) == expected, 'Frozen source/input changed: ' + path)
    require(sha256(plan['selection']['checkpoint']) == plan['selection']['checkpoint_sha256'], 'Selected model changed')
    return plan


def row_key(row):
    return row['case'], row['regime'], row['scene_variant']


def load_rows(out, plan, *, repair=False):
    """Keep complete committed rows; repair only an interrupted final write."""
    path = Path(out) / 'rows.jsonl'
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        require(repair, 'Interrupted JSONL tail; resume explicitly to retain committed rows')
        cut = raw.rfind(b'\n') + 1
        write(Path(out) / 'TAIL_RECOVERY.json', dict(recorded_at=now(), discarded_bytes=len(raw) - cut,
            discarded_sha256=hashlib.sha256(raw[cut:]).hexdigest(), committed_bytes=cut))
        with path.open('r+b') as stream:
            stream.truncate(cut); stream.flush(); os.fsync(stream.fileno())
        raw = raw[:cut]
    rows = [json.loads(line) for line in raw.splitlines()]
    entries = {e['case']: e for e in plan['cases']}
    require(len({row_key(r) for r in rows}) == len(rows), 'Duplicate committed result')
    for row in rows:
        require(row['case'] in entries and row['regime'] in plan['regimes']
                and row['scene_variant'] in plan['scene_variants'], 'Unexpected metric row')
        entry_ = entries[row['case']]
        require(row['checkpoint_sha256'] == plan['selection']['checkpoint_sha256']
                and row['selection_sha256'] == plan['selection_manifest_sha256'], 'Wrong committed model/selection')
        require(all(row[k] == entry_[k] for k in ('mode', 'dataset', 'recording', 'sequence_id', 'start', 'source_first'))
                and row['seed'] == entry_['case_seed'], 'Committed source identity differs')
        expected = (Path(out) / 'motions' / f"case{row['case']:05d}_{row['regime']}_{row['scene_variant']}.npz").resolve()
        require(Path(row['artifact']).resolve() == expected and sha256(expected) == row['artifact_sha256'],
                'Committed prediction artifact differs')
    return rows


def make_data(plan, out):
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    config = plan['config']
    options = dict(frames=192, window_stride=64, memory_protocol=config['scene']['memory_protocol'],
                   quality_config=config['gt_reliability'])
    if plan['split'] == 'full-validation':
        options.update(window_index=plan['window_index'], exo_max_empty_frames=config.get('exo_max_empty_frames'))
    data = Native30Data(plan['manifest'], 'test' if plan['split'] == 'rich-test' else 'validation', **options)
    data.runtime_cache_root = Path(out) / 'runtime_cache'
    data.scene_memory.cache_root = data.runtime_cache_root / 'static_maps'
    data.scene_memory.cached_streams = 1
    data.canonical_surface_cache = True
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    return data


def build_inputs(data, builder, entry_, regime, config, device):
    import numpy as np
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from . import reference_temporal
    from .validation_review import inference_inputs, assert_input_device
    meta, _ = data.members[entry_['group']][entry_['member']]
    require(Path(meta['path']).resolve() == Path(entry_['path']).resolve(), 'Data member ordering differs')
    require(sha256(Path(entry_['path']) / 'metadata.json') == entry_['metadata_sha256'], 'Source metadata changed')
    sample, identity = data.sample(entry_['group'], entry_['mode'], member=entry_['member'], start=entry_['start'], yaw=0)
    require(identity['source_start_30fps'] == entry_['source_first'] and identity['sequence_id'] == entry_['sequence_id'],
            'Sample source clock differs')
    forbidden = ('motion', 'joints', 'contact_target', 'contact_valid', 'support_points')
    sensors = {k: v for k, v in sample.items() if k not in forbidden and not k.startswith('gt_')}
    selected = builder if regime == 'clean' else Native30ObservationBuilder(builder.skin_cache, builder.mounts,
                                                        dict(builder.noise_config, clean_probability=0.))
    with reference_temporal.installed():
        observed, scene, asset = selected.build(data, sensors, identity, entry_['mode'], entry_['joints'],
            noise_rng=None if regime == 'clean' else np.random.default_rng(entry_['case_seed']))
        batch = collate_observations([(observed, scene, asset, entry_['mode'])], device=device)
        known = inference_inputs(config, batch)
    require(not any(k in known for k in forbidden) and not any(k.startswith('gt_') for k in known), 'GT leaked into model inputs')
    assert_input_device(known, device)
    return sample, identity, observed, known


def summarize(out, plan, rows, *, verified=False):
    groups = {}
    for mode in ('ego', 'exo'):
        for dataset in ('rich', 'trumans', 'egobody'):
            for regime in plan['regimes']:
                for scene in plan['scene_variants']:
                    selected = [r for r in rows if (r['mode'], r['dataset'], r['regime'], r['scene_variant'])
                                == (mode, dataset, regime, scene)]
                    if not selected:
                        continue
                    metrics = {}
                    for name in sorted({k for r in selected for k in r['metrics']}):
                        valid = [r for r in selected if r['metrics'][name] is not None]
                        if not valid:
                            continue
                        per_recording = defaultdict(list)
                        for row in valid:
                            per_recording[row['recording']].append(row['metrics'][name])
                        counted = [r for r in valid if r['counts'].get(name, 0) > 0]
                        count = sum(r['counts'][name] for r in counted)
                        metrics[name] = dict(window_mean=statistics.mean(r['metrics'][name] for r in valid),
                            recording_macro=statistics.mean(statistics.mean(v) for v in per_recording.values()),
                            count_weighted=sum(r['metrics'][name] * r['counts'][name] for r in counted) / count if count else None,
                            valid_samples=count, windows=len(valid), recordings=len(per_recording))
                    planned = sum(e['mode'] == mode and e['dataset'] == dataset for e in plan['cases'])
                    groups['/'.join((mode, dataset, regime, scene))] = dict(predictions=len(selected), planned_windows=planned, metrics=metrics)
    state = ('completed_verified' if len(rows) == plan['planned_predictions'] else 'partial_verified') if verified else (
        'sampling_complete_pending_CPU' if len(rows) == plan['planned_predictions'] else 'partial')
    result = dict(protocol=PROTOCOL, status=state,
        checked_at=now(), split=plan['split'], selection=plan['selection'], completed_predictions=len(rows),
        planned_predictions=plan['planned_predictions'], groups=groups, optimizer_updates=0,
        model_selection_allowed=False, aggregation='Equal-window and equal-recording macro; valid-sample weighting additionally reported where denominators exist.',
        scene_ablation='Same checkpoint, input observations/reference, seed and DDIM20; only use_scene changes. Analytical physics OFF for both.')
    write(Path(out) / 'SUMMARY.json', result)
    lines = ['冻结模型正式指标；未完成结果不能作为完整测试集结果。', '',
        f"split={plan['split']}；完成{len(rows)}/{plan['planned_predictions']}；physics OFF；模型由VAL预先确定。", '',
        '| mode/dataset/regime/scene | 完成/计划 | World MPJPE mm | 遮挡 mm | 可见 mm |',
        '|---|---:|---:|---:|---:|']
    for key, group in groups.items():
        def fmt(name):
            value = group['metrics'].get(name, {}).get('window_mean')
            return 'NA' if value is None else f'{value:.3f}'
        lines.append('| ' + ' | '.join((key, f"{group['predictions']}/{group['planned_windows']}",
                     fmt('w_mpjpe_mm'), fmt('hidden_mpjpe_mm'), fmt('visible_mpjpe_mm'))) + ' |')
    lines += ['', plan['disclosure'], 'CPU_RECHECK通过后才确认完整指标；TEST不用于改模型或调系数。']
    (Path(out) / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n')
    return result


def evaluate(out, *, resume=False, max_predictions=0, memory_fraction=.7, status_every=10, device='cuda'):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_refinement import sample_motion
    from experiments.offline_camera_retrain_v1.train_native30 import model_digest
    from . import validation_review as common
    from .reference_review import metrics
    from .scene_comparison import scene_off_guard
    out = Path(out).resolve(); plan = verify(out)
    require(status_every > 0 and 0 < memory_fraction <= 1 and max_predictions >= 0, 'Invalid runtime budget')
    require(device in ('cpu', 'cuda'), 'device must be cpu/cuda')
    require(resume or not (out / 'rows.jsonl').exists(), 'Existing results require --resume')
    rows = load_rows(out, plan, repair=resume); done = {row_key(r) for r in rows}
    status(out, 'waiting_gpu' if device == 'cuda' else 'loading_CPU', split=plan['split'], device=device,
           completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
    with common.gpu_lock() if device == 'cuda' else nullcontext():
        verify(out)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(2)
        if device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        model, identity = load_selected(plan['selection'], device)
        require(model_digest(model) == plan['model_tensor_sha256'], 'Frozen model tensors differ')
        data, builder = make_data(plan, out), common.observation_builder(plan['config'], device=device)
        (out / 'motions').mkdir(exist_ok=True)
        started, generated = time.monotonic(), 0
        for entry_ in plan['cases']:
            for regime in plan['regimes']:
                if all((entry_['case'], regime, scene) in done for scene in plan['scene_variants']):
                    continue
                sample, source_identity, observed, known = build_inputs(data, builder, entry_, regime, plan['config'], device)
                input_sha = common.tensor_fingerprint(known)
                require(all(r['input_sha256'] == input_sha for r in rows if r['case'] == entry_['case'] and r['regime'] == regime),
                        'Resumed/paired full input hash differs')
                for scene in plan['scene_variants']:
                    if (entry_['case'], regime, scene) in done:
                        continue
                    status(out, 'sampling', split=plan['split'], current_case=entry_['case'], mode=entry_['mode'],
                           regime=regime, scene_variant=scene, completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
                    tick = time.monotonic()
                    with (scene_off_guard(model) if scene == 'scene_off' else nullcontext()) as guards, torch.no_grad():
                        pred = sample_motion(model, known, plan['config'], steps=20,
                                             seed=entry_['case_seed'], use_scene=scene == 'scene_on')
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    require(bool(torch.isfinite(pred).all()) and common.tensor_fingerprint(known) == input_sha,
                            'Nonfinite prediction or mutated inference input')
                    sampling_s = time.monotonic() - tick
                    arrays = dict(prediction=pred[0].cpu().numpy(), truth=sample['motion'], rest=sample['rest'],
                        observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                        reference_motion=known['reference_motion'][0].cpu().numpy(), identity_json=np.array(json.dumps(source_identity)),
                        input_sha256=np.array(input_sha), checkpoint_sha256=np.array(identity['checkpoint_sha256']),
                        case_seed=np.array(entry_['case_seed']), use_scene=np.array(scene == 'scene_on'))
                    artifact = out / 'motions' / f"case{entry_['case']:05d}_{regime}_{scene}.npz"
                    with artifact.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
                    artifact.with_suffix('.tmp').replace(artifact)
                    values, counts = metrics(arrays)
                    row = {k: entry_[k] for k in ('case', 'mode', 'dataset', 'recording', 'sequence_id', 'start', 'source_first')}
                    row.update(regime=regime, scene_variant=scene, seed=entry_['case_seed'], input_sha256=input_sha,
                        selection_sha256=plan['selection_manifest_sha256'], checkpoint_sha256=identity['checkpoint_sha256'],
                        artifact=str(artifact), artifact_sha256=sha256(artifact), metrics=values, counts=counts,
                        scene_off_guard_counts=guards, sampling_s=sampling_s, checked_at=now(), device=device)
                    with (out / 'rows.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                    rows.append(row); done.add(row_key(row)); generated += 1
                    if generated % status_every == 0:
                        summarize(out, plan, rows)
                        print(json.dumps(dict(completed=len(rows), planned=plan['planned_predictions'],
                                              last_sampling_s=sampling_s, elapsed_s=time.monotonic() - started)), flush=True)
                    if max_predictions and generated >= max_predictions:
                        require(model_digest(model) == plan['model_tensor_sha256'], 'Model tensors changed')
                        summarize(out, plan, rows)
                        status(out, 'bounded_stop', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
                        return
                del known, sample, observed, pred
                gc.collect()
        require(model_digest(model) == plan['model_tensor_sha256'], 'Frozen inference changed model tensors')
        verify(out)
    summarize(out, plan, rows)
    status(out, 'sampling_completed_pending_CPU', completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])


def recheck(out):
    import numpy as np
    import torch
    from .reference_review import metrics
    from .refinement_evaluation import numpy_core
    from .validation_review import observation_builder, tensor_fingerprint
    require(not torch.cuda.is_initialized(), 'CPU recheck must not initialize CUDA')
    torch.set_num_threads(2)
    out = Path(out).resolve(); plan = verify(out); rows = load_rows(out, plan)
    require(rows, 'No committed prediction is available for CPU recheck')
    data, builder = make_data(plan, out), observation_builder(plan['config'], device='cpu')
    entries = {e['case']: e for e in plan['cases']}
    maximum, independent, paired, checked_inputs = 0., defaultdict(float), {}, set()
    for row in rows:
        key = row['case'], row['regime']
        with np.load(row['artifact'], allow_pickle=False) as arrays:
            require(str(arrays['input_sha256']) == row['input_sha256']
                    and str(arrays['checkpoint_sha256']) == plan['selection']['checkpoint_sha256'], 'Saved result identity differs')
            actual, counts = metrics(arrays)
            require(counts == row['counts'], 'Metric denominators changed')
            for name, value in row['metrics'].items():
                require((value is None) == (actual[name] is None), 'Metric availability changed')
                if value is not None:
                    maximum = max(maximum, abs(value - actual[name]))
            for name, value in numpy_core(arrays).items():
                if value is not None:
                    independent[name] = max(independent[name], abs(value - actual[name]))
            digest = hashlib.sha256()
            for field in ('truth', 'rest', 'observations', 'observation_meta', 'reference_motion'):
                digest.update(np.ascontiguousarray(arrays[field]).tobytes())
            fingerprint = digest.hexdigest()
            require(key not in paired or paired[key] == fingerprint, 'Scene ON/OFF saved inputs differ')
            paired[key] = fingerprint
            if key not in checked_inputs:
                sample, _, observed, known = build_inputs(data, builder, entries[row['case']], row['regime'], plan['config'], 'cpu')
                require(np.array_equal(sample['motion'], arrays['truth']) and np.array_equal(sample['rest'], arrays['rest'])
                        and np.array_equal(observed['trajectory'], arrays['observations'])
                        and np.array_equal(observed['observation_meta'], arrays['observation_meta']), 'Reloaded source/truth/observations changed')
                # Raw observation/reference arrays are the device-independent
                # proof; recursive full input hashes are paired on the GPU.
                require(np.allclose(known['reference_motion'][0].numpy(), arrays['reference_motion'], rtol=0, atol=2e-5),
                        'CPU observation-derived reference differs')
                checked_inputs.add(key)
        if row['scene_variant'] == 'scene_off':
            require(row['scene_off_guard_counts'] == dict(scene_tokens=0, local_geometry=0, guide_motion=0), 'Scene-off guards differ')
    limits = dict(hidden_mpjpe_mm=.02, visible_mpjpe_mm=.02, mpjpe_mm=.02, mpjve_mm_s=.05,
                  acceleration_error_mm_s2=2., foot_joint_speed_gt_low_m_s=.0001)
    passed = maximum < 1e-6 and all(independent[k] <= limits[k] for k in independent)
    report = dict(status='independent_CPU_passed' if passed else 'independent_CPU_FAILED', checked_at=now(),
        completed_predictions=len(rows), planned_predictions=plan['planned_predictions'],
        complete=len(rows) == plan['planned_predictions'], CUDA_initialized=False,
        maximum_recompute_difference=maximum, independent_numpy_max_difference=dict(independent),
        independent_tolerances=limits, reloaded_observation_regimes=len(checked_inputs), paired_saved_inputs_exact=True,
        rows_sha256=sha256(out / 'rows.jsonl') if rows else None, optimizer_updates=0, model_selection_allowed=False)
    write(out / 'CPU_RECHECK.json', report)
    summarize(out, plan, rows, verified=passed)
    status(out, 'completed_verified' if report['complete'] and passed else 'partial_verified' if passed else 'CPU_FAILED',
           completed_predictions=len(rows), planned_predictions=plan['planned_predictions'])
    require(passed, 'Independent CPU metrics failed')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('entry', 'freeze', 'evaluate', 'recheck', 'report', 'status'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--split', choices=('rich-test', 'full-validation'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--test-plan', type=Path, default=DEFAULT_TEST_PLAN)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-predictions', type=int, default=0)
    parser.add_argument('--memory-fraction', type=float, default=.7)
    parser.add_argument('--status-every', type=int, default=10)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'entry':
            require(args.selection is not None and args.split is not None, 'entry requires --selection and --split')
            result = entry(args.out, args.selection, split=args.split, config_path=args.config, test_plan=args.test_plan)
            print(json.dumps({k: result[k] for k in ('state', 'coverage', 'planned_predictions')}))
        elif args.command == 'freeze':
            result = freeze(args.out)
            print(json.dumps(dict(state=result['state'], selection_sha256=result['selection_manifest_sha256'])))
        elif args.command == 'evaluate':
            evaluate(args.out, resume=args.resume, max_predictions=args.max_predictions,
                     memory_fraction=args.memory_fraction, status_every=args.status_every, device=args.device)
        elif args.command == 'recheck':
            print(json.dumps(recheck(args.out)))
        elif args.command == 'report':
            plan = verify(args.out); summarize(args.out, plan, load_rows(args.out, plan))
        else:
            print(json.dumps(read(args.out / 'STATUS.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
