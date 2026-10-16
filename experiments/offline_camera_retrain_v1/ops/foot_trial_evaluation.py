"""Matched VAL evaluation for frozen short foot-training candidates.

Primary comparisons disable the independent physical solver for every model;
a preselected five-window secondary set crosses the solver ON. Cohort selection
uses recording identities and legal indexed starts, never model outcomes.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

from .motion_ablation import BASE, FULL, load_module, now, read, require, sha, surface_metrics, write

DEFAULT_OUT = BASE / 'runs/native30_live_monitor_oct16/foot_trial_evaluation'
SUPPORT_PLAN = BASE / 'runs/native30_live_monitor_oct16/support_ablation/PLAN.json'
EFFECTIVE = BASE / 'runs/native30_gait_v2_ego55k_oct15/prepared/effective_config.json'
METRIC_KEYS = ('w_mpjpe_mm', 'pa_mpjpe_mm', 'observed_head_position_error_mm',
    'foot_joint_horizontal_speed_gt_low_m_s', 'foot_joint_vertical_speed_gt_low_m_s',
    'gt_foot_joint_horizontal_speed_gt_low_m_s', 'gt_foot_joint_vertical_speed_gt_low_m_s',
    'foot_joint_y_abs_error_gt_low_mm', 'foot_joint_y_signed_error_gt_low_mm', 'root_y_abs_error_mm', 'root_y_signed_error_mm',
    'predicted_root_y_mean_m', 'gt_root_y_mean_m',
    'body_joint_vertical_extent_abs_error_mm', 'predicted_body_joint_vertical_extent_mean_m',
    'gt_body_joint_vertical_extent_mean_m', 'head_to_foot_vertical_span_abs_error_mm',
    'predicted_head_to_foot_vertical_span_mean_m', 'gt_head_to_foot_vertical_span_mean_m',
    'bilateral_motion_rms_error_mm', 'bilateral_motion_rms_m', 'gt_bilateral_motion_rms_m',
    'bilateral_motion_rms_ratio', 'bilateral_stride_change_abs_error_mm',
    'bilateral_stride_change_mean_m', 'gt_bilateral_stride_change_mean_m',
    'support_calibrated_absolute_error_mm', 'support_signed_height_mm', 'support_gt_signed_height_mm')


def cohort(out):
    """Five existing diagnostics plus nine evenly selected other RICH recordings."""
    original = read(SUPPORT_PLAN)
    selected = [dict(entry, source_case=entry['case'], selection='existing diagnostic core')
                for entry in original['entries']]
    used = {e['recording'] or e['sequence_id'] for e in selected if e['dataset'] == 'rich'}
    groups = collections.defaultdict(list)
    for entry in read(FULL / 'PLAN.json')['plan']:
        if entry['recording'] not in used:
            groups[entry['recording']].append(entry)
    names = sorted(groups)
    require(len(names) >= 9, 'Insufficient independent remaining RICH recordings')
    # Nearest integer positions including both ends; fixed before any outcomes.
    indices = [round(i * (len(names) - 1) / 8) for i in range(9)]
    require(len(set(indices)) == 9, 'Nonunique evenly selected recordings')
    for i in indices:
        entries = sorted(groups[names[i]], key=lambda e: e['start'])
        source = entries[len(entries) // 2]
        selected.append(dict(source, source_case=source['case'],
            selection='recording-name equal spacing; upper middle legal indexed start'))
    for index, entry in enumerate(selected):
        entry['case'] = index
        entry['label'] = entry['dataset'] + ' / ' + entry['sequence_id']
    require(len(selected) == 14, 'Unexpected cohort size')
    recordings = [(e['dataset'], e['recording'] or e['sequence_id']) for e in selected]
    require(len(recordings) == len(set(recordings)), 'Repeated recording in cohort')
    result = dict(protocol='foot-trial-diverse-val-cohort-v1', entries=selected,
        unique_windows=14, independent_recordings=14, datasets=dict(collections.Counter(e['dataset'] for e in selected)),
        regimes=['clean', 'noisy'], frames=192, fps=30, ddim_steps=20,
        primary=dict(guidance_enabled=False, cases=list(range(14))),
        secondary=dict(guidance_enabled=True, cases=list(range(5))),
        planned_predictions_per_model=38,
        selection='Five pre-existing diagnostic windows, including pushup/yoga non-upright cases; nine other RICH recordings selected by evenly spaced sorted recording names, each upper-middle legal indexed window. VAL development cohort, not unbiased benchmark or TEST.',
        source_sha256={str(SUPPORT_PLAN): sha(SUPPORT_PLAN), str(FULL / 'PLAN.json'): sha(FULL / 'PLAN.json')})
    path = out / 'COHORT.json'
    if path.exists():
        require(read(path) == result, 'Immutable cohort changed')
    else:
        write(path, result)
    return result


def extended_metrics(prediction, truth, rest):
    """CPU pose extent and root-local bilateral movement; no scene assumptions."""
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from experiments.offline_camera_retrain_v1.orientation_supervision import global_rotations

    p = torch.as_tensor(np.array(prediction, dtype=np.float32, copy=True))[None]
    g = torch.as_tensor(np.array(truth, dtype=np.float32, copy=True))[None]
    r = torch.as_tensor(np.array(rest, dtype=np.float32, copy=True))[None]
    with torch.no_grad():
        jp = forward_kinematics(p, r)[0].numpy().astype(np.float64)
        jg = forward_kinematics(g, r)[0].numpy().astype(np.float64)
        rp = global_rotations(p)[0, :, 0].numpy().astype(np.float64)
        rg = global_rotations(g)[0, :, 0].numpy().astype(np.float64)
    root_y = jp[:, 0, 1] - jg[:, 0, 1]
    extent_p = jp[..., 1].max(-1) - jp[..., 1].min(-1)
    extent_g = jg[..., 1].max(-1) - jg[..., 1].min(-1)
    head_foot_p = jp[:, 15, 1] - jp[:, (10, 11), 1].min(-1)
    head_foot_g = jg[:, 15, 1] - jg[:, (10, 11), 1].min(-1)
    # Both predicted and target feet use their own root orientation; avoid
    # interpreting a whole-body turn as bilateral foot articulation.
    separation_p = np.einsum('tji,tj->ti', rp, jp[:, 10] - jp[:, 11])
    separation_g = np.einsum('tji,tj->ti', rg, jg[:, 10] - jg[:, 11])
    rms = lambda x: float(np.sqrt(np.square(x - x.mean(0, keepdims=True)).sum(-1).mean()))
    rms_p, rms_g = rms(separation_p), rms(separation_g)
    changes_p, changes_g, errors = [], [], []
    for lag in (6, 12):
        dp = separation_p[lag:] - separation_p[:-lag]
        dg = separation_g[lag:] - separation_g[:-lag]
        changes_p.append(float(np.linalg.norm(dp, axis=-1).mean()))
        changes_g.append(float(np.linalg.norm(dg, axis=-1).mean()))
        errors.append(float(np.abs(np.linalg.norm(dp, axis=-1) - np.linalg.norm(dg, axis=-1)).mean()))
    return dict(root_y_signed_error_mm=float(root_y.mean() * 1000),
        root_y_abs_error_mm=float(np.abs(root_y).mean() * 1000),
        predicted_root_y_mean_m=float(jp[:, 0, 1].mean()), gt_root_y_mean_m=float(jg[:, 0, 1].mean()),
        predicted_body_joint_vertical_extent_mean_m=float(extent_p.mean()),
        gt_body_joint_vertical_extent_mean_m=float(extent_g.mean()),
        body_joint_vertical_extent_abs_error_mm=float(np.abs(extent_p - extent_g).mean() * 1000),
        predicted_head_to_foot_vertical_span_mean_m=float(head_foot_p.mean()),
        gt_head_to_foot_vertical_span_mean_m=float(head_foot_g.mean()),
        head_to_foot_vertical_span_abs_error_mm=float(np.abs(head_foot_p - head_foot_g).mean() * 1000),
        bilateral_motion_rms_m=rms_p, gt_bilateral_motion_rms_m=rms_g,
        bilateral_motion_rms_error_mm=abs(rms_p - rms_g) * 1000,
        bilateral_motion_rms_ratio=rms_p / rms_g if rms_g > .01 else None,
        bilateral_stride_change_mean_m=float(np.mean(changes_p)),
        gt_bilateral_stride_change_mean_m=float(np.mean(changes_g)),
        bilateral_stride_change_abs_error_mm=float(np.mean(errors) * 1000))


def self_check(out):
    import numpy as np
    import torch
    source = BASE / 'runs/native30_live_monitor_oct16/foot_stage_diagnostics/motions/case00_clean.npz'
    with np.load(source, allow_pickle=False) as saved:
        truth, rest = saved['truth'], saved['rest']
    same = extended_metrics(truth, truth, rest)
    moved = truth.copy()
    moved[:, :3] += np.asarray([0, .05, 0], np.float32)
    shifted = extended_metrics(moved, truth, rest)
    frozen = np.repeat(truth[:1], len(truth), axis=0)
    static = extended_metrics(frozen, truth, rest)
    require(same['root_y_abs_error_mm'] == 0 and same['bilateral_motion_rms_error_mm'] == 0,
            'GT self metrics must be zero')
    require(abs(shifted['root_y_abs_error_mm'] - 100) < .002, '10cm translation unit mismatch')
    require(shifted['body_joint_vertical_extent_abs_error_mm'] < .002, 'Vertical extent must ignore translation')
    require(static['bilateral_motion_rms_m'] < 1e-7 and static['bilateral_stride_change_mean_m'] < 1e-7,
            'Frozen pose should have no bilateral articulation')
    require(same['gt_bilateral_motion_rms_m'] > .01, 'Self-check source lacks foot articulation')
    result = dict(status='passed', CUDA_initialized=torch.cuda.is_initialized(),
        source=str(source), source_sha256=sha(source), source_code_sha256=sha(Path(__file__)),
        gt_self=same, root_10cm_shift=shifted, frozen_pose=static)
    write(out / 'CPU_METRIC_CHECK.json', result)
    return result


def freeze_models(out, manifest, selected):
    """Seal model declarations; actual checkpoints are pinned only on execution.

    models.json maps label to {checkpoint, expected_step, optional metadata}.
    This permits the parent phase before trial checkpoints exist, while the
    evaluation cohort/code and future stop steps are already immutable.
    """
    require(isinstance(manifest, dict) and 2 <= len(manifest) <= 3, 'Expected two trial arms, optionally one baseline')
    helper = load_module('foot_trial_freeze_helpers', FULL / 'runner.py')
    helper.verify_sources()
    frozen = {}
    directory = out / 'checkpoints'
    directory.mkdir(exist_ok=True)
    for name, item in manifest.items():
        require(re.fullmatch(r'[a-z][a-z0-9_]{0,39}', name) is not None, 'Unsafe model label')
        require(isinstance(item, dict) and isinstance(item.get('expected_step'), int),
                'Each model declaration requires checkpoint and integer expected_step')
        metadata = dict(item)
        source = Path(metadata['checkpoint']).resolve()
        target = directory / (name + '.pt')
        frozen[name] = dict(source=str(source), checkpoint=str(target), expected_step=metadata['expected_step'],
            metadata=metadata,
            evaluation_config='Unmodified original architecture/config; guidance_enabled overridden equally at inference only.')
    dependencies = [Path(__file__), Path(__file__).with_name('motion_ablation.py'),
                    Path(__file__).with_name('support_metrics.py'), FULL / 'runner.py', EFFECTIVE,
                    BASE / 'runs/native30_final_review_oct16/io_recovery.py']
    plan = dict(protocol='frozen-short-foot-trial-evaluation-v1', cohort_sha256=sha(out / 'COHORT.json'),
        models=frozen, inference_config=str(EFFECTIVE), inference_config_sha256=sha(EFFECTIVE),
        primary='All 14 VAL windows × clean/noisy with physical guidance OFF for every model.',
        secondary='Same five preselected core windows × clean/noisy with physical guidance ON for every model.',
        planned_predictions=len(frozen) * selected['planned_predictions_per_model'],
        dependency_sha256={str(p): sha(p) for p in dependencies},
        targets='GT and contact labels never passed into inference. GT used for matched CPU evaluation only.',
        metric_scope='22 body joints; horizontal/vertical foot speed on GT 3D-speed<.2 m/s frames. Body height is current-pose joint vertical extent/head-foot span, not mesh stature. Sparse skin-anchor surface proxies are not full-mesh collision or self-collision.',
        surface_scope='Same GT-associated reliable visible static upward patches for every compared output; calibrated pred-GT signed deviation, not a global inferred floor.',
        aggregation='Window arithmetic means by dataset/regime/inference guidance; paired changes on identical windows/inputs/seeds; no TEST inference or ranking.')
    if (out / 'PLAN.json').exists():
        require(read(out / 'PLAN.json') == plan, 'Immutable evaluation plan differs')
    else:
        write(out / 'PLAN.json', plan)
    return plan


def pin_checkpoint(out, name, declaration, helper):
    """Validate the declared stop step before publishing a per-model freeze."""
    import torch
    metadata = out / 'checkpoints' / (name + '.json')
    target = Path(declaration['checkpoint'])
    if metadata.exists():
        result = read(metadata)
        require(result['declaration'] == declaration, 'Existing checkpoint declaration changed')
        require(sha(target) == result['checkpoint_sha256'], 'Pinned checkpoint changed')
        return result
    require(not target.exists(), 'Unresolved checkpoint pin without metadata')
    source = Path(declaration['source'])
    require(source.exists(), 'Declared model is not ready: ' + name)
    pending = target.with_suffix('.pending')
    require(not pending.exists(), 'Unresolved pending checkpoint')
    os.link(source, pending)
    try:
        state = torch.load(pending, map_location='cpu', mmap=True, weights_only=False)
        require(int(state['step']) == declaration['expected_step'],
                f"{name}: expected completed step {declaration['expected_step']}, found {state['step']}; unfinished checkpoint not pinned")
        digest = sha(pending)
        requested = declaration['metadata'].get('checkpoint_sha256')
        require(requested is None or requested == digest, 'Requested checkpoint SHA differs: ' + name)
        result = dict(declaration=declaration, checkpoint=str(target), checkpoint_sha256=digest,
            step=int(state['step']), model_tensor_sha256=helper.model_tensor_digest(state['model']),
            source_checkpoint_bytes=pending.stat().st_size, frozen_at=now())
        del state
        pending.replace(target)
        write(metadata, result)
        return result
    finally:
        pending.unlink(missing_ok=True)


def summarize(out, rows, plan, complete=False):
    groups = {}
    for guidance in ('off', 'on'):
        for dataset in ('all', 'rich', 'nonrich', 'trumans', 'egobody'):
            for regime in ('clean', 'noisy'):
                name = '/'.join((guidance, dataset, regime))
                groups[name] = {}
                for model in plan['models']:
                    subset = [r for r in rows if r['model'] == model and r['guidance'] == guidance
                        and r['regime'] == regime and (dataset == 'all' or r['dataset'] == dataset
                            or dataset == 'nonrich' and r['dataset'] != 'rich')]
                    means = {k: statistics.mean(v) if (v := [r['metrics'][k] for r in subset if r['metrics'].get(k) is not None]) else None for k in METRIC_KEYS}
                    groups[name][model] = dict(windows=len(subset), metrics=means)
    comparisons = {}
    names = list(plan['models'])
    indexed = {(r['model'], r['case'], r['regime'], r['guidance']): r for r in rows}
    for before in names:
        for after in names:
            if names.index(after) <= names.index(before):
                continue
            for guidance in ('off', 'on'):
                pairs = [(r, indexed[(after, r['case'], r['regime'], guidance)]) for r in rows
                    if r['model'] == before and r['guidance'] == guidance and (after, r['case'], r['regime'], guidance) in indexed]
                comparisons[f'{after} minus {before} / physical {guidance}'] = dict(pairs=len(pairs),
                    mean_paired_delta={k: statistics.mean(v) if (v := [b['metrics'][k] - a['metrics'][k] for a, b in pairs if a['metrics'].get(k) is not None and b['metrics'].get(k) is not None]) else None for k in METRIC_KEYS})
    write(out / 'SUMMARY.json', dict(status='completed' if complete else 'partial', rows=len(rows),
        planned_predictions=plan['planned_predictions'], groups=groups, paired_comparisons=comparisons))
    lines = ['# 脚步短训：统一推理口径配对验证', '',
        '**' + ('计划内验证完成。' if complete else '部分结果，尚不可选最终方案。') + '**', '',
        '主评价各模型全部关闭独立物理修正；五个核心窗另交叉开启，区分训练权重变化与推理后处理。14个不同录制窗口，包含12个RICH、1个TRUMANS、1个EgoBody，均为VAL。', '',
        '脚速是GT三维足速<0.2m/s时的水平/垂直速度。身体高度指标是当前姿态关节高度范围，并非mesh身高。场景指标是稀疏足锚点对同一可靠可见局部表面的GT校准偏差，不是全mesh穿模率。', '']
    for dataset in ('rich', 'nonrich'):
        for regime in ('clean', 'noisy'):
            lines += [f'## 主评价 physics OFF / {dataset} / {regime}', '',
                '| 模型 | 窗数 | World mm | PA mm | 头对输入 mm | 水平足速 m/s | 垂直足速 m/s | 脚高误差 mm | 双足运动RMS / GT |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for model, item in groups['off/' + dataset + '/' + regime].items():
                keys = ('w_mpjpe_mm', 'pa_mpjpe_mm', 'observed_head_position_error_mm', 'foot_joint_horizontal_speed_gt_low_m_s', 'foot_joint_vertical_speed_gt_low_m_s', 'foot_joint_y_abs_error_gt_low_mm', 'bilateral_motion_rms_ratio')
                values = ['—' if item['metrics'][k] is None else f"{item['metrics'][k]:.4f}" for k in keys]
                lines.append('| ' + model + ' | ' + str(item['windows']) + ' | ' + ' | '.join(values) + ' |')
            lines.append('')
    lines += ['完整分数据集、clean/noisy、physics OFF/ON、根高/当前姿态高度/stride/表面校准指标及同窗配对差值见SUMMARY.json。',
        '双足相对运动RMS及0.2/0.4秒stride变化与GT共同报告，避免把冻结脚步误当滑步改善。不存在TEST集评价或论文排名结论。', '']
    (out / 'REPORT_zh.md').write_text('\n'.join(lines))


def run(out, selected, plan, memory_fraction, only=None):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.prepare_native30 import make_model, window_options
    from experiments.offline_camera_retrain_v1.native30_data import Native30Data
    from experiments.offline_camera_retrain_v1.native30_assets import Native30SkinAnchorCache
    from experiments.offline_camera_retrain_v1.native30_observations import Native30ObservationBuilder
    from experiments.offline_camera_retrain_v1.unified_observations import collate_observations
    from experiments.offline_camera_retrain_v1.native30_refinement import model_inputs, sample_motion
    from experiments.offline_camera_retrain_v1 import native30_surface_cache as surfaces, native30_memory as memory
    from experiments.offline_camera_retrain_v1.native30_surface_cache import CanonicalSurfaceCache
    from .support_metrics import support_metrics

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    helper = load_module('foot_trial_eval_helpers', FULL / 'runner.py')
    effective = helper.verify_sources()
    for path, digest in plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Evaluation dependency changed: ' + path)
    labels = list(plan['models']) if only is None else [label for label in plan['models'] if label in only]
    require(labels and (only is None or set(labels) == set(only)), 'Unknown or empty --only model labels')
    checkpoints = {label: pin_checkpoint(out, label, plan['models'][label], helper) for label in labels}
    # A missing/unfinished future arm fails before this call touches CUDA.
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    cfg = effective['config']
    io = load_module('foot_trial_crc_reads', BASE / 'runs/native30_final_review_oct16/io_recovery.py')
    surfaces.np = io.NumpyProxy(surfaces.np, out / 'IO_READS.jsonl')
    memory.np = io.NumpyProxy(memory.np, out / 'IO_READS.jsonl')
    data = Native30Data(BASE / cfg['train_manifest'], 'validation', cfg['seed'] + 1, frames=cfg['frames'],
        memory_protocol=cfg['scene']['memory_protocol'], quality_config=cfg['gt_reliability'],
        exo_max_empty_frames=cfg['exo_max_empty_frames'], **window_options(cfg))
    data.runtime_cache_root = out / 'native30_runtime_cache_v2'
    cache_path = data.scene_memory._cache_path
    data.scene_memory._cache_path = lambda *args: (p if (p := cache_path(*args)) is not None and p.exists() else None)
    data.surface_cache = CanonicalSurfaceCache(data, limit=1)
    data.scene_memory.cached_streams = 1
    builder = Native30ObservationBuilder(Native30SkinAnchorCache(), read(BASE / cfg['mount_defaults']), cfg['observation']['input_noise'])
    noisy = Native30ObservationBuilder(builder.skin_cache, builder.mounts, dict(builder.noise_config, clean_probability=0.))
    rows = []
    row_path = out / 'rows.jsonl'
    if row_path.exists():
        raw = row_path.read_bytes()
        require(not raw or raw.endswith(b'\n'), 'Partial row tail; preserve before explicit recovery')
        rows = [json.loads(line) for line in raw.splitlines()]
    done = {(r['model'], r['case'], r['regime'], r['guidance']): r for r in rows}
    require(len(done) == len(rows), 'Duplicate rows')
    entries_by_case = {entry['case']: entry for entry in selected['entries']}
    expected = {(label, entry['case'], regime, guidance)
        for label in plan['models'] for entry in selected['entries'] for regime in selected['regimes']
        for guidance in ('off', 'on')
        if guidance == 'off' or entry['case'] in selected['secondary']['cases']}
    require(set(done).issubset(expected), 'Unplanned saved model/window/regime/guidance row')
    fingerprints, metric_denominators = {}, {}
    for row in rows:
        require(row['model'] in plan['models'], 'Unplanned model in old rows')
        entry = entries_by_case[row['case']]
        require(all(row[key] == entry[key] for key in ('dataset', 'sequence_id', 'start', 'source_first', 'case_seed')),
                'Saved row cohort identity differs')
        require(sha(row['artifact']) == row['artifact_sha256'], 'Saved artifact changed')
        frozen = read(out / 'checkpoints' / (row['model'] + '.json'))
        require(frozen['declaration'] == plan['models'][row['model']], 'Saved checkpoint declaration changed')
        require(sha(frozen['checkpoint']) == frozen['checkpoint_sha256'] == row['checkpoint_sha256'], 'Mixed checkpoint')
        key = row['case'], row['regime']
        require(key not in fingerprints or fingerprints[key] == row['input_sha256'], 'Saved input mismatch')
        fingerprints[key] = row['input_sha256']
        denominator = (row['metrics'].get('support_evaluation_anchor_count', 0), row['metrics']['gt_low_speed_foot_joint_samples'])
        require(key not in metric_denominators or metric_denominators[key] == denominator, 'Saved GT evaluation denominators differ')
        metric_denominators[key] = denominator
    protocol_path = out / 'PROTOCOL.json'
    previous = read(protocol_path) if protocol_path.exists() else None
    protocol = dict(status='running', pid=os.getpid(), started_at=now(), plan_sha256=sha(out / 'PLAN.json'),
        optimizer_updates=0, resumed_rows=len(rows), resources_before=helper.resources(), models={}, phase_models=labels)
    if previous:
        require(previous['plan_sha256'] == protocol['plan_sha256'], 'Resumed plan mismatch')
        with (out / 'ATTEMPTS.jsonl').open('a') as stream:
            stream.write(json.dumps(previous, ensure_ascii=False) + '\n')
        protocol['models'].update(previous.get('models', {}))
    write(protocol_path, protocol)
    (out / 'motions').mkdir(exist_ok=True)
    for label, checkpoint in checkpoints.items():
        require(sha(checkpoint['checkpoint']) == checkpoint['checkpoint_sha256'], 'Frozen checkpoint changed')
        state = torch.load(checkpoint['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        before = helper.model_tensor_digest(state['model'])
        require(before == checkpoint['model_tensor_sha256'], 'Checkpoint tensor identity differs')
        model = make_model(cfg).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        del state
        for entry in selected['entries']:
            pending = [(regime, guidance) for regime in selected['regimes'] for guidance in ('off', 'on')
                if (guidance == 'off' or entry['case'] in selected['secondary']['cases'])
                and (label, entry['case'], regime, guidance) not in done]
            if not pending:
                continue
            members = [i for i, (meta, _) in enumerate(data.members[entry['group']]) if meta['path'] == entry['path']]
            require(len(members) == 1, 'Nonunique source window')
            sample, identity = data.sample(entry['group'], 'ego', member=members[0], start=entry['start'], yaw=0)
            require(identity['sequence_id'] == entry['sequence_id'] and identity['source_start_30fps'] == entry['source_first'], 'Source window identity mismatch')
            for regime in selected['regimes']:
                modes = [g for r, g in pending if r == regime]
                if not modes:
                    continue
                observed, scene, asset = (builder if regime == 'clean' else noisy).build(data, sample, identity, 'ego', (15,),
                    noise_rng=None if regime == 'clean' else np.random.default_rng(entry['case_seed']))
                batch = collate_observations([(observed, scene, asset, 'ego')])
                known = model_inputs(cfg, batch)
                require(not any(k in known for k in ('motion', 'joints', 'contact_target', 'contact_valid', 'gt_position_weight', 'support_points')), 'GT target leaked into inference')
                fingerprint = hashlib.sha256()
                for key, value in sorted(known.items()):
                    if isinstance(value, torch.Tensor):
                        array = value.detach().cpu().contiguous().numpy()
                        fingerprint.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
                        fingerprint.update(array.tobytes())
                input_sha = fingerprint.hexdigest()
                key = entry['case'], regime
                require(key not in fingerprints or fingerprints[key] == input_sha, 'Input differed between models or physics modes')
                fingerprints[key] = input_sha
                for guidance in modes:
                    write(out / 'STATUS.json', dict(status='sampling', checked_at=now(), pid=os.getpid(), model=label,
                        case=entry['case'], sequence_id=entry['sequence_id'], regime=regime, guidance=guidance,
                        completed_predictions=len(rows), planned_predictions=plan['planned_predictions']))
                    model.guidance_enabled = guidance == 'on'
                    started = time.monotonic()
                    with torch.no_grad():
                        result = sample_motion(model, known, cfg, steps=selected['ddim_steps'], seed=entry['case_seed'], use_scene=True)
                    prediction = result[0].detach().cpu().numpy()
                    artifact = out / 'motions' / f"{label}_case{entry['case']:02d}_{regime}_physical_{guidance}.npz"
                    with artifact.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream, prediction=prediction, truth=sample['motion'], rest=sample['rest'],
                            observations=observed['trajectory'], observation_meta=observed['observation_meta'],
                            identity_json=np.asarray(json.dumps(identity)), input_sha256=np.asarray(input_sha),
                            model=np.asarray(label), guidance=np.asarray(guidance), checkpoint_sha256=np.asarray(checkpoint['checkpoint_sha256']))
                        stream.flush()
                        os.fsync(stream.fileno())
                    artifact.with_suffix('.tmp').replace(artifact)
                    # All numerical evaluation reopens committed arrays on CPU.
                    with np.load(artifact, allow_pickle=False) as saved:
                        metrics = support_metrics(saved['prediction'], saved['truth'], saved['rest'], saved['observations'], saved['observation_meta'], asset)
                        metrics.update(extended_metrics(saved['prediction'], saved['truth'], saved['rest']))
                        metrics.update(surface_metrics(saved['prediction'], saved['truth'], asset, scene))
                    denominator = (metrics.get('support_evaluation_anchor_count', 0), metrics['gt_low_speed_foot_joint_samples'])
                    require(key not in metric_denominators or metric_denominators[key] == denominator, 'GT evaluation masks differed across models')
                    metric_denominators[key] = denominator
                    row = dict(model=label, checkpoint_sha256=checkpoint['checkpoint_sha256'], checkpoint_step=checkpoint['step'],
                        case=entry['case'], dataset=entry['dataset'], sequence_id=entry['sequence_id'], recording=entry['recording'],
                        start=entry['start'], source_first=entry['source_first'], case_seed=entry['case_seed'], regime=regime,
                        guidance=guidance, input_sha256=input_sha, metrics=metrics, artifact=str(artifact), artifact_sha256=sha(artifact),
                        seconds=time.monotonic() - started, checked_at=now())
                    with row_path.open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    done[label, entry['case'], regime, guidance] = row
                    summarize(out, rows, plan)
                    print(json.dumps(dict(event='prediction_complete', model=label, case=entry['case'], regime=regime,
                        guidance=guidance, completed=len(rows), world_mm=metrics['w_mpjpe_mm'],
                        foot_horizontal_m_s=metrics['foot_joint_horizontal_speed_gt_low_m_s'])), flush=True)
                    del result
                del known, batch, observed, scene, asset
                gc.collect()
                torch.cuda.empty_cache()
            data.arrays.clear()
        after = helper.model_tensor_digest(model.state_dict())
        require(before == after, 'Frozen candidate tensors mutated')
        protocol['models'][label] = dict(model_tensor_digest_before=before, model_tensor_digest_after=after)
        write(protocol_path, protocol)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    phase_expected = {key for key in expected if key[0] in labels}
    require(phase_expected.issubset(done), 'Incomplete requested phase coverage')
    complete = set(done) == expected
    require(not complete or len(rows) == plan['planned_predictions'], 'Unexpected complete coverage count')
    helper.verify_sources()
    for path, digest in plan['dependency_sha256'].items():
        require(sha(path) == digest, 'Evaluation source changed while running: ' + path)
    for item in checkpoints.values():
        require(sha(item['checkpoint']) == item['checkpoint_sha256'], 'Frozen checkpoint changed during evaluation')
    protocol.update(status='completed' if complete else 'phase_completed', finished_at=now(), resources_after=helper.resources())
    write(protocol_path, protocol)
    verified = dict(status='passed_complete' if complete else 'passed_phase', optimizer_updates=0,
        completed_predictions=len(rows), fixed_model_tensors=True, inputs_matched_across_models_and_modes=True,
        metrics_from_committed_arrays=True, sealed_training_sources_unchanged=True,
        primary_physical_off=True, secondary_preselected_core_physical_on=True,
        phase_models=labels, phase_predictions=len(phase_expected),
        independent_CPU_reverification='Not performed by this runner; its metrics are CPU computed after reopening each artifact.')
    write(out / 'VERIFY.json', verified)
    write(out / ('VERIFY_PHASE_' + '_'.join(labels) + '.json'), verified)
    summarize(out, rows, plan, complete=complete)
    write(out / 'STATUS.json', dict(status='completed' if complete else 'phase_completed', checked_at=now(), pid=os.getpid(),
        completed_predictions=len(rows), planned_predictions=plan['planned_predictions'], phase_models=labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--models', type=Path, help='JSON mapping model label to checkpoint path or checkpoint metadata')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare cohort and optional checkpoint plan; never initializes CUDA')
    parser.add_argument('--self-check', action='store_true', help='Run CPU metric invariants only')
    parser.add_argument('--memory-fraction', type=float, default=.22)
    parser.add_argument('--only', nargs='+', help='Execute named models only; future declared checkpoints need not exist yet')
    args = parser.parse_args()
    require(0 < args.memory_fraction <= .22, 'Memory fraction must be in (0,.22]')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.self_check:
            print(json.dumps(self_check(out), ensure_ascii=False))
            return
        selected = cohort(out)
        if args.models is None:
            require(args.prepare_only, 'Supply --models to execute frozen evaluation')
            print(json.dumps(selected, ensure_ascii=False, indent=2))
            return
        plan = freeze_models(out, read(args.models), selected)
        if args.prepare_only:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        try:
            run(out, selected, plan, args.memory_fraction, args.only)
        except Exception as error:
            write(out / 'STATUS.json', dict(status='failed', checked_at=now(), pid=os.getpid(), error=repr(error)))
            raise


if __name__ == '__main__':
    main()
