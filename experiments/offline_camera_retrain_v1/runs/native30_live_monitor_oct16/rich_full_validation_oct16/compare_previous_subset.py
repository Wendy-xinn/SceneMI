#!/usr/bin/env python3
"""Strict CPU-only 12600/current matched RICH validation comparison.

The watcher imports only the standard library until evaluation finishes, then
executes this same script with one CPU thread and GPU visibility disabled.
All training files, evaluation rows and saved motions are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'balanced_validation_oct16'
BASE = OUT.parents[2]
REPO = BASE.parents[1]
RUN = OUT.parents[1] / 'native30_gait_v2_ego55k_oct15'
FIELDS = ('dataset', 'recording', 'sequence_id', 'start', 'source_first', 'case_seed')
INPUT_ARRAYS = ('truth', 'rest', 'observations', 'observation_meta', 'reference')
sys.path.insert(0, str(REPO))


def now():
    return datetime.now().astimezone().isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def identity(entry):
    return tuple(entry[key] for key in FIELDS)


def read_rows(folder):
    path = folder / 'rows.jsonl'
    raw = path.read_bytes() if path.exists() else b''
    output = {}
    ignored_tail = False
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            ignored_tail = True
            continue
        row = json.loads(line)
        if row['dataset'] != 'rich':
            continue
        key = (identity(row), row['regime'])
        require(key not in output, 'Duplicate RICH row identity/regime in ' + str(folder))
        output[key] = row
    return output, dict(path=str(path), snapshot_bytes=len(raw), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                       incomplete_tail_ignored=ignored_tail)


def planned_pairs(old_plan, new_plan):
    old = [entry for entry in old_plan['plan'] if entry['dataset'] == 'rich']
    new = {identity(entry): entry for entry in new_plan['plan']}
    require(len(old) == 10 and len({identity(e) for e in old}) == 10, 'Old RICH cohort must be 10 unique windows')
    require(len(new) == len(new_plan['plan']), 'New PLAN has repeated window identities')
    require(all(identity(e) in new for e in old), 'Old RICH window absent from current full-grid PLAN')
    for entry in old:
        counterpart = new[identity(entry)]
        require({k: v for k, v in entry.items() if k != 'case'} ==
                {k: v for k, v in counterpart.items() if k != 'case'},
                'Matched PLAN source metadata, hashes or input settings differ')
    return [(entry, new[identity(entry)]) for entry in old]


def compare_numbers(expected, actual, path, errors):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), path + ': dictionary keys differ')
        for key in expected:
            compare_numbers(expected[key], actual[key], path + '/' + str(key), errors)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), path + ': list sizes differ')
        for index, value in enumerate(expected):
            compare_numbers(value, actual[index], path + '/' + str(index), errors)
    elif expected is None or isinstance(expected, (bool, int)):
        require(expected == actual, path + ': non-float values differ')
    else:
        require(math.isfinite(float(expected)) and math.isfinite(float(actual)), path + ': nonfinite metric')
        error = abs(float(expected) - float(actual))
        require(math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=3e-5),
                path + ': CPU metric differs from saved row ' + repr((expected, actual)))
        errors.append(error)


def check_sources():
    effective = load(RUN / 'prepared/effective_config.json')
    hashes = dict(effective['code_sha256'])
    foot_path = BASE / effective['config']['run_control']['foot_metrics_source']
    hashes[str(foot_path)] = effective['config']['run_control']['foot_metrics_sha256']
    for path, digest in hashes.items():
        require(sha(path) == digest, 'Frozen evaluation metric dependency changed: ' + path)
    return hashes, effective


def inputs_equal(left, right):
    import numpy as np
    results = {}
    for key in INPUT_ARRAYS:
        a, b = left[key], right[key]
        same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        require(same, 'Matched inputs/GT differ: ' + key)
        require(np.isfinite(a).all(), 'Nonfinite matched input array: ' + key)
        results[key] = dict(shape=list(a.shape), dtype=str(a.dtype), exact_array_equal=True,
            old_bytes_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),
            current_bytes_sha256=hashlib.sha256(b.tobytes(order='C')).hexdigest())
    return results


def recompute(arrays, metric_module, foot_module):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.native30_validation import raw_metrics
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    pred = torch.from_numpy(arrays['prediction'])[None]
    truth = torch.from_numpy(arrays['truth'])[None]
    rest = torch.from_numpy(arrays['rest'])[None]
    batch = dict(motion=truth, rest=rest)
    conditioned = torch.from_numpy(arrays['observation_meta'][..., 0] > 0)[None]
    require(pred.device.type == 'cpu' and truth.device.type == 'cpu', 'Unexpected GPU tensor')
    with torch.no_grad():
        raw = raw_metrics(pred, batch, conditioned)
        extra = foot_module.extra_metrics(pred, batch)
        pj, gj = forward_kinematics(pred, rest), forward_kinematics(truth, rest)
        standard = metric_module.standard_motion_metrics(pj[0].numpy(), gj[0].numpy(), fps=30.)
        frames, joints = pj.shape[1:3]
        counts = {key: frames * joints for key in standard}
        counts.update(root_position_error_mm=frames, mpjve_mm_s=(frames - 1) * joints,
                      acceleration_error_mm_s2=(frames - 2) * joints)
        benchmark = {key: dict(value=value, count=counts[key]) for key, value in standard.items()}
        speed = pj.diff(dim=1)[:, :, (10, 11)].norm(dim=-1) * 30
        true_speed = gj.diff(dim=1)[:, :, (10, 11)].norm(dim=-1) * 30
        low = true_speed < .2
        error = pj[:, :, 0] - gj[:, :, 0]
        extra.update(gt_low_speed_count=int(low.sum()), gt_foot_velocity_samples=int(low.numel()),
            gt_foot_speed_on_gt_low_speed_m_s=float(true_speed[low].mean()) if low.any() else None,
            pred_speed_over_point2_on_gt_low_speed_count=int(((speed > .2) & low).sum()),
            root_terminal_horizontal_error_change_m=float((error[:, -1] - error[:, 0])[:, (0, 2)].norm(dim=-1).mean()))
    return dict(metrics=raw, extra_metrics=extra, benchmark_metrics=benchmark)


def paired_summary(pairs, report):
    result = {}
    lower_is_better = {
        'mpjpe_mm', 'w_mpjpe_mm', 'root_relative_mpjpe_mm', 'pa_mpjpe_mm',
        'conditioned_mpjpe_mm', 'unconditioned_mpjpe_mm', 'root_position_error_mm',
        'root_orientation_deg', 'head_orientation_deg', 'regular_rotation_deg',
        'mpjve_mm_s', 'acceleration_error_mm_s2', 'body_local_foot_error_mm',
        'foot_world_error_mm', 'foot_velocity_RMSE_m_s', 'swing_velocity_RMSE_m_s',
        'foot_acceleration_RMSE_m_s2', 'articulation_log_ratio_error',
        'bilateral_sagittal_correlation_error', 'foot_speed_during_GT_stance_m_s',
        'pred_speed_over_point2_on_gt_low_speed_fraction', 'root_terminal_horizontal_error_change_m',
    }
    for regime in ('clean', 'noisy'):
        selected = [p for p in pairs if p['regime'] == regime]
        old = [dict(recording=p['recording'], normalized_metrics=report.normalized_metrics(p['old_recomputed'])) for p in selected]
        current = [dict(recording=p['recording'], normalized_metrics=report.normalized_metrics(p['current_recomputed'])) for p in selected]
        deltas = []
        wins = {}
        for p, left, right in zip(selected, old, current):
            common = left['normalized_metrics'].keys() & right['normalized_metrics'].keys()
            delta = {}
            for key in common:
                a, b = left['normalized_metrics'][key], right['normalized_metrics'][key]
                require(a['count'] == b['count'], 'Matched effective counts differ: ' + key)
                delta[key] = dict(value=b['value'] - a['value'], count=a['count'])
                if key in lower_is_better:
                    target = wins.setdefault(key, dict(improved=0, equal=0, worsened=0))
                    target['improved' if b['value'] < a['value'] else 'worsened' if b['value'] > a['value'] else 'equal'] += 1
            deltas.append(dict(recording=p['recording'], normalized_metrics=delta))
        delta_summary = report.group(deltas)
        # A signed change in window RMSE is not itself a samplewise RMSE.
        for subtotal in [delta_summary['overall'], *delta_summary['per_recording'].values()]:
            for metric in subtotal['metrics'].values():
                metric.pop('count_pooled_RMSE', None)
        result[regime] = dict(matched_windows=len(selected), old=report.group(old), current=report.group(current),
            paired_current_minus_old=delta_summary, window_direction_counts_lower_is_better=wins)
    return result


def compare(require_complete=False):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    hashes, effective = check_sources()
    metric_module = module('matched_standard_metrics', BASE / 'standard_motion_metrics.py')
    foot_module = module('matched_original_foot_metrics', BASE / effective['config']['run_control']['foot_metrics_source'])
    report = module('matched_reporting_helpers', OUT / 'report.py')
    old_plan, new_plan = load(OLD / 'PLAN.json'), load(OUT / 'PLAN.json')
    planned = planned_pairs(old_plan, new_plan)
    old_protocol, new_protocol = load(OLD / 'PROTOCOL.json'), load(OUT / 'PROTOCOL.json')
    require(old_protocol['status'] == 'completed' and old_protocol['checkpoint_step'] == 12600, 'Old experiment identity differs')
    for key in ('effective_config_sha256', 'frames', 'fps', 'ddim_steps', 'scene', 'mode', 'regimes'):
        require(old_protocol[key] == new_protocol[key], 'Evaluation protocols differ on ' + key)
    require(new_protocol['fps'] == 30, 'Explicit 30Hz protocol required')
    old_rows, old_source = read_rows(OLD)
    new_rows, new_source = read_rows(OUT)
    pairs, pending = [], []
    errors = []
    for left_entry, right_entry in planned:
        for regime in ('clean', 'noisy'):
            key = (identity(left_entry), regime)
            require(key in old_rows, 'Old saved result missing for ' + repr(key))
            if key not in new_rows:
                pending.append(dict(old_case=left_entry['case'], current_case=right_entry['case'], regime=regime,
                                    sequence_id=left_entry['sequence_id'], start=left_entry['start']))
                continue
            left_row, right_row = old_rows[key], new_rows[key]
            for row, folder, protocol in ((left_row, OLD, old_protocol), (right_row, OUT, new_protocol)):
                require(row['checkpoint_step'] == protocol['checkpoint_step'], 'Row checkpoint identity differs')
                require(sha(folder / row['motion_file']) == row['motion_sha256'], 'Saved artifact digest differs')
            with np.load(OLD / left_row['motion_file'], allow_pickle=False) as left_file:
                left = {name: left_file[name] for name in left_file.files}
            with np.load(OUT / right_row['motion_file'], allow_pickle=False) as right_file:
                right = {name: right_file[name] for name in right_file.files}
            require(int(left['step']) == left_row['checkpoint_step'] and int(right['step']) == right_row['checkpoint_step'],
                    'Artifact checkpoint step differs from row')
            equality = inputs_equal(left, right)
            left_metrics = recompute(left, metric_module, foot_module)
            right_metrics = recompute(right, metric_module, foot_module)
            for label, row, computed in (('old', left_row, left_metrics), ('current', right_row, right_metrics)):
                for container in ('metrics', 'extra_metrics'):
                    compare_numbers(row[container], computed[container], label + '/' + container, errors)
                if label == 'current':
                    compare_numbers(row['benchmark_metrics'], computed['benchmark_metrics'], label + '/benchmark_metrics', errors)
            pairs.append(dict(old_case=left_entry['case'], current_case=right_entry['case'], regime=regime,
                recording=left_entry['recording'], identity=dict(zip(FIELDS, identity(left_entry))),
                arrays=equality, old_artifact=dict(path=str(OLD / left_row['motion_file']), sha256=left_row['motion_sha256']),
                current_artifact=dict(path=str(OUT / right_row['motion_file']), sha256=right_row['motion_sha256']),
                old_recomputed=left_metrics, current_recomputed=right_metrics))
    summary_path = OUT / 'SUMMARY.json'
    summary = load(summary_path) if summary_path.exists() else {}
    final = (not pending and not old_source['incomplete_tail_ignored'] and not new_source['incomplete_tail_ignored'] and
        new_protocol['status'] == 'completed' and load(OUT / 'STATUS.json')['status'] == 'completed' and
        summary.get('is_final_complete') is True and not summary.get('read_audit', {}).get('integrity_issues'))
    if require_complete:
        require(final and load(OUT / 'STATUS.json')['status'] == 'completed', 'Full run not complete; matched comparison cannot be final')
        require(summary['is_final_complete'] and not summary['read_audit']['integrity_issues'], 'Full-run SUMMARY has not passed final coverage')
    check_sources()
    result = dict(generated_at=now(), status='complete' if final else 'partial_no_conclusion', is_final_complete=final,
        expected_matched_windows=10, expected_paired_predictions=20, completed_paired_predictions=len(pairs), pending=pending,
        checkpoint_steps=dict(old=old_protocol['checkpoint_step'], current=new_protocol['checkpoint_step']),
        checkpoint_sha256=dict(old=old_protocol['checkpoint_sha256'], current=new_protocol['checkpoint_sha256']),
        identity_fields=list(FIELDS), exact_input_arrays=list(INPUT_ARRAYS),
        plan_mapping=[dict(old_case=a['case'], current_case=b['case'], identity=dict(zip(FIELDS, identity(a)))) for a, b in planned],
        source_snapshots=[old_source, new_source], sealed_metric_sources=hashes,
        comparison_script_sha256=sha(Path(__file__)), report_helpers_sha256=sha(OUT / 'report.py'),
        CPU_only=True, cpu_threads=1, explicit_metric_fps=30,
        saved_row_vs_CPU_recomputed_max_abs_difference=max(errors, default=0),
        aggregates=paired_summary(pairs, report), paired_windows=pairs,
        interpretation_limits=[
            'Only the same preselected 10 RICH windows are compared; never compare old 10-window mean with current 206-window mean.',
            'Every matched pair has identical dataset/recording/sequence/start/source_first/case_seed and exactly equal GT/rest/observations/meta/reference arrays.',
            'Both sides use the same sealed metric code with explicit fps=30; no model rerun, no smoothing, no posthoc alignment except metric-defined root-relative/PA.',
            'These exposed development VAL windows are not an untouched TEST or a guaranteed external-method benchmark.',
            'GT low-speed foot phase is a kinematic proxy, not true contact; sliding and foot-position metrics do not quantify collision.',
            'No overall improvement claim is issued while any pair is missing or the full current protocol is unfinished.'])
    write(OUT / 'MATCHED_12600_VS_CURRENT.json', result)
    render(result)
    print(json.dumps(dict(status=result['status'], matched_predictions=len(pairs), expected=20,
                         checkpoint_steps=result['checkpoint_steps']), ensure_ascii=False), flush=True)
    return result


def render(result):
    lines = ['# 相同 RICH 窗口的训练阶段对照', '',
        f"状态 **{result['status']}**；已完成 {result['completed_paired_predictions']}/20 对预测。旧权重 {result['checkpoint_steps']['old']} 步，新权重 {result['checkpoint_steps']['current']} 步。", '',
        '严格限定为原先选定的 10 个 RICH 窗口，每窗 clean/noisy；不拿旧 10 窗平均与新 206 窗平均比较。录制、窗口起点、原始帧起点、seed 均一致，GT、rest、观测、观测 meta 和 reference 五数组逐项相等。', '']
    if not result['is_final_complete']:
        lines += ['**尚未齐全，当前表仅为工具核验，不对训练是否改善作结论。**', '']
    metrics = [('w_mpjpe_mm', '全局 MPJPE (mm)'), ('root_relative_mpjpe_mm', 'Root-relative MPJPE (mm)'),
        ('pa_mpjpe_mm', 'PA-MPJPE (mm)'), ('root_position_error_mm', '根位置 (mm)'),
        ('root_orientation_deg', '根朝向 (°)'), ('head_orientation_deg', '头朝向 (°)'),
        ('mpjve_mm_s', '关节速度误差 (mm/s)'), ('acceleration_error_mm_s2', '关节加速度误差 (mm/s²)'),
        ('body_local_foot_error_mm', '身体局部足误差 (mm)'), ('foot_world_error_mm', '世界足误差 (mm)'),
        ('foot_speed_during_GT_stance_m_s', 'GT 低速期预测足速 (m/s)'),
        ('pred_speed_over_point2_on_gt_low_speed_fraction', 'GT 低速期预测>0.2m/s 比例'),
        ('root_terminal_horizontal_error_change_m', '根新增漂移 (m)')]
    for regime, data in result['aggregates'].items():
        lines += [f'## {regime}', '', f"已配对 {data['matched_windows']}/10 个窗口。优先使用显式计数加权；无 count 的足部额外指标使用等窗口平均。", '',
            '| 指标 | 12600 | 当前权重 | 当前−12600 | 改善窗口/全部 |', '|---|---:|---:|---:|---:|']
        for key, label in metrics:
            def metric(stage):
                item = data[stage]['overall']['metrics'].get(key, {})
                return item.get('count_weighted_mean') if item.get('count_weighted_mean') is not None else item.get('equal_window_mean')
            a, b, delta = metric('old'), metric('current'), metric('paired_current_minus_old')
            if a is None or b is None:
                continue
            wins = data['window_direction_counts_lower_is_better'][key]
            if 'fraction' in key:
                values = f'{a:.1%} | {b:.1%} | {100 * delta:+.1f} 百分点'
            else:
                values = f'{a:.3f} | {b:.3f} | {delta:+.3f}'
            lines.append(f"| {label} | {values} | {wins['improved']}/{data['matched_windows']} |")
        lines.append('')
    lines += [f"全部指标在 CPU 单线程重算，两侧共用冻结源码。旧结果补算标准动作指标，显式 fps=30。与保存行对应指标的最大绝对重算差 {result['saved_row_vs_CPU_recomputed_max_abs_difference']:.3g}。", '',
        '每对输入数组 SHA、预测文件 SHA、逐窗指标、有效计数、等录制宏平均与配对差保存在 [MATCHED_12600_VS_CURRENT.json](MATCHED_12600_VS_CURRENT.json)。位置、朝向、速度、脚滑应分别判断，不能用总 loss 或单个指标代替所有质量。']
    temporary = OUT / f'MATCHED_12600_VS_CURRENT.md.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(OUT / 'MATCHED_12600_VS_CURRENT.md')


def self_check():
    import numpy as np
    arrays = {name: np.arange(6, dtype=np.float32).reshape(2, 3) for name in INPUT_ARRAYS}
    equality = inputs_equal(arrays, {k: v.copy() for k, v in arrays.items()})
    require(all(v['exact_array_equal'] for v in equality.values()), 'Exact equality self-check failed')
    altered = {k: v.copy() for k, v in arrays.items()}
    altered['reference'][0, 0] += .01
    try:
        inputs_equal(arrays, altered)
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    require(mismatch_rejected, 'Changed reference was not rejected')
    planned = planned_pairs(load(OLD / 'PLAN.json'), load(OUT / 'PLAN.json'))
    metric_module = module('matched_metric_selfcheck', BASE / 'standard_motion_metrics.py')
    trajectory = np.arange(5, dtype=np.float64)[:, None, None] * np.array([[[.01, 0., 0.]]])
    truth = np.repeat(trajectory, 22, axis=1)
    pred = truth + np.arange(5)[:, None, None] * np.array([[[.001, 0., 0.]]])
    result = metric_module.standard_motion_metrics(pred, truth, fps=30.)
    require(math.isclose(result['mpjve_mm_s'], 30., rel_tol=1e-8), '30Hz velocity unit self-check failed')
    require(result['root_relative_mpjpe_mm'] < 1e-8, 'Root-relative translation self-check failed')
    check = dict(checked_at=now(), CPU_only=True, exact_array_equality=True, changed_reference_rejected=True,
        all_old_10_windows_present_in_new_plan=True, matched_windows=len(planned), fps30_velocity_units_correct=True,
        root_relative_translation_removed=True, script_sha256=sha(Path(__file__)))
    write(OUT / 'MATCHED_SELF_CHECK.json', check)
    print(json.dumps(check, ensure_ascii=False, indent=2))


def wait_complete():
    started = now()
    while True:
        protocol = load(OUT / 'PROTOCOL.json')
        status = load(OUT / 'STATUS.json')
        write(OUT / 'MATCHED_WATCH.json', dict(status='waiting_for_runner_completion', started_at=started,
            checked_at=now(), pid=os.getpid(), CPU_only=True, wait_phase_imports='stdlib only',
            torch_imported='torch' in sys.modules, command=[sys.executable, str(Path(__file__)), '--require-complete'],
            current_checkpoint_step=protocol['checkpoint_step'], completed_predictions=status.get('completed_predictions')))
        if protocol.get('status') == 'completed' and status.get('status') == 'completed':
            # Runner publishes PROTOCOL just before its final SUMMARY. Wait until
            # that final exact-coverage report is available to avoid a race.
            summary_path = OUT / 'SUMMARY.json'
            if summary_path.exists() and load(summary_path).get('is_final_complete'):
                write(OUT / 'MATCHED_WATCH.json', dict(status='executing_CPU_comparison', started_at=started,
                    checked_at=now(), pid=os.getpid(), CPU_only=True, cpu_threads=1))
                os.execv(sys.executable, [sys.executable, '-u', str(Path(__file__)), '--require-complete'])
        if status.get('status') == 'failed':
            write(OUT / 'MATCHED_WATCH.json', dict(status='runner_failed_no_final_comparison', started_at=started,
                checked_at=now(), pid=os.getpid(), runner_error=status.get('error')))
            raise RuntimeError('Current evaluator failed; matched comparison not finalized')
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--wait-complete', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.wait_complete:
        wait_complete()
    else:
        try:
            result = compare(args.require_complete)
            if args.require_complete:
                write(OUT / 'MATCHED_WATCH.json', dict(status='completed', checked_at=now(), pid=os.getpid(),
                    CPU_only=True, cpu_threads=1, matched_predictions=result['completed_paired_predictions']))
        except BaseException as error:
            write(OUT / 'MATCHED_ERROR.json', dict(status='failed', checked_at=now(), error=repr(error)))
            raise
