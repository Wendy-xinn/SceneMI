#!/usr/bin/env python3
"""CPU-only full-RICH evaluator reporting; reads the exact frozen PLAN identities."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path, optional=False):
    if optional and not path.exists():
        return {}, None
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    n = (len(values) - 1) * q
    lo, hi = math.floor(n), math.ceil(n)
    return values[lo] + (values[hi] - values[lo]) * (n - lo)


def normalized_metrics(row):
    output = {}
    for container in ('metrics', 'extra_metrics', 'foot_metrics', 'diagnostics', 'benchmark_metrics', 'motion_metrics'):
        for name, metric in row.get(container, {}).items():
            value, count = (metric.get('value'), metric.get('count')) if isinstance(metric, dict) else (metric, None)
            if finite(value) and (count is None or finite(count) and count > 0):
                output[name] = dict(value=float(value), count=float(count) if count is not None else None)
    extras = row.get('extra_metrics', {})
    slow_count = extras.get('gt_low_speed_count')
    velocity_count = extras.get('gt_foot_velocity_samples')
    if finite(slow_count) and slow_count > 0:
        if 'gt_foot_speed_on_gt_low_speed_m_s' in output:
            output['gt_foot_speed_on_gt_low_speed_m_s']['count'] = slow_count
        over_count = extras.get('pred_speed_over_point2_on_gt_low_speed_count')
        if finite(over_count):
            output['pred_speed_over_point2_on_gt_low_speed_fraction'] = dict(value=over_count / slow_count, count=slow_count)
    if finite(velocity_count) and velocity_count > 0:
        if 'foot_velocity_RMSE_m_s' in output:
            output['foot_velocity_RMSE_m_s']['count'] = velocity_count
        if finite(slow_count) and velocity_count > slow_count and 'swing_velocity_RMSE_m_s' in output:
            output['swing_velocity_RMSE_m_s']['count'] = velocity_count - slow_count
    return output


def activity(row):
    info = row.get('gt_activity', row.get('activity', {}))
    if not isinstance(info, dict):
        return 'unknown'
    path = info.get('gt_root_horizontal_path_m', info.get('root_horizontal_path_m'))
    amp = info.get('gt_local_foot_RMS_left_right_m', info.get('body_local_foot_RMS_m'))
    if not finite(path):
        return 'unknown'
    if path >= .5:
        return 'moving'
    if path < .25 and isinstance(amp, list) and len(amp) == 2 and all(finite(x) and x < .08 for x in amp):
        return 'near_static'
    return 'mixed'


def summarize(rows):
    metrics = defaultdict(list)
    for row in rows:
        for name, metric in row['normalized_metrics'].items():
            metrics[name].append(metric)
    output = {}
    for name, items in sorted(metrics.items()):
        values = [i['value'] for i in items]
        counted = [i for i in items if i['count'] is not None]
        total = math.fsum(i['count'] for i in counted)
        output[name] = dict(windows_with_value=len(items), equal_window_mean=statistics.mean(values),
            count_weighted_mean=math.fsum(i['value'] * i['count'] for i in counted) / total if total else None,
            counted_windows=len(counted), effective_count_sum=total,
            median=quantile(values, .5), p10=quantile(values, .1), p90=quantile(values, .9),
            p95=quantile(values, .95), min=min(values), max=max(values))
        if 'rmse' in name.lower() and total:
            output[name]['count_pooled_RMSE'] = math.sqrt(math.fsum(i['value'] ** 2 * i['count'] for i in counted) / total)
    return dict(sampling_calls=len(rows), unique_recordings=len({r['recording'] for r in rows}), metrics=output)


def group(rows):
    recordings = sorted({r['recording'] for r in rows})
    by_recording = {recording: summarize([r for r in rows if r['recording'] == recording]) for recording in recordings}
    names = set().union(*(s['metrics'].keys() for s in by_recording.values())) if by_recording else set()
    macro = {}
    for name in sorted(names):
        summaries = [s['metrics'][name] for s in by_recording.values() if name in s['metrics']]
        counted = [s['count_weighted_mean'] for s in summaries if s['count_weighted_mean'] is not None]
        macro[name] = dict(recordings_with_value=len(summaries),
            equal_recording_of_equal_window_mean=statistics.mean(s['equal_window_mean'] for s in summaries),
            equal_recording_of_count_weighted_mean=statistics.mean(counted) if counted else None)
    return dict(overall=summarize(rows), recording_macro=macro, per_recording=by_recording)


def main(folder=None):
    folder = Path(folder) if folder is not None else Path(__file__).resolve().parent
    plan, plan_source = read_json(folder / 'PLAN.json')
    protocol, protocol_source = read_json(folder / 'PROTOCOL.json', optional=True)
    entries = plan.get('plan', plan.get('cases', []))
    issues = []
    case_ids = [str(e.get('case', i)) for i, e in enumerate(entries)]
    if len(set(case_ids)) != len(case_ids):
        issues.append('PLAN contains duplicate case IDs')
    if not entries or any(e.get('dataset') != 'rich' for e in entries):
        issues.append('PLAN must contain nonempty RICH-only entries')
    by_case = dict(zip(case_ids, entries))
    regimes = plan.get('regimes', ['clean', 'noisy'])
    if len(set(regimes)) != len(regimes) or set(regimes) != {'clean', 'noisy'}:
        issues.append('PLAN regime list must be exactly clean/noisy')
    expected = {(case, regime) for case in by_case for regime in regimes}
    rows_path = folder / 'rows.jsonl'
    raw = rows_path.read_bytes() if rows_path.exists() else b''
    lines = raw.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith(b'\n'))
    if incomplete_tail:
        lines.pop()
    unique, duplicates = {}, 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as error:
            issues.append(f'Completed line {i + 1} invalid JSON: {error}')
            continue
        if not isinstance(row, dict):
            issues.append(f'Completed line {i + 1} must be a JSON object')
            continue
        key = (str(row.get('case', row.get('case_index', ''))), row.get('regime'))
        if key not in expected:
            issues.append(f'Unexpected case/regime {key}')
            continue
        entry = by_case[key[0]]
        for identity in ('dataset', 'recording', 'sequence_id', 'start', 'source_first', 'case_seed'):
            if identity in entry and row.get(identity) != entry[identity]:
                issues.append(f'Identity mismatch {identity} on case/regime {key}')
        if 'checkpoint_step' not in row:
            issues.append(f'Missing checkpoint step on case/regime {key}')
        for metric in ('w_mpjpe_mm', 'root_relative_mpjpe_mm', 'pa_mpjpe_mm', 'mpjve_mm_s', 'acceleration_error_mm_s2'):
            value = row.get('benchmark_metrics', {}).get(metric)
            if not isinstance(value, dict) or not finite(value.get('value')) or not finite(value.get('count')) or value['count'] <= 0:
                issues.append(f'Missing/invalid benchmark metric {metric} on case/regime {key}')
        if key in unique:
            if unique[key] == row:
                duplicates += 1
            else:
                issues.append(f'Conflicting duplicate case/regime {key}')
            continue
        unique[key] = row
    normalized = [dict(row, case=int(case), dataset='rich', recording=by_case[case]['recording'],
        normalized_metrics=normalized_metrics(row), activity_stratum=activity(row))
        for (case, regime), row in unique.items()]
    checkpoint_steps = {r.get('checkpoint_step') for r in normalized}
    if len(checkpoint_steps) > 1:
        issues.append('Rows contain multiple checkpoint steps')
    if checkpoint_steps and 'checkpoint_step' in protocol and checkpoint_steps != {protocol['checkpoint_step']}:
        issues.append('Row checkpoint step disagrees with PROTOCOL')
    hashes = {r['checkpoint_sha256'] for r in normalized if 'checkpoint_sha256' in r}
    if len(hashes) > 1:
        issues.append('Rows contain multiple checkpoint hashes')
    if hashes and 'checkpoint_sha256' in protocol and hashes != {protocol['checkpoint_sha256']}:
        issues.append('Row checkpoint hash disagrees with PROTOCOL')
    if incomplete_tail:
        issues.append('Incomplete JSONL tail is excluded; final completion requires a complete tail')
    complete_rows = unique.keys() == expected
    complete = complete_rows and not issues and protocol.get('status') == 'completed'
    status = ('complete' if complete else 'integrity_issue' if issues else
              'all_rows_collected_waiting_final_checks' if complete_rows else
              'in_progress' if normalized else 'prepared_no_results')
    aggregates = {}
    for regime in regimes:
        selected = [r for r in normalized if r['regime'] == regime]
        aggregates[regime] = dict(all=group(selected), by_gt_activity={
            label: group([r for r in selected if r['activity_stratum'] == label])
            for label in ('moving', 'near_static', 'mixed', 'unknown')})
    paired = []
    for case in by_case:
        if (case, 'clean') in unique and (case, 'noisy') in unique:
            clean = normalized_metrics(unique[case, 'clean'])
            noisy = normalized_metrics(unique[case, 'noisy'])
            paired.append(dict(recording=by_case[case]['recording'], normalized_metrics={
                k: dict(value=noisy[k]['value'] - clean[k]['value'], count=None) for k in clean.keys() & noisy.keys()}))
    expected_recordings = len({e['recording'] for e in entries})
    result = dict(generated_at=datetime.now().astimezone().isoformat(), status=status, is_final_complete=complete,
        checkpoint_protocol=protocol, row_checkpoint_steps=sorted(checkpoint_steps, key=str), row_checkpoint_sha256=sorted(hashes),
        coverage=dict(expected_calls=len(expected), completed_calls=len(unique), all_rows_collected=complete_rows,
            expected_windows=len(entries), expected_recordings=expected_recordings,
            completed_recordings=len({r['recording'] for r in normalized}),
            windows_with_both_regimes=sum(all((case, regime) in unique for regime in regimes) for case in by_case),
            expected_windows_per_recording=dict(Counter(e['recording'] for e in entries)),
            source_sequences=plan.get('selection', {}).get('source_sequences'),
            source_coverage=plan.get('source_coverage', []),
            covered_unique_frames=plan.get('covered_unique_frames'),
            window_frame_occurrences=plan.get('window_frame_occurrences'),
            completed_calls_per_regime=dict(Counter(r['regime'] for r in normalized)),
            missing=[dict(case=case, regime=regime) for case, regime in sorted(expected - unique.keys())]),
        sources=[s for s in (plan_source, protocol_source,
            dict(path=str(rows_path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())) if s],
        read_audit=dict(incomplete_tail_ignored=incomplete_tail, identical_duplicate_rows_ignored=duplicates, integrity_issues=issues),
        aggregates=aggregates, paired_noisy_minus_clean=group(paired),
        per_window_metrics=[dict(case=r['case'], regime=r['regime'], recording=r['recording'],
            sequence_id=r.get('sequence_id'), start=r.get('start'), activity_stratum=r['activity_stratum'],
            metrics=r['normalized_metrics']) for r in normalized],
        interpretation_limits=[
            'All eligible RICH windows from this frozen local official-validation index; not TEST and not automatically another paper\'s benchmark protocol.',
            '192-frame, stride-64 windows overlap. Window count is not independent recording count; recording macro is reported separately.',
            'Clean/noisy are paired versions of identical windows, not independent samples.',
            'All-window count-weighted, equal-window and equal-recording means answer different weighting questions.',
            'Count weighting uses only explicit matching effective counts; uncounted extras remain equal-window only.',
            'Velocity/acceleration derivatives are evaluated inside each window; overlapping physical frames can occur repeatedly.',
            'GT foot speed<0.2m/s is a low-speed kinematic proxy, not ground-truth contact; sliding is not penetration.',
            'GT strata: moving root-horizontal path>=0.5m; near-static path<0.25m and both local-foot RMS<0.08m; otherwise mixed.',
            'Per-window quantiles are across window means, not per-frame/joint percentiles or confidence intervals.',
            'Root-relative and PA metrics remove specified transforms; global error remains separately reported.',
            'Compare other ego methods only after matching split, inputs, scene scope, body/joint definitions, fps, alignment and metric units.'])
    atomic_json(folder / 'SUMMARY.json', result)
    render_report(folder, result)
    print(json.dumps(dict(status=status, completed=len(unique), expected=len(expected), issues=issues), ensure_ascii=False))


def render_report(folder, result):
    coverage = result['coverage']
    metrics = [
        ('w_mpjpe_mm', '全局 MPJPE (mm)'),
        ('root_relative_mpjpe_mm', 'Root-relative MPJPE (mm)'),
        ('pa_mpjpe_mm', 'PA-MPJPE (mm)'),
        ('mpjve_mm_s', '关节速度误差 (mm/s)'),
        ('acceleration_error_mm_s2', '关节加速度误差 (mm/s²)'),
        ('root_position_error_mm', '根位置误差 (mm)'),
        ('root_orientation_deg', '根朝向误差 (°)'),
        ('body_local_foot_error_mm', '身体局部足误差 (mm)'),
        ('foot_world_error_mm', '世界足误差 (mm)'),
        ('foot_speed_during_GT_stance_m_s', 'GT 低速期足速 (m/s)'),
        ('foot_velocity_RMSE_m_s', '足速度 RMSE (m/s)'),
    ]
    def display(value, digits=3):
        return '—' if value is None else f'{value:.{digits}f}'
    lines = ['# RICH 全验证集报告', '',
        f"生成时间 {result['generated_at']}。状态 **{result['status']}**；已完成 **{coverage['completed_calls']}/{coverage['expected_calls']}** 次采样，**{coverage['windows_with_both_regimes']}/{coverage['expected_windows']}** 个窗口已完成 clean/noisy。计划覆盖 **{coverage['expected_recordings']}** 个录制。", '',
        '所有数目与身份均来自 PLAN.json 的实际条目集合。统一冻结 checkpoint；训练继续独立推进。clean/noisy 分开报告。', '']
    sources = coverage.get('source_coverage', [])
    if sources:
        short = [s for s in sources if not s.get('legal_windows')]
        uncovered_tails = sum(s.get('uncovered_frames', 0) for s in sources if s.get('legal_windows'))
        source_frames = sum(s.get('frames', 0) for s in sources)
        short_note = '；'.join(f"{s['sequence_id']} 仅 {s['frames']} 帧" for s in short)
        lines += [f"“全量”指当前索引所有合法 192 帧窗口：源 manifest 共 {len(sources)} 条、{source_frames} 帧，网格覆盖 {coverage.get('covered_unique_frames')} 个唯一帧。{short_note}，不足一窗；其余序列共 {uncovered_tails} 个尾帧不在固定网格。没有补尾窗或 padding。", '']
    if not result['is_final_complete']:
        lines += ['**尚未得到完整最终结果，下表仅为已完成部分；全体条目齐全、身份核对通过且协议标记 completed 后才视为完成。**', '']
    if result['read_audit']['integrity_issues']:
        lines += ['完整性问题：', '', *[f'- {x}' for x in result['read_audit']['integrity_issues']], '']
    for regime, data in result['aggregates'].items():
        all_data = data['all']
        summary = all_data['overall']
        lines += [f'## {regime}', '',
            f"已完成 {summary['sampling_calls']} 个窗口，涉及 {summary['unique_recordings']} 个录制。计数加权按指标实际有效计数；录制宏平均先在每个录制内按有效计数求均值，再让每个录制等权。没有显式 count 的额外指标不虚构 count。", '',
            '| 指标 | 全窗计数加权 | 等窗口 | 等录制宏平均 | 窗口中位数 | 窗口 P90 | 窗口最大 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for key, label in metrics:
            m = summary['metrics'].get(key)
            if m:
                macro = all_data['recording_macro'].get(key, {})
                macro_value = macro.get('equal_recording_of_count_weighted_mean')
                if macro_value is None:
                    macro_value = macro.get('equal_recording_of_equal_window_mean')
                lines.append(f"| {label} | {display(m['count_weighted_mean'])} | {display(m['equal_window_mean'])} | {display(macro_value)} | {display(m['median'])} | {display(m['p90'])} | {display(m['max'])} |")
        shown = {key for key, label in metrics}
        other = [key for key in summary['metrics'] if key not in shown and any(token in key.lower() for token in ('mpjpe', 'mpjve', 'velocity_error', 'accel'))]
        if other:
            lines += ['', '其他标准动作指标（字段原名；定义和单位见评估指标源码与协议）：', '', '| 字段 | 计数加权 | 等窗口 |', '|---|---:|---:|']
            for key in other:
                m = summary['metrics'][key]
                lines.append(f"| {key} | {display(m['count_weighted_mean'])} | {display(m['equal_window_mean'])} |")
        lines += ['', '### GT 动作分层与滑步', '',
            '分层用于解释全体结果，不删除任何窗口。moving：GT 根水平路径≥0.5m；near_static：路径<0.25m 且左右脚身体局部 RMS 均<0.08m；其余 mixed。', '',
            '| GT 分层 | 窗口数 | 录制数 | MPJPE (mm) | 局部足误差 (mm) | GT低速期预测足速 (m/s) | GT低速期预测>0.2m/s比例 | 根新增漂移 (m) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for label, strata in data['by_gt_activity'].items():
            m = strata['overall']['metrics']
            if not strata['overall']['sampling_calls']:
                continue
            def stat(key):
                value = m.get(key, {})
                return value.get('count_weighted_mean') if value.get('count_weighted_mean') is not None else value.get('equal_window_mean')
            slip = stat('pred_speed_over_point2_on_gt_low_speed_fraction')
            slip_text = '—' if slip is None else f'{slip:.1%}'
            lines.append(f"| {label} | {strata['overall']['sampling_calls']} | {strata['overall']['unique_recordings']} | {display(stat('mpjpe_mm'))} | {display(stat('body_local_foot_error_mm'))} | {display(stat('foot_speed_during_GT_stance_m_s'))} | {slip_text} | {display(stat('root_terminal_horizontal_error_change_m'))} |")
        lines += ['', '### 各录制 MPJPE', '', '| 录制 | 窗口数 | 全局 MPJPE 计数加权 (mm) |', '|---|---:|---:|']
        for recording, recording_data in sorted(all_data['per_recording'].items()):
            m = recording_data['metrics'].get('mpjpe_mm', {})
            lines.append(f"| {recording} | {recording_data['sampling_calls']} | {display(m.get('count_weighted_mean'))} |")
        lines.append('')
    lines += ['## 统计范围与对比口径', '',
        '- 本次覆盖冻结本地 RICH 官方 VAL 索引中的全部可用 192 帧窗口，步长 64 帧。相邻窗口有重叠，窗口数不等于独立录制数；同时保留等录制宏平均。',
        '- 全局、root-relative 和 PA 指标应分别解读。使用 GT 头部生成的 clean/noisy 观测不自动等同真实视频头轨迹输入。场景输入范围、身体骨架、帧率和划分亦需与对比方法匹配。',
        '- 足速是 GT 低速运动学代理，不是真接触标签；足高度包络与可观察平面代理不能替代全场景人体穿透率。近静态窗口会压低平均足速，须同时查看 moving 分层。',
        '- 未给出独立样本置信区间；窗口 P90 指每窗平均指标的分位数。完整计数、P10/P50/P90/P95、录制宏平均、逐窗指标及 clean/noisy 配对差异见 JSON。', '',
        '结果：[SUMMARY.json](SUMMARY.json)；计划：[PLAN.json](PLAN.json)；协议：[PROTOCOL.json](PROTOCOL.json)；监督趋势：[LOSS_TREND_zh.md](LOSS_TREND_zh.md)。']
    temporary = folder / f'REPORT_zh.md.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(folder / 'REPORT_zh.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().run_dir)
