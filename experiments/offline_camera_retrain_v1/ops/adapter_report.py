"""CPU-only factual adapter-stage report; run as a module with --run [--watch]."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import shutil
import time
from urllib.parse import urlparse


def read(run, name):
    path = run / name
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def moment(value):
    return datetime.fromisoformat(value) if value else None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def terminal_error(status, job, launch):
    """A previous supervisor's JOB must not terminate a new STATUS/PID."""
    if launch.get('launched_at') and status.get('updated_at') and moment(status['updated_at']) < moment(launch['launched_at']):
        return None
    if status.get('status') in ('failed', 'stopped'):
        return status.get('error', f"实际状态为 {status['status']}")
    if job.get('status') == 'failed' and job.get('child_pid') == status.get('pid') and status.get('pid') is not None and job.get('supervisor_pid') == launch.get('supervisor_pid'):
        return f"当前任务退出，returncode={job.get('returncode')}，详见 run.log"
    return None


def write(run, body):
    path = run / 'RESULTS_zh.md'
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(body + '\n')
    os.replace(temporary, path)


def number(value):
    return '无有效样本' if value is None else f'{value:.3f}'


def pair(first, final, passed):
    return f'{number(first)} → {number(final)} {"✓" if passed else "失败"}'


def failures(value, prefix='验收'):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            if child is False and (key == 'passed' or key.endswith(('_pass', '_exact'))):
                result.append(f'{prefix}.{key}')
            elif isinstance(child, (dict, list)):
                result.extend(failures(child, f'{prefix}.{key}'))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            label = '/'.join(str(child[k]) for k in ('dataset', 'regime') if k in child) if isinstance(child, dict) else ''
            result.extend(failures(child, f'{prefix}[{label or index}]'))
    return result


def render(run):
    status, comparison, gates, cpu, alignment, protocol, launch = [read(run, name) for name in
        ('STATUS.json', 'COMPARISON.json', 'ACCEPTANCE.json', 'CPU_RECHECK.json', 'ALIGNMENT_DIAGNOSTICS.json', 'training/protocol.json', 'LAUNCH.json')]
    require(status.get('status') == 'completed', '尚未完成，拒绝生成结论')
    require(cpu.get('status') == 'independent_CPU_primary_metrics_passed' and cpu.get('CUDA_initialized') is False and cpu.get('GPU_used_for_recheck') is False and cpu.get('matched_input_arrays') is True, '独立 CPU 核验尚未通过')
    require(comparison.get('status') == 'matched_primary_metrics_CPU_verified' and comparison.get('independent_CPU_metric_recheck_pending') is False and comparison.get('TEST_used') is False, '缺少已核验开发验证比较')
    require(comparison.get('CPU_recheck_sha256') == digest(run / 'CPU_RECHECK.json') and comparison.get('acceptance') == gates, '比较、验收或 CPU 记录身份不一致')
    require(cpu.get('final_checkpoint_sha256') == digest(run / 'training/last.pt'), '最终 checkpoint SHA 不匹配')
    require(cpu.get('frozen_base_sha256') == status.get('frozen_base_sha256') == protocol.get('frozen_base_sha256') and cpu.get('frozen_base_sha256'), '冻结主干身份不一致')
    require(gates.get('EGO_serialized_outputs_exact') == cpu.get('EGO_serialized_outputs_exact') == status.get('EGO_serialized_outputs_exact'), 'EGO 一致性记录不同')
    for path, expected in cpu.get('files_sha256', {}).items():
        require(digest(Path(path)) == expected, f'验证文件 SHA 不匹配：{path}')
    steps = status.get('completed_stage_updates')
    trace = [json.loads(line) for line in (run / 'training/trace.jsonl').read_text().splitlines() if line.strip()]
    require(isinstance(steps, int) and steps > 0 and steps == protocol.get('completed_steps') == protocol.get('planned_steps'), '更新预算未完成')
    require([row['step'] for row in trace] == list(range(1, steps + 1)), '实际更新记录缺失或重复')
    require(all(math.isfinite(row['update_s']) and row['update_s'] >= 0 for row in trace), '更新耗时无效')
    actual_s = sum(row['update_s'] for row in trace)
    elapsed = (moment(status['updated_at']) - moment(launch['launched_at'])).total_seconds()
    require(elapsed >= 0, '启动/完成时间顺序错误')
    passed = gates.get('status') == 'pass' and not failures(gates)
    lines = [f'本轮已完成 {steps} 次内部 adapter 更新；工程验收{"通过" if passed else "未通过"}。不自动替换正式模型。',
        f'本报告只使用本轮固定开发验证：{comparison.get("per_window_rows")} 次匹配预测，未使用 TEST 选择模型或调参。',
        f'本次启动：{launch["launched_at"]}；最终复核完成：{status["updated_at"]}。',
        f'实际更新累计 {actual_s / 60:.1f} 分钟；本次启动至最终 CPU 复核完成 {elapsed / 60:.1f} 分钟，后者包含准备、等待和验证。',
        '', '|输入|隐藏关节 MPJPE：适配前 → 后（mm）|等数据集宏平均门槛|', '|---|---:|---|']
    for row in gates['hidden_macro']:
        lines.append(f'|{row["regime"]}|{number(row["baseline_hidden_mm"])} → {number(row["final_hidden_mm"])}|{"通过" if row["passed"] else "失败"}（至少改善 5%）|')
    lines += ['', '各数据集内按有效关节数加权；脚部 RMS/stride 误差按窗口平均。低速脚指标使用 GT 关节速度代理，不是实测接触。',
        '逐数据集门槛：隐藏关节最多增加 max(5mm,5%)；可见关节 max(1mm,2%)；低速脚速度 max(0.01m/s,5%)；双足 RMS/stride 误差 max(5mm,10%)。', '',
        '|数据集/输入|隐藏关节 mm|可见关节 mm|低速脚速度 m/s|双足运动 RMS 误差 mm|stride 误差 mm|', '|---|---:|---:|---:|---:|---:|']
    for row in gates['groups']:
        motion = row['motion_guards']
        cells = [f'{row["dataset"]}/{row["regime"]}（{row["windows"]}窗）',
            pair(row['baseline_hidden_mm'], row['final_hidden_mm'], row['hidden_no_material_regression_pass']),
            pair(row['baseline_visible_mm'], row['final_visible_mm'], row['visible_guard_pass']),
            pair(row['baseline_low_speed_foot_m_s'], row['final_low_speed_foot_m_s'], row['foot_speed_guard_pass'])]
        cells += [pair(motion[key]['baseline'], motion[key]['final'], motion[key]['passed']) for key in ('bilateral_motion_rms_error_mm', 'bilateral_stride_change_abs_error_mm')]
        lines.append('|' + '|'.join(cells) + '|')
    failed = failures(gates)
    lines += ['', '未通过项：' + ('；'.join(failed) if failed else ('无。' if passed else '验收总状态不是 pass。'))]
    checks = read(run, 'prepared/input_checks.json')
    lines += ['', f'EGO 保存输出逐值一致：{cpu.get("EGO_serialized_outputs_exact")}；非零 adapter 的 EGO 旁路预检：{checks.get("nonzero_adapter_EGO_bypass_exact", "无记录")}；无人体动作标签的独立 EXO 加载预检：{checks.get("frozen_EXO_without_body_labels", "无记录")}。',
        f'冻结主干 SHA：`{cpu["frozen_base_sha256"]}`；最终完整 checkpoint SHA：`{cpu["final_checkpoint_sha256"]}`。']
    deployment = read(run, 'DEPLOYMENT_LOAD_CHECK.json')
    final_load = steps == 1000 and deployment.get('step') == 1000 and deployment.get('checkpoint_sha256') == cpu['final_checkpoint_sha256'] and deployment.get('status') == 'passed'
    lines.append('最终独立部署加载：' + ('[已通过](DEPLOYMENT_LOAD_CHECK.json)。' if final_load else '待核验；首步或其他 SHA 的加载记录不代表最终模型。'))
    rows = [row for row in alignment.get('rows', []) if row.get('mode') == 'exo' and row.get('source') == 'metrics']
    lines += ['', '尺度诊断仅解释本轮 EXO 开发窗口：每窗口一个固定刚体/相似变换，GT 对齐不用于修改预测。表中先按数据集平均，再对数据集等权；RMS 是平方误差开方，MPJPE 是欧氏距离平均，两者不可替代。', '',
        '|阶段/输入|世界 RMS mm|GT 刚体 RMS mm|GT 相似 RMS mm|可见拟合尺度|', '|---|---:|---:|---:|---:|']
    for stage in ('baseline', 'final'):
        for regime in ('clean', 'noisy'):
            chosen = [row for row in rows if row['model'] == stage and row['regime'] == regime]
            require(chosen, '缺少本轮尺度诊断')
            def macro(key):
                groups = [[row['metrics'][key] for row in chosen if row['dataset'] == dataset and row['metrics'][key] is not None] for dataset in sorted({row['dataset'] for row in chosen})]
                return sum(sum(group) / len(group) for group in groups) / len(groups) if all(groups) else None
            values = [macro(key) for key in ('raw_gt_world_rms_mm', 'diagnostic_gt_window_rigid_fit_weighted_rms_residual_mm', 'diagnostic_gt_window_similarity_fit_weighted_rms_residual_mm', 'observed_window_robust_similarity_fit_scale')]
            lines.append('|' + stage + '/' + regime + '|' + '|'.join(map(number, values)) + '|')
    server = read(run, 'visuals/viser/SERVER.json')
    try:
        os.kill(server['pid'], 0)
        url = urlparse(server['url'])
        require(url.hostname in ('localhost', '127.0.0.1'), '只接受本机查看器')
        with socket.create_connection((url.hostname, url.port), timeout=.5):
            viewer = f'[当前 Viser]({server["url"]})'
    except (KeyError, OSError, ValueError):
        viewer = '[已准备的回放清单](VIEWER_CASES.json)（未确认服务正在运行）'
    lines += ['', viewer + '。原始证据：[验收](ACCEPTANCE.json)、[匹配比较](COMPARISON.json)、[独立 CPU 复核](CPU_RECHECK.json)、[尺度诊断](ALIGNMENT_DIAGNOSTICS.json)。']
    storage = read(run, 'STORAGE.json')
    storage['after'] = dict(checked_at=datetime.now().astimezone().isoformat(), stage='after completed stage and CPU recheck',
        disks={path:dict(zip(('total', 'used', 'free'), shutil.disk_usage(path))) for path in ('/', '/mnt/d', '/mnt/e') if Path(path).exists()},
        retained_artifacts={str(path.relative_to(run)):path.stat().st_size for path in [run/'training/last.pt', *run.glob('*/validation_latest.npz')] if path.exists()})
    target = run/'STORAGE.json'
    temporary = target.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(storage, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, target)
    write(run, '\n'.join(lines))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    while True:
        status, job, launch = [read(args.run, name) for name in ('STATUS.json', 'JOB.json', 'LAUNCH.json')]
        error = terminal_error(status, job, launch)
        stale = launch.get('launched_at') and status.get('updated_at') and moment(status['updated_at']) < moment(launch['launched_at'])
        if error:
            write(args.run, f'本轮未形成可验收结果，未宣称通过。\n\n实际错误：`{error}`\n\n请查阅 [状态](STATUS.json)、[当前任务](JOB.json) 和 [运行日志](run.log)。')
            return 1
        if status.get('status') == 'completed' and not stale:
            try:
                return render(args.run)
            except (AssertionError, OSError, ValueError, KeyError, TypeError) as exc:
                write(args.run, f'结果身份或复核未通过，拒绝生成成功结论。\n\n实际错误：`{exc}`')
                return 1
        if not args.watch:
            raise RuntimeError(f'实验尚未完成：{status.get("status", "无状态")}，拒绝生成结论')
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
