"""Create a paired Markdown comparison from two exhaustive validation sweeps."""
import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ('mpjpe_cm', 'MPJPE (cm)', 'lower'),
    ('head_cm', '头部误差 (cm)', 'lower'),
    ('relative_head_cm', '头路径相对误差 (cm)', 'lower'),
    ('slowest_foot_horizontal_cm_per_frame', '较慢脚水平速度 (cm/帧)', 'lower'),
    ('gt_stance_slide_cm_frame', 'GT 支撑相位脚滑 (cm/帧)', 'lower'),
    ('foot_relative_speed_ratio', '脚相对骨盆速度 / GT', 'target'),
    ('root_path_length_ratio', '根部路径 / GT', 'target'),
    ('foot_separation_rms_cm', '脚间距变化 RMS (cm)', 'raw'),
    ('both_feet_moving_fraction', '双脚同时移动比例（观察量）', 'raw'),
    ('support_floating_m', '悬浮代理 (m)', 'lower'),
    ('support_penetration_m', '穿透代理 (m)', 'lower'),
    ('coordination_relation_mse', '协调关系误差', 'lower'),
    ('coordination_amplitude_m2', '协调幅度误差 (m²)', 'lower'),
)


def load(path):
    rows = [json.loads(line) for line in (Path(path) / 'rows.jsonl').read_text().splitlines() if line]
    key = lambda r: (r['group'], r['identity']['sequence_id'], r['identity']['source_start_30fps'])
    out = {key(r): r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f'duplicate rows in {path}')
    return out


def stats(rows):
    if not rows:
        return None
    result = {'n': len(rows), 'sequences': len({(r['group'], r['identity']['sequence_id']) for r in rows})}
    for key, _, _ in METRICS:
        result[key] = float(np.mean([r['metrics'][key] for r in rows]))
    return result


def fmt(value, key):
    if key.endswith('fraction') or key.endswith('ratio'):
        return f'{value:.3f}'
    if key in ('coordination_relation_mse', 'coordination_amplitude_m2'):
        return f'{value:.5f}'
    return f'{value:.3f}'


def section(name, old, new):
    lines = [f'### {name}（{len(old)} 个窗口）', '', '| 指标 | 30k | 55k | 变化 |', '|---|---:|---:|---:|']
    a, b = stats(old), stats(new)
    for key, label, direction in METRICS:
        x, y = a[key], b[key]
        if direction == 'target':
            # For ratios, report distance from the physically meaningful target 1.
            delta = abs(y - 1.) - abs(x - 1.)
            change = f'{delta:+.3f}（距1的差）'
        else:
            delta = (y - x) / max(abs(x), 1e-9) * 100
            change = f'{delta:+.1f}%'
        lines.append(f'| {label} | {fmt(x, key)} | {fmt(y, key)} | {change} |')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = load(args.old), load(args.new)
    if set(old) != set(new):
        raise ValueError(f'window identity mismatch: old={len(old)} new={len(new)}')
    for key in old:
        if old[key]['activity'] != new[key]['activity']:
            raise ValueError(f'GT activity strata mismatch: {key}')
    old_rows, new_rows = list(old.values()), list(new.values())
    lines = ['# SceneMI 原始 30k vs 脚步方案 55k：全量验证对比', '',
             '协议完全一致：validation、128帧（6.4秒）、头部 mask、ego 可见场景、DDIM20、seed777；'
             '共 3,310 个审计候选窗口、112 条序列，按 `(group, sequence_id, source_start_30fps)` 逐窗口配对。', '',
             '变化百分比按 `(55k-30k)/|30k|`；脚速度、误差和滑动率越低越好。脚相对骨盆速度比和根部路径比报告离目标 1 的变化，'
             '因为过低也可能代表动作冻结；脚间距与双脚同时移动是原始观察量，必须结合 GT 和播放判断。悬浮/穿透是 GT 高度代理，不是真实 SDF 碰撞。', '']
    lines += section('总体（窗口等权）', old_rows, new_rows) + ['']
    for group in ('trumans', 'camera_wearer', 'interactee'):
        a = [r for r in old_rows if r['group'] == group]; b = [r for r in new_rows if r['group'] == group]
        lines += section(group, a, b) + ['']
    for strata in ('walking_proxy', 'upper_body_active_proxy', 'low_activity_proxy', 'other_proxy'):
        a = [r for r in old_rows if r['activity'][strata]]; b = [r for r in new_rows if r['activity'][strata]]
        lines += section(strata, a, b) + ['']
    lines += ['## 序列等权摘要', '', '| 子集 | 30k MPJPE | 55k MPJPE | 30k 较慢脚 | 55k 较慢脚 | 30k 脚速比 | 55k 脚速比 |', '|---|---:|---:|---:|---:|---:|---:|']
    subsets = [('overall', old_rows, new_rows)]
    for group in ('trumans', 'camera_wearer', 'interactee'):
        subsets.append((group, [r for r in old_rows if r['group'] == group], [r for r in new_rows if r['group'] == group]))
    for name, a, b in subsets:
        # Sequence-equal weighting, unlike the main tables.
        def seqmean(rows, key):
            groups = {}
            for r in rows: groups.setdefault((r['group'], r['identity']['sequence_id']), []).append(r['metrics'][key])
            return float(np.mean([np.mean(v) for v in groups.values()]))
        lines.append(f'| {name} | {seqmean(a,"mpjpe_cm"):.3f} | {seqmean(b,"mpjpe_cm"):.3f} | {seqmean(a,"slowest_foot_horizontal_cm_per_frame"):.3f} | {seqmean(b,"slowest_foot_horizontal_cm_per_frame"):.3f} | {seqmean(a,"foot_relative_speed_ratio"):.3f} | {seqmean(b,"foot_relative_speed_ratio"):.3f} |')
    lines += ['', '## 解释边界', '', '- 55k 是从已修复脚步方案继续训练后的权重；本表证明相对最初 30k 的整体变化，不单独证明新增全身协调损失的收益。', '- 两个版本使用同一组窗口和同一噪声 seed，差异可做严格配对；窗口之间有重叠，不能当作独立样本。', '- CondMDI/旧 exo 微调模型不放入这张主表：其输入为 1 秒 GT 历史、2 秒未来、固定 Kinect exo 协议和 DDIM50，与这里的头部 mask、6.4 秒 ego-visible 任务不等价；应另列为外部参考。', '']
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()
