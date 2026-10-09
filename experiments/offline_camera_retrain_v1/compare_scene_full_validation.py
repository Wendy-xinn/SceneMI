"""Strictly paired scene-on/off report from exhaustive validation archives."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from experiments.offline_camera_retrain_v1.evaluate_full_validation import summarize
from experiments.offline_camera_retrain_v1.gallery import write_gallery


SPECS = (
    ('mpjpe_cm', 'MPJPE', 'cm', 1, True),
    ('head_cm', '真值头位置误差', 'cm', 1, True),
    ('relative_head_cm', '头路径相对误差', 'cm', 1, True),
    ('slowest_foot_horizontal_cm_per_frame', '较慢脚水平位移', 'cm/帧', 1, True),
    ('gt_stance_slide_cm_frame', 'GT支撑相位脚滑', 'cm/帧', 1, True),
    ('foot_relative_speed_ratio', '脚相对骨盆速度/GT', '比值，参考1', 1, False),
    ('root_path_length_ratio', '根部路径长度/GT', '比值，参考1', 1, False),
    ('foot_separation_rms_cm', '脚间距动态RMS', 'cm', 1, False),
    ('gt_foot_separation_rms_cm', 'GT脚间距动态RMS', 'cm', 1, False),
    ('both_feet_moving_fraction', '双脚同时明显移动帧比例', '%', 100, False),
    ('support_floating_m', '悬浮高度代理', 'cm', 100, True),
    ('support_penetration_m', '穿透高度代理', 'cm', 100, True),
    ('coordination_relation_mse', '全身关系误差', 'MSE', 1, True),
    ('coordination_amplitude_m2', '全身活动幅度误差', 'm²', 1, True),
)


def row_key(row):
    return row['group'], row['identity']['sequence_id'], row['identity']['source_start_30fps']


def verify_pair(on_path, off_path):
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (on_path, off_path)]
    for path in (on_path, off_path):
        status = json.loads((path / 'status.json').read_text())
        if status['status'] != 'completed' or not status['full_coverage']:
            raise ValueError(f'Incomplete full sweep: {path}')
    on, off = manifests
    if not on.get('use_scene', True) or off.get('use_scene', True):
        raise ValueError('Expected scene-on then scene-off')
    for key in ('checkpoint_sha256', 'checkpoint_step', 'cumulative_step', 'skeleton_profile',
                'split', 'length', 'fps', 'control', 'ddim_steps', 'seed', 'seed_policy', 'windows'):
        if on[key] != off[key]:
            raise ValueError(f'Pairing protocol mismatch: {key}')
    maps = []
    for path, manifest in zip((on_path, off_path), manifests):
        rows = [json.loads(s) for s in (path / 'rows.jsonl').read_text().splitlines() if s]
        mapping = {row_key(r): r for r in rows}
        if len(mapping) != len(rows) or len(rows) != manifest['eligible_windows']:
            raise ValueError('Duplicate or missing windows')
        maps.append(mapping)
    if maps[0].keys() != maps[1].keys():
        raise ValueError('Different window identities')
    for key, row in maps[0].items():
        other = maps[1][key]
        if row['activity'] != other['activity'] or row['identity'] != other['identity']:
            raise ValueError('Different GT activity/identity')
        for metric in row['metrics']:
            if not np.isfinite(row['metrics'][metric]) or not np.isfinite(other['metrics'][metric]):
                raise ValueError('Nonfinite metric')
        with np.load(on_path / row['motion_file']) as a, np.load(off_path / other['motion_file']) as b:
            np.testing.assert_array_equal(a['truth'], b['truth'])
            np.testing.assert_array_equal(a['camera'], b['camera'])
    return on, maps[0], maps[1]


def paired_stats(a, b):
    result = {}
    for key, _, _, _, lower in SPECS:
        if not lower:
            continue
        delta = np.array([x['metrics'][key] - y['metrics'][key] for x, y in zip(a, b)])
        sequence = {}
        for row, difference in zip(a, delta):
            sequence.setdefault(row_key(row)[:2], []).append(float(difference))
        means = np.array([np.mean(v) for v in sequence.values()])
        result[key] = dict(scene_on_minus_off_mean=float(delta.mean()),
                           improved_windows=int((delta < 0).sum()), windows=len(delta),
                           improved_sequences=int((means < 0).sum()), sequences=len(means))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-scene', type=Path, required=True)
    parser.add_argument('--without-scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-gallery', type=Path, required=True)
    args = parser.parse_args()
    manifest, on, off = verify_pair(args.with_scene, args.without_scene)
    sections = [('全部窗口', list(on))]
    for group in ('trumans', 'camera_wearer', 'interactee'):
        sections.append((group, [k for k in on if k[0] == group]))
    for label, flag in [('走动代理子集', 'walking_proxy'), ('上肢活动代理子集', 'upper_body_active_proxy'),
                        ('低活动代理子集', 'low_activity_proxy'), ('其他窗口', 'other_proxy')]:
        sections.append((label, [k for k in on if on[k]['activity'][flag]]))
    notes = ('同一脚步方案55k权重，仅推理时关闭全局occupancy场景嵌入和局部BPS；不是分别训练的有/无场景模型。'
             '头部mask、相机、GT、窗口和初始噪声不变。训练时已有10%场景dropout。'
             '完整窗口和GT/相机数组逐项一致检查通过。旧有场景manifest未记录use_scene字段，'
             '其原脚本默认开启场景；三个来源的回放检查逐数复现原有场景预测。'
             '仅一个采样种子；窗口重叠，窗口胜率不是独立样本显著性检验。'
             '高度与接触指标是GT关节代理，不是真实场景碰撞；更低脚速必须结合活动幅度，防止奖励冻结。'
             '相关性/幅度误差也不是独立的人类自然性评分，活动分层不是人工动作标签。')
    title = '55k 有场景 / 无场景：全量配对验证'
    md = [f'# {title}', '', f'覆盖 {len(on):,} 个窗口、{len({k[:2] for k in on})} 条序列。'
          f'128帧/20Hz，头部条件，DDIM{manifest["ddim_steps"]}，seed{manifest["seed"]}。', '', notes, '',
          '“有−无”小于0表示数值降低。比值/活动观察量并非越小越好，需对照GT；均值接近1不代表每窗都接近1。', '']
    body = [f'<h1>{title}</h1>', f'<p>{html.escape(md[2])}</p>', f'<p>{html.escape(notes)}</p>',
            '<p><a href="comparison.csv">下载表格CSV</a> · <a href="comparison.json">完整数据JSON</a> · '
            '<a href="fixed_walking.html">原来的5段走动：有/无场景</a> · '
            '<a href="fixed12.html">原固定12段：有/无场景</a> · '
            '<a href="largest_effects.html">场景有帮助和有退步的极端片段</a></p>']
    flat, reports = [], {}
    for name, keys in sections:
        a, b = [on[k] for k in keys], [off[k] for k in keys]
        if not a:
            continue
        sa, sb = summarize(a), summarize(b)
        paired = paired_stats(a, b)
        reports[name] = dict(with_scene=sa, without_scene=sb, paired=paired)
        for aggregation, agg_label in [('window_mean', '窗口等权'), ('sequence_mean', '序列等权')]:
            heading = f'{name} · {agg_label}（{len(keys)}窗 / {sa["sequences"]}序列）'
            md += [f'## {heading}', '', '| 指标 | 单位 | 有场景 | 无场景 | 有−无 | 相对无场景变化 |',
                   '|---|---|---:|---:|---:|---:|']
            body += [f'<h2>{heading}</h2><table><tr><th>指标</th><th>单位</th><th>有场景</th><th>无场景</th><th>有−无</th><th>相对无场景变化</th></tr>']
            for key, label, unit, scale, lower in SPECS:
                x, y = sa[aggregation][key] * scale, sb[aggregation][key] * scale
                percent = (x - y) / abs(y) * 100 if abs(y) > 1e-12 and lower else None
                change = f'{percent:+.1f}%' if percent is not None else '观察量'
                precision = 6 if key == 'coordination_amplitude_m2' else 4 if key == 'coordination_relation_mse' else 3
                values = [label, unit, f'{x:.{precision}f}', f'{y:.{precision}f}', f'{x-y:+.{precision}f}', change]
                md.append('| ' + ' | '.join(values) + ' |')
                body.append('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in values) + '</tr>')
                flat.append(dict(subset=name, aggregation=aggregation, metric=key, unit=unit,
                                 with_scene=x, without_scene=y, delta_on_minus_off=x-y, relative_percent=percent))
            md += ['']; body += ['</table>']
        text = '；'.join(f'{label}：有场景更好的窗口 {paired[key]["improved_windows"]}/{len(keys)}，序列均值 {paired[key]["improved_sequences"]}/{sa["sequences"]}'
                        for key, label in [('mpjpe_cm', 'MPJPE'), ('head_cm', '头误差'),
                                           ('slowest_foot_horizontal_cm_per_frame', '较慢脚位移')])
        md += [text, '']; body += [f'<p>{text}</p>']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'REPORT_zh.md').write_text('\n'.join(md))
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title>'
        '<style>body{font:16px system-ui;margin:30px;max-width:1400px}table{border-collapse:collapse;margin-bottom:20px}td,th{padding:8px 14px;border:1px solid #ccc;text-align:right}td:first-child{text-align:left}a{color:#176ac2}p{line-height:1.65}</style>' + '\n'.join(body))
    (args.output / 'comparison.json').write_text(json.dumps(dict(notes=notes, sections=reports,
        checkpoint_sha256=manifest['checkpoint_sha256'], windows=len(on)), indent=2, allow_nan=False))
    with (args.output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    # Preserve original diagnostic clips rather than replacing them with nicer results.
    page = args.fixed_gallery.read_text()
    fixed = json.loads(page.split('const data=', 1)[1].split(';\nconst $=', 1)[0])['cases']
    fixed_keys = [(c['group'], c['identity']['sequence_id'], c['identity']['source_start_30fps']) for c in fixed]
    if any(k not in on for k in fixed_keys):
        raise ValueError('Original diagnostic clip missing from full sweep')
    delta_order = sorted(on, key=lambda k: on[k]['metrics']['mpjpe_cm'] - off[k]['metrics']['mpjpe_cm'])
    selections = [('fixed12', fixed_keys), ('fixed_walking', [k for k in fixed_keys if on[k]['activity']['walking_proxy']]),
                  ('largest_effects', delta_order[:4] + delta_order[-4:])]
    for label, keys in selections:
        cases = []
        for key in keys:
            a, b = on[key], off[key]
            with np.load(args.with_scene / a['motion_file']) as x, np.load(args.without_scene / b['motion_file']) as y:
                cases.append(dict(group=a['group'], identity=a['identity'], camera=x['camera'].round(4).tolist(),
                    tracks={'GT': x['truth'].round(4).tolist(), '55k_with_scene': x['prediction'].round(4).tolist(),
                            '55k_without_scene': y['prediction'].round(4).tolist()},
                    metrics={'55k_with_scene': a['metrics'], '55k_without_scene': b['metrics']}))
        write_gallery(args.output / f'{label}.html', cases, f'55k scene on/off · {label} · full-sweep seed777')
    print(json.dumps(reports['全部窗口']['paired'], indent=2))
    print(args.output / 'index.html')


if __name__ == '__main__':
    main()
