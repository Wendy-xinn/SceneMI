#!/usr/bin/env python3
"""CPU-only snapshot of existing training records, with no training imports."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1] / 'native30_gait_v2_ego55k_oct15'


def snapshot(path):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    complete = [line for line in lines if line.endswith(b'\n')]
    return [json.loads(line) for line in complete], dict(
        path=str(path), snapshot_bytes=len(raw),
        snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        complete_records=len(complete), incomplete_tail_ignored=len(lines) != len(complete))


def stratum(record):
    pair = '+'.join(sorted(i['dataset'] for i in record['identities']))
    return f"{pair}|low_t={sum(t < 100 for t in record['timesteps'])}|scene={int(record['scene'])}"


def main():
    trace, provenance = snapshot(RUN / 'training/trace.jsonl')
    effective_path = RUN / 'prepared/effective_config.json'
    effective_raw = effective_path.read_bytes()
    effective = json.loads(effective_raw)
    obs_co = effective['observation_coefficients']
    geometry_co = effective['config']['geometry']['coefficients']
    gait_co = effective['config']['gait_supervision']['coefficients']
    raw_gait_co = effective['config']['gait_supervision']['raw_gait_weight']
    assert [r['step'] for r in trace] == list(range(1, trace[-1]['step'] + 1)), 'Missing/repeated updates'
    groups = defaultdict(list)
    for row in trace:
        base, observation, geometry = (row['terms'][name] for name in ('base', 'observation', 'geometry'))
        values = {'total': row['loss'], 'weighted/base_body_and_gait': base['total'],
                  'weighted/observation': math.fsum(obs_co[k] * v for k, v in observation.items()),
                  'weighted/scene_geometry': math.fsum(geometry_co[k] * v for k, v in geometry.items())}
        values.update({f'raw/base/{k}': v for k, v in base.items()})
        values.update({f'raw/observation/{k}': v for k, v in observation.items()})
        values.update({f'raw/geometry/{k}': v for k, v in geometry.items()})
        values.update({f'weighted/observation/{k}': obs_co[k] * v for k, v in observation.items()})
        values.update({f'weighted/geometry/{k}': geometry_co[k] * v for k, v in geometry.items()})
        values['weighted/gait_final_total'] = base['gait_total']
        values['weighted/gait_raw_total'] = raw_gait_co * base['gait_raw_total']
        for key, coefficient in gait_co.items():
            values[f'weighted/gait_final/{key}'] = coefficient * base[f'gait_{key}']
            values[f'weighted/gait_raw/{key}'] = raw_gait_co * coefficient * base[f'gait_raw_{key}']
        values['audit/three_group_sum_minus_total'] = math.fsum(values[f'weighted/{k}'] for k in
            ('base_body_and_gait', 'observation', 'scene_geometry')) - row['loss']
        values['audit/final_gait_sum_minus_total'] = math.fsum(co * base[f'gait_{k}'] for k, co in gait_co.items()) - base['gait_total']
        values['audit/raw_gait_sum_minus_total'] = math.fsum(co * base[f'gait_raw_{k}'] for k, co in gait_co.items()) - base['gait_raw_total']
        groups[(row['step'] - 1) // 1000].append(dict(step=row['step'], stratum=stratum(row), values=values,
            gradient_norm=row['gradient_norm'], low_noise_samples=sum(t < 100 for t in row['timesteps']),
            scene=row['scene'], datasets='+'.join(sorted(i['dataset'] for i in row['identities']))))
    complete_groups = [g for g in groups.values() if len(g) == 1000]
    common = set.intersection(*(set(r['stratum'] for r in g) for g in complete_groups))
    mixture = Counter(r['stratum'] for g in complete_groups for r in g if r['stratum'] in common)
    weights = {k: v / sum(mixture.values()) for k, v in mixture.items()}
    keys = list(trace[0]['terms']['base'])
    del keys  # Values below include the full raw and weighted logged term sets.
    blocks = []
    nonfinite = []
    for group in groups.values():
        values = defaultdict(list)
        stratified = defaultdict(lambda: defaultdict(list))
        for r in group:
            for key, value in r['values'].items():
                if not math.isfinite(value):
                    nonfinite.append(dict(step=r['step'], term=key))
                    continue
                values[key].append(value)
                stratified[key][r['stratum']].append(value)
        terms = {}
        for key, raw_values in values.items():
            used = set(stratified[key]).intersection(weights)
            coverage = math.fsum(weights[s] for s in used)
            standardized = math.fsum(weights[s] * statistics.mean(stratified[key][s]) for s in used) / coverage
            ordered = sorted(raw_values)
            terms[key] = dict(mean=statistics.mean(raw_values), median=statistics.median(raw_values),
                p90=ordered[int(.9 * (len(ordered) - 1))],
                composition_standardized_mean=standardized, standardization_weight_coverage=coverage)
        blocks.append(dict(first_step=group[0]['step'], last_step=group[-1]['step'], count=len(group),
            complete_1000_update_block=len(group) == 1000, terms=terms,
            dataset_pair_counts=dict(Counter(r['datasets'] for r in group)),
            low_t_sample_fraction=sum(r['low_noise_samples'] for r in group) / (2 * len(group)),
            scene_off_fraction=sum(not r['scene'] for r in group) / len(group),
            gradient_norm_median_before_clip=statistics.median(r['gradient_norm'] for r in group),
            nonfinite_gradient_count=sum(not math.isfinite(r['gradient_norm']) for r in group)))
    full = [b for b in blocks if b['complete_1000_update_block']]
    recent = full[-4:]
    comparisons = {}
    for key in recent[-1]['terms']:
        means = [b['terms'][key]['mean'] for b in recent]
        standardized = [b['terms'][key]['composition_standardized_mean'] for b in recent]
        baseline, previous, latest = means[0], means[-2], means[-1]
        delta = None if not baseline else 100 * (latest / baseline - 1)
        latest_delta = None if not previous else 100 * (latest / previous - 1)
        if not baseline and not latest:
            classification = 'zero'
        elif all(b < a for a, b in zip(means, means[1:])):
            classification = 'decreased_in_all_recent_complete_blocks'
        elif delta is not None and abs(delta) < 3:
            classification = 'approximately_flat_with_fluctuations'
        elif delta is not None and delta < 0:
            classification = 'overall_decrease_with_fluctuations'
        else:
            classification = 'overall_increase_with_fluctuations'
        comparisons[key] = dict(means=means, standardized_means=standardized,
            oldest_to_latest_percent=delta, previous_to_latest_percent=latest_delta,
            descriptive_classification=classification)
    validation, validation_provenance = snapshot(RUN / 'training/validation.jsonl')
    rich_fixed = []
    for row in validation:
        for case in row['cases']:
            if case['dataset'] == 'rich' and case['panel'] == 'fixed' and case.get('scene_variant', 'scene_on') == 'scene_on':
                rich_fixed.append(dict(step=row['step'], regime=case['regime'], sequence_id=case['sequence_id'],
                    start=case['start'], case_seed=case['case_seed'], metrics=case['metrics']))
    result = dict(reviewed_at=datetime.now().astimezone().isoformat(), source=provenance,
        effective_config_source=dict(path=str(effective_path), sha256=hashlib.sha256(effective_raw).hexdigest()),
        validation_source=validation_provenance,
        status_snapshot=json.loads((RUN / 'continuous_training_oct16/STATUS.json').read_text()),
        latest_trace_step=trace[-1]['step'], latest_complete_block_step=full[-1]['last_step'],
        recent_complete_blocks=[dict(first_step=b['first_step'], last_step=b['last_step']) for b in recent],
        blocks=blocks, recent_comparisons=comparisons, rich_fixed_validation=rich_fixed,
        coefficients=dict(observation=obs_co, geometry=geometry_co, gait=gait_co, raw_gait_outer=raw_gait_co),
        standardization_weights=weights,
        numeric_health=dict(nonfinite_logged_values=nonfinite,
            max_abs_three_group_sum_error=max(abs(r['values']['audit/three_group_sum_minus_total']) for g in groups.values() for r in g),
            max_abs_final_gait_sum_error=max(abs(r['values']['audit/final_gait_sum_minus_total']) for g in groups.values() for r in g),
            max_abs_raw_gait_sum_error=max(abs(r['values']['audit/raw_gait_sum_minus_total']) for g in groups.values() for r in g)),
        limitations=[
            'Training losses average mixed-corpus batches and do not identify per-RICH-sample losses.',
            'Standardization controls dataset pair, count of diffusion t<100 samples and scene dropout; it does not match motion content, exact t or observation perturbations.',
            'All raw logged terms have their existing noise, reliability and validity gates; raw means before outer weighting, not ungated physical metric.',
            'Gait totals and base totals contain their subterms; nested rows must not be summed again.',
            'Global rotation-matrix MSE and head-position MSE are not individually logged, so the base subtotal cannot be completely reconstructed from raw terms.',
            'GT foot-envelope penetration and observed-plane proxy do not measure whole-scene collision or full support-surface penetration.',
            'Short-window training decrease does not establish held-out RICH generalization or benchmark comparability.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'LOSS_TREND.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    selected = [
        ('total', '总 loss'), ('weighted/base_body_and_gait', '身体与步态（已加权）'),
        ('weighted/observation', '观测一致性（已加权）'), ('weighted/scene_geometry', '场景几何（已加权）'),
        ('weighted/gait_final_total', '最终预测步态小计'), ('weighted/gait_raw_total', '原始预测步态贡献（×0.5）'),
        ('raw/base/x0', '最终动作 x0'), ('raw/base/fk_position_m2', 'FK 位置'),
        ('raw/base/fk_velocity_m2_s2', 'FK 速度'), ('raw/base/fk_acceleration_m2_s4', 'FK 加速度'),
        ('raw/base/root_absolute_chordal', '根朝向'), ('raw/base/head_absolute_chordal', '头朝向'),
        ('raw/base/neck_absolute_chordal', '颈朝向'), ('raw/base/gait_stance_velocity', '支撑期足速误差'),
        ('raw/base/gait_swing_velocity', '摆腿足速误差'), ('raw/base/gait_bilateral_stride', '左右脚相对步幅'),
        ('raw/base/gait_bilateral_amplitude', '左右脚相对幅度'), ('raw/base/gait_bilateral_relation', '左右脚协调关系'),
        ('raw/base/gait_support_penetration', 'GT 足高度包络穿透代理'),
        ('raw/geometry/contact', '场景接触分类'), ('raw/geometry/contact_distance_m2', '可观察接触距离'),
        ('raw/geometry/contact_relative_slip_m2_s2', '可观察接触相对滑动'),
        ('raw/geometry/observed_plane_proxy_m2', '可观察平面代理'),
        ('raw/geometry/correspondence', '场景对应分类'), ('raw/geometry/support_relation_m2', '支撑关系'),
    ]
    classifications = {'decreased_in_all_recent_complete_blocks': '各段下降',
        'overall_decrease_with_fluctuations': '总体下降、有震荡',
        'approximately_flat_with_fluctuations': '近似平台、有震荡',
        'overall_increase_with_fluctuations': '总体回升、有震荡', 'zero': '为零'}
    labels = [f"{b['last_step'] // 1000}k" for b in recent]
    lines = [f"# 监督趋势复核（{result['reviewed_at']}）", '',
        f"只读日志至 {trace[-1]['step']} 步。下表只比较最近四个完整千步段（每段 1000 次更新），末尾未完整段单独存入 JSON。主线状态 {result['status_snapshot'].get('status')}，IO 重启 {result['status_snapshot'].get('consecutive_io_restarts')} 次。没有加载模型、使用 GPU、修改训练配置或状态。", '',
        '结论：总 loss、身体与步态、观测监督仍在下降；并非每一个分项都单调下降，场景接触、加速度、左右脚关系等会震荡。下降说明训练目标仍在被拟合，不能替代全量 RICH 验证。', '',
        '| 项目 | ' + ' | '.join(labels) + ' | 首末变化 | 描述 |',
        '|---|' + '---:|' * len(labels) + '---:|---|']
    for key, label in selected:
        c = comparisons[key]
        values = ' | '.join(f'{v:.6g}' for v in c['means'])
        delta = '—' if c['oldest_to_latest_percent'] is None else f"{c['oldest_to_latest_percent']:+.1f}%"
        lines.append(f"| {label} | {values} | {delta} | {classifications[c['descriptive_classification']]} |")
    lines += ['', '前三个“已加权”小计之和与总 loss 一致；步态小计已经包含于身体与步态，不再重复相加。其余表内分项使用日志原值，单位和外层权重不同，数值大小不能相互排序。原始网络输出的 raw 表示修正前，不表示没有门控。', '',
        '| 完整千步段 | 总 loss | 同构成总 loss | 低噪声样本比例 | 场景关闭比例 |', '|---|---:|---:|---:|---:|']
    for block in recent:
        lines.append(f"| {block['first_step']}–{block['last_step']} | {block['terms']['total']['mean']:.6f} | {block['terms']['total']['composition_standardized_mean']:.6f} | {block['low_t_sample_fraction']:.3f} | {block['scene_off_fraction']:.3f} |")
    health = result['numeric_health']
    lines += ['',
        f"全部已记录分项非有限值数量 {len(nonfinite)}；三项带权和与总 loss 最大绝对误差 {health['max_abs_three_group_sum_error']:.3g}，final/raw 步态小计核对残差 {health['max_abs_final_gait_sum_error']:.3g}/{health['max_abs_raw_gait_sum_error']:.3g}。这核验的是日志口径与数值状态。", '',
        '同构成统计按数据集对、每批 t<100 的样本数和场景 dropout 标准化，窗口内容及精确 t 仍会变化。不能从混合批训练 loss 单独推断 RICH 的 loss；当前日志没有逐样本损失。', '',
        'RICH 固定验证只有一个固定窗口，必须与本次全量 RICH 评估分开。固定窗口的最近可比较 MPJPE（mm）：', '', '| 步数 | clean | noisy |', '|---:|---:|---:|']
    val_steps = sorted({r['step'] for r in rich_fixed})[-5:]
    for step in val_steps:
        by_regime = {r['regime']: r for r in rich_fixed if r['step'] == step}
        lines.append(f"| {step} | {by_regime['clean']['metrics']['mpjpe_mm']['value']:.2f} | {by_regime['noisy']['metrics']['mpjpe_mm']['value']:.2f} |")
    lines += ['',
        'GT 足高度包络的穿透项只相对于 GT 足高度，场景平面项只覆盖可信可观察区域；它们下降不等于整个人体不穿场景。完整分项、权重、统计和日志快照 SHA 见 [LOSS_TREND.json](LOSS_TREND.json)。', '',
        '与其他 ego 方法比较还需匹配数据划分、输入轨迹类型、场景可见范围、帧率、关节定义及误差对齐方式。当前训练趋势不提供跨方法排名。']
    (OUT / 'LOSS_TREND_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(latest_trace_step=trace[-1]['step'], complete_block_step=full[-1]['last_step'],
        recent_blocks=labels, main_groups={k: comparisons[k] for k in
            ('total', 'weighted/base_body_and_gait', 'weighted/observation', 'weighted/scene_geometry')},
        numeric_health=health), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
