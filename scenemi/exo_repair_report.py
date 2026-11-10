"""Summarize sealed repair validation, with paired source-family uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scenemi.config import ROOT
from scenemi.data.amass_prepare import atomic_json, sha256


def paired_interval(rows, condition, control, field, rng):
    groups=defaultdict(dict)
    for row in rows:
        if row['condition']==condition and row['variant'] in ('repair',control):
            groups[row['case_id']][row['variant']]=row
    families=defaultdict(list)
    for values in groups.values():
        if set(values)!=set(('repair',control)):
            raise ValueError('Unpaired repair/control case')
        repair,reference=values['repair'],values[control]
        if repair['identity']!=reference['identity']:
            raise ValueError('Paired estimates or observation masks differ')
        families[repair['family']].append(repair['metrics'][field]-reference['metrics'][field])
    delta=np.asarray([np.mean(families[key]) for key in sorted(families)])
    samples=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(-1)
    return dict(delta_repair_minus_control_mm=float(delta.mean()),
        percentile_95_interval_mm=np.quantile(samples,[.025,.975]).tolist(),
        families=len(delta),families_improved=int((delta<0).sum()),
        bootstrap_unit='independent_source_family',resamples=5000)


def report(run):
    status=json.loads((run/'STATUS.json').read_text())
    if status['state']!='complete':raise ValueError('Full pilot has not completed')
    result_file=run/'full_validation/RESULT.json'
    result=json.loads(result_file.read_text())
    if not result['full_validation'] or result['test_used']:
        raise ValueError('Expected completed full validation without test selection')
    seal=json.loads((run/'PROTOCOL.json').read_text())
    if status['contract_sha256']!=seal['sha256']:
        raise ValueError('Run and protocol differ')
    for path,digest in seal['sources'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Sealed source changed: '+path)
    checkpoint=run/'training/last.pt'
    if sha256(checkpoint)!=status['final_checkpoint_sha256']:
        raise ValueError('Final checkpoint no longer matches evaluation')
    metrics_file=run/'full_validation/metrics.jsonl'
    rows=[json.loads(line) for line in metrics_file.read_text().splitlines()]
    expected=result['windows_per_condition']*4*3
    if len(rows)!=expected or len({(r['condition'],r['variant'],r['case_id']) for r in rows})!=expected:
        raise ValueError('Missing or duplicate paired metric rows')
    rng=np.random.default_rng(20261026)
    intervals={condition:{control:paired_interval(rows,condition,control,'repair_region_mpjpe_mm',rng)
        for control in ('initial','temporal_smoothing')} for condition in ('tracking','occlusion','gap')}
    record=dict(candidate_gate_passed=result['candidate_gate_passed'],checks=result['checks'],
        full_validation=True,families=result['families'],windows_per_condition=result['windows_per_condition'],
        paired_region_intervals=intervals,bootstrap_seed=20261026,conditions=result['conditions'],
        checkpoint_sha256=sha256(checkpoint),contract_sha256=seal['sha256'],test_used=False,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in
            [result_file,metrics_file,run/'PROTOCOL.json',run/'STATUS.json',Path(__file__).resolve()]},
        limitations=['Synthetic native pose/shape estimates, richer than historical partial-joint inputs.',
            'Reliability is declared synthetic metadata; real-export proxy calibration remains unvalidated.',
            'Exact high-confidence preservation is enforced by construction, not learned.',
            'No frozen-prior-feature ablation or real-video accuracy evaluation in this pilot.'])
    atomic_json(run/'SUMMARY.json',record)
    table=['# EXO 初始化修复：固定 2k 完整验证','',
        f"验证动作 {result['families']} 条，每条件 {result['windows_per_condition']} 窗口；未使用 test。",
        f"候选门槛：{'通过' if result['candidate_gate_passed'] else '未全部通过'}。固定最后 2k，未按中途指标选模型。",'',
        '| 条件 | 初始化 MPJPE / mm | 时间平滑 MPJPE / mm | 修复 MPJPE / mm |',
        '| --- | ---: | ---: | ---: |']
    labels=dict(identity='理想准确估计',tracking='含噪跟踪',occlusion='局部遮挡',gap='整人缺帧')
    for condition in labels:
        values=result['conditions'][condition]
        numbers=[values[v]['family_macro_mean']['mpjpe_mm'] for v in ('initial','temporal_smoothing','repair')]
        table.append(f'| {labels[condition]} | '+ ' | '.join(f'{v:.3f}' for v in numbers)+' |')
    table+=['','以下只统计事先定义的待修复区域。负差值表示修复优于对照；区间按源动作 family 配对重采样。','',
        '| 区域 | 初始化 / mm | 时间平滑 / mm | 修复 / mm | 相对平滑差值的 95% 区间 / mm |',
        '| --- | ---: | ---: | ---: | --- |']
    for condition in ('occlusion','gap'):
        values=result['conditions'][condition]
        numbers=[values[v]['family_macro_mean']['repair_region_mpjpe_mm'] for v in ('initial','temporal_smoothing','repair')]
        low,high=intervals[condition]['temporal_smoothing']['percentile_95_interval_mm']
        table.append(f'| {labels[condition]} | '+' | '.join(f'{v:.3f}' for v in numbers)+f' | [{low:.3f}, {high:.3f}] |')
    table+=['','门槛逐项结果：','']+[f"- {key}: {value}" for key,value in result['checks'].items()]
    table+=['','这是 AMASS 合成原生姿态/体形估计的小试，输入比旧部分关节生成更丰富；不能直接比较两轮数字来归因架构收益。',
        '准确输入的保持由显式保留通路实现；缺帧初值只由已有估计插值。真实 ZIP 接入、实际可靠度、先验特征消融尚未验证。',
        '实验配置、输入约定与代码见 [短试协议](../../docs/EXO_REPAIR_PILOT_zh.md)。','']
    (run/'SUMMARY_zh.md').write_text('\n'.join(table))
    print(json.dumps(dict(summary=str(run/'SUMMARY_zh.md'),candidate_gate_passed=result['candidate_gate_passed'],
        paired_region_intervals=intervals),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/amass_exo_repair128')
    args=parser.parse_args();report(args.run.resolve())


if __name__=='__main__':
    main()
