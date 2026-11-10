"""Paired validation comparison and local report for a completed scene repair.

Bootstrap source families within predeclared corpus/action strata, preserving
the original dataset/category weights. The interval is conditional on selected
checkpoints and is not a test-set or multi-training-seed confidence interval.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

METRICS = ('mpjpe_mm','pa_mpjpe_mm','root_error_mm','head_error_mm')
DATASETS = ('egobody','trumans','grail')
CATEGORIES = ('curb','sitting','slope','stair_p1','stair_p2')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def paired_comparison(baseline, candidate, *, draws=5000, seed=20261025):
    a,b=baseline['rows'],candidate['rows']
    if not a or len(a)!=len(b) or any(x['identity']!=y['identity'] for x,y in zip(a,b)):
        raise ValueError('Comparison requires exactly the same ordered validation identities')
    families=defaultdict(list)
    for x,y in zip(a,b):
        identity=x['identity']; dataset=identity['dataset']
        if dataset not in DATASETS or identity['split']!='validation' or identity['synthetic_chair']:
            raise ValueError('Unexpected validation membership or synthetic geometry')
        category=identity['category'] if dataset=='grail' else dataset
        family=identity['corpus'],identity['family']
        values=np.array([[row['metrics'][m] for m in METRICS] for row in (x,y)],dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite metrics')
        families[(dataset,category,family)].append(values)
    strata=defaultdict(list)
    for (dataset,category,family),values in families.items():
        strata[(dataset,category)].append(np.mean(values,axis=0))
    expected={('egobody','egobody'),('trumans','trumans')}|{('grail',c) for c in CATEGORIES}
    if set(strata)!=expected or draws<100:
        raise ValueError('All predeclared corpus/action strata and >=100 draws are required')
    rng=np.random.default_rng(seed); corpus=defaultdict(list); boot=defaultdict(list)
    counts={}
    for (dataset,category),values in sorted(strata.items()):
        values=np.stack(values); corpus[dataset].append(values.mean(0))
        delta=values[:,0]-values[:,1]
        boot[dataset].append(delta[rng.integers(len(values),size=(draws,len(values)))].mean(1))
        counts[dataset+'/'+category]=len(values)
    per_dataset={}; overall=[]; sampled=[]
    for dataset in DATASETS:
        values=np.stack(corpus[dataset]).mean(0)
        samples=np.stack(boot[dataset]).mean(0)
        overall.append(values); sampled.append(samples)
        per_dataset[dataset]=dict(baseline=dict(zip(METRICS,values[0].tolist())),
            candidate=dict(zip(METRICS,values[1].tolist())),
            baseline_minus_candidate=dict(zip(METRICS,(values[0]-values[1]).tolist())),
            paired_family_bootstrap_95pct={m:np.quantile(samples[:,i],[.025,.975]).tolist()
                                          for i,m in enumerate(METRICS)})
    values=np.stack(overall).mean(0); samples=np.stack(sampled).mean(0)
    for index,reference in enumerate((baseline,candidate)):
        if abs(values[index,0]-reference['selection_score_mm'])>1e-8:
            raise ValueError('Independent aggregation disagrees with the selection score')
    return dict(windows=len(a),source_families=len(families),stratum_family_counts=counts,
        per_dataset=per_dataset,dataset_macro=dict(
            baseline=dict(zip(METRICS,values[0].tolist())),candidate=dict(zip(METRICS,values[1].tolist())),
            baseline_minus_candidate=dict(zip(METRICS,(values[0]-values[1]).tolist())),
            paired_family_bootstrap_95pct={m:np.quantile(samples[:,i],[.025,.975]).tolist()
                                          for i,m in enumerate(METRICS)}),
        bootstrap=dict(draws=draws,seed=seed,unit='paired source families within fixed corpus/action strata',
            scope='Conditional selected-checkpoint validation uncertainty; no selection-bias adjustment, independent test or training-seed inference.'))


def render(result):
    lines=['# 冻结先验场景修复：与独立无场景基线比较','',
        f"原无场景 30k；修复分支选中额外 {result['selected_step']} 步。单位 mm，越低越好。",'',
        '| 数据集 | 无场景 MPJPE | 场景修复 MPJPE | 无场景 PA | 场景修复 PA | 无场景 root | 场景修复 root |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,group in list(result['per_dataset'].items())+[('三数据集等权',result['dataset_macro'])]:
        lines.append('| '+name+' | '+' | '.join(f"{group[c][m]:.2f}" for m in METRICS[:3]
                        for c in ('baseline','candidate'))+' |')
    group=result['dataset_macro']; delta=group['baseline_minus_candidate']['mpjpe_mm']
    interval=group['paired_family_bootstrap_95pct']['mpjpe_mm']
    lines+=['',f'总体 MPJPE 改善（无场景－修复）：{delta:.3f} mm；配对家族重采样 95% 范围 [{interval[0]:.3f}, {interval[1]:.3f}] mm。',
        '验证集点估计超过基线。' if delta>1e-8 else '当前未超过基线，不支持场景整体更好的结论。','',
        f"全部 {result['windows']} 窗口、{result['source_families']} 来源家族逐条配对；三数据集等权，GRAIL 五类动作等权。",'',
        '这是探索性选模验证结果。场景修复比基线多了适配预算，后续需要额外预算匹配的续训对照和独立测试；',
        '上述范围以当前选定权重为条件，不包含挑选 checkpoint 的偏差或训练随机性的影响。',
        '不使用 RICH、真值个人体形或验证集补椅子；MPJPE 改善不能替代接触、穿透和实际可视化核查。',
        '逐窗数值、身份、源文件哈希和重采样设置见 COMPARISON.json。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--draws',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261025)
    a=p.parse_args(argv);root=Path(__file__).resolve().parents[1]
    run=a.run.resolve();out=run/'evaluation';out.mkdir(exist_ok=True)
    training=run/'training/STATUS.json';finished=run/'selection/RESULT.json'
    contract=run/'training/CONTRACT.json';seal=json.loads(contract.read_text())
    policy=dict(protocol='paired-scene-repair-report-v1',contract_sha256=seal['sha256'],
        draws=a.draws,seed=a.seed,metrics=list(METRICS),source_sha256=digest(__file__),test_used=False)
    record=out/'REPORT_PROTOCOL.json'
    if record.exists() and json.loads(record.read_text())!=policy:
        raise ValueError('Started reporting protocol changed')
    write_json(record,policy)
    while not finished.exists():
        status=json.loads(training.read_text())
        if not a.watch:
            raise ValueError('Final comprehensive selection is not complete')
        if status['state'] in ('failed','stopped'):
            write_json(out/'REPORT_STATUS.json',dict(state='blocked_by_training',training=status));return
        try:
            os.kill(status['pid'],0)
        except ProcessLookupError:
            write_json(out/'REPORT_STATUS.json',dict(state='training_process_absent',training=status));return
        write_json(out/'REPORT_STATUS.json',dict(state='waiting_for_comprehensive_selection',
            completed_updates=status['completed_updates'],updated_at_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(30)
    selection=json.loads(finished.read_text());best=selection['best']
    candidate_path=run/'selection/comprehensive'/f"step_{best['step']:06d}.json"
    baseline_path=root/seal['config']['base']['comprehensive_reference']
    baseline=json.loads(baseline_path.read_text());candidate=json.loads(candidate_path.read_text())
    if candidate['contract_sha256']!=seal['sha256'] or candidate['checkpoint_sha256']!=best['checkpoint_sha256']:
        raise ValueError('Final selected metrics do not match the sealed candidate')
    if digest(baseline_path)!=seal['base_references']['comprehensive_reference']:
        raise ValueError('Pinned independent baseline metrics changed')
    weight=run/'selection/candidates'/f"step_{best['step']:06d}.pt"
    if digest(weight)!=best['checkpoint_sha256']:
        raise ValueError('Selected adapter checkpoint changed')
    if digest(root/seal['config']['base']['checkpoint'])!=seal['config']['base']['sha256']:
        raise ValueError('Frozen full base checkpoint changed')
    for name,expected in seal['sources'].items():
        if digest(root/name)!=expected:
            raise ValueError('Sealed repair source changed: '+name)
    result=paired_comparison(baseline,candidate,draws=a.draws,seed=a.seed)
    result.update(selected_step=best['step'],policy=policy,baseline_file=str(baseline_path),
        baseline_sha256=digest(baseline_path),candidate_file=str(candidate_path),candidate_sha256=digest(candidate_path),
        selected_checkpoint_sha256=best['checkpoint_sha256'],verified_training_sources=len(seal['sources']))
    write_json(out/'COMPARISON.json',result);(out/'REPORT_zh.md').write_text(render(result))
    write_json(out/'REPORT_STATUS.json',dict(state='complete',baseline_minus_candidate_mm=
        result['dataset_macro']['baseline_minus_candidate']['mpjpe_mm']))
    print(json.dumps(result['dataset_macro'],indent=2),flush=True)


if __name__=='__main__':
    main()
