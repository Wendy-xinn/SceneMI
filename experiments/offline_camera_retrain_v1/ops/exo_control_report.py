"""CPU-only, paired reference/control/adapter report for a completed EXO stage.

The final stage is judged against both its unadapted generator and its repaired
input reference. No prediction is aligned to GT and no checkpoint is selected.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .dependencies import sha256

DATASETS = ('rich', 'trumans', 'egobody')
REGIMES = ('clean', 'noisy')
LABELS = ('input_reference', 'control3000', 'adapter_final')
PROTOCOL = 'corrected-exo-reference-control-final-usability-v1'
CORE = ('w_mpjpe_mm', 'visible_mpjpe_mm', 'hidden_mpjpe_mm')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(row):
    return tuple(row[k] for k in ('dataset', 'recording', 'sequence_id', 'start', 'source_first'))


def metric_summary(rows, metric):
    """Count weighting within recordings, then equal recording weighting.

    Missing values require zero counts. They never silently become zero errors
    or reduce the denominator of a pooled error incorrectly.
    """
    records = defaultdict(list)
    for row in rows:
        value, count = row['metrics'][metric], row['counts'][metric]
        require(type(count) is int and count >= 0, 'Invalid metric count: ' + metric)
        require((value is None) == (count == 0), 'Metric missingness/count differs: ' + metric)
        if count:
            require(isinstance(value, (int, float)) and math.isfinite(value), 'Nonfinite metric: ' + metric)
            records[(row['dataset'], row['recording'])].append((value, count))
    total = sum(n for values in records.values() for _, n in values)
    per_record = [sum(v*n for v,n in values)/sum(n for _,n in values) for values in records.values()]
    return dict(equal_recording=statistics.mean(per_record) if per_record else None,
        pooled=sum(v*n for values in records.values() for v,n in values)/total if total else None,
        count=total, valid_recordings=len(records), valid_windows=sum(len(v) for v in records.values()),
        total_windows=len(rows), total_recordings=len({(r['dataset'], r['recording']) for r in rows}))


def aggregate(rows):
    groups = {}
    for scope in ('all64', 'new55', 'old9'):
        for dataset in DATASETS:
            for regime in REGIMES:
                for label in LABELS:
                    chosen = [r for r in rows if r['model'] == label and r['dataset'] == dataset
                        and r['regime'] == regime and (scope == 'all64' or r['panel'] == scope)]
                    if chosen:
                        keys = set.intersection(*(set(r['metrics']) for r in chosen))
                        groups['/'.join((scope,dataset,regime,label))] = {
                            key: metric_summary(chosen,key) for key in sorted(keys)}
    macros = {}
    for scope in ('all64','new55','old9'):
        for regime in REGIMES:
            for label in LABELS:
                selected = [groups.get('/'.join((scope,d,regime,label))) for d in DATASETS]
                if not all(selected):
                    continue
                common = set.intersection(*(set(g) for g in selected))
                macros['/'.join((scope,regime,label))] = {metric: {
                    reduction: statistics.mean(g[metric][reduction] for g in selected)
                        if all(g[metric][reduction] is not None for g in selected) else None
                    for reduction in ('equal_recording','pooled')} for metric in sorted(common)}
    return groups,macros


def paired_winrates(rows):
    """A recording votes only when both paired estimates have valid samples."""
    indexed = {}
    for row in rows:
        key = identity(row),row['regime'],row['model']
        require(key not in indexed,'Duplicate paired output identity')
        indexed[key] = row
    result = []
    for comparator in LABELS[:2]:
        for scope in ('all64','new55','old9'):
            for dataset in DATASETS:
                for regime in REGIMES:
                    candidates = [r for r in rows if r['model'] == 'adapter_final' and r['dataset'] == dataset
                        and r['regime'] == regime and (scope == 'all64' or r['panel'] == scope)]
                    metrics = {}
                    for metric in CORE:
                        paired = defaultdict(list)
                        for row in candidates:
                            other = indexed.get((identity(row),regime,comparator))
                            require(other is not None,'Missing paired comparator')
                            require(row['counts'][metric] == other['counts'][metric],'Paired denominator differs')
                            a,b = row['metrics'][metric],other['metrics'][metric]
                            if a is not None and b is not None:
                                paired[row['recording']].append((a-b,row['counts'][metric]))
                        changes = [sum(delta*n for delta,n in values)/sum(n for _,n in values)
                            for values in paired.values()]
                        metrics[metric] = dict(paired_recordings=len(changes),wins=sum(v < -1e-9 for v in changes),
                            ties=sum(abs(v) <= 1e-9 for v in changes),losses=sum(v > 1e-9 for v in changes),
                            paired_windows=sum(len(v) for v in paired.values()),
                            mean_delta_mm=statistics.mean(changes) if changes else None)
                    if candidates:
                        result.append(dict(scope=scope,dataset=dataset,regime=regime,comparator=comparator,metrics=metrics))
    return result


def usability(groups,macros,*,ego_exact,inherited_foot_pass,cpu_verified,base_unchanged):
    hidden_macro=[]; corpus=[]; passed=all((ego_exact,inherited_foot_pass,cpu_verified,base_unchanged))
    for regime in REGIMES:
        for comparator in LABELS[:2]:
            def macro(label):
                return macros.get('/'.join(('all64',regime,label)),{}).get('hidden_mpjpe_mm',{}).get('equal_recording')
            old,new=macro(comparator),macro('adapter_final')
            ok=old is not None and new is not None and new <= .95*old
            hidden_macro.append(dict(regime=regime,comparator=comparator,baseline_hidden_mm=old,
                final_hidden_mm=new,passed=ok))
            passed = passed and ok
        for dataset in DATASETS:
            def value(label,metric):
                return groups.get('/'.join(('all64',dataset,regime,label)),{}).get(metric,{}).get('equal_recording')
            current=value('adapter_final','hidden_mpjpe_mm');guards=[]
            for comparator in LABELS[:2]:
                old=value(comparator,'hidden_mpjpe_mm')
                ok=old is not None and current is not None and current <= old+max(5.,.05*old)
                guards.append(dict(comparator=comparator,baseline_mm=old,final_mm=current,passed=ok))
                passed=passed and ok
            observed=value('adapter_final','visible_mpjpe_mm');old=value('control3000','visible_mpjpe_mm')
            ok=old is not None and observed is not None and observed <= old+max(1.,.02*old)
            passed=passed and ok
            corpus.append(dict(dataset=dataset,regime=regime,hidden_guards=guards,
                observed_guard=dict(baseline_mm=old,final_mm=observed,passed=ok)))
    return dict(status='pass' if passed else 'failed',hidden_macro=hidden_macro,corpus_guards=corpus,
        EGO_exact=ego_exact,inherited_foot_guards_pass=inherited_foot_pass,CPU_verified=cpu_verified,
        frozen_base_unchanged=base_unchanged,automatic_promotion=False,
        aggregation='Equal recordings within each dataset; equal three-dataset macro. Pooled values are separate diagnostics.',
        scope='Registered final-step engineering screen on historical VAL; not TEST or proof of generalization.')


def motion_metrics(arrays):
    import numpy as np
    import torch
    from experiments.offline_camera_retrain_v1.supervision import forward_kinematics
    from .reference_review import temporal_metrics
    with torch.no_grad():
        rest=torch.from_numpy(np.array(arrays['rest'],dtype=np.float32,copy=True))[None]
        p,g=(forward_kinematics(torch.from_numpy(np.array(arrays[k],dtype=np.float32,copy=True))[None],rest)[0]
             .numpy().astype(np.float64) for k in ('prediction','truth'))
    require(p.shape == g.shape and p.ndim == 3 and p.shape[1:] == (22,3) and len(p) >= 3,'Invalid saved body sequence')
    require(np.isfinite(p).all() and np.isfinite(g).all(),'Nonfinite saved body sequence')
    visible=np.asarray(arrays['observation_meta'])[...,0]>0
    require(visible.shape == p.shape[:2],'Observation mask shape differs')
    errors=np.linalg.norm(p-g,axis=-1)*1000
    speed_p,speed_g=(np.diff(v[:,[10,11]],axis=0)*30 for v in (p,g))
    low=np.linalg.norm(speed_g,axis=-1)<.2
    measures=dict(w_mpjpe_mm=(errors,np.ones_like(visible)),visible_mpjpe_mm=(errors,visible),
        hidden_mpjpe_mm=(errors,~visible),root_position_error_mm=(errors[:,0],np.ones(len(p),dtype=bool)),
        mpjve_mm_s=(np.linalg.norm(np.diff(p-g,axis=0)*30,axis=-1)*1000,np.ones((len(p)-1,22),dtype=bool)),
        acceleration_error_mm_s2=(np.linalg.norm(np.diff(p-g,n=2,axis=0)*900,axis=-1)*1000,np.ones((len(p)-2,22),dtype=bool)),
        foot_joint_speed_gt_low_m_s=(np.linalg.norm(speed_p,axis=-1),low))
    values={k:float(a[m].mean()) if m.any() else None for k,(a,m) in measures.items()}
    counts={k:int(m.sum()) for k,(_,m) in measures.items()}
    temporal=temporal_metrics(arrays);values.update(temporal)
    # Quantiles, step-count diagnostics and rotational errors are summarized as
    # window observations, never pooled as if every frame were an independent trial.
    counts.update({k:1 if v is not None else 0 for k,v in temporal.items()})
    return values,counts


def _last_record(path):
    lines=[line for line in Path(path).read_text().splitlines() if line.strip()]
    require(bool(lines),'Missing validation record')
    return json.loads(lines[-1])


def render(run,baseline):
    import numpy as np
    import torch
    from .adapt import now,write
    run,baseline=Path(run).resolve(),Path(baseline).resolve()
    require(not torch.cuda.is_initialized(),'Report must start in a CPU-only process')
    torch.set_num_threads(2)
    cpu=read(run/'CPU_RECHECK.json');old_gates=read(run/'ACCEPTANCE.json');comparison=read(run/'COMPARISON.json')
    require(cpu.get('status') == 'independent_CPU_primary_metrics_passed' and cpu.get('CUDA_initialized') is False
        and cpu.get('GPU_used_for_recheck') is False and cpu.get('matched_input_arrays') is True,'Stage CPU audit missing')
    require(comparison.get('CPU_recheck_sha256') == sha256(run/'CPU_RECHECK.json')
        and comparison.get('acceptance') == old_gates,'Stage audit linkage differs')
    config=read(run/'config.json');panel=read(run/'prepared/validation_plan.json')['plan']
    require(len(panel) == 67 and sum(r['mode'] == 'exo' for r in panel) == 64,'Need registered64 EXO +3 EGO panel')
    require(all(row.get('case',i) == i for i,row in enumerate(panel)),'Validation JSON case/index differs')
    require(config['adaptation']['reference_policy'] == 'exo-local-so3-reference-gap-fill-v1'
        and config['physical_guidance']['enabled'] is False,'Reference/physics configuration differs')
    before=_last_record(run/'baseline_validation/validation.jsonl');after=_last_record(run/'training/validation.jsonl')
    require(before['step'] == 0 and after['step'] == config['adaptation']['updates'],'Final update clock differs')
    for record in (before,after):
        require(len(record['cases']) == 2*len(panel),'Validation coverage differs')
        require(len({(r['case'],r['regime']) for r in record['cases']}) == 2*len(panel),'Duplicate validation row')
    old_rows={(r['case'],r['regime']):r for r in before['cases']};new_rows={(r['case'],r['regime']):r for r in after['cases']}
    baseline_cpu=read(baseline/'CPU_RECHECK.json');baseline_plan=read(baseline/'PLAN.json')
    require(baseline_cpu.get('status') == 'independent_cpu_passed','Standalone baseline CPU audit missing')
    baseline_rows=[json.loads(line) for line in (baseline/'rows.jsonl').read_text().splitlines() if line.strip()]
    require(len(baseline_rows) == baseline_plan['planned_rows'] == 384,'Standalone baseline incomplete')
    require(baseline_cpu['rows_sha256'] == sha256(baseline/'rows.jsonl'),'Standalone baseline row digest differs')
    basemap={(identity(r),r['regime'],r['model']):r for r in baseline_rows}
    require(len(basemap) == len(baseline_rows),'Duplicate standalone baseline identity')
    files={};rows=[];ego_exact=True;matched_external=0;maximum_metric_difference=0.
    source_paths=[Path(__file__),run/'config.json',run/'STUDY.json',run/'prepared/validation_plan.json',
        run/'prepared/effective_config.json',run/'CPU_RECHECK.json',run/'COMPARISON.json',run/'ACCEPTANCE.json',
        baseline/'PLAN.json',baseline/'rows.jsonl',baseline/'CPU_RECHECK.json']
    for folder in ('baseline_validation','training'):
        source_paths.extend([run/folder/'validation.jsonl',run/folder/'validation_latest.npz'])
    for path,expected in cpu.get('files_sha256',{}).items():
        require(sha256(path) == expected,'Stage CPU-audited file changed: '+path)
    require(sha256(run/'training/last.pt') == cpu['final_checkpoint_sha256'],'Final checkpoint changed')
    files[str(run/'training/last.pt')]=cpu['final_checkpoint_sha256']
    with np.load(run/'baseline_validation/validation_latest.npz',allow_pickle=False) as a, \
         np.load(run/'training/validation_latest.npz',allow_pickle=False) as b:
        require(int(a['step']) == 0 and int(b['step']) == after['step'],'Saved array update clock differs')
        for i,entry in enumerate(panel):
            for suffix in ('truth','rest'):
                require(np.array_equal(a[f'{i}_{suffix}'],b[f'{i}_{suffix}']),'Paired truth/rest changed')
            for regime in REGIMES:
                for suffix in ('observations','observation_meta','reference_motion'):
                    require(np.array_equal(a[f'{i}_{regime}_{suffix}'],b[f'{i}_{regime}_{suffix}']),'Paired input/reference changed')
                old,new=old_rows[i,regime],new_rows[i,regime]
                require(identity(old) == identity(new) == identity(entry)
                    and old['case_seed'] == new['case_seed'] == entry['case_seed'],'Validation source/seed differs')
                if entry['mode'] == 'ego':
                    ego_exact=ego_exact and np.array_equal(a[f'{i}_{regime}_prediction'],b[f'{i}_{regime}_prediction'])
                    continue
                for label,arrays,suffix,jsonrow in (
                    ('input_reference',a,'reference_motion',old),('control3000',a,'prediction',old),('adapter_final',b,'prediction',new)):
                    payload=dict(prediction=arrays[f'{i}_{regime}_{suffix}'],truth=a[f'{i}_truth'],rest=a[f'{i}_rest'],
                        observations=a[f'{i}_{regime}_observations'],observation_meta=a[f'{i}_{regime}_observation_meta'])
                    values,counts=motion_metrics(payload)
                    stored=jsonrow['reference_metrics'] if label == 'input_reference' else jsonrow['metrics']
                    for metric,savedkey in [('w_mpjpe_mm','mpjpe_mm'),('visible_mpjpe_mm','conditioned_mpjpe_mm'),('hidden_mpjpe_mm','unconditioned_mpjpe_mm')]:
                        saved=stored[savedkey];require(saved['count'] == counts[metric],'Primary denominator mismatch')
                        value=values[metric];require((value is None) == (saved['value'] is None),'Primary missingness mismatch')
                        if value is not None:
                            difference=abs(value-saved['value']);maximum_metric_difference=max(maximum_metric_difference,difference)
                            require(difference <= max(.002,abs(value)*2e-5),'Primary metric CPU difference')
                    if label != 'adapter_final':
                        external=basemap.get((identity(entry),regime,label));require(external is not None,'Missing standalone comparator')
                        require(external['seed'] == entry['case_seed'],'Standalone DDIM seed differs')
                        path=Path(external['artifact']);require(sha256(path) == external['artifact_sha256'],'Standalone artifact changed')
                        files[str(path)]=external['artifact_sha256']
                        with np.load(path,allow_pickle=False) as saved:
                            for key,value in payload.items():
                                require(np.array_equal(saved[key],value),'Standalone/stage comparator differs: '+key)
                            require(np.array_equal(saved['reference_motion'],a[f'{i}_{regime}_reference_motion']),'Standalone repaired reference differs')
                        matched_external+=1
                    rows.append(dict(dataset=entry['dataset'],recording=entry['recording'],sequence_id=entry['sequence_id'],
                        start=entry['start'],source_first=entry['source_first'],case=i,regime=regime,case_seed=entry['case_seed'],
                        panel=entry['panel'],model=label,metrics=values,counts=counts))
    require(len(rows) == 384 and matched_external == 256,'EXO paired report coverage differs')
    require(ego_exact == cpu['EGO_serialized_outputs_exact'],'EGO CPU evidence differs')
    groups,macros=aggregate(rows)
    foot_pass=all(g['foot_speed_guard_pass'] and all(v['passed'] for v in g['motion_guards'].values()) for g in old_gates['groups'])
    require(len(old_gates['groups']) == 6,'Inherited foot guard coverage differs')
    base_unchanged=cpu['frozen_base_sha256'] == read(run/'prepared/effective_config.json')['adaptation_parent']['base_model_sha256']
    gates=usability(groups,macros,ego_exact=ego_exact,inherited_foot_pass=foot_pass,cpu_verified=True,base_unchanged=base_unchanged)
    files.update({str(path):sha256(path) for path in source_paths})
    require(not torch.cuda.is_initialized(),'Report unexpectedly initialized CUDA')
    result=dict(protocol=PROTOCOL,status='cpu_verified',checked_at=now(),run=str(run),baseline=str(baseline),
        source_sha256=files,rows=rows,groups=groups,macros=macros,paired_recording_winrates=paired_winrates(rows),
        usability=gates,CPU_initialized_CUDA=False,source_inference_rerun=False,TEST_used=False,
        maximum_primary_metric_recheck_difference=maximum_metric_difference,matched_standalone_reference_control_arrays=matched_external,
        missingness='Position evidence unavailable; includes natural occlusion and simulated gaps, not exclusively physical occlusion.',
        metric_definition='World FK joint errors without GT alignment. Derivatives at30Hz. Foot speed uses GT-low-speed proxy, not contact labels.',
        temporal_aggregation='Per-window angular quantiles/counts are equal-recording window diagnostics, not frame-pooled quantiles.',
        scope='64 distinct historical VAL recordings, one window each; new55/old9 reported separately; three EGO anchors are retention checks.')
    write(run/'USABILITY_REVIEW.json',result)
    fmt=lambda v:'无有效样本' if v is None else f'{v:.3f}'
    lines=[f'本轮最终适配模型的强工程验收：{"通过" if gates["status"] == "pass" else "未通过"}。未自动替换模型。',
        '固定64条EXO验证录制、每条一个窗口，分别报告clean/noisy；另3条EGO锚点检验保持。全为历史VAL，未使用TEST。',
        '下表每格为修复后输入参考 → 未适配control → 最终适配；数据集内等录制平均，macro对三个数据集等权。单位mm。',
        '', '|数据集/输入|全身|可见位置关节|缺失/不可用位置关节|','|---|---:|---:|---:|']
    for dataset in (*DATASETS,'macro'):
        for regime in REGIMES:
            def value(label,metric):
                if dataset == 'macro':return macros['/'.join(('all64',regime,label))][metric]['equal_recording']
                return groups['/'.join(('all64',dataset,regime,label))][metric]['equal_recording']
            cells=[' → '.join(fmt(value(label,metric)) for label in LABELS) for metric in CORE]
            lines.append('|'+dataset+'/'+regime+'|'+'|'.join(cells)+'|')
    lines+=['', '强验收要求：clean和noisy的缺失关节等数据集宏平均均须比control及输入参考各改善至少5%；每个数据集/输入相对两者最多退化max(5mm,5%)；可见误差相对control最多增加max(1mm,2%)。同时要求EGO逐值一致、冻结基座保持及原有脚部工程门槛通过。',
        f'EGO输出逐值一致：{ego_exact}；脚部原门槛通过：{foot_pass}；冻结基座保持：{base_unchanged}。',
        '缺失不是纯物理遮挡。输出还包含原有观测IK、锚点和学习式场景反馈，收益不能全部归因于纯网络。低速脚代理不证明无漂浮或mesh穿模。',
        '', '逐录制胜负、new55/old9、有效分母、直接池化结果和手腕/时序诊断见[完整CPU报告](USABILITY_REVIEW.json)。等录制与池化结果分别保存，未混用。']
    temporary=run/'USABILITY_zh.md.tmp';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(run/'USABILITY_zh.md')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args(argv);result=render(args.run,args.baseline)
    print(json.dumps(dict(status=result['status'],usability=result['usability']['status'],rows=len(result['rows']))))


if __name__ == '__main__':
    main()
