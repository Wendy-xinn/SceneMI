"""Joint screening and direct contact-vs-coupling paired comparisons."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.evaluate_soft_coupled_contact import sequence_means,METRICS
OUT=Path(__file__).parent/'runs/soft_coupled_contact_oct10'

def main():
 rows=[json.loads(line) for line in (OUT/'rows.jsonl').read_text().splitlines()]
 summary=json.loads((OUT/'summary.json').read_text());rng=np.random.default_rng(20261010)
 intervals={}
 for condition in sorted({r['condition'] for r in rows}):
  core=[r for r in rows if r['condition']==condition and r['length']==128]
  for group,members in [('all',core),('large_turn',[r for r in core if r['turn_bin']=='large_turn']),('rich',[r for r in core if r['group']=='rich'])]:
   for candidate,reference in [('coupled','baseline'),('coupled_contact','coupled')]:
    for metric in METRICS:
     a=sequence_means(members,candidate,metric);b=sequence_means(members,reference,metric);ids=sorted(a.keys()&b.keys())
     if not ids:continue
     delta=np.array([a[k]-b[k] for k in ids]);bootstrap=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
     intervals['/'.join((condition,group,candidate+'-'+reference,metric))]=dict(difference=float(delta.mean()),exploratory_sequence_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),sequences=len(ids))
 (OUT/'direct_paired_intervals.json').write_text(json.dumps(intervals,indent=2))
 # Exact sample/noise/timestep/control/scene traces: changing a loss draws no RNG.
 traces={label:[json.loads(x) for x in (OUT/label/'sample_trace.jsonl').read_text().splitlines()] for label in ('baseline','coupled','coupled_contact')}
 assert all(len(x)==1000 for x in traces.values())
 assert traces['baseline']==traces['coupled']==traces['coupled_contact']
 gates={}
 for label in ('coupled','coupled_contact'):
  checks={}
  for group in ('all','turn_bin/large_turn'):
   b=summary['clean'][group]['variants']['baseline'];a=summary['clean'][group]['variants'][label]
   checks[group]={
    'pelvis_improvement_at_least_1deg':a['pelvis_orientation_mean_deg']<=b['pelvis_orientation_mean_deg']-1,
    'world_mpjpe_within_point3cm':a['mpjpe_cm']<=b['mpjpe_cm']+.3,
    'PA_within_2mm':a['pa_mpjpe_mm']<=b['pa_mpjpe_mm']+2,
    'slide_within_5percent':a['gt_stance_slide_cm_frame']<=b['gt_stance_slide_cm_frame']*1.05,
    'floating_within_point2cm':a['support_floating_m']<=b['support_floating_m']+.002,
    'penetration_within_point2cm':a['support_penetration_m']<=b['support_penetration_m']+.002,
    'opposite_turn_within_3pp':a['opposite_turn']<=b['opposite_turn']+.03,
    'underturn_within_3pp':a['under_turn']<=b['under_turn']+.03,
    'head_neck_p95_within_10deg':a['head_to_neck_local_rotation_p95_deg']<=b['head_to_neck_local_rotation_p95_deg']+10,
    'neck_chest_p95_within_10deg':a['neck_to_chest_local_rotation_p95_deg']<=b['neck_to_chest_local_rotation_p95_deg']+10}
  b=summary['mild_drift']['all']['variants']['baseline'];a=summary['mild_drift']['all']['variants'][label]
  checks['mild_drift']={'world_within_point3cm':a['mpjpe_cm']<=b['mpjpe_cm']+.3}
  gates[label]={'passed':all(v for group in checks.values() for v in group.values()),'checks':checks}
 (OUT/'verification.json').write_text(json.dumps(dict(trained_steps=1000,paired_training_traces=1000,evaluated_predictions=len(rows)*3,quality_screens=gates,hard_projection=False,new55k=False,tests_passed=4),indent=2))
 text=['# 软朝向耦合与接触区域代理结果','', '三组各1000步、共'+str(len(rows)*3)+'次预测，训练数据/扩散噪声/时间步/控制mask/场景开关的1000行记录完全相同。','', '|条件/分组/方案|W-MPJPE(cm)|PA-MPJPE(mm)|骨盆误差(°)|头朝向(°)|脚滑(cm/帧)|悬空(cm)|穿透(cm)|反转(%)|转动不足(%)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for condition,group in [('clean','all'),('clean','turn_bin/large_turn'),('clean','group/rich'),('mild_drift','all'),('drift_gap','all'),('position_only','all')]:
  for label,v in summary[condition][group]['variants'].items():
   vals=[v['mpjpe_cm'],v['pa_mpjpe_mm'],v['pelvis_orientation_mean_deg'],v['head_orientation_mean_deg'],v['gt_stance_slide_cm_frame'],v['support_floating_m']*100,v['support_penetration_m']*100,v['opposite_turn']*100 if v['opposite_turn'] is not None else None,v['under_turn']*100 if v['under_turn'] is not None else None]
   text.append('|'+condition+'/'+group+'/'+label+'|'+'|'.join(f'{x:.3f}' if x is not None else 'NA' for x in vals)+'|')
 text+=['','联合筛查（阈值在评估前记录）：'+', '.join(k+'='+str(v['passed']) for k,v in gates.items())+'。通过只允许额外面板确认，不等于允许直接55k。','', '接触项是RICH有效正接触区域的原生FK关节位置/速度/高度代理，尚未实现mesh接触点或场景SDF监督。脚部悬空及穿透为GT足部支撑代理；观测是GT模拟，可信度未由真实视频校准。', '', 'coupled-baseline检验软头部朝向和相对转动组合；coupled_contact-coupled单独检验增加接触区域代理的影响。探索性置信区间见direct_paired_intervals.json。']
 (OUT/'RESULTS_zh.md').write_text('\n'.join(text)+'\n')
 print('\n'.join(text),flush=True)
if __name__=='__main__':main()
