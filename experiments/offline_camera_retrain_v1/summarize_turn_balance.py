"""Write paired diagnostic intervals and a report without accepting a candidate by head error alone."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).parent/'runs/turn_balance_oct09'
LABELS={'orientation_v2':'v2软条件','orientation_v3':'v3时间归一化','v3_head_final':'v3末尾硬旋转','v3_head_each_step':'v3全程硬旋转','v3_head_late_step':'v3低噪声硬旋转','v3_head_trained':'v3训练期硬旋转'}

def main():
 s=json.loads((P/'summary.json').read_text());rows=[json.loads(line) for line in (P/'rows.jsonl').read_text().splitlines()];core=[r for r in rows if r['length']==128];rng=np.random.default_rng(1009);intervals={}
 for label in LABELS:
  if label=='orientation_v2':continue
  intervals[label]={}
  for metric in ['mpjpe_cm','pelvis_orientation_mean_deg','gt_stance_slide_cm_frame','support_floating_m','both_feet_moving_fraction','head_to_neck_local_rotation_p95_deg']:
   byseq=defaultdict(list)
   for row in core:byseq[row['identity']['sequence_id']].append(row['variants'][label][metric]-row['variants']['orientation_v2'][metric])
   delta=np.array([np.mean(values) for values in byseq.values()]);boot=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
   intervals[label][metric]={'difference':float(delta.mean()),'exploratory_sequence_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
 (P/'paired_intervals.json').write_text(json.dumps(intervals,indent=2))
 traces={}
 for label,path in [('v2',P.parent/'turn_repair_oct08/orientation_v2/sample_trace.jsonl'),('v3',P/'orientation_v3/sample_trace.jsonl'),('v3_hard',P/'orientation_v3_hard/sample_trace.jsonl')]:traces[label]=path.read_text().splitlines()
 assert traces['v2']==traces['v3']==traces['v3_hard'],'Training sample trace mismatch'
 assert len(rows)==384 and all(len(r['variants'])==6 for r in rows)
 lines=['# 时间尺度与头部硬约束结果（2026-10-09）','','两组1k短训及固定配对评估完成。两组从同一原55k初始化，训练样本记录与此前v2逐行一致；同一学习率、控制mask分布、噪声和数据协议。27项相关回归测试通过。原55k未覆盖，未启动新55k。','','## 实验设置','','v3只归一化累计转向损失的时间尺度；128帧的目标强度与v2完全相同。v3_hard在相同v3目标下，对已观测的虚拟相机头部旋转进行可微投影，使局部头部姿态损失能向根部及上身旋转传播梯度。真实PV没有已知可靠标定，始终保留软条件。','','复用48条验证人物序列、144个128帧窗口（不包含原展示序列），每窗两组配对噪声，DDIM20步。同组噪声的v2预测已与旧缓存核对。另加同一批人物中24个64帧、24个192帧开发探针。共192个窗口、384个窗口/噪声组合、2304份指标；末尾投影直接复用v3生成，其余共1920次采样。总体表只使用原144个128帧窗口，序列内平均后序列等权，四个数据/角色组各12条。此为开发诊断，不是未参与调参的独立测试集。','','所有硬旋转方法用输入相机姿态及已知安装关系，不读取验证GT姿态或GT接触。末尾、全程、低噪声(alpha≥0.8)三个版本是仅推理期消融；训练期版在训练和每步DDIM中都投影。只求解头部关节15局部旋转，没有通过根平移贴齐相机位置。头部旋转误差归零是约束定义，不能当作模型质量提高的独立证据。','','## 总体与大转身','','各项指标越小越好；悬浮为GT脚高度估计支撑面、扣除5cm容差后的代理，不是真实mesh接触距离。头对颈部旋转为每窗局部旋转角95分位的平均，不是某一解剖轴或全样本95分位。']
 specs=[('mpjpe_cm','W-MPJPE cm',1),('head_orientation_mean_deg','头旋转°',1),('pelvis_orientation_mean_deg','骨盆旋转°',1),('gt_stance_slide_cm_frame','GT相位脚滑 cm/帧',1),('support_floating_m','悬浮代理 cm',100),('both_feet_moving_fraction','双脚同时移动 %',100),('head_to_neck_local_rotation_p95_deg','头对颈部旋转°',1)]
 for name,title in [('all','总体144窗'),('turn_bin/large_turn','大转身（GT净转角≥60°）')]:
  g=s[name];lines+=['',f'### {title}：{g["windows"]}窗、{g["sequences"]}条人物序列','', '| 方法 | '+' | '.join(label for _,label,_ in specs)+' |','|---|'+'---:|'*len(specs)]
  for label in LABELS:lines.append('| '+LABELS[label]+' | '+' | '.join(f'{g["variants"][label][key]*scale:.3f}' for key,_,scale in specs)+' |')
 lines+=['','## 分域与窗口长度','','下面保留全部方案，避免只按头部误差挑例。']
 for group in ['group/trumans','group/camera_wearer','group/interactee','group/rich','length/64','length/192']:
  g=s[group];lines+=['',f'### {group}：{g["windows"]}窗、{g["sequences"]}条人物序列','','| 方法 | 骨盆旋转° | 脚滑 cm/帧 | 悬浮代理 cm | 头对颈部旋转° |','|---|---:|---:|---:|---:|']
  for label in LABELS:
   m=g['variants'][label];lines.append(f'| {LABELS[label]} | {m["pelvis_orientation_mean_deg"]:.3f} | {m["gt_stance_slide_cm_frame"]:.3f} | {100*m["support_floating_m"]:.3f} | {m["head_to_neck_local_rotation_p95_deg"]:.3f} |')
 lines+=['','## 记录','','- [全部分组指标](summary.json)','- [相对v2的探索性配对区间](paired_intervals.json)：以人物序列bootstrap，不视作跨录制独立性保证。','- [相机标定与最终投影检查](head_constraint_audit.json)：已知虚拟相机的GT头部标定残差小于0.04°；实拍PV最大约33°，故未硬投影。最终单独修改关节15不会改变22关节FK位置，不能用它改善脚步。','- 固定抽样与checkpoint记录manifest.json；原生动作、相机和rest保存在motions；训练及评估日志本地保留。','', '下一方案选择必须同时检查骨盆转向、步态支撑、位置和颈部补偿，不能仅按受硬约束控制的头部误差排序。候选结论见后续人工审核段落。','']
 (P/'RESULTS_zh.md').write_text('\n'.join(lines))
 (P/'verification.json').write_text(json.dumps({'pipeline_completed':True,'same_training_sample_trace':True,'windows':192,'evaluated_predictions':2304,'regression_tests':27,'quality_accepted':False,'reason':'Paired results produced; physical/motion quality requires joint review, head-error zero alone is not acceptance'},indent=2))
 print('Report and paired intervals complete; quality not automatically accepted.')
if __name__=='__main__':main()
