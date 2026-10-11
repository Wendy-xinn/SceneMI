"""Score predeclared foot component and full turn goal; never modifies outputs."""
import json
from pathlib import Path
import torch,shutil
H=Path(__file__).parent;O=H/'runs/fresh55k_support_coverage_oct11'
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--scale-body-scene',action='store_true');parser.add_argument('--variant',default='scale_calibrated',choices=['scale_calibrated','body_local','angular_balanced','root_facing']);args=parser.parse_args()
 global O
 variant=args.variant if args.scale_body_scene else 'coverage_support'
 if args.scale_body_scene:O=H/'runs/control_scale_body_scene_oct11'
 if variant in ('angular_balanced','root_facing'):O=O/('root_facing_eval' if variant=='root_facing' else 'angular_balance_eval')
 checkpoint_root=O.parent if variant in ('angular_balanced','root_facing') else O
 reference='support_reference' if args.scale_body_scene else 'scene_reference'
 rows=json.loads((O/'rows.json').read_text());s=json.loads((O/'summary.json').read_text());d={r['index']:r for r in rows if r['scope']=='demo'};p=json.loads((O/'pre_registered_evaluation.json').read_text());foot=[];turn=[]
 def add(out,name,good):out.append(dict(check=name,passed=bool(good)))
 for i,r in d.items():
  a=r['metrics'][variant];b=r['metrics']['original55k'];fa=a['observed_floor_support'];fb=b['observed_floor_support']
  for phase in ('full','post80'):add(foot,f'{i}_{phase}_native_slide',fa[phase]['contact_vertex_slide_cm_frame']<=fb[phase]['contact_vertex_slide_cm_frame']*1.02)
  add(foot,f'{i}_contact_frames',fa['observed_contact_frame_fraction']>=.95);add(foot,f'{i}_height_p95',fa['lowest_foot_height_p95_cm']<=2.5);add(foot,f'{i}_height_mean',fa['lowest_foot_height_mean_cm']>=-2)
  add(foot,f'{i}_foot_amplitude',a['foot_relative_speed_ratio']>=b['foot_relative_speed_ratio']*.85)
  add(turn,f'{i}_turn_extent',abs(a['root_turn_progress_error_deg'])<=30);add(turn,f'{i}_absolute_final_heading',a['horizontal_root_yaw_final_abs_deg']<=45)
 a=d[906]['metrics'][variant];add(foot,'906_negative_root_and_head',a['pelvis_turn_deg']<0 and a['head_turn_deg']<0);add(foot,'906_root_peak',a['root_step_max_deg']<=7)
 for panel in ('means','confirmation_means'):
  for metric in ('gt_stance_slide_cm_frame','leg_angular_accel_p95_deg_frame2'):add(foot,panel+'_'+metric,s[panel][variant][metric]<=s[panel]['original55k'][metric]*1.02)
 add(foot,'primary_heading',s['means'][variant]['pelvis_orientation_mean_deg']<=s['means']['original55k']['pelvis_orientation_mean_deg']*.95)
 result=dict(foot_component_passed=all(x['passed'] for x in foot),full_joint_goal_passed=all(x['passed'] for x in foot+turn),foot_checks=foot,turn_checks=turn,pre_registered_protocol=p,scope='exposed development confirmation, not an independent benchmark')
 (O/(variant+'_acceptance.json' if args.scale_body_scene else 'acceptance.json')).write_text(json.dumps(result,indent=2))
 cp=torch.load(checkpoint_root/variant/'last.pt',map_location='cpu',weights_only=False);state=cp['model'];assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());trace=[json.loads(x) for x in (checkpoint_root/variant/'trace.jsonl').read_text().splitlines()];ref=[json.loads(x) for x in (H/('runs/fresh55k_support_coverage_oct11/coverage_support/trace.jsonl' if args.scale_body_scene else 'runs/fresh55k_scene_physics_oct11/winding_scene/trace.jsonl')).read_text().splitlines()]
 assert len(trace)==len(ref)==600
 match=all(all(a[k]==b[k] for k in ('step','group','length','identities','t','noise_checksum','use_scene','replay_schedule','replay_applied')) for a,b in zip(trace,ref));assert match
 (O/(variant+'_storage_and_pair_audit.json' if args.scale_body_scene else 'storage_and_pair_audit.json')).write_text(json.dumps(dict(checkpoint_bytes=(checkpoint_root/variant/'last.pt').stat().st_size,checkpoint_keys=list(cp),one_checkpoint_only=len(list((checkpoint_root/variant).glob('*.pt')))==1,finite_weights=True,matched_all_600_sample_noise_replay_steps=match,free_disk_bytes=shutil.disk_usage(O).free,optimizer_removed=cp['optimizer_removed']),indent=2))
 text='# '+variant+'：配对短训结果\n\n'+f"脚部组件验收：{result['foot_component_passed']}；完整朝向＋脚步目标：{result['full_joint_goal_passed']}。均从原55k独立开始，不接着旧候选训练。全部128帧由噪声生成；没有GT身体初态、历史或推理GT投影。\n\n"
 text+='|例子/版本|净转向°|末帧水平朝向误差°|转向幅度误差°|接触帧比例|足部最低点末段平均高度cm|接触点滑动全段/末段 cm/帧|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for i,r in d.items():
  for v in ('original55k',reference,variant):
   a=r['metrics'][v];f=a['observed_floor_support'];text+=f"|{i}/{v}|{a['pelvis_turn_deg']:.2f}|{a['horizontal_root_yaw_final_abs_deg']:.2f}|{a['root_turn_progress_error_deg']:.2f}|{f['observed_contact_frame_fraction']:.3f}|{f['lowest_foot_height_post80_mean_cm']:.2f}|{f['full']['contact_vertex_slide_cm_frame']:.3f}/{f['post80']['contact_vertex_slide_cm_frame']:.3f}|\n"
 text+='\n所有版本使用完整native足部顶点同样重评分。接触滑动只在接触点对上计算，必须一起看帧覆盖、点对数和动作幅度；GT支撑期关节脚滑仍另外保留。主面板16窗口×2种子，另32窗口×2种子确认，加850/906，共98条开发评估；不能宣称独立测试性能。\n\n|主面板指标|原55k|上轮参考|本组候选|\n|---|---:|---:|---:|\n'
 for title,m in [('朝向误差°','pelvis_orientation_mean_deg'),('GT支撑期脚滑cm/帧','gt_stance_slide_cm_frame'),('腿角加速度P95','leg_angular_accel_p95_deg_frame2'),('PA-MPJPE mm','pa_mpjpe_mm'),('世界MPJPE cm','mpjpe_cm')]:text+='|'+title+'|'+'|'.join(f"{s['means'][v][m]:.3f}" for v in ('original55k',reference,variant))+'|\n'
 if args.scale_body_scene and variant not in ('angular_balanced','root_facing'):
  text+='\n本组新增容差归一化的head位置/绝对朝向及head-body耦合目标，系数由TRAIN32条件编码器梯度审计缩放到.1。body_local另加当前生成x0关节的局部表面查询，所有GT身体/接触仅训练标签与评分；body_local并非在scale_calibrated上继续训练。\n'
 if variant=='angular_balanced':text+='\n本组分别提高角度系数：位置.02、head绝对旋转.02、head-body相对旋转.10；身体耦合噪声权重从alpha改为.25+.75alpha，系数依据TRAIN32完整主干分项梯度。没有加载body-local适配器；其他目标和采样保持一致。\n'
 if variant=='root_facing':text+='\n本组在角度平衡目标上新增TRAIN骨盆绝对SO(3)朝向监督，容差15度，系数见协议，噪声权重.25+.75alpha；GT只作训练标签，不用GT身体初始化。没有加载body-local适配器；其他目标和随机安排不变。\n'
 text+='\n未通过检查：'+', '.join(x['check'] for x in foot+turn if not x['passed'])+'。没有仅靠平均指标下降宣称解决。\n'
 if args.scale_body_scene and variant not in ('angular_balanced','root_facing'):
  text+='\n本组头部目标已改变，body_local在每个扩散步按初步生成x0的FK身体重新查询观测局部场景，额外一次网络前向。这里只是距离/方向条件，没有全场景SDF或人体碰撞证书。未知表面保留mask，动态物体只查当帧。600步采样/噪声/回放严格配对，GT仅训练标签与评分；未引入GT初态或GT安装变换进行生成。\n'
 elif not args.scale_body_scene:
  text+='\n本组头部目标与上轮相同，只检验支撑覆盖。局部BPS仍为相机固定网格，没有身体随姿态查询或碰撞证书。\n'
 text+='\n一组仅一个最终eval-only checkpoint，原55k未改。\n'
 camera_path=O/'generated_camera_diagnostic.json'
 if camera_path.exists():
  camera=json.loads(camera_path.read_text());text+='\n## 生成相机对应观测的残差\n\n|例子/版本|位置均值cm|朝向均值°|\n|---|---:|---:|\n'
  for row in camera['rows']:
   for v in ('original55k',reference,variant):
    m=row['metrics'][v];text+=f"|{row['index']}/{v}|{m['position_mean_cm']:.2f}|{m['orientation_mean_deg']:.2f}|\n"
  text+='\n由生成native头部FK重建相机；GT首帧固定安装标定仅用于绘图/评分，不参与生成或位置朝向重置。残差证明现有软条件没有保证遵循观测，本组是否匹配观测应由这些残差判断。相机及头关节仍须先做坐标/安装变换，再按观测位置与角度的误差尺度和置信度设鲁棒绝对目标；不能直接把相机当成头关节或让骨盆强制与头同向。\n'
 (O/(variant+'_ASSESSMENT_zh.md' if args.scale_body_scene else 'ASSESSMENT_zh.md')).write_text(text);print(json.dumps(result))
if __name__=='__main__':main()
