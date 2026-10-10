"""Score predeclared foot component and full turn goal; never modifies outputs."""
import json
from pathlib import Path
import torch,shutil
H=Path(__file__).parent;O=H/'runs/fresh55k_support_coverage_oct11'
def main():
 rows=json.loads((O/'rows.json').read_text());s=json.loads((O/'summary.json').read_text());d={r['index']:r for r in rows if r['scope']=='demo'};p=json.loads((O/'pre_registered_evaluation.json').read_text());foot=[];turn=[]
 def add(out,name,good):out.append(dict(check=name,passed=bool(good)))
 for i,r in d.items():
  a=r['metrics']['coverage_support'];b=r['metrics']['original55k'];fa=a['observed_floor_support'];fb=b['observed_floor_support']
  for phase in ('full','post80'):add(foot,f'{i}_{phase}_native_slide',fa[phase]['contact_vertex_slide_cm_frame']<=fb[phase]['contact_vertex_slide_cm_frame']*1.02)
  add(foot,f'{i}_contact_frames',fa['observed_contact_frame_fraction']>=.95);add(foot,f'{i}_height_p95',fa['lowest_foot_height_p95_cm']<=2.5);add(foot,f'{i}_height_mean',fa['lowest_foot_height_mean_cm']>=-2)
  add(foot,f'{i}_foot_amplitude',a['foot_relative_speed_ratio']>=b['foot_relative_speed_ratio']*.85)
  add(turn,f'{i}_turn_extent',abs(a['root_turn_progress_error_deg'])<=30);add(turn,f'{i}_absolute_final_heading',a['horizontal_root_yaw_final_abs_deg']<=45)
 a=d[906]['metrics']['coverage_support'];add(foot,'906_negative_root_and_head',a['pelvis_turn_deg']<0 and a['head_turn_deg']<0);add(foot,'906_root_peak',a['root_step_max_deg']<=7)
 for panel in ('means','confirmation_means'):
  for metric in ('gt_stance_slide_cm_frame','leg_angular_accel_p95_deg_frame2'):add(foot,panel+'_'+metric,s[panel]['coverage_support'][metric]<=s[panel]['original55k'][metric]*1.02)
 add(foot,'primary_heading',s['means']['coverage_support']['pelvis_orientation_mean_deg']<=s['means']['original55k']['pelvis_orientation_mean_deg']*.95)
 result=dict(foot_component_passed=all(x['passed'] for x in foot),full_joint_goal_passed=all(x['passed'] for x in foot+turn),foot_checks=foot,turn_checks=turn,pre_registered_protocol=p,scope='exposed development confirmation, not an independent benchmark')
 (O/'acceptance.json').write_text(json.dumps(result,indent=2))
 cp=torch.load(O/'coverage_support/last.pt',map_location='cpu',weights_only=False);state=cp['model'];assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());trace=[json.loads(x) for x in (O/'coverage_support/trace.jsonl').read_text().splitlines()];ref=[json.loads(x) for x in (H/'runs/fresh55k_scene_physics_oct11/winding_scene/trace.jsonl').read_text().splitlines()]
 assert len(trace)==len(ref)==600
 match=all(all(a[k]==b[k] for k in ('step','group','length','identities','t','noise_checksum','use_scene','replay_schedule','replay_applied')) for a,b in zip(trace,ref));assert match
 (O/'storage_and_pair_audit.json').write_text(json.dumps(dict(checkpoint_bytes=(O/'coverage_support/last.pt').stat().st_size,checkpoint_keys=list(cp),one_checkpoint_only=len(list((O/'coverage_support').glob('*.pt')))==1,finite_weights=True,matched_all_600_sample_noise_replay_steps=match,free_disk_bytes=shutil.disk_usage(O).free,optimizer_removed=cp['optimizer_removed']),indent=2))
 text='# 支撑覆盖修复：配对短训结果\n\n'+f"脚部组件验收：{result['foot_component_passed']}；完整朝向＋脚步目标：{result['full_joint_goal_passed']}。均从原55k独立开始，不接着旧候选训练。全部128帧由噪声生成；没有GT身体初态、历史或推理GT投影。\n\n"
 text+='|例子/版本|净转向°|末帧水平朝向误差°|转向幅度误差°|接触帧比例|足部最低点末段平均高度cm|接触点滑动全段/末段 cm/帧|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for i,r in d.items():
  for v in ('original55k','scene_reference','coverage_support'):
   a=r['metrics'][v];f=a['observed_floor_support'];text+=f"|{i}/{v}|{a['pelvis_turn_deg']:.2f}|{a['horizontal_root_yaw_final_abs_deg']:.2f}|{a['root_turn_progress_error_deg']:.2f}|{f['observed_contact_frame_fraction']:.3f}|{f['lowest_foot_height_post80_mean_cm']:.2f}|{f['full']['contact_vertex_slide_cm_frame']:.3f}/{f['post80']['contact_vertex_slide_cm_frame']:.3f}|\n"
 text+='\n所有版本使用完整native足部顶点同样重评分。接触滑动只在接触点对上计算，必须一起看帧覆盖、点对数和动作幅度；GT支撑期关节脚滑仍另外保留。主面板16窗口×2种子，另32窗口×2种子确认，加850/906，共98条开发评估；不能宣称独立测试性能。\n\n|主面板指标|原55k|上轮场景参考|本轮支撑覆盖|\n|---|---:|---:|---:|\n'
 for title,m in [('朝向误差°','pelvis_orientation_mean_deg'),('GT支撑期脚滑cm/帧','gt_stance_slide_cm_frame'),('腿角加速度P95','leg_angular_accel_p95_deg_frame2'),('PA-MPJPE mm','pa_mpjpe_mm'),('世界MPJPE cm','mpjpe_cm')]:text+='|'+title+'|'+'|'.join(f"{s['means'][v][m]:.3f}" for v in ('original55k','scene_reference','coverage_support'))+'|\n'
 text+='\n未通过检查：'+', '.join(x['check'] for x in foot+turn if not x['passed'])+'。没有仅靠平均指标下降宣称解决。\n\n头部软条件、绝对位置/朝向监督与上轮相同，本次仅检验支撑覆盖补丁，不能归因于加强头部约束。control_gradient_audit.json显示两个固定开发例子的输出梯度尺度不均衡，不能代表全分布参数梯度。下一步头部目标需单独按位置/角度容差与轨迹置信度校准；必须先明确相机→头部安装变换，不将相机直接硬吸附到头关节，不引入GT身体初态。局部BPS仍以相机固定采样网格为中心，未跟随生成身体做全场景碰撞查询；沙发/膝/臀部碰撞没有保证。\n\n一组仅一个最终eval-only checkpoint，600步采样/噪声/回放与上轮全部匹配。原55k未改，代码及紧凑记录备份GitHub，大权重与生成数组留本地。\n'
 camera_path=O/'generated_camera_diagnostic.json'
 if camera_path.exists():
  camera=json.loads(camera_path.read_text());text+='\n## 生成相机对应观测的残差\n\n|例子/版本|位置均值cm|朝向均值°|\n|---|---:|---:|\n'
  for row in camera['rows']:
   for v in ('original55k','scene_reference','coverage_support'):
    m=row['metrics'][v];text+=f"|{row['index']}/{v}|{m['position_mean_cm']:.2f}|{m['orientation_mean_deg']:.2f}|\n"
  text+='\n由生成native头部FK重建相机；GT首帧固定安装标定仅用于绘图/评分，不参与生成或位置朝向重置。残差证明现有软条件没有保证遵循观测，本次支撑补丁也没有加强头部约束。相机及头关节仍须先做坐标/安装变换，再按观测位置与角度的误差尺度和置信度设鲁棒绝对目标；不能直接把相机当成头关节或让骨盆强制与头同向。\n'
 (O/'ASSESSMENT_zh.md').write_text(text);print(json.dumps(result))
if __name__=='__main__':main()
