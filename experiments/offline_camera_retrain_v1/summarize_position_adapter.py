"""Report paired, sequence-balanced adapter results without selecting examples."""
import json
from pathlib import Path
OUT=Path(__file__).parent/'runs/position_adapter_head_oct10'
def main():
 s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'invariance_audit.json').read_text())
 text=['# 独立位置支路评估（10月10日）','',f"完整输入384次预测最大逐元素差异：{audit['maximum_complete_motion_difference']:.8g}。原模型权重SHA256一致。",'', '|输入/模型|世界MPJPE(cm)|PA-MPJPE(mm)|骨盆朝向(°)|脚滑(cm/帧)|悬空(cm)|穿透(cm)|反转率(%)|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for c in ('clean','position_only'):
  for label,v in s[c]['all']['variants'].items():
   vals=[v['mpjpe_cm'],v['pa_mpjpe_mm'],v['pelvis_orientation_mean_deg'],v['gt_stance_slide_cm_frame'],v['support_floating_m']*100,v['support_penetration_m']*100,v['opposite_turn']*100]
   text.append('|'+c+'/'+label+'|'+'|'.join(f'{x:.3f}' for x in vals)+'|')
 b=s['position_only']['all']['variants']['joint_all'];a=s['position_only']['all']['variants']['adapter']
 improved=a['mpjpe_cm']<b['mpjpe_cm'] and a['gt_stance_slide_cm_frame']<=b['gt_stance_slide_cm_frame'] and a['support_floating_m']<=b['support_floating_m'] and a['support_penetration_m']<=b['support_penetration_m']
 text+=['','仅位置同时改善误差、脚滑及支撑代理：'+str(improved)+'。这只是探索性筛查，不能证明完整输入转身已解决。','', '核心128帧144窗口按人体序列平均，另检查64/192帧；192窗口×2种子×2条件×2模型=1536次预测。观测为GT模拟理想关节轨迹，场景缓存未加入估计误差，尚不能代表真实视频输入。悬空和穿透为GT支撑足代理指标。','', '独立48窗口面板评估与否另行记录。未启动新的55k；未加硬投影。']
 (OUT/'RESULTS_zh.md').write_text('\n'.join(text)+'\n')
 (OUT/'verification.json').write_text(json.dumps(dict(evaluation_completed=True,quality_screen_passed=improved,complete_input_unchanged=audit['maximum_complete_motion_difference']<=1e-6,new_55k_started=False,hard_projection=False),indent=2))
 print('\n'.join(text))
if __name__=='__main__':main()
