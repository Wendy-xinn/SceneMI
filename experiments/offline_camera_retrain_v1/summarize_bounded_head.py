"""Write paired screening results, never approve a long run automatically."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.evaluate_bounded_head import sequence_means, METRICS

OUT=Path(__file__).parent/'runs/bounded_head_adaptation_oct10'
LABELS=('joint_all','joint_staged','rotation_only','mild_only')


def main():
    status=json.loads((OUT/'evaluation_status.json').read_text())
    assert status['status']=='completed' and status['windows']==192
    rows=[json.loads(line) for line in (OUT/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==2368 and status['evaluated_predictions']==9472
    traces={label:[json.loads(line) for line in (OUT/label/'sample_trace.jsonl').read_text().splitlines()] for label in LABELS}
    assert all(len(v)==1000 for v in traces.values())
    assert all(v==traces['joint_all'] for v in traces.values())
    summary=json.loads((OUT/'summary.json').read_text())
    paired={};rng=np.random.default_rng(20261010)
    for condition in ('clean','mild_drift','drift','drift_gap','position_only','low_confidence'):
        for subset in ('all','large_turn'):
            members=[row for row in rows if row['condition']==condition and row['length']==128 and (subset=='all' or row['turn_bin']=='large_turn')]
            for candidate,reference in [('joint_staged','joint_all'),('rotation_only','joint_staged'),('mild_only','joint_staged')]:
                for metric in METRICS:
                    a=sequence_means(members,candidate,metric);b=sequence_means(members,reference,metric);ids=sorted(set(a)&set(b))
                    if not ids:continue
                    d=np.array([a[k]-b[k] for k in ids]);boot=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
                    paired['/'.join((condition,subset,candidate+'-'+reference,metric))]=dict(difference=float(d.mean()),exploratory_sequence_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    (OUT/'joint_paired_intervals.json').write_text(json.dumps(paired,indent=2))
    def table(condition,subset='all'):
        data=summary[condition][subset]['variants'];s='| 方案 | W-MPJPE cm | PA-MPJPE mm | 骨盆朝向 ° | 脚滑 cm/帧 | 悬浮 cm | 穿透代理 cm | 反向转身 | 转角不足 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for label in LABELS:
            v=data[label]
            def percent(key):return '—' if v[key] is None else f'{100*v[key]:.1f}%'
            s+=f"| {label} | {v['mpjpe_cm']:.3f} | {v['pa_mpjpe_mm']:.2f} | {v['pelvis_orientation_mean_deg']:.2f} | {v['gt_stance_slide_cm_frame']:.3f} | {100*v['support_floating_m']:.3f} | {100*v['support_penetration_m']:.3f} | {percent('opposite_turn')} | {percent('under_turn')} |\n"
        return s
    report="""# 温和头部条件与分阶段适配结果

四组各1000步，相同55k起点与数据/扩散噪声/控制/scene抽样。joint_all全模型从第1步更新；其余先200步仅更新条件编码器，再800步联合更新；编码器和解冻后的主体分别warmup100步，随后按共同1000步cosine下降，最低0.1倍lr。orientation_v5/所有GT动作与场景数据监督保持不变，无新增头部损失或硬投影。

joint_staged使用完整干净关节轨迹；rotation_only只去掉20%头部轨迹朝向；mild_only只在20%头部轨迹添加范数≤1cm、yaw≤3°的平滑误差。配额在原控制mask之前计数，不叠加扰动，不改变训练可信度。四组1000条逐步trace含数据、noisy输入hash、timestep、control hash和scene开关，必须完全相同后生成报告。

192个固定开发窗口、2噪声、DDIM20；核心144个128帧、另64/192帧各24窗口。六种全窗口条件，另32窗口无控制诊断，总计9472次预测；主表先对序列内平均再序列等权，bootstrap为开发集上的探索性区间。所有模型评估使用同一种理想解剖头部输入，仍非真实视频估计实验。

## 干净输入（144窗口）

"""+table('clean')+'\n## 大转身干净输入\n\n'+table('clean','turn_bin/large_turn')+'\n## 温和漂移（1cm、3°）\n\n'+table('mild_drift')+'\n## 更强漂移与半秒缺失\n\n'+table('drift_gap')+'\n## 只有位置，没有朝向\n\n'+table('position_only')
    report+="""
## 验收与边界

是否通过联合验收见ASSESSMENT_zh.md；不能仅以位置误差或悬浮的单项改善入选。需同时看骨盆/胸/颈、反向转身/转角不足、脚滑、悬浮和穿透。后两项是由GT足部估计支撑面及容差得到的代理，不是真实mesh碰撞验证。

本轮使用训练标签构造理想关节观测；未知真人安装关系、视频估计误差与真实可信度没有经过标定。场景缓存保持干净，条件噪声实验不是联合带噪声场景重建。低可信度测试仅诊断元数据变化，训练中没有新增可靠性学习。这是开发短训，不是独立测试或长训收敛结论。
"""
    (OUT/'RESULTS_zh.md').write_text(report)
    verification=json.loads((OUT/'verification.json').read_text());verification.update(paired_training_traces=1000,completed_prediction_count=9472,head_hard_projection=False,long_training_started=False)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    print('saved',OUT/'RESULTS_zh.md')

if __name__=='__main__':main()
