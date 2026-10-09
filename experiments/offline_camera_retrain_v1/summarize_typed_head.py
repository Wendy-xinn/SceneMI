"""Write paired screening results, never approve a long run automatically."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.evaluate_typed_head import sequence_means, METRICS

OUT=Path(__file__).parent/'runs/typed_head_track_oct10'
LABELS=('camera_control','joint_clean','joint_noise','joint_reliable')


def main():
    status=json.loads((OUT/'evaluation_status.json').read_text())
    assert status['status']=='completed' and status['windows']==192
    rows=[json.loads(line) for line in (OUT/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==1984 and status['evaluated_predictions']==7936
    traces={label:[json.loads(line) for line in (OUT/label/'sample_trace.jsonl').read_text().splitlines()] for label in LABELS}
    assert all(len(v)==500 for v in traces.values())
    assert all(v==traces['camera_control'] for v in traces.values())
    summary=json.loads((OUT/'summary.json').read_text())
    paired={};rng=np.random.default_rng(20261010)
    for condition in ('clean','drift','drift_gap','position_only','low_confidence'):
        for subset in ('all','large_turn'):
            members=[row for row in rows if row['condition']==condition and row['length']==128 and (subset=='all' or row['turn_bin']=='large_turn')]
            for candidate,reference in [('joint_noise','joint_clean'),('joint_reliable','joint_noise'),('joint_reliable','joint_clean')]:
                for metric in METRICS:
                    a=sequence_means(members,candidate,metric);b=sequence_means(members,reference,metric);ids=sorted(set(a)&set(b))
                    if not ids:continue
                    d=np.array([a[k]-b[k] for k in ids]);boot=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
                    paired['/'.join((condition,subset,candidate+'-'+reference,metric))]=dict(difference=float(d.mean()),exploratory_sequence_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    (OUT/'joint_paired_intervals.json').write_text(json.dumps(paired,indent=2))
    def table(condition,subset='all'):
        data=summary[condition][subset]['variants'];s='| 方案 | W-MPJPE cm | PA-MPJPE mm | 骨盆朝向 ° | 脚滑 cm/帧 | 悬浮 cm | 反向转身 | 转角不足 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for label in LABELS:
            v=data[label]
            def percent(key):return '—' if v[key] is None else f'{100*v[key]:.1f}%'
            s+=f"| {label} | {v['mpjpe_cm']:.3f} | {v['pa_mpjpe_mm']:.2f} | {v['pelvis_orientation_mean_deg']:.2f} | {v['gt_stance_slide_cm_frame']:.3f} | {100*v['support_floating_m']:.3f} | {percent('opposite_turn')} | {percent('under_turn')} |\n"
        return s
    report='''# 头部观测类型与可靠性短训结果

四组各500步，相同55k起点、样本/扩散噪声/控制与场景抽样；已有orientation_v5监督保持不变。四组每一步sample_trace完全一致。192个固定开发窗口、2个种子、DDIM20；144个128帧核心窗口，另24个64帧、24个192帧窗口。五种全窗口输入条件，另32窗口无控制诊断，合计7936次预测。主表先按序列求均值，再对序列均衡平均；bootstrap为开发集上的探索性区间，不代替独立测试。

camera_control输入相机轨迹；其余输入理想人体头部关节。joint_noise训练含偏差/漂移/缺失/只有位置；joint_reliable另外输入模拟置信度和增加软观测损失。可靠性与软损失在这一组共同变化，不能把收益单独归因到其中一项。

## 干净输入（144窗口）

'''+table('clean')+'\n## 大转身干净输入\n\n'+table('clean','turn_bin/large_turn')+'\n## 漂移和半秒缺失（144窗口）\n\n'+table('drift_gap')+'\n## 大转身漂移和缺失\n\n'+table('drift_gap','turn_bin/large_turn')+'\n## 只有头部位置，没有旋转\n\n'+table('position_only')
    report+='''
## 解释边界

- head_joint条件来自GT，只代表正确解剖语义的理想观测。相机→关节函数已经通过非单位安装旋转、旋转平移偏置验证，但真实PV标定/视频估计尚未验证，不能用GT反推安装关系作为实际推理方案。
- 置信度来自已知注入扰动的模拟器，属于合成可信度实验，实际估计器置信度未经校准。低可信度条件仅改变metadata，用于诊断模型是否利用可靠性。
- 场景与相机缓存保持干净，本轮是条件输入鲁棒性，不代表噪声场景重建的整个链路。
- 500步同时有warmup与cosine下降，只用于低成本筛选；没有充分收敛或正式泛化结论。相机和关节组输入点不同，应结合同类型joint_clean/joint_noise/joint_reliable配对结果判断，不把理想输入视为数据处理错误。
- 尚未自动选用任何方案，也未启动55k。颈/胸、脚滑、穿透、悬浮、不同角色与窗口长度完整数据见summary.json和配对区间；进入下一步需要联合检查这些指标。
'''
    (OUT/'RESULTS_zh.md').write_text(report)
    verification=json.loads((OUT/'verification.json').read_text());verification.update(paired_training_traces=500,completed_prediction_count=7936,head_hard_projection=False,long_training_started=False)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    print('saved',OUT/'RESULTS_zh.md')

if __name__=='__main__':main()
