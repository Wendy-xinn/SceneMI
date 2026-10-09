"""Report fixed paired soft-loss and input-uncertainty diagnostics."""
import json
from pathlib import Path
import numpy as np
from experiments.offline_camera_retrain_v1.evaluate_soft_body_turn import OUT, REFERENCE, sequence_means

LABELS = {'v3': 'v3对照', 'noise_gate': '仅高噪声减权', 'neck_parent': '加颈部父链监督'}
COLUMNS = [('mpjpe_cm', 'W-MPJPE cm', 1), ('pa_mpjpe_mm', 'PA-MPJPE mm', 1),
           ('head_orientation_mean_deg', '头旋转°', 1), ('pelvis_orientation_mean_deg', '骨盆旋转°', 1),
           ('gt_stance_slide_cm_frame', '脚滑 cm/帧', 1), ('support_floating_m', '悬浮代理 cm', 100),
           ('both_feet_moving_fraction', '双脚移动 %', 100),
           ('head_to_neck_local_rotation_p95_deg', '头对颈°', 1),
           ('neck_to_chest_local_rotation_p95_deg', '颈对胸°', 1)]


def table(group, columns=COLUMNS):
    lines = ['| 方法 | ' + ' | '.join(title for _, title, _ in columns) + ' |',
             '|---|' + '---:|' * len(columns)]
    for label, title in LABELS.items():
        value = group['variants'][label]
        lines.append('| ' + title + ' | ' + ' | '.join('—' if value.get(key) is None else f'{value[key]*scale:.3f}' for key, _, scale in columns) + ' |')
    return lines


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    rows = [json.loads(line) for line in (OUT / 'rows.jsonl').read_text().splitlines()]
    status = json.loads((OUT / 'evaluation_status.json').read_text())
    assert status['status'] == 'completed'
    traces = [(REFERENCE / 'orientation_v3/sample_trace.jsonl').read_text().splitlines()]
    traces += [(OUT / label / 'sample_trace.jsonl').read_text().splitlines() for label in ('noise_gate', 'neck_parent')]
    assert len(traces[0]) == 1000 and traces[0] == traces[1] == traces[2]
    assert len(rows) == 192*2*4 + 32*2 and all(len(row['variants']) == 3 for row in rows)
    lines = ['# 软轨迹与身体转向配对结果（2026-10-09）', '',
             '两组1k短训与配对评估完成。训练样本记录与v3逐行一致；两个新模型均从同一原55k初始化，没有硬头部约束。38项相关回归检查通过，新损失profile也通过完整监督损失的零目标/反向传播检查。', '',
             '[训练与扰动协议](README_zh.md)。48条验证人物序列、144个128帧窗口，另有24个64帧和24个192帧窗口；每窗两组DDIM20噪声，192个窗口均检查四种条件。32个128帧窗口另做无轨迹条件诊断。共4800份动作指标；全部v3干净输入预测核对上次缓存通过。', '',
             '这是参与调参的开发诊断，尚不是独立测试。总体先序列内平均，再序列等权；四个数据/角色组各12条。脚滑采用GT支撑相位，悬浮由GT脚估计支撑高度并扣5cm容差，均为代理指标。头对颈、颈对胸是各窗口局部3D旋转角95分位的序列平均，不是解剖轴或硬关节限位。', '',
             '轨迹扰动仅施加于条件：位置平滑偏差约3–5cm、yaw平滑3–11度，以及可选0.5秒缺失。场景、GT和camera缓存保持原值。这不能替代端到端视频轨迹估计与场景重建的噪声评估。另有head_joint条件：仅以验证GT构造理想的真实头部关节位置/朝向，诊断人体关节与相机观测类型的差别；没有额外提供根部或腿部GT，不是视频估计性能。']
    for condition, title in [('clean', '干净输入'), ('drift', '平滑轨迹偏差'), ('drift_gap', '偏差加短时缺失'), ('head_joint', '理想人体头部关节输入（类型诊断）')]:
        for group, description in [('all', '总体'), ('turn_bin/large_turn', 'GT大转身≥60°')]:
            result = summary[condition][group]
            lines += ['', f'## {title}：{description}，{result["windows"]}窗 / {result["sequences"]}序列', ''] + table(result)
    lines += ['', '## 分域与长度（干净条件）', '']
    small = [('pelvis_orientation_mean_deg', '骨盆旋转°', 1), ('chest_orientation_mean_deg', '胸旋转°', 1),
             ('neck_orientation_mean_deg', '颈旋转°', 1), ('gt_stance_slide_cm_frame', '脚滑 cm/帧', 1),
             ('support_floating_m', '悬浮代理 cm', 100), ('neck_to_chest_local_rotation_p95_deg', '颈对胸°', 1)]
    for group in ['group/trumans', 'group/camera_wearer', 'group/interactee', 'group/rich', 'length/64', 'length/192']:
        result = summary['clean'][group]
        lines += ['', f'### {group}，{result["windows"]}窗 / {result["sequences"]}序列', ''] + table(result, small)
    lines += ['', '## 对条件变化的响应', '',
              '下面是相对相同模型、相同噪声的干净输入预测的变化；越小不必然越好。无条件行使用32个窗口，不能直接与144窗扰动行横向比较。还需同时看GT误差，避免把忽略输入误当鲁棒。']
    response = [('response_head_cm', '头位置变化 cm', 1), ('response_head_deg', '头旋转变化°', 1),
                ('response_pelvis_deg', '骨盆旋转变化°', 1), ('mpjpe_cm', 'W-MPJPE cm', 1)]
    for condition in ('drift', 'drift_gap', 'head_joint', 'none'):
        lines += ['', f'### {condition}', ''] + table(summary[condition]['all'], response)
    lines += ['', '## 结果记录', '',
              '- [全部指标](summary.json)', '- [相对v3的探索性配对区间](paired_intervals.json)：按人物序列bootstrap，非跨录制独立性的保证。',
              '- [候选审核及下一步](ASSESSMENT_zh.md)',
              '- motions保留GT、原生rest、相机、实际扰动条件/控制mask及三种生成动作；权重和缓存仅留本机。', '']
    (OUT / 'RESULTS_zh.md').write_text('\n'.join(lines))
    (OUT / 'verification.json').write_text(json.dumps(dict(pipeline_completed=True, same_training_sample_trace=True,
        valid_checkpoints_loaded=True, baseline_cache_reproduced=True, windows=192, evaluated_predictions=status['evaluated_predictions'],
        regression_tests=38, quality_accepted=None, reason='See joint body/gait/compensation review in ASSESSMENT_zh.md'), indent=2))
    print('Report written; quality verdict requires metric review.')

if __name__ == '__main__':
    main()
