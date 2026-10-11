"""Compact report of fixed paired arms; verdict uses preregistered gates."""
import json,statistics
from pathlib import Path
H=Path(__file__).parent;O=H/'runs/control_scale_body_scene_oct11'
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root-facing',action='store_true');parser.add_argument('--angular-balance',action='store_true');args=parser.parse_args()
    if args.root_facing:args.angular_balance=True
    global O
    if args.angular_balance:O=O/('root_facing_eval' if args.root_facing else 'angular_balance_eval')
    rows=json.loads((O/'rows.json').read_text());summary=json.loads((O/'summary.json').read_text());camera=json.loads((O/'generated_camera_diagnostic.json').read_text());labels=('original55k','support_reference','scale_calibrated','body_local');
    if args.angular_balance:labels=('original55k','support_reference','scale_calibrated','angular_balanced')
    if args.root_facing:labels=('original55k','support_reference','angular_balanced','root_facing')
    metrics=('pelvis_orientation_mean_deg','head_orientation_mean_deg','gt_stance_slide_cm_frame','leg_angular_accel_p95_deg_frame2','mpjpe_cm','pa_mpjpe_mm');groups={}
    for scope in ('panel','confirmation'):
        groups[scope]={}
        for dataset in sorted({r['identity']['dataset'] for r in rows if r['scope']==scope}):
            subset=[r for r in rows if r['scope']==scope and r['identity']['dataset']==dataset]
            groups[scope][dataset]=dict(seed_cases=len(subset),means={v:{m:statistics.mean(r['metrics'][v][m] for r in subset) for m in metrics} for v in labels})
    (O/'per_dataset_summary.json').write_text(json.dumps(groups,indent=2))
    verdict={v:json.loads((O/f'{v}_acceptance.json').read_text()) for v in (('root_facing',) if args.root_facing else ('angular_balanced',) if args.angular_balance else ('scale_calibrated','body_local'))}
    text='# '+('骨盆绝对朝向监督' if args.root_facing else '角度分项平衡与高噪声身体耦合' if args.angular_balance else '约束尺度与身体局部场景')+'：验收结果\n\n'
    for v,a in verdict.items():text+=f"{v}：脚部组件验收 **{a['foot_component_passed']}**；完整朝向＋脚步验收 **{a['full_joint_goal_passed']}**。\n\n"
    if args.root_facing:text+='本次第四组root_facing从原55k独立600步，在角度平衡目标上仅新增TRAIN骨盆绝对旋转监督。容差15度、系数.20、噪声权重.25+.75alpha，由TRAIN完整主干梯度选择。GT只作为训练标签，推理没有GT身体初态、历史或投影；不加载局部适配器，其他目标与随机安排不变。\n\n'
    elif args.angular_balance:text+='本次第三组angular_balanced仍从原55k独立600步；角度系数提高10倍，身体耦合高噪声权重保留.25，依据TRAIN32完整主干分项梯度选择。没有加载身体局部适配器；原相机/静态记忆/当前动态BPS、足部支撑和其他目标保持一致。\n\n'
    text+='所有短训候选均从原55k独立600步，未接着前一候选训练。全部128帧、DDIM20、纯噪声初始化；没有GT身体前缀、姿态初始化或推理投影。原55k未修改。\n\n'
    text+='## 850/906固定例子\n\n|例子/版本|净转向°|末帧朝向误差°|转向幅度误差°|接触帧比例|原生接触点脚滑：全段/末48帧 cm/帧|末48帧最低足点高度cm|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        if r['scope']!='demo':continue
        for v in labels:
            a=r['metrics'][v];f=a['observed_floor_support'];text+=f"|{r['index']}/{v}|{a['pelvis_turn_deg']:.1f}|{a['horizontal_root_yaw_final_abs_deg']:.1f}|{a['root_turn_progress_error_deg']:.1f}|{f['observed_contact_frame_fraction']:.3f}|{f['full']['contact_vertex_slide_cm_frame']:.3f}/{f['post80']['contact_vertex_slide_cm_frame']:.3f}|{f['lowest_foot_height_post80_mean_cm']:.2f}|\n"
    text+='\n接触点滑动采用相同的全native足部顶点，在已观测地面±2cm且两帧都接触的点对评分；需与接触帧覆盖、接触点对数、脚步幅度一起解释，不能通过离地降低条件滑动。另保留GT支撑期关节脚滑。联合门槛还要求转向幅度误差≤30°、末帧水平root朝向误差≤45°，不存在靠平均指标改善通过验收。\n\n'
    for panel,title in [('means','主面板16窗口×2种子'),('confirmation_means','确认面板32窗口×2种子')]:
        text+=f'## {title}\n\n|版本|身体朝向误差°|头部朝向误差°|GT支撑期脚滑cm/帧|腿角加速度P95|世界MPJPE cm|PA-MPJPE mm|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for v in labels:text+='|'+v+'|'+'|'.join(f'{summary[panel][v][m]:.3f}' for m in metrics)+'|\n'
        text+='\n'
    text+='这些是已经曝光的开发/确认窗口，共98条固定种子评估，不是独立测试性能。按三个数据集拆开的均值见per_dataset_summary.json。\n\n## 生成相机对应输入的残差\n\n|例子/版本|位置均值cm|朝向均值°|\n|---|---:|---:|\n'
    for r in camera['rows']:
        for v in labels:
            a=r['metrics'][v];text+=f"|{r['index']}/{v}|{a['position_mean_cm']:.2f}|{a['orientation_mean_deg']:.2f}|\n"
    text+='\n固定安装偏置仅由GT首帧作绘图/评分标定，不用于生成。这些残差不是部署时相机安装估计性能。5cm/5°是损失归一化尺度，不是推理硬误差上限；head-body目标不是让骨盆与头始终同向。\n\n## 身体场景是否真正生效\n\n'
    p=O/'body_local_intervention_audit.json'
    if p.exists():
        j=json.loads(p.read_text());text+=f"适配器末层权重范数 {j['adapter_last_weight_norm']:.6f}。固定权重/噪声，仅将新增适配器输出静音，未重新训练：\n\n|例子|FK平均/最大改变cm|局部表面覆盖|局部条件embedding平均L2|GT破坏后最大输出差|\n|---|---:|---:|---:|---:|\n"
        for r in j['rows']:text+=f"|{r['index']}|{r['adapter_mute_FK_difference_mean_cm']:.3f}/{r['adapter_mute_FK_difference_max_cm']:.3f}|{r['query']['available_fraction']:.3f}|{r['query']['embedding_l2_mean']:.6f}|{r['GT_poison_max_diff']:.1f}|\n"
        text+='\n这个诊断只能证明条件改变输出，不能证明改变合理或消除了穿透。\n\n'
    if args.angular_balance:text+='本组没有新增身体局部适配器，保留原相机BPS条件；身体局部查询与静音审计属于此前独立body_local组，详见父目录报告。它并未证明能够纠正朝向。\n\n'
    else:text+='静态表面按各帧first-observed时间截断；动态物体只查当帧，不留下旧几何。查询来自每个扩散步初步生成x0的FK22，覆盖身体关节周围，不读取GT身体。世界/anchor查询距离检查最大差约2e-7m。当前为最近表面方向/距离条件，没有有符号碰撞项；关节在沙发内部不一定能仅靠最近表面距离被识别。部分EgoBody窗口连诊断GT身体周围也没有0.75m内的观测，未知区域按mask处理。\n\n'
    text+='## 结论与保留\n\n'
    selected='root_facing' if args.root_facing else ('angular_balanced' if args.angular_balance else 'body_local')
    if verdict[selected]['full_joint_goal_passed']:text+='候选达到本次固定开发门槛，还需独立验证和人工查看；未证明完整人体碰撞或tracker成功。\n'
    else:text+='本轮尚不能采用为“已修好朝向且脚步不退化”的版本。具体失败检查见候选acceptance.json；保留原55k作为正式基线，不启动长训或用局部平均改善替代验收。下一项应检查更直接的身体表面接触/穿透目标及训练覆盖，不能仅继续提高头部权重。\n'
    text+='\n每组一个最终eval-only checkpoint；四组累计约1.82GiB，前三组约1.36GiB，前两组约0.91GiB，无optimizer/best/编号副本。初始尺度过强的51步预检没有保存权重，记录保留；原数据和活动缓存均未删除。源代码、协议与小记录备份GitHub，权重/数组留本地。\n'
    (O/'ASSESSMENT_zh.md').write_text(text)
    print('WROTE',O/'ASSESSMENT_zh.md')
if __name__=='__main__':main()
