"""Record predetermined joint acceptance, lineage and storage evidence."""
import argparse,json,shutil
from pathlib import Path
import torch
H=Path(__file__).parent;O=H/'runs/fresh55k_replay_oct11'
def main():
    global O
    parser=argparse.ArgumentParser();parser.add_argument('--winding',action='store_true');args=parser.parse_args()
    if args.winding:O=H/'runs/fresh55k_winding_oct11'
    labels=['winding_guard','guard_terminal'] if args.winding else ['denoise_control','rollout_replay','denoise_terminal','replay_terminal']
    rows=json.loads((O/'rows.json').read_text());summary=json.loads((O/'summary.json').read_text());demos={r['index']:r['metrics'] for r in rows if r['scope']=='demo'}
    registered=json.loads((O/'pre_registered_evaluation.json').read_text());g=registered['adoption_gate'];audit={}
    for label in labels:
        m=summary['means'][label];b=summary['means']['original55k'];d=demos[906][label];e=demos[850][label]
        values={
            'panel_joint_slide_vs_original_ratio_max':m['gt_stance_slide_cm_frame']/b['gt_stance_slide_cm_frame'],
            'panel_native_sole_slide_vs_original_ratio_max':m['native_sole_stance_horizontal_cm_frame']/b['native_sole_stance_horizontal_cm_frame'],
            'panel_leg_accel_vs_original_ratio_max':m['leg_angular_accel_p95_deg_frame2']/b['leg_angular_accel_p95_deg_frame2'],
            '906_negative_net_turn':d['pelvis_turn_deg']<0,
            '906_root_peak_deg_max':d['root_step_max_deg'],
            '906_slide_vs_original_ratio_max':d['gt_stance_slide_cm_frame']/demos[906]['original55k']['gt_stance_slide_cm_frame'],
            '850_slide_vs_original_ratio_max':e['gt_stance_slide_cm_frame']/demos[850]['original55k']['gt_stance_slide_cm_frame'],
            '906_floating_proxy_cm_max':d['support_floating_m']*100,
            '906_contact_proxy_fraction_min':d['contact_coverage'],
            '850_floating_proxy_vs_original_cm_margin':(e['support_floating_m']-demos[850]['original55k']['support_floating_m'])*100,
            'panel_heading_vs_original_ratio_max':m['pelvis_orientation_mean_deg']/b['pelvis_orientation_mean_deg'],
            '906_post80_slide_vs_original_ratio_max':d['post80_stance_slide_cm_frame']/demos[906]['original55k']['post80_stance_slide_cm_frame']}
        gates={k:dict(value=v,threshold=g[k],passed=(v==g[k] if isinstance(g[k],bool) else (v>=g[k] if k.endswith('_min') else v<=g[k]))) for k,v in values.items()}
        c=summary['confirmation_means'];mapping={'heading_vs_original_max':'pelvis_orientation_mean_deg','joint_slide_vs_original_max':'gt_stance_slide_cm_frame','native_sole_slide_vs_original_max':'native_sole_stance_horizontal_cm_frame','leg_accel_vs_original_max':'leg_angular_accel_p95_deg_frame2'}
        for k,key in mapping.items():
            v=c[label][key]/c['original55k'][key];t=registered['confirmation_gate'][k];gates['confirmation_'+k]=dict(value=v,threshold=t,passed=v<=t)
        audit[label]=dict(adopt=all(x['passed'] for x in gates.values()),gates=gates)
    trace_paths={'replay_reference':H/'runs/fresh55k_replay_oct11/rollout_replay/trace.jsonl','winding_guard':O/'winding_guard/trace.jsonl'} if args.winding else {v:O/v/'trace.jsonl' for v in ['denoise_control','rollout_replay']}
    traces={v:[json.loads(x) for x in path.read_text().splitlines()] for v,path in trace_paths.items()}
    keys=['step','group','length','identities','t','noise_checksum','use_scene','replay_schedule','lr']
    matching=len(list(traces.values())[0])==len(list(traces.values())[1]) and all(all(x[k]==y[k] for k in keys) for x,y in zip(*traces.values()))
    train_only=all('/train/' in i['scene_bundle'] for v in traces.values() for x in v for i in x['identities'])
    weights=[]
    for v in (['winding_guard'] if args.winding else ['denoise_control','rollout_replay']):
        f=O/v/'last.pt';cp=torch.load(f,map_location='cpu',weights_only=False)
        weights.append(dict(variant=v,bytes=f.stat().st_size,step=cp['step'],evaluation_only=cp.get('evaluation_only'),optimizer_present='optimizer' in cp,all_tensors_finite=all(torch.isfinite(x).all().item() for x in cp['model'].values())))
    result=dict(variants=audit,trace_steps={k:len(v) for k,v in traces.items()},paired_samples_noise_schedule_match=matching,train_scene_bundles_only=train_only,weights=weights,free_disk_bytes=shutil.disk_usage(O).free,storage='two final evaluation-only last.pt; no optimizer/best/numbered weights; source/preprocessing dependencies untouched')
    (O/'acceptance_and_storage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    names={'original55k':'原55k','denoise_control':'标准加噪600','rollout_replay':'生成回放600标准DDIM','denoise_terminal':'标准加噪末端一次','replay_terminal':'回放末端一次','replay_reference':'回放参考标准DDIM','winding_guard':'累计/尾部600标准DDIM','guard_terminal':'累计/尾部末端一次'}
    report='# 从原55k重新开始：'+('累计转向/速率尾部对照验收' if args.winding else '生成回放对照验收')+'\n\n'
    report+='通过联合门槛的分支：'+('、'.join(names[k] for k,v in audit.items() if v['adopt']) or '**没有；不替换正式55k**')+'。不是仅靠降低朝向误差挑候选。门槛在本轮评分前登记，详细数值见acceptance_and_storage.json。\n\n'
    if not args.winding:report+='追加terminal是标准加噪组850/906预览后、terminal评分前登记的探索（terminal_exploration_protocol.json），复用同两份权重。原55k先纯噪声DDIM20生成整段，然后候选t0前向一次；输出仍是学习得到的修正，无GT反馈，也没有迭代IK。主比较标准DDIM与此附加探索分开解释。\n\n'
    if args.winding:report+='本组仅新增一份模型，从原55k初始化，**没有加载回放参考权重**；回放参考600步只用于配对消融比较。样本、噪声、回放计划相同。新增head/root累计路径在高噪声下仍以0.25+0.75alpha加权，global根部/颈部/头部/腿部的相对旋转误差和超速项取时间top10%，不让少数尖峰被128帧均值稀释。已测试相同终点的左右不同绕转及单帧尖峰；这仍是监督假设，不是硬关节投影或已证实根因。标准DDIM和末端一次修正均在训练前登记。\n\n'
    report+='## 来源与方法\n\n两个分支都直接加载原55k，源码没有读取之前候选。两组用相同转向、native足底高度/速度/连续支撑位移、头/颈旋转目标，不冻结旧生成的世界脚部位置。控制组GT加噪，回放组50%批次将冻结原55k从TRAIN已知输入、纯噪声DDIM20得到的动作重新加噪；另外50%保持普通扩散训练。这是固定55k生成轨迹回放，**不是当前模型在线on-policy回放**，也不是推理时GT优化。\n\n600步/组、batch2、完整128帧、AdamW lr2e−5、50步warmup、cosine最低0.1，时长平方根组采样，场景dropout0.1。目标GT只作训练监督，已知输入仅相机/场景与配置身体模板。没有GT身体前缀/初始姿态/接触条件，也没有头部或足部硬投影。标准去噪训练使用GT加噪是训练分布；推理全部128帧从噪声开始。\n\n'
    report+='## 固定850/906完整128帧\n\n脚滑单位cm/帧，浮起为GT相对支撑高度容差5cm以外的均值代理，不是绝对离地高度。接触覆盖仅为脚部关节高度落在GT参考高度±5cm内的比例，不含速度，不能当作稳定支撑。额外joint_low_and_slow_fraction要求高度范围与3D位移<1cm/帧同时满足，也只是评分代理。末段固定80–127帧。\n\n|例子/模型|净转向°|根部峰值°/帧|脚滑|浮起cm|接触覆盖|末段脚滑|末段浮起cm|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for i,metrics in demos.items():
        for label,m in metrics.items():report+=f"|{i}/{names[label]}|{m['pelvis_turn_deg']:.1f}|{m['root_step_max_deg']:.2f}|{m['gt_stance_slide_cm_frame']:.3f}|{m['support_floating_m']*100:.3f}|{m['contact_coverage']:.3f}|{m['post80_stance_slide_cm_frame']:.3f}|{m['post80_floating_proxy_cm']:.3f}|\n"
    for scope,title in [('means','16条原开发序列×2种子'),('confirmation_means','另32条固定开发窗口×2种子')]:
        report+=f'\n## {title}\n\n|模型|世界MPJPE cm|PA-MPJPE mm|骨盆朝向°|关节脚滑 cm/帧|native足底滑 cm/帧|腿角加速度P95°/帧²|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for label,m in summary[scope].items():report+=f"|{names[label]}|{m['mpjpe_cm']:.3f}|{m['pa_mpjpe_mm']:.2f}|{m['pelvis_orientation_mean_deg']:.2f}|{m['gt_stance_slide_cm_frame']:.3f}|{m['native_sole_stance_horizontal_cm_frame']:.3f}|{m['leg_angular_accel_p95_deg_frame2']:.3f}|\n"
    report+='\n这48条是固定的开发身体序列（EgoBody角色共享录制），不是48条独立录制或未经开发的holdout。判定使用预登记平均指标和固定示例，不改阈值挑种子；不是所有val的结论。全部同噪声种子、batch1、DDIM20评估；原55k850/906与缓存原生成吻合。身体GT污染输出差见evaluation_protocol.json。\n\n'
    report+=f"## 审计与存储\n\n配对样本/时刻/噪声校验/场景/学习率匹配：{matching}；train场景包检查：{train_only}。这里只保存{len(weights)}份最终evaluation-only权重和两组示例薄数组，没有optimizer/best/编号副本；各权重模型tensor有限：{all(x['all_tensors_finite'] for x in weights)}。正式55k和其他仍需复现的对照未删除；活动预处理依赖未动。\n\n"
    report+='场景输入仍为既有exact20因果静态记忆+当帧动态物体，未将动态物体累积为ghost geometry。本轮足底包络来自训练GT，不是新增场景SDF碰撞约束，不能据此宣称沙发穿透或物理tracking已经解决。\n'
    (O/'ASSESSMENT_zh.md').write_text(report)
if __name__=='__main__':main()
