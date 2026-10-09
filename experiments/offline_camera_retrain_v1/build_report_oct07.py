"""Build a Chinese report with exact corpus counts and completed paired metrics."""
import json,html,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;OUT=HERE/'runs/report55k_fullval_oct07'
LABELS={'trumans':'TRUMANS','egobody':'EgoBody（两角色）','camera_wearer':'EgoBody 相机佩戴者','interactee':'EgoBody 交互者','rich':'RICH'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def duration(frames):
 s=frames/20;return f'{s/3600:.3f} 小时（{s/60:.1f} 分钟）'
def aggregate_extra(rows):
 from experiments.offline_camera_retrain_v1.evaluate_report_oct07 import aggregate
 return {'all_eligible':aggregate(rows),'primary_128':aggregate([r for r in rows if r['window']['length']==128]),'short_64_supplement':aggregate([r for r in rows if r['window']['length']==64])}
def main():
 stats=json.loads((OUT/'dataset_stats.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());status=json.loads((OUT/'status.json').read_text());rows=[json.loads(l) for l in (OUT/'rows.jsonl').read_text().splitlines() if l]
 full=status['status']=='completed' and len(rows)==manifest['eligible_windows'];summary=aggregate_extra(rows)
 (OUT/'metrics_detailed.json').write_text(json.dumps(dict(full_coverage=full,protocol=manifest['protocol'],**summary),indent=2))
 cfg=manifest['config'];lines=['# 三数据集 SceneMI 55k 实验汇报（2026-10-07）','',f'**评估状态：{"全量完成" if full else "进行中，以下指标为部分结果，禁止作为全量结果引用"}；{len(rows):,} / {manifest["eligible_windows"]:,} 个窗口。**','',
 '## 可直接用于汇报的结论','',
 '本次完成了 TRUMANS、EgoBody 和 RICH 的三域混合训练，共 55,000 步、220,000 次窗口抽样。训练从随机初始化开始，模型使用头部为主的混合身体轨迹条件和 ego 可见场景条件。旧版 RICH 人体动作曾被抽到 5 Hz 再插值到 20 Hz，因此损失了高频动作细节；SMPL-X→SMPL 还使用参数复用的近似转换。当前结果是这一代理数据版本的工程实验，不能作为最终统一 SMPL 模型的结论。',
 '', '场景比较使用同一 checkpoint、同一验证窗口、同一头部控制及完全相同的初始扩散噪声；只切换全局 occupancy 和局部 BPS 两条场景分支。该比较检验模型是否使用场景条件，以及场景对生成的影响，不是两个独立训练模型的比较。', '',
 '## 数据规模和分布','',
 '下面统计本实验实际预处理的数据，不是三个公开数据集的全部发布规模。帧数按人物轨迹计，时长 = 帧数 / 20 Hz；重叠窗口不重复累计原始帧。EgoBody 两角色的“人物时长”与去重后录制时长分别列出。', '']
 corpus=[]
 for split,label in [('train','训练'),('validation','验证'),('test','测试（本次未评估）')]:
  for g in ['trumans','egobody','rich']:
   if g=='rich':
    if split=='test':continue
    r=stats[split][g]
   else:r=stats[split]['prepared_manifest_corpus'][g]
   corpus.append([label,LABELS[g],r['sequences'],f'{r["frames_20hz"]:,}',duration(r['frames_20hz']),len(r['scenes'])])
 lines.extend([table(['划分','数据集','预处理轨迹数','20 Hz 帧数','人物时长','场景数'],corpus),'',
 '训练采样最低需要 64 帧：TRUMANS 的 469 条训练预处理轨迹中 457 条可用；EgoBody 的 213 条中 188 条可用（佩戴者 89、交互者 99）；RICH 62 条全部可用。这是“已预处理”与“实际进入采样”的区别。TRUMANS 的 12 条排除轨迹没有合格的审计窗口起点，并不都是短序列；EgoBody 的 25 条训练排除轨迹为 60 帧短轨迹。', '',
 table(['划分','可采样数据组','轨迹数','帧数','人物时长'],[[label,LABELS[g],stats[split][g]['sequences'],f'{stats[split][g]["frames_20hz"]:,}',duration(stats[split][g]['frames_20hz'])] for split,label in [('train','训练'),('validation','验证')] for g in ['trumans','camera_wearer','interactee','rich']]),'',
 'EgoBody 训练可采样轨迹覆盖 64 个 recording，合并两个角色时间区间后的录制时长为 '+f'{stats["train"]["egobody"]["union_recording_hours"]:.3f} 小时；验证覆盖 {stats["validation"]["egobody"]["unique_recordings"]} 个 recording，去重录制时长为 {stats["validation"]["egobody"]["union_recording_hours"]:.3f} 小时。', '',
 '旧 RICH 训练动作张量有 24,210 帧、验证有 11,728 帧，但其中大量 20 Hz 帧来自 5 Hz 插值，不应称为同等数量的独立高频观测。RICH 未接入 test split。','',
 '### 序列长度分布（可采样轨迹）','',table(['划分','数据集','最短/中位/P95/最长（秒）'],[[label,LABELS[g],'/'.join(f'{stats[split][g]["duration_s"][k]:.2f}' for k in ['min','median','p95','max'])] for split,label in [('train','训练'),('validation','验证')] for g in ['trumans','egobody','rich']]),'',
 '### 场景分布（可采样轨迹）',''])
 scene_csv=[]
 for split,label in [('train','训练'),('validation','验证')]:
  for g in ['trumans','egobody','rich']:
   rr=[]
   for scene,r in stats[split][g]['scenes'].items():
    rr.append([scene,r['sequences'],f'{r["frames"]:,}',f'{r["person_hours"]:.3f}']);scene_csv.append([split,g,scene,r['sequences'],r['frames'],r['person_hours']])
   lines.extend([f'#### {label} · {LABELS[g]}','',table(['场景','轨迹数','帧数','人物小时'],rr),''])
 with (OUT/'scene_distribution.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['split','dataset','scene','sequences','frames_20hz','person_hours']);w.writerows(scene_csv)
 lines.extend(['TRUMANS/EgoBody 的预处理划分检查：train 与 validation/test 的源序列或 recording 无交集，train 与 validation 的 scene_family 也无交集。RICH 使用所接入的 train/val 划分，其场景可能跨划分共享，不把三个域统一称为未见场景评估。详见 split_audit.json。','', '## 详细实验设置','',table(['项目','实际设置'],[
 ['权重','canonical_smpl_3dataset_20hz_55k_oct07/last.pt；第 55,000 步'],['初始化','随机初始化；没有续接旧权重'],['网络','SceneMI AdaGN Temporal U-Net + 全局 occupancy ViT + 相机中心 BPS MLP'],['U-Net','latent_dim=256；dim_mults=(1,2,4)'],['动作表示','201 维：根平移 3 + 22×6 残差旋转 + 22×3 辅助全局关节；FK 解码评估'],['坐标','相机首帧位置和水平朝向锚定；内部位置以 2 m 缩放，指标恢复为米'],['训练预算','55,000 步；batch=4；220,000 次窗口抽样'],['优化器','AdamW；lr=1e-4；weight_decay=0.01；梯度范数裁剪=1.0'],['学习率','500 步 warmup；余弦衰减；最低乘子 0.1'],['精度/设备','float32；单张 RTX 5090'],['随机种子','训练 seed=2026'],['动作频率','统一张量 20 Hz；旧 RICH 为 5→20 Hz 插值'],['窗口长度','64/128/192 帧，即 3.2/6.4/9.6 秒'],['长度采样','实际约 30%/45%/25%，不是先前讨论的 25%/50%/25%'],['数据域采样','TRUMANS / EgoBody / RICH 各约 1/3；EgoBody 两角色各约 1/6'],['域内采样','按数据组×长度分别打乱、无放回遍历窗口，遍历完再重排'],['场景 dropout','每个 batch 10% 概率共同关闭 occupancy 与 BPS'],['扩散','1000 步 cosine noise schedule；预测 x0；25% 样本额外使用 t∈[0,99] 低噪声'],['检查点','每 1000 步保存；每 5000 步 DDIM20 抽样评估；保留推理权重'],['训练耗时','到第 55k 步约 3 小时 32 分；含最终抽样评估约 3 小时 35 分']]),'',
 '实际可采样训练帧量中 TRUMANS 约占 81.88%、EgoBody 约占 15.16%、RICH 约占 2.95%，但三域各获得约 1/3 的训练步数。因此 RICH 被显著重采样；三域平衡是实验设计，不是按帧数自然比例采样。','', '### 控制条件概率','',table(['条件','概率'],[[k,f'{v*100:.0f}%'] for k,v in cfg['control_mode_probabilities'].items()]),'',
 '### 55k 的窗口抽样与覆盖','',table(['数据组','64/128/192 帧的实际训练步数','对应窗口抽样次数（batch=4）'],[[LABELS[g],'/'.join(str(cfg['schedule_bucket_steps'][f'{g}/{L}']) for L in [64,128,192]),'/'.join(str(cfg['schedule_bucket_steps'][f'{g}/{L}']*4) for L in [64,128,192])] for g in ['trumans','camera_wearer','interactee','rich']]),'',
 '各桶的计划抽样次数均大于其候选窗口数；采样器按桶无放回遍历，55k 已完成，因此按该确定性采样日程覆盖每个可用窗口至少一轮。完整候选窗口数见 config.json 的 train_summary；这不等于覆盖公开数据集全部原始动作。','',
 '### 场景条件','',
 '- occupancy：24×48×48，覆盖约 12.8×4.8×12.8 m；只编码已观察表面，0 表示未知。\n- BPS：每帧 67 个相机中心固定锚点，查询可见场景的最近表面；不用 GT 身体位置查询。\n- TRUMANS/EgoBody 的可见地图为 5 Hz 投影来源；RICH 为 5 Hz 网格首交遮挡结果的静态可见并集。\n- 虚拟相机视场角 66.56°×40.49°；RICH 相机来自头部旋转和眼部附近偏置，不是真实头戴相机观测。\n- 场景来自整段序列的离线可见并集；不能称为严格在线因果输入。\n- TRUMANS 默认分支还加入窗口起点的可见动态物体快照；当前不是完整时变他人/物体交互条件。','',
 '### 损失设置','',
 'loss_profile=coordination_v1。基础项为 x0、旋转矩阵、FK 位置/速度/加速度、头部位置、辅助关节与 FK 一致性；额外包含 stance/swing 脚速、脚高、相对脚位置、多帧 stride displacement、相位无关支撑与 planting、短窗口肢段相关关系与运动幅度。其支撑项使用训练 GT/FK，推理时不输入真值支撑或脚接触。','',
 '基础系数：x0 1；rotation 0.2；FK position 2；FK velocity 10；FK acceleration 2；head 1；branch consistency 0.5。coordination_v1 用相位无关支撑替代旧 GT 相位 foot-stillness 项。gait 系数为 stance velocity 0.10、swing velocity 0.05、support height 2、swing height 0.5、relative position 1、stride displacement 2；phase-free 系数为 support envelope 4、penetration 2、planting 0.10；coordination relation 0.05、amplitude 5。','',
 '## 全验证评估协议','',
 f'权重 SHA256：`{manifest["checkpoint_sha256"]}`。validation split；DDIM20；float32；完整头部位置/旋转条件；两场景变体权重完全相同。batch=16，batch seed=777+该 batch 首窗口编号；每一对有/无场景使用相同噪声，固定当前 batch 划分可复现。','',
 '遍历审计清单中每个 128 帧起点；对不足 128 帧但至少 64 帧的序列，遍历其 64 帧候选起点作为补充。EgoBody 21 条 60 帧（3 秒）预处理轨迹不能满足训练最短 64 帧，排除并明确列出，不用重复末帧强行补齐。','',
 table(['数据组','覆盖序列','128 帧窗口','64 帧补充窗口'],[[LABELS[g],len({r['sequence_id'] for r in manifest['windows'] if r['group']==g}),sum(r['group']==g and r['length']==128 for r in manifest['windows']),sum(r['group']==g and r['length']==64 for r in manifest['windows'])] for g in ['trumans','camera_wearer','interactee','rich']]),'',
 '指标定义：\n- MPJPE：相机首帧锚定坐标内，生成 FK 22 关节与该版本 GT 的欧氏距离均值；不做骨盆对齐，保留全球位置误差。\n- WA-MPJPE：按连续 100 帧及末尾短片段分别拟合一个旋转、平移和尺度的相似变换，对该片段所有帧和 22 关节一起对齐，再算误差；不是每帧单独对齐。另存整窗口 WA 以便复核。参考 [GVHMR 的评估实现](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/eval/eval_utils.py)。\n- 头部误差：生成头关节相对 GT 头关节的误差，不等同于相机到头关节的解剖偏置。\n- 脚滑：GT 静止脚时刻的生成脚位移（cm/帧）；不同合理步态相位可能受罚。\n- 接触覆盖、支撑悬空及穿透：GT 脚高度代理，不能称为真实场景 mesh 碰撞。\n- 双脚水平相对位置变化 RMS、相对脚速度和肢段协调为诊断指标，不是通用自然度分数。','',
 '窗口相互重叠，因此同时提供窗口平均和每序列等权平均。RICH 起点密集，不能把所有窗口直接混成一个指标后声称三数据域等权。这是本实验的 22 关节条件生成评估，不是公开 RICH/EMDB 重建基准的完整标准协议，不能直接与论文表格比较。以下按三个数据集分别统计，EgoBody 另外拆成两个角色。','',
 '## 评估结果',''])
 for protocol,label in [('primary_128','主协议：所有 128 帧候选窗口'),('all_eligible','覆盖全部可采样序列：128 帧 + 短序列 64 帧补充')]:
  lines.extend([f'### {label}',''])
  for weighting,title in [('sequence_mean','每序列等权平均（建议汇报主表）'),('window_mean','每窗口平均（完整遍历对照表）')]:
   rr=[]
   for g in ['trumans','egobody','camera_wearer','interactee','rich']:
    group=summary[protocol][g]
    for variant,vlabel in [('scene_on','有场景'),('scene_off','无场景')]:
     if variant not in group['variants']:continue
     m=group['variants'][variant][weighting]
     rr.append([LABELS[g],vlabel,group['sequences'],group['windows'],f'{m["mpjpe_cm"]:.2f}',f'{m["wa_mpjpe_cm"]:.2f}',f'{m["head_cm"]:.2f}',f'{m["gt_stance_slide_cm_frame"]:.3f}',f'{m["contact_coverage"]*100:.1f}%',f'{m["support_floating_m"]*100:.2f}'])
   lines.extend([f'**{title}**','',table(['数据集','场景','序列','窗口','MPJPE cm','WA-MPJPE cm','头误差 cm','脚滑 cm/帧','接触覆盖','悬空 cm'],rr),''])
 lines.extend(['## Viser 汇报示例','',
 '[打开有/无场景对照：http://127.0.0.1:8792](http://127.0.0.1:8792)。绿色为有场景生成，红色为无场景生成，蓝色为 GT，紫色为相机轨迹。可播放、拖动帧、切换其他片段及同坐标叠加。右侧环境点云只是显示参照，不输入无场景分支。','',
 '默认选择 EgoBody 验证窗口 1121（recording_20220215_S22_S21_02_camera_wearer_1411-3931，源 30 fps 起点 3151）。它不是随机代表样本，而是为展示可见条件效应选取的案例。第二个是 TRUMANS 窗口 558，用于观察步态。',''])
 for idx in [1121,558]:
  row=next((r for r in rows if r['window']['index']==idx),None)
  if not row:continue
  on=row['scene_on'];off=row['scene_off']
  lines.extend([f'### 窗口 {idx}：{row["window"]["sequence_id"]}','',table(['诊断','有场景','无场景'],[['MPJPE cm',f'{on["mpjpe_cm"]:.2f}',f'{off["mpjpe_cm"]:.2f}'],['WA-MPJPE cm',f'{on["wa_mpjpe_cm"]:.2f}',f'{off["wa_mpjpe_cm"]:.2f}'],['头误差 cm',f'{on["head_cm"]:.2f}',f'{off["head_cm"]:.2f}'],['双脚水平相对位置变化 RMS cm',f'{on["foot_separation_rms_cm"]:.2f}',f'{off["foot_separation_rms_cm"]:.2f}'],['支撑悬空代理 cm',f'{on["support_floating_m"]*100:.2f}',f'{off["support_floating_m"]*100:.2f}']]),'',f'两生成的全身平均差异 {row["scene_effect_cm"]:.2f} cm；去除每帧骨盆平移后仍有 {row["root_relative_scene_effect_cm"]:.2f} cm 差异。因此场景影响不只是整个人平移，肢体配置/步态幅度也改变。此案例不能独立证明模型学会了可靠避障。',''])
 lines.extend(['## 数据质量与结论边界','',
 '1. 旧 RICH 的 30→5→20 Hz 动作流程会丢失细节；不能把旧 55k 的 RICH 异常唯一归因于这一因素。\n2. 当前 SMPL-X→SMPL 是姿态/shape 参数复用和骨盆对齐的代理，不是官方网格 transfer fitting；此前对应关节对齐审计平均差约 4 cm、P95 约 8.8–8.9 cm。建议先做官方转换小样对照，再确定下一版监督。\n3. 内部 FK 一致不代表与原始 SMPL-X 完全一致；WA 对齐也会消去尺度/位置误差，必须与原始 MPJPE、头部、支撑指标联合解读。\n4. 直接从原始 30 Hz 动作重采样到 20 Hz 的版本是另一轮独立训练；光轴和 5 Hz 可见场景保持面外协议。该版本尚未用于这里的 55k 全验证结果。\n5. 当前评估为单种子、头部控制协议，不涵盖所有 mask、所有长度及所有随机生成。\n6. 场景消融能证明该条件影响当前模型输出；无法仅凭它证明场景物理正确、改善所有动作，或加入 RICH 一定优于两数据集训练。','',
 '## 可复核文件','',
 '- `manifest.json`：完整窗口身份、checkpoint hash、原始训练 config、配对种子规则。\n- `dataset_stats.json` / `scene_distribution.csv`：数据规模及场景分布。\n- `excluded_short_validation.json`：21 条排除的短验证轨迹及原因。\n- `split_audit.json`：TRUMANS/EgoBody 的源序列与场景划分检查。\n- `rows.jsonl`：每窗口两变体指标和输入身份。\n- `metrics_detailed.json`：主 128 帧、补充 64 帧、全可用序列的分开聚合。\n- `motions/*.npz`：原始生成动作表示、生成 FK、GT 和相机轨迹。\n- `viser_selected_cases.json`：可视化示例身份与指标。\n- `wa_reference_verification.json`：WA 实现与 GVHMR 对齐函数的交叉核对。\n- `datasets/TRUMANS`、`datasets/EgoBody`、`datasets/RICH`：分数据集指标和逐窗记录。\n- `sequence_metrics.csv`：每序列指标，方便表格软件进一步分析。',''])
 if full:
  cm=summary['all_eligible'];changes={g:(1-cm[g]['variants']['scene_on']['sequence_mean']['mpjpe_cm']/cm[g]['variants']['scene_off']['sequence_mean']['mpjpe_cm'])*100 for g in ['trumans','egobody','rich']}
  camera_on=cm['camera_wearer']['variants']['scene_on']['sequence_mean']['head_cm'];camera_off=cm['camera_wearer']['variants']['scene_off']['sequence_mean']['head_cm']
  interpretation=f'全可采样验证序列中，场景开启使 TRUMANS / EgoBody / RICH 的 MPJPE 分别下降 {changes["trumans"]:.1f}% / {changes["egobody"]:.1f}% / {changes["rich"]:.1f}%。但不是所有指标都改善：EgoBody 佩戴者的头部误差从无场景 {camera_off:.2f} cm 增至有场景 {camera_on:.2f} cm。RICH 有场景 MPJPE 仍为 56.88 cm、GT 支撑悬空代理约 79.91 cm，生成质量仍未通过验收。结论应是场景条件影响输出且在部分指标上有收益，而不是全面提升或已经解决物理合理性。'
  lines[10:10]=[interpretation,'']
 lines.extend(['## 汇报图表','','![实际数据规模](dataset_size.png)','','![全量场景消融指标](scene_ablation_metrics.png)',''])
 text='\n'.join(lines);(OUT/'REPORT_zh.md').write_text(text)
 # HTML uses a local optional markdown package, with a readable fallback.
 try:
  import markdown
  body=markdown.markdown(text,extensions=['tables','fenced_code'])
 except ImportError:body='<pre>'+html.escape(text)+'</pre>'
 css='body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.65;color:#223}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd;padding:7px;text-align:left}th{background:#eef3f8}h1,h2{color:#164867}pre{white-space:pre-wrap}a{color:#176ac2}'
 (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>三数据集55k汇报</title><style>'+css+'</style>'+body)
 # Short version for oral presentation, with the full tables stored above.
 brief=['# 两点汇报：三数据集 55k','',f'评估状态：{"完成" if full else "进行中"}，{len(rows):,}/{manifest["eligible_windows"]:,} 个候选窗口。','', '旧版 RICH 人体动作经过 5 Hz 抽帧后插值到 20 Hz，SMPL-X→SMPL 仍是近似转换。此轮用于验证三域混训和场景条件作用，不宣称最终数据版本已解决。','', '## 全可采样验证序列，每序列等权','']
 rr=[]
 for g in ['trumans','egobody','rich']:
  group=summary['all_eligible'][g]
  for v,label in [('scene_on','有场景'),('scene_off','无场景')]:
   if v in group['variants']:
    m=group['variants'][v]['sequence_mean'];rr.append([LABELS[g],label,f'{m["mpjpe_cm"]:.2f}',f'{m["wa_mpjpe_cm"]:.2f}',f'{m["head_cm"]:.2f}',f'{m["support_floating_m"]*100:.2f}'])
 brief.extend([table(['数据集','场景','MPJPE cm','WA-MPJPE cm','头误差 cm','悬空代理 cm'],rr),'', '覆盖 TRUMANS 40、EgoBody 78、RICH 28 条可采样验证轨迹；EgoBody 另有 21 条 60 帧短轨迹不满足最短 64 帧，未评估。WA 使用每100帧统一相似对齐，包含末尾短片段；22关节。','', '训练：55k、batch4、从零、AdamW lr1e-4、float32；三域各约1/3；窗口64/128/192帧=30/45/25%；头部主导混合mask；场景dropout10%。实际约3小时35分钟。','', '[Viser 场景对照](http://127.0.0.1:8792)：默认 EgoBody 窗口1121。可观察双脚相对位置动态幅度和支撑姿态变化；示例按效应明显选取，不能替代全量指标。','', '下一步：完成原始30→20Hz版本对照；官方SMPL-X→SMPL拟合小样与脚端误差审计；再决定正式重训。'])
 if full:brief.extend(['',interpretation])
 brief.extend(['','## 实际可采样数据规模','',table(['数据集','训练轨迹 / 帧数 / 人物小时','验证轨迹 / 帧数 / 人物小时'],[[LABELS[g],f'{stats["train"][g]["sequences"]} / {stats["train"][g]["frames_20hz"]:,} / {stats["train"][g]["person_hours"]:.3f}',f'{stats["validation"][g]["sequences"]} / {stats["validation"][g]["frames_20hz"]:,} / {stats["validation"][g]["person_hours"]:.3f}'] for g in ['trumans','egobody','rich']]),'', 'EgoBody 人物时长累计两个角色，不等于录制时长；旧 RICH 20 Hz 帧数包含插值。完整预处理规模、场景/长度分布、角色拆分、损失系数和排除项见 REPORT_zh.md。'])
 (OUT/'BRIEF_zh.md').write_text('\n'.join(brief));print('report written',full,len(rows))
if __name__=='__main__':main()
