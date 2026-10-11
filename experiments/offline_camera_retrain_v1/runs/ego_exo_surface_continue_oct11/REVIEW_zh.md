# 3000 步续训结果与可视化验收

这轮训练和自动评估已经完成，不再追加优化。用户要求后续先查看动作可视化，再决定续训或结构修改。当前候选保留可恢复的 last.pt；原 55k 保留。

播放器：http://127.0.0.1:8781 。蓝色 GT、灰色原 55k、绿色续训 3k、可选红色为续训模型关闭全部场景。同坐标叠加与并排比较均可，黄色只显示实际可用输入位置/轨迹，灰色点云是当前模式实际输入的已观测表面，橙色为当前可见动态点。并排仅平移显示，不改输入或评分。GT mesh 是共享 22 关节表示的原生解码参考，SMPL-X 手指采用平均手型。

全部预登记的 12 个 validation 开发窗口、18 个 ego/exo 模式案例均显示，没有按表现挑选。128 帧、20 Hz、相同 seed=2026101149、20 步 DDIM，无 GT 身体前缀。第一例重生成与缓存 motion 最大差为 0；实际计数是 40 次 core 前向，即每个 DDIM 步两次。模型 SHA256：344011ff6192382c8bad4982ba9abad10ab301d2700247806bf7f76dae5fd0b2。

## 原 55k 与续训候选

世界坐标未对齐 MPJPE，单位 mm，各分组 3 个窗口：

| 输入分组 | 原 55k | 续训 3k |
|---|---:|---:|
| TRUMANS ego | 171.64 | 174.63 |
| EgoBody wearer ego | 106.77 | 132.92 |
| EgoBody interactee ego | 77.25 | 86.32 |
| RICH ego | 197.85 | 205.94 |
| EgoBody wearer exo | 667.24 | 258.98 |
| EgoBody interactee exo | 356.03 | 202.76 |

12 个 ego 平均 138.38→149.95 mm；PA 88.43→93.95 mm；GT 支撑期脚 marker 水平速度 125.07→140.04 mm/s。ego 保持没有通过这组开发面板。6 个 exo 平均 511.63→230.87 mm，PA 79.52→72.26 mm，脚 marker 速度 428.95→169.20 mm/s，但原 55k 的 exo 是零样本对照，不能归因为场景 attention 的收益。续训 exo 的输入关节误差仍约 248.50 mm，不能视为可靠轨迹恢复完成。

这是小开发面板，不是全量验证或独立测试集。转向符合门槛的 ego 只有 3 个窗口，不能用它替代原 55k 的全量转向统计。

视觉诊断也有明显失败：04 camera_wearer/exo 的第 0 帧，GT 为低位姿态，续训候选仍接近站姿，朝向与参考明显不同。平均 exo 误差下降不能代替坐/蹲姿态、方向与可见轨迹的验收。preview_exo.png 保留这一失败例，没有只展示收益最大的案例。

正式 3000 个优化 step，共 6000 样本；实际 exo 为 698 样本、349 个 step。三域采样和来源身份详见 protocol.json/trace.jsonl。评价缓存位于 evaluation/。

## 场景消融

固定同一续训 checkpoint、输入和噪声，只在推理改变支路；这是条件敏感性检查，不是分别重训的结构消融。每个模式都在完整面板执行三个消融，共 54 次生成：

| 模式 | 完整模型 | 关闭全部场景 | 关闭全局 ViT | 关闭新增 attention 输出 |
|---|---:|---:|---:|---:|
| ego 全局 MPJPE mm | 149.95 | 155.05 | 152.61 | 149.46 |
| exo 全局 MPJPE mm | 230.87 | 236.72 | 236.63 | 230.87 |
| ego 输出相对完整模型变化 mm | 0 | 76.71 | 72.68 | 3.44 |
| exo 输出相对完整模型变化 mm | 0 | 49.24 | 52.28 | 4.50 |

关闭全部场景：use_scene=False，屏蔽全局、legacy BPS 和生成身体局部支路。关闭全局：将 ViT 的输出严格清零，保留其他场景条件。关闭新增 attention：仅将 surface_projection 输出清零，仍保留生成身体的 BPS 与全局 ViT。不是用零 occupancy 代替屏蔽，后者仍可能产生非零 ViT embedding。

完整模型相对关闭全部场景的世界误差，在 ego 9/12、exo 3/6 案例更低；相对关闭新增 attention，在 ego 1/12、exo 4/6 案例更低。当前 attention 输出影响小，平均误差无稳定收益。全局场景确实改变输出，但这没有证明接触或真实场景穿模改善；ego 开启全场景的脚 marker 速度反而高于关闭场景的 108.04 mm/s。

## 条件如何影响生成与尚缺的设计

全局 ViT embedding 与 diffusion timestep embedding 相加，进入每个 temporal residual block 的 MLP，生成 GroupNorm 特征的 scale/shift：h'=(1+scale)·GN(h)+shift。局部 BPS 的 MLP 特征与新增 attention 特征相加，再拼入每帧动作通道。动作损失的梯度训练这些路径如何改变干净动作预测；输入场景不等同施加碰撞约束。

当前新增 attention 使用固定 anchor ID embedding 作 query，67 个 context 经接触门控后平均汇总为单个 32 维每帧条件；BPS 本身仍保留锚点顺序。proposal 查询位置没有训练梯度回到第一遍生成。没有基于观测表面的穿模误差、区域修正目标、已知自由空间通道或明确的局部几何修复。现有脚部支撑惩罚来自 GT 高度代理，RICH contact BCE 是接触分类。

后续设计应在视觉验收后决定：保留身体区域 token，让姿态特征参与局部查询；先建立可靠的表面/自由空间约束及几何评估，再验证扩散后段的短窗口 root/pose refinement 与速度/加速度一致性。部分点云无符号距离不能直接作为穿模深度，67 个锚点也不足以作为完整碰撞检测采样。

## 重现与存储

导出及消融：PYTHONPATH=. python -m experiments.offline_camera_retrain_v1.prepare_unified_review 。播放：PYTHONPATH=. python -m experiments.offline_camera_retrain_v1.view_unified_review --port 8781 。使用 scenemi 环境。

visual_review/manifest.json 保存完整案例、输入来源、模型 hash、mesh/FK 一致性检查和各消融指标，summary.json 与 ablation_rows.json 保存汇总/逐例指标；mesh、motion 与截图留本地，代码及紧凑报告备份 user-backup。播放器按案例加载，避免一次加载全部 mesh。

浏览器检查通过：18 个案例全部可选、末帧 seek、exo 切换与指标匹配、播放推进、锚点/场景/无场景身体开关、叠加与并排切换，无 JavaScript 异常；最后调整观察视角后再次核对渲染和 exo 切换。browser_checks.json 记录最终 viewer hash。显示 mesh 与指标 FK 的逐例一致性检查均通过。
