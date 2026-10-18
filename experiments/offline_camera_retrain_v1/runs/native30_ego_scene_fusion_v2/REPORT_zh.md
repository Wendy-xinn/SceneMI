# 身体区域与场景的主干融合实验

当前状态：结构实现、真实 TRAIN GPU 预检、两步配对和恢复审计已通过，正在执行两臂各 1000 更新。尚无本轮生成质量结论，未替换正式 41k/55k。

## 实现

在第二次共享 U-Net 的瓶颈中，让 22 个身体区域分别查询全局场景 patch 和局部场景关系。查询结合主干时间特征、身体区域身份和已有粗预测的位置/旋转；全局、局部 attention 分开，局部输出按 alpha_bar² 控制强度。输出投影零初始化，共新增 413,504 参数，不增加场景编码、FK 或几何查询次数。

原场景编码器、已有场景条件、接触/脚步监督、观测约束和 FK 保留；训练关闭预测后的迭代解析物理修正。两次 U-Net 和 DDIM20 的定义不变。代码复用 ops/scene_fusion.py、ops/ego_scene_trial.py、ops/ego_scene_evaluation.py，不在实验目录复制脚本。

## 训练与固定验证

- 起点是新版 native30 长训中的 best41k，正式 best41k 与 last55k 的字节校验均保持不变。
- 原架构 control 与 fusion 两臂使用同一数据、观测误差、扩散噪声、监督、dropout 和 RNG。主干 LR 2e-6，新融合 LR 1e-4，warmup100，固定日程 3000 更新；1000 是开发检查边界。
- TRAIN 数据集权重 RICH/TRUMANS/EgoBody=0.5/0.25/0.25，EGO head-only，batch2，FP32。两臂都重置 AdamW；每臂只有一个原子覆盖的可恢复 last.pt。
- VAL 固定 43 个不同录制，各取一个合法 L192 窗口：RICH27、TRUMANS8、EgoBody8。RICH28条VAL中一条仅165帧，不能按L192评估，明确排除。TRAIN/VAL按录制（包括EgoBody跨角色）无重合。
- 先用 dev12 检查明显退化；confirmation31 保留给最终阶段。最终用两个预先固定的 DDIM 采样种子，观测噪声保持不变，平均两次结果，不选择最佳采样。两次采样不是两个独立训练重复。

## 当前证据

30 项定向检查通过，含 CUDA 确定性反传。真实 TRAIN 输入的 scene-on/off 初始输出均与原模型逐值一致，新增输出层收到非零梯度；两臂两步之后实际输入及随机状态完全配对。独立评估审计确认阶段权重通过硬链接固定，续训原子替换 last.pt 不会破坏旧阶段验证身份。

实现检查不代表生成收益。最终按事先登记的 ACCEPTANCE.json，同时比较匹配 control 和原 41k；要求 RICH 改善、配对不确定性支持该改善，并通过其他数据集、脚步、加速度及头部误差门槛。GT 足高/低速脚速是关节级代理，不能写成完整场景穿模率。

入口：[后台流水线](PIPELINE.json) · [训练配置](PLAN.json) · [TensorBoard](http://localhost:6006/) · [真实输入预检](PREFLIGHT.json) · [独立审计](ARCHITECTURE_AUDIT_zh.md) · [固定案例](../native30_ego_scene_fusion_cohort/COHORT.json)。
