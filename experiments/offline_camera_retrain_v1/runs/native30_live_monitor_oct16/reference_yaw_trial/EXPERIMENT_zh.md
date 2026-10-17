# 头部水平朝向参考：1000步独立训练对照

用户在2026-10-17授权尝试“头部俯仰不再直接成为全身初始参考朝向”，同时保持后台55k主线。本实验是独立续训分支，不替换主训练。

## 改动与边界

仅ego且缺少可靠躯干位置观测时，将参考root朝向投影到Y轴水平转向，按新朝向重新计算参考root位置与局部关节旋转。真实输入头部位置、完整头部旋转、置信度和所有既有监督保留；可靠躯干、EXO及外部完整reference旁路。头部pitch/roll仍进入观测和参考头部完整全局朝向。没有给最终生成root增加硬直立约束，没有假设固定平地，也没有接入tracker。

近竖直头部前向的水平投影不足0.1时，在同一离线窗口内插值可用heading、端点保持；全窗没有可用heading时回identity。这是参考约定，不是已知的真实骨盆朝向。root参考置信沿用原值，降低置信将是另一项变量。

10项CPU不变量通过：保留头位置/完整朝向/置信、EXO与可靠torso/外部prior等价、近竖直与缺失输入有限、上下台阶头Y轨迹保留、最终soft-anchor仍允许倾斜root。实际弯腰输入也显示：新reference可能从旧参考的倾斜抬脚变成直立参考下穿。该变更只消除朝向耦合，蹲/弯腰仍需网络学习；不能把未经训练的参考当成物理解决方案。

## 训练比较

- 父状态：同一24,500步模型、Adam、采样器和全套RNG。
- 对照：复用上一轮已匹配审计的原reference、物理ON、24,500→25,500步1000更新及其38份预测。
- 候选：相同父状态续训1000步，只安装yaw-only reference；独立物理修正保持ON，场景、贴头、IK、loss系数和55k全局LR时程保留。
- 同批数据、监督、几何、timestep/dropout、学习率和RNG逐步核对；reference、残差target和noisy residual因实验变量必然不同，不把它们伪称相同输入。
- 一个原子last.pt，不新增编号/best权重。checkpoint顶层记录reference_variant与本训练PLAN SHA；original effective_config原样保留以保证真实resume，不能忽略variant直接按原模型解释候选。

新准备封存显式动态入口和静态import闭包（85份依赖），原trainer仍额外执行历史准备source seal。新代码统一在ops模块中，不生成run目录临时Python脚本。

## 验证与输出

沿用已预定14个不同录制VAL窗口：12 RICH、1 TRUMANS、1 EgoBody，含pushup/yoga反例。主评价统一推理物理OFF，5核心窗另交叉ON，clean/noisy分开。control 38份预测保留原SHA、逐数组字节验证后包装公共指纹；新增候选38份，总76份独立CPU复核。

公共输入指纹排除reference_motion、reference_confidence、observation_anchor_reference；三项派生reference单列SHA，要求本轮reference confidence不变。保留非reference与reference tensor数组供CPU独立hash复核，GT只用于评估。

联合报告World/PA、GT低速期脚速、连续段漂移、脚高误差、双足幅值与stride、可见表面GT校准偏差及头误差。GT低速并不是真实接触标签；稀疏足锚点偏差也不是全mesh穿模率。判据是脚步/高度/表面误差共同改善且身体/头误差不明显退化，不能仅凭脚动得更多选模型。1000步续训未见收益也不能证明全新参数化从零训练必然失败。

后台流水线：训练→匹配审计→统一评价→独立CPU复算→预选三案例两模型Viser菜单。仅CPU核验后发布视图，不自动切换主线或进入EXO。状态见[STATUS.json](STATUS.json)，评估见evaluation/REPORT_zh.md（生成前没有最终效果结论）。

## 已有失败对照清理

上一轮no_physics权重、38份预测和旧8802播放缓存已按用户指令删除，只保留配置、逐窗指标、身份/seed与审计记录；该评估目录专属surface派生缓存也已删除。共释放2,139,009,024 bytes。原始数据和共享场景缓存未动，父状态和control及其76份预测SHA不变。本次仍需要它们复用，不能当作垃圾删除。[清理清单](../paired_foot_trial/CLEANUP.json)。

另清理已拒绝的去贴头/弱直立/组合和投影尝试数组、旧8801播放缓存，以及这些评估目录本地可重建surface缓存，释放655,388,672 bytes。baseline10份预测、四份逐阶段数组与19k诊断权重和所有compact记录保留，逐项核验SHA；没有删除任何封存源码。两轮共释放2,794,397,696 bytes，旧8801与8802已下线。[执行记录](../../../ops/cleanup_oct17_rejected_support_EXECUTED.json)。
