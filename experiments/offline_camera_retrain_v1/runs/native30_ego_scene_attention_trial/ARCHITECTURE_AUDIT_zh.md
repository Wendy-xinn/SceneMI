# 场景注意力与反馈许可的独立设计审计

审计时间：2026-10-18 22:15–22:20（Asia/Shanghai）。本记录只读审查已冻结源码和既有证据，不启动训练、评估或 GPU，不修改源码、PLAN、旧实验或权重。机器可读绑定见 [ARCHITECTURE_AUDIT.json](ARCHITECTURE_AUDIT.json)。

## 可以得出的结论

候选确实在共享 Temporal U-Net 的第二次前向中增加了显式场景 cross-attention；它不是逐关节 Transformer。新 local K/V 直接读取原始 region context，没有旧局部条件与反馈使用的 `alpha_bar²` 幅度门控。因此本轮同时改变了融合位置、融合形式和局部信息在高噪声时刻的接入强度。该差异属于本次方案的定义，不是已经确认的实现错误；500 步的好坏只能评价这一具体方案，不能推出所有主干 cross-attention 都有效或无效。

初始化与场景关闭的实现符合预期：输出投影严格零初始化，新分支仅在 `use_scene=True` 且第二次 `_core_pass` 时调用；既有真实 TRAIN 输入 GPU 零更新预检记录两种 `use_scene` 设置均与原模型逐值相等、最大差为 0。训练后 `use_scene=False` 仍绕过新增分支，但两臂主干已经各自更新，不能因此要求 attention500 与 control500 的 scene-off 输出继续相等。

## 注意力实际读取什么

| 项目 | 当前实现 | 解释范围 |
| --- | --- | --- |
| 注入位置 | `core.unet.mid_block2` 的输出，第二次 `_core_pass` | 第一次无梯度 proposal 保留，用于构建局部场景查询 |
| Query | `[B,1024,48]` bottleneck 经线性投影后形成 `[B×48,1,64]` | 192 帧降采样为 48 个全身时间特征；每个 query 已融合身体通道和时间邻域 |
| Key / Value | 16 个全局 patch + 同时间 bin 的 22 个身体区域 context + 1 个 null | 每个时间 bin 对 39 个场景证据 token 注意；不是对其他时间 query 注意 |
| 局部汇聚 | 192 帧 region context 按有效性加权汇聚到 48 个 bin | 保留有效 mask；空 bin 为 0；null 避免全 mask 非有限值 |
| 新参数 | 148,800；token 维度 64；4 heads；attention dropout=0 | 原主干 122,667,536 参数；两臂主干均可训练 |
| 接入门控 | `use_scene`、第二次前向、key validity、是否有任何证据 | 新 attention 自身没有显式 `alpha_bar²` 乘法 |

源码：[构造与零初始化](../../ops/scene_attention.py:32)、[局部汇聚与 Q/K/V](../../ops/scene_attention.py:62)、[原始 context 缓存](../../ops/scene_attention.py:150)、[第二次前向门控](../../ops/scene_attention.py:158)。Temporal U-Net 的 residual blocks 接收时刻/条件嵌入，bottleneck query 继承这些信息，见 [ResidualTemporalBlock](../../../../model/mdm_scene_unet.py:167) 与 [TemporalUnet.forward](../../../../model/mdm_scene_unet.py:328)。没有开启原 U-Net 的 self-attention。

当前 raw region context 包含原近场 anchor context 和已启用的宽范围关系 context；local key 的有效性是原 `region_valid`，它会与 `relation_valid` 做 OR。不能把新增的 22 个 local token 全部解释为“0.6m 内真实接触证据”。见 [关系 context 合并](../../native30_scene_relations.py:128)。没有新增 ViT 编码或邻域查询调用。

## 扩散时刻调度的差异

这里用 `alpha_bar = model.alphas[timestep]` 表示代码中的累计信号系数，避免把平方根信号幅度与累计系数混用。

| 通路 | 当前显式幅度调度 |
| --- | --- |
| 旧 `region_condition(context)` | 乘 `alpha_bar²` |
| 旧局部 BPS | 乘 `alpha_bar²` |
| 旧 learned root / rotation feedback | 乘 `alpha_bar²`，再乘预测接触、对应置信度与证据有效性 |
| 新主干 scene attention 的 raw local K/V | 有效性加权汇聚，无额外 `alpha_bar²` 幅度门控 |

旧通路见 [native30_model.forward](../../native30_model.py:250) 与 [反馈残差](../../native30_model.py:267)。新通路在取回原始 context 时没有经过 `region_condition` 或 `fine`，见 [缓存位置](../../ops/scene_attention.py:150)。

这不等于新分支不知道时刻：Q 已有 U-Net 时刻条件；原 local context 的生成也使用 `sqrt(alpha_bar)` 特征、随 `alpha_bar` 变化的预测 skin / 可信观测混合查询中心；关系 context 的查询与特征同样包含 `alpha_bar`。这些属于条件输入，不等同于把整条输出幅度硬乘 `alpha_bar²`。在高噪声步，旧局部条件幅度被压低，新路仍可能读取完整 raw local context；该差异需要随结果一并报告。见 [局部查询时刻特征](../../native30_model.py:195) 与 [关系查询时刻特征](../../native30_scene_relations.py:96)。

已冻结 `attention_spec` 记录了 hook、维度、pass 与 key/value 类型，但没有单独声明上述局部幅度差异。本审计作为补充声明，不回写原 PLAN 或 metadata，不改变本次处理。

## 500 步 pilot 能回答的问题

两臂严格从同一保留的 41k 权重开始，使用相同 EGO/head-only 输入构造、训练参考方案、采样、噪声、扩散时刻、场景 dropout 和监督；两臂都关闭解析物理修正，保留原观测 anchor 与 IK。AdamW 均重置，FP32、batch=2、学习率 `2e-5`、25 步 warmup、固定 1500 步调度，当前 pilot 停在 500 步。所有模型参数都在可训练参数集合中；新增分支零初始化不会冻结原主干。见 [训练初始化与恢复](../../ops/scene_attention_trial.py:298) 和 [输入及更新记录](../../ops/scene_attention_trial.py:360)。

主比较必须是 attention500 对 matched control500；base41k 只能作为额外参照。这个比较回答：“在相同短训条件下，这一第二次前向、64 维、4-head、全身 bottleneck query、未额外衰减 raw local K/V 的实现是否有帮助”。不能将结论替换为逐关节 token 方案、其他融合层、时刻门控方案、从头训练或充分收敛后的结论。

500 次更新只有 1000 个训练窗口输入。训练混合仍沿用原采样分布，约 Trumans 62.7%、EgoBody 两组共 26.9%、RICH 10.4%，不是三个数据集等量训练；单个数据集的改善或退化需要单列。场景仍有 0.1 dropout。开发/确认面板各每数据集 2 个窗口，共 12 个不同 recording 的验证窗口；这是小规模验证，不是完整 RICH test 或方法排名。禁止用这些 pilot 结果声称最终收敛、最终 checkpoint 最优或模型已可发布。

本审计读取的状态快照中 control 已完成 500 步，attention 正在运行（读取时 40 步）；本记录不评价尚未完成的 500 步质量。既有 `MATCHED_AUDIT.json` 此时确认的是两臂 2 步更新，不应误标成 500 步配对审计。两步 `UPDATE_SANITY.json` 显示两臂均有 315 个原参数张量改变，attention 输出投影范数已非零；这说明计算图和更新通路工作，不代表生成质量已经改善。

## 反馈政策是否只改变反馈权限

当前 `scene_feedback_policy.py` 限定原始架构，两种政策都用 `learned_only`（解析物理修正关闭）：

- `near_only` 保留 `local_geometry` 返回的 `context`、BPS 和原 `valid` 对象；复制 geometry 字典，只把 `region_valid` 换成按关节汇聚的本地 anchor 有效性。宽关系 context、`relation_valid` 和 U-Net 的局部条件仍保留。
- `feedback_off` 只设置 `model.feedback_enabled=False`。权重、参考、观测、相机、全局与局部场景条件构造不变。

见 [政策实现](../../ops/scene_feedback_policy.py:35)。对于相同模型状态、同一 `x_t`、同一输入批次，这确实只改变 learned feedback 的许可；不删除可供生成器读取的宽关系信息。当前代码中，修改的 `region_valid` 用于原 root / rotation feedback 的 support mask；本轮解析物理修正关闭，不会借它触发另一个 solver 差异。

需要限制“场景条件不变”的表述：一次反馈残差会改变当前 `x0`，随后 DDIM 更新改变下一步 `x_t`；后续 proposal、查询中心与实际得到的 context 数值可以随之变化。这是闭环采样的预期因果链。这里不改变场景条件的算法和证据可用范围，不能声称两条完整采样轨迹在每一步都读取逐值相同的场景条件。

`near_only` 的最大半径为 0.6m，围绕实际查询中心；高噪声时查询中心可能混入可靠观测，而非完全等于预测 skin 点。该 mask 表示“该身体区域有本地邻域证据”，不表示接触标签为真、脚已踩稳或没有穿模。

## 证据与封印范围

本次读取并重新计算训练 PLAN 中 1122 个依赖路径的 SHA-256，全部与封印一致。源码、PLAN、预检与审计记录的 SHA 绑定在 JSON。既有 [PREFLIGHT.json](PREFLIGHT.json) 记录真实 TRAIN 输入的零更新等价检查；[MATCHED_AUDIT.json](MATCHED_AUDIT.json) 与 [UPDATE_SANITY.json](UPDATE_SANITY.json) 记录两步配对与更新检查。CPU 测试源覆盖零初始化、RNG 保留、第二次调用、scene-off、非零输出后的场景依赖、空输入有限性和 strict state 恢复；本次只读取其实现，不重跑测试，见 [attention 测试](../../ops/test_scene_attention.py:61) 与 [反馈政策测试](../../ops/test_scene_feedback_policy.py:31)。

结论性质是设计与实现范围审计，不是新增质量实验，也不替代 500 步完成后的独立配对预测和指标审计。
