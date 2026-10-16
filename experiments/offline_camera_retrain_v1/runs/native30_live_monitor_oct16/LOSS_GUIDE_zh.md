# 总 loss 与分项：当前实际训练口径

TensorBoard 打开 **Custom Scalars**。最上面只有一条 **Total loss**，就是训练实际反传的 `trace.jsonl → loss`；下面只有三项**已经加权的贡献**。详细项默认折叠。

| 首屏曲线 | 含义 | 如何从已有日志得到 |
|---|---|---|
| `loss/total` | 实际总损失 | 原始 `loss`，不重新计算 |
| `loss/weighted/base_body_and_gait` | 基础扩散、身体姿态与步态监督 | 原始 `terms.base.total`，内部权重已包含 |
| `loss/weighted/observation` | 观测轨迹一致性与身体转动监督 | 六个 `terms.observation` 乘**实际标定系数**后相加 |
| `loss/weighted/scene_geometry` | 接触、支撑与场景几何监督 | 七个 `terms.geometry` 乘训练配置系数后相加 |

三项满足：

```text
Total loss = 基础身体与步态 + 观测一致性 + 场景几何
```

这些只是原始训练日志的显示换算，没有新增 loss，没有修改训练、系数、模型或梯度。贡献大小表示该项在总 loss 中的数值大小，不等于它的梯度影响大小。

## 真实数值核对

核对第 1–8900 步全部 8900 条日志，三项加总与实际总 loss 的最大绝对差为 **3.7943×10⁻⁷**，符合 float32 舍入差异。第 8900 步示例：

| 已加权贡献 | 数值 |
|---|---:|
| 基础身体与步态 | 0.674195886 |
| 观测一致性 | 0.017463656 |
| 场景几何 | 0.296804154 |
| 三项之和 | 0.988463696 |
| 训练实际 Total loss | 0.988463759 |
| 加总减实际值 | −6.35×10⁻⁸ |

[LOSS_ACCOUNTING_AUDIT.json](LOSS_ACCOUNTING_AUDIT.json) 保存覆盖的日志字节数与 SHA256、所有实际系数、最大残差和该示例，防止拿后续增长的日志冒充同一核对快照。默认折叠的“详细：核对与轮换验证”中也有逐步残差曲线。

## 基础身体与步态包括什么

当前基础小计来自 `native30_losses.py:base_losses`。它已经包括以下带权项：

| 项目 | 外层系数 |
|---|---:|
| 最终动作 x0、原始网络输出 raw_x0 | 1、0.5 |
| 全局旋转矩阵误差、FK 关节位置误差 | 0.2、2 |
| FK 速度、加速度误差 | 0.025、0.0000125 |
| 头位置、raw 直接关节与 FK 一致性 | 1、0.5 |
| 根、头、颈的绝对朝向 | 0.25、0.5、0.25 |
| 头身相对朝向、相对转动 | 0.1、0.02 |
| 最终预测的 gait_total | 1 |
| 原始预测的 gait_raw_total | 0.5 |

**raw 指修正前的网络输出，不是“没有乘任何系数”。** `gait_total` 与 `gait_raw_total` 都已包含内部步态系数；raw 小计在基础总和中再乘 0.5。因此不能把 `base.total`、`gait_total` 和各步态子项再放在一起相加。

当前 V2 会替换旧的 `0.075×foot_velocity_m2_s2` 和 `0.1×planting_m2_s2`：二者仍被记录为诊断值，但这两个旧项**不再直接加入当前 total**，脚步速度和支撑由 V2 步态项承担。

当前步态小计内部系数如下，final 与 raw 使用相同内部系数：

| 步态项 | 系数 |
|---|---:|
| stance_velocity / swing_velocity | 0.1 / 0.05 |
| relative_position / stride_displacement | 1 / 2 |
| coordination_amplitude / coordination_relation | 5 / 0.05 |
| support_height / swing_height | 2 / 0.5 |
| support_envelope / support_penetration | 4 / 2 |
| phase_free_planting | 0.1 |
| bilateral_stride / bilateral_amplitude / bilateral_relation | 1 / 5 / 0.05 |

日志中的各步态项已经过其 GT 可靠性、时序有效性和 alpha_bar 门控，显示时不重复应用这些门控。全部 8900 步的 final/raw 步态小计与内部带权和分别在 3.70×10⁻⁷、7.73×10⁻⁷ 以内吻合。

一个日志限制：基础小计内部的“全局旋转矩阵误差”和“头位置误差”没有单独记录。因此仅凭现有 JSONL 不能把基础小计的每个底层项独立完整还原；监控直接使用原始 `base.total`，不伪造这两条曲线。

## 观测项的实际标定系数

以下来自当前主线 `prepared/effective_config.json → observation_coefficients`，**不是**配置中的 `maximum_coefficients` 上限。

| 原始字段 | 含义 | 实际系数 |
|---|---|---:|
| position | 与可用观测位置一致 | 0.09597684428424401 |
| rotation | 与可用观测朝向一致 | 0.025 |
| velocity | 与观测线速度一致 | 0.00029011997255170386 |
| acceleration | 与观测线加速度一致 | 0.00000316126911095216 |
| angular_velocity | 与观测角速度一致 | 0.0023217302368079363 |
| body_turn_rate | 与训练 GT 的身体根转动速度一致 | 0.0007816471124286364 |

原 `terms.observation` 已含可用性 mask、置信度、可靠性、尺度归一化和相应噪声门控，不能简单叫“原始米/弧度误差”。`loss/details/weighted_observation/*` 才是乘上述外层系数后的贡献；六项之和就是首屏观测分项。

## 场景几何项的外层系数

| 原始字段 | 含义 | 系数 |
|---|---|---:|
| contact | 身体区域接触分类 | 0.2 |
| contact_distance_m2 | GT 标定的接触区域到可观察支撑距离 | 2 |
| contact_relative_slip_m2_s2 | 接触相对滑动 | 0.02 |
| observed_plane_proxy_m2 | GT 标定的可观察表面平面代理 | 0.1 |
| correspondence | 身体区域与近邻场景对应分类 | 0.01 |
| feedback_smoothness | 根/旋转反馈的时序平滑 | 0.0001 |
| support_relation_m2 | 与 GT 支撑关系的差异 | 1 |

`loss/details/weighted_scene/*` 是乘系数后的七项贡献，已经包含于首屏场景分项。它们不是完整场景 SDF，也不代表对未观察区域做真碰撞检测。推理内部物理优化的能量有另一套前向优化系数；它没有作为一个额外顶层 loss 直接加进此处 total，不应把那套能量权重混在本表。

## 旧 55k 为什么单独放

旧 55k 的日志 `losses.total` 使用 `coordination_v1`：

```text
x0 + 0.2×局部旋转矩阵误差 + 2×FK位置
   + 10×逐帧位移误差 + 2×逐帧二阶差分误差
   + 头位置 + 0.5×直接关节/FK一致性
   + gait_total + phase_free_support_total + coordination_total
   + 0.1×contact_bce
```

该 profile 已移除旧的 GT-phase-specific `10×foot_stillness`，用 phase-free 支撑小计替代。三个小计都已加权，不能再把其子项叠加。2751 条旧训练记录完整核对后，总和最大绝对残差为 **2.9941×10⁻⁷**。

旧版是 20Hz、B4、多条件混合；新版是 30Hz、B2、当前 head-only ego，旋转、时间导数单位、噪声门控、观察/场景监督均有区别。相同名字和同一步数不代表同一个训练目标，不能以两个总 loss 的高低给模型质量排名。旧版总 loss 默认放在折叠的“旧55k”区，质量比较以明确匹配协议的验证和可视化为准。
