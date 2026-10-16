# 当前头部高度与约束复核

复核对象：持续训练主线 `native30_gait_v2_ego55k_oct15`，固定验证快照 **10000 步**。本记录只评分已有结果，不改训练、配置、模型或原始观测。

## 高度究竟保留了什么

当前坐标变换为 `p_window = (p_world - camera_world[0]) @ yaw_anchor.T`。[native30_data.py:165–176](../../../native30_data.py#L165) 确实减去首帧相机的完整 XYZ，包含 Y；anchor 是纯绕 Y 的旋转，另有纯 yaw 增广，**重力方向没有被相机俯仰旋转掉**。原 anchor 实现在 [data_loader.py:20–24](../../../../../docs/local_dependencies/diffusion-motion-inbetweening/experiments/offline_sequence_v1/data_loader.py#L20)。

- **保留**窗口内头部上升/下降，及身体、头部与观测场景之间的相对高度。场景与人体减同一个 origin：[native30_observations.py:124](../../../native30_observations.py#L124)、[native30_memory.py:138–139](../../../native30_memory.py#L138)。
- **不直接输入**首帧相机的原世界 Y。首帧 camera Y=0；但经过安装偏移反算后的解剖 head Y 通常不为 0：[native30_observations.py:71–76](../../../native30_observations.py#L71)。
- 因此，窗口内从站到蹲的降高信号确实存在；整段已坐着且脚下地形未知时，头轨迹本身不直接给出“离当前脚下地面 1.2m”。已观察的远地板可以给上下文，但不能自动认作脚下同高度。原世界海拔也不能普遍当作离地高度。

## 当前是软条件，加显式软校正

不仅有 loss/条件编码。当前 [config.json:204–214](../../native30_gait_v2_ego55k_oct15/config.json#L204) 开启 observation anchor gain=.9、joint anchor gain=.9、IK 6 次；物理引导仍开 2 次。

1. 观测轨迹通过置信度编码输入网络：[native30_model.py:90–107](../../../native30_model.py#L90)。reference 用观测和 rest 骨架构造，可在初始化时几乎精确对齐 head，但它是可修改的先验：[native30_refinement.py:67–116](../../../native30_refinement.py#L67)。
2. 每个 proposal/final x0 有显式头部误差平移。系数为 `alpha_bar * .9 * available * confidence * .05²/(.05²+sigma²)`；不把 head orientation 精确赋值给身体：[native30_refinement.py:144–165](../../../native30_refinement.py#L144)、[native30_model.py:249](../../../native30_model.py#L249)、[同文件:282](../../../native30_model.py#L282)。
3. 再有 6 次阻尼的观测位置 IK，按可观测祖先链/uncertainty 加权，不读取地板：[native30_observation_ik.py:14–51](../../../native30_observation_ik.py#L14)。
4. 物理候选另有**硬接受条件**：可信观测不能比修正前恶化超过 2mm。这是保护误差的上界，不是把 head 精确锁到观测。可信要求 sigma<5cm；EgoBody 当前 mount 下界约6.95cm，不满足该条件：[native30_physical_guidance.py:76–79](../../../native30_physical_guidance.py#L76)、[同文件:138–143](../../../native30_physical_guidance.py#L138)。

最终采样没有调用旧 `project_head_pose` 的精确投影。[native30_refinement.py:235–252](../../../native30_refinement.py#L235) 虚拟 clean 的软校正实际压得很紧，不能仅凭接近零的误差认定代码做了硬赋值。

## 为何 TRUMANS clean 约0.001mm，而 EgoBody 约50mm

以下单位均为 **mm**，同一固定 10000 步快照，按有有效 head 观测帧计算平均欧氏距离。GT 仅用于评分；关节由 CPU NumPy FK 重算。

| 固定窗 | 观测 head → GT | 预测 head → 观测 | 预测 head → GT |
|---|---:|---:|---:|
| TRUMANS clean | <0.001 | 0.001 | 0.001 |
| EgoBody clean | 41.555 | 9.228 | 49.848 |
| RICH clean | <0.001 | 0.002 | 0.002 |
| TRUMANS noisy | 23.873 | 0.922 | 23.532 |
| EgoBody noisy | 73.645 | 14.609 | 82.678 |
| RICH noisy | 15.886 | 3.556 | 17.886 |

虚拟 camera 的实际 mount 在导出时声明 position/rotation uncertainty=0：[native30_export.py:219–221](../../../native30_export.py#L219)。真实 EgoBody PV 使用 TRAIN 冻结近似 mount，position_p90=0.0694848m：[TRAIN_camera_mounts.json:25](../../../data/native30_official_memory_v3/TRAIN_camera_mounts.json#L25)。即使 clean 未额外加噪，也有该 uncertainty 下界：[native30_observations.py:135–138](../../../native30_observations.py#L135)。在 alpha=1、完整置信度时，单次头部平移系数分别为虚拟 **.9**、EgoBody **.3070**。

这组 EgoBody clean 的观测本身已距 GT 41.6mm，预测约50mm不能全归为模型不遵循 head；其预测距实际观测约9.2mm。`clean` 代表没有额外模拟扰动，不代表真实 PV→解剖头部转换与人体拟合 GT 完全一致。上述标量是直接测量，不能按相加关系拆解误差向量。

## 历史硬约束失败的确切边界

旧方案同时精确设 head rotation，并用 root translation 精确对齐 head position：[hard_head_coordination.py:12–23](../../../hard_head_coordination.py#L12)。历史验收记载它保留错误的 pelvis 转向，把误差转移到颈部扭转和脚滑；850/906 的头相对颈 p95 为106.9°/141.6°，后续全身求解仍未通过腿部联合验收：[历史 ASSESSMENT:11–24](../../../archive/historical_trials_oct16/hierarchical_constraint_oct11/ASSESSMENT_zh.md#L11)。

这支持“不应通过锁头掩盖全身错误”，但该实验**没有单独隔离精确 head position 的作用**，不能推论任何位置硬约束都必然失败。当前也没有精确 head rotation 投影。当前快照未记录 raw residual、IK 前后、物理前后的全部中间 motion，不能据此确定某个模块导致腿伸直。

## 复现

在 SceneMI 仓库根目录运行：

```bash
/home/wenxin/miniconda3/envs/scenemi/bin/python experiments/offline_camera_retrain_v1/runs/native30_live_monitor_oct16/gait_review_current/head_review.py
```

默认只读 `FROZEN_VALIDATION.npz`、配置及源码，将 JSON 输出到 stdout；加 `--output PATH` 才写指定文件。[HEAD_CONSTRAINT_REVIEW.json](HEAD_CONSTRAINT_REVIEW.json) 包含快照/源码 SHA256、实际开关和全部数值。脚本不加载模型、不用 GPU、不改训练。NumPy FK 已与现有 CPU `supervision.forward_kinematics` 在12组已存 motion 上比对，最大坐标差 **0.000358mm**。
