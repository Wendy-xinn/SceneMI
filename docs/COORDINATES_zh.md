# 相机、头部与人体表示约定

**新 AMASS 部署约束（2026-10-21 用户确认）：EGO 只有场景和相机轨迹，无身高/人体标定；EXO 另有估计的 3D 人体/关节。** 新模型已实现观测到 16D 形状的预测；真实 rest/betas 只作标签。相机转头部使用预测形状的外参，不读取目标 betas。阶段 A 关闭场景；AMASS EXO 没有真实外部相机，明确标记相机不可用。

## 固定的是什么

保留虚拟相机轨迹作为输入。写 `T_a_b` 表示把 b 坐标转换到 a；位姿中的平移以米计。

```text
T_world_camera = T_world_head @ T_head_camera
T_world_head   = T_world_camera @ inverse(T_head_camera)
T_world_CPF    = T_world_camera @ inverse(T_head_camera) @ T_head_CPF
```

固定的是**一个标定 profile 内、头部局部坐标下**的 `T_head_camera`。世界坐标偏移随头部旋转：`p_camera = p_head + R_head @ t_head_camera`。不同人体形状、SMPL/H/X 模板或真实相机设备不能无条件共用同一个数值。

既有虚拟相机使用头部局部眼睛偏移，加最终相机光轴 **+Z 方向的 2 cm** clearance。若 face 旋转为 `R_hc`，完整局部平移是 `eye_offset_head + 0.02 * R_hc[:,2]`。眼睛偏移已处于头部坐标，不能再乘一次 face 旋转。实现见 `scenemi/geometry/camera.py`，已测试非单位 face 旋转下的正反变换。

现有 raw30 导出从前至多 20 帧的原生网格眼睛位置反算并取中位数，随后固定，实际数值写在 `actual_virtual_mount`；已有缓存以该元数据为准，不重算或替换。新 AMASS 接口为每个原生形状从 rest 网格眼睛位置建立固定 profile，同样采用光轴和 clearance 约定；这是明确的新标定方式，不保证数值与旧 posed-median 标定逐位相同。真实设备外参须由设备标定或 TRAIN 数据估计并冻结，不用 VAL/TEST 动作重新拟合。

如果位置偏移非零而某帧相机旋转缺失，就不能唯一恢复头部位置。既有 `native30_observations.inverse_head` 会降低相应可用性；不能用干净 GT 旋转悄悄补齐。

## 与 EgoAllo 的 CPF 对齐

EgoAllo 使用 SMPL-H 双眼四个顶点的中点和头部方向作为 CPF。其静态外参来自形状化 rest 网格，**没有本仓库的 2 cm clearance**。它使用的 root 排除式 joint index 14 对应本仓库含 root 的 head index 15。

保留本仓库的相机输入并通过上式转换 CPF，就能给对比方法提供同一传感器信息；CPF 转换不能额外读目标动作真值。要显式声明使用了已知人体形状或哪种标定 profile，两种方法保持一致。真实设备若没有标定，不能靠这个公式自动得到准确 CPF。

## SMPL、SMPL-H、SMPL-X 如何共存

共同模型输出仅覆盖含 pelvis 的前 22 个身体关节，使用共享名称、父子层级和局部 SO(3)。这允许同一主干处理三种模型，但不表示模型参数和网格可以直接互换。

| 部分 | 处理方式 |
|---|---|
| 身体关节旋转 | 核对前 22 个关节的层级后共享；包含 root 朝向 |
| rest 骨架和骨长 | 用各自原生模型及形状回归。旧模型逐样本输入真值；新模型须用预测形状得到 rest，真实 rest 只作监督，不套另一模型的骨架 |
| 形状系数 | AMASS SMPL-H 此处完整保留 16 维；已有 SMPL/X 管线用其声明的 10 维。不同模型的 betas 不直接复制或静默截断 |
| 手、脸、顶点和 skin anchors | 留在原生拓扑，使用对应模型的索引、回归器与权重；不属于当前 22 关节预测 |
| body conditioning | 既有 69D 接口：rest 66 + 类型 2 + 尺度 1。SMPL-H 的身体层映射至 SMPL family，数据 profile 仍明确记为 smplh |

人体参数有兼容路径，但完整网格跨模型转换需要另行拟合/重定向。当前新接口支持 SMPL-H rest/相机/201D 动作，不声称原有 SMPL/X skin cache 已自动支持 SMPL-H 全网格。

## 世界坐标与 201D 动作

统一使用右手 Y-up、米；AMASS Z-up 世界转换 `(x,y,z) -> (x,z,-y)`。转换作用于世界 pelvis 位置及 root 全局旋转，不能把所有局部关节旋转和原生 rest 骨骼都乘一遍。

201D 的布局保持既有模型：`3 root + 22*6 local_rotation + 22*3 global_joint`。位置通道除以 2，旋转 6D 是旋转矩阵前两**列**相对单位旋转的残差。

旧数据/模型用 `(world_pelvis - native_rest_pelvis)/2` 作为 root。本次 AMASS 使用新的 `201d_common_neutral_pelvis_v1`：设 native SMPL-H 模板的零形状 pelvis 为固定 `r0`，按 `rest = native_rest(beta) - native_rest(beta)[0] + r0` 平移整个 rest，root 为 `(world_pelvis - r0)/2`。真实与预测骨架的根参考因此相同，骨长及局部相机偏移仍来自各自原生形状。FK 世界关节不变，root 表示不再依赖未知真实 betas。实现见 `scenemi/geometry/shape.py` 与 `scenemi/data/amass_dataset.py`。

向原生 SMPL-H 导出时先恢复 `world_pelvis = root*2 + r0`，再计算 `native_trans = world_pelvis - predicted_native_rest[0]`；不能把新 root 直接当作原生 `trans`。此处应使用预测 native pelvis，不能取测试真值。旧数据封印和既有模型继续使用旧语义，两种表示不可混用。

窗口原点与 heading anchor、输入相机、GT 和后续场景必须使用同一刚体变换。所有随机 heading/平移增强都作用于它们的世界部分；形状与头部局部外参不随窗口重新拟合。

预测 FK 与目标 FK 分别使用 `details['rest']` 与 `batch['target_rest']`，两者已经放到同一个固定 pelvis 参考下。回归检查覆盖“局部旋转相同、骨长不同”的情况，避免误用预测骨架解码真值而消掉形状误差。形状、相机偏移在每个窗口内固定；长序列跨窗口的形状一致性仍需推理策略和评估，不能假定各窗口预测完全一致。
