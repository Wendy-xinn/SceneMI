# EXO 场景融合只读审计

结论：目前不能把 noisy yoga 的失败归因于“局部查询错了”或“物理反馈过强”，也没有发现已经坐实的错轴/米与半米转换错误。**第一轮 proposal 在局部查询前，就已经同时受全局 CLS、静态 patch attention、动态对象 attention 影响。** 下一步应当在原 10k 权重上逐支隔离，不继续加训练步数。本次没有运行额外推理、训练或 GPU 任务，没有更改封存源码、权重、历史计划。

## 实际融合路径

| 分支 | 实际代码入口 | 注入位置 | 现有独立开关及限制 |
|---|---|---|---|
| 全局 occupancy CLS | `native30_model.py:138–147` → `model/mdm_scene_unet.py:879–897` | ViT 的 CLS 输出直接加到 `t_emb`；经 U-Net condition MLP，在各尺度残差块以 AdaGN scale/shift 调制 | 没有 CLS 专用公开开关；`patch_enabled=False` **仍保留 CLS** |
| 静态 patch attention | `native30_model.py:163–170` | 以当前 noisy/reference 或 proposal 生成 22 个关节 query，读取 16 个场景 patch；投影后加到 64 维 observation/camera/reference 条件 | `patch_enabled=False` 只关这一支 |
| 动态 owner summary attention | `native30_model.py:171–180` | 当前帧可见对象中心/范围/速度等 token，经 attention 后也加到同一 64 维条件 | 没有 dynamic 专用开关；不受 `patch_enabled` 控制 |
| 局部几何与关系 | `native30_model.py:193,251–254`、`native30_scene_relations.py:91–129` | proposal FK/皮肤锚点查询后形成 region condition，加进逐帧 32 维特征，同时输入 BPS | `local_enabled=False` 关局部查询、region/BPS，以及随后反馈；保留前三支 |
| 学习的显式反馈 | `native30_model.py:267–282` | 在 raw motion 后施加有界 root 和 SO(3) 增量，再由 FK 重建关节 | `feedback_enabled=False` 仅关此项，局部条件与 BPS 仍在 |

`use_scene=False` 是以上多分支的总开关；若分析物理 guidance 原本开启，也会同时绕过该项。当前 10k 配置的解析物理 guidance 已关闭。`native30_overnight.evaluate(scene_modes=...)` 只是 full/off 的调用标签，不是分支选择机制。原 backbone 的 `attention=False` 指 U-Net 内部 temporal attention 配置，不会关闭外部 `patch_attention`、`dynamic_attention`、`surface_attention` 或 `relation_attention`。

第一遍 `_core_pass` 在 `native30_model.py:246` 运行，局部查询到 `251` 才开始；refinement 在 `native30_refinement.py:230` 用 `reference + noisy residual` 形成这一遍的 patch/dynamic query。CLS 不依赖该 query，直接参与同一次 core。第二遍仍然使用相同 CLS，并把 query 改成第一遍 proposal。因此“原始 first-core 已经坏了”不能靠关闭最终反馈直接证明或排除全局场景问题。

## 已核对的坐标和尺度

- EXO 轨迹位置和 EXO 场景点都用同一 `origin/anchor` 变换，见 `native30_observations.py:95,102,108`。轨迹除以 2 保存，场景点保持米；局部查询用 `trajectory[..., :3] * 2` 恢复米，FK/skin points 也为米。
- `occupancy_from_points` 在 `unified_geometry.py:190–196` 以 `[Y,X,Z]` 存储，X/Z 范围为 `[-6.4,6.4)` 米，Y 为 `[-3.1,1.7)` 米，中心 Y=-0.7。ViT 展平 `(h,w)` 对应 `(Xpatch,Zpatch)`；valid reshape 和 patch centres 的顺序一致，未发现 X/Z 交换。
- canonical surface cache 在 `native30_surface_cache.py:50,68,84–87` 以及 `YawSurface` 中把场景变换回当前窗口 yaw。局部 delta 使用相同坐标系和旋转，静态核对未见缺一次 yaw 或重复 yaw。

这只是代码路径核对，不能代替逐源 bundle 的外参/重建数据检查，也不能证明所有序列的场景与人体已经正确对齐。

## 需要验证的具体风险

1. CLS 直接进入全层 AdaGN；静态/dynamic 场景输出与 observation/reference 条件直接相加。前三支没有显式参考保持约束，也没有按 reference 置信度、人体到 patch 距离或 diffusion timestep 限制场景注入幅度。可靠 reference 能被另一条条件路径覆盖；这条通路确实存在，但尚未证明哪支触发 yoga。
2. “空 patch=UNKNOWN”的输出 mask 在 ViT transformer **之后** 才应用。空 patch 的 embedding/位置编码此前仍参与 token/CLS 计算，CLS 本身也没有 observed-coverage gate。因此外部场景覆盖方式变化可能形成全局先验偏移，尚无分支隔离或数据分布测量证明它就是根因。
3. 当前坐标原点是观察相机；EGO 训练与 EXO 的相机距离、位置、场景覆盖可能明显不同。二者坐标仍可各自数学一致，但全局 occupancy 的统计分布不一定一致。这属于待核实的分布问题，不能直接当成数据错位。

## 最小下一步，只讨论，尚未执行

使用原始 natural10k `step_010000.pt`，SHA 为 `3247383297ca2d7129755c6cd96c82eb2009cb2758aa8e22b6e5d0acb04c66b5`。保持 `use_scene=True`、原输入/参考/场景/种子、局部查询/BPS/region/feedback 和解析物理 guidance 配置不变，分别声明三种单支对照：

1. **CLS off**：进程内包装 `model.scene_tokens`，只把返回 tuple 的 summary 替换为 `zeros_like(summary)`，保留 tokens、valid、dynamic。必须在 `sample_refinement` 填充 `_sampling_scene` 缓存前安装。
2. **static patch off**：暂时设 `model.patch_enabled=False`，结束后恢复；不会关闭全局 CLS 或局部场景。
3. **dynamic summary off**：在 `model.dynamic_condition` 上挂临时 forward hook，返回同形状零张量。保留原始动态点云在局部查询中的作用。

复用 `ops/scene_feedback_policy.py` 的 process-scoped extension/contextmanager 模式，接入现有 frozen paired evaluator，扩展到一个可复用入口；不修改封存模型文件，不另造每个案例脚本。hooks/flags 用 finally 恢复，采样前后要求 `_sampling_scene` 为空。三种单支都不解释时，再声明“关闭前三支、保留局部场景”的组合对照，必要时再区分仅第一 core 和两遍 core，不同时堆改动。

验收沿用已经声明的三数据集 clean/noisy VAL 身份与相同 DDIM seeds，不只看 yoga：报告遮挡/可见关节误差、比 reference 更差的窗口数、低速 GT 足部速度、加速度，并读取第一 core raw/FK 诊断。推理开关不得读取 GT；GT 只用于已有指标。单个坏例好转也不能直接作为最终版或论文收益。

## 已有 150+150 对照的边界

现有 paired VAL 的 noisy yoga 隐藏关节误差为：原 10k **585.78 mm**，control150 **582.14 mm**，nondegradation150 **581.46 mm**；scene OFF 分别 **68.83 / 69.55 / 67.79 mm**。nondegradation 的 scene ON 足速由原版 **0.522** 增到 **0.551 m/s**，加速度误差由 **7139.5** 增到 **7375.0 mm/s²**。因此新 loss 没有修复这个故障，也不能因 TRAIN excess-error 下降而宣称方案有效。数据来自已有 `paired_validation_cuda/DIRECTION_REVIEW.json`，本次未重算。

精确源码 SHA、行号、seal 匹配结果、已有指标来源 SHA、已证事实与未验证边界见同目录 `SCENE_FUSION_AUDIT.json`。本报告不提升任何候选 checkpoint。
